"""Manifest construction and verification, independently runnable."""

import argparse
import csv
import random
from collections import Counter, defaultdict
from statistics import median, quantiles

from PIL import Image, ImageDraw, ImageOps

from .paths import DATASET_DIR, figure_path, open_result, read_result_rows, result_path


def build_manifest() -> None:
    """Select 6,000 images with seed 42 and the historical quotas."""
    inventory_path = result_path("image_inventory_deduplicated.csv")
    output_path = result_path("dataset_manifest.csv")

    labels = {"Human": 0, "Latent Diffusion": 1}
    quotas = {"train": 210, "validation": 30, "test": 60}
    rng = random.Random(42)

    groups = defaultdict(list)
    styles = set()

    with inventory_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["source"] in labels and row["readable"] == "True":
                source = row["source"]
                style = row["style"]
                styles.add(style)
                groups[row["original_split"], source, style].append(row)

    if len(styles) != 10:
        raise ValueError(f"Expected 10 styles, {len(styles)} found.")

    manifest = []
    counts = Counter()
    train_pool_size = quotas["train"] + quotas["validation"]

    for style in sorted(styles):
        for source, label in labels.items():
            train_pool = sorted(
                groups["train", source, style], key=lambda row: row["path"]
            )
            test_pool = sorted(
                groups["test", source, style], key=lambda row: row["path"]
            )

            if len(train_pool) < train_pool_size or len(test_pool) < quotas["test"]:
                raise ValueError(f"Insufficient images for {source} / {style}.")

            train_sample = rng.sample(train_pool, train_pool_size)
            selections = {
                "train": train_sample[:quotas["train"]],
                "validation": train_sample[quotas["train"]:],
                "test": rng.sample(test_pool, quotas["test"]),
            }

            for split, selected_rows in selections.items():
                for row in selected_rows:
                    manifest.append({
                        "image_id": f"img_{len(manifest) + 1:06d}",
                        "path": row["path"],
                        "source": source,
                        "style": style,
                        "label": label,
                        "original_split": row["original_split"],
                        "split": split,
                    })
                    counts[split, source, style] += 1

    paths = {row["path"] for row in manifest}
    if len(paths) != len(manifest):
        raise ValueError("The same path appears more than once in the manifest.")

    for style in styles:
        for source in labels:
            for split, expected in quotas.items():
                if counts[split, source, style] != expected:
                    raise ValueError(f"Unexpected count for {split} / {source} / {style}.")

    with open_result(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id", "path", "source", "style",
                "label", "original_split", "split",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    totals = Counter(row["split"] for row in manifest)
    for split in quotas:
        print(f"{split}: {totals[split]} images")

    print(f"Total: {len(manifest)} images")
    print(f"Unique paths : {len(paths)}")
    print(f"Manifest saved : {output_path.name}")


def _print_size_statistics(inventory_rows: list[dict[str, str]]) -> None:
    """Describe raw file sizes with the historical quartile definition."""
    sizes = defaultdict(list)
    for row in inventory_rows:
        if row["readable"] == "True":
            sizes[row["source"]].append(int(row["file_size_bytes"]))

    print("FILE SIZES (KiB)")
    print("Source | Q1 | Median | Q3")
    for source, values in sorted(sizes.items()):
        q1, _, q3 = quantiles(values, n=4, method="inclusive")
        print(f"{source} | {q1 / 1024:.1f} | {median(values) / 1024:.1f} | {q3 / 1024:.1f}")


def _verify_exact_exclusions(
    inventory_rows: list[dict[str, str]],
    cleaned_rows: list[dict[str, str]],
    exclusions: list[dict[str, str]],
) -> None:
    """Verify that the log exactly describes the removed paths."""
    inventory_counts = Counter((row["source"], row["original_split"]) for row in inventory_rows)
    cleaned_counts = Counter((row["source"], row["original_split"]) for row in cleaned_rows)

    print("\nCLEANING")
    print("Source | Original split | Before | After | Excluded")
    for source, split in sorted(inventory_counts):
        original_count = inventory_counts[source, split]
        retained_count = cleaned_counts[source, split]
        print(
            f"{source} | {split} | {original_count} | {retained_count} | "
            f"{original_count - retained_count}"
        )

    inventory_paths = {row["path"] for row in inventory_rows}
    cleaned_by_path = {row["path"]: row for row in cleaned_rows}
    excluded_paths = {row["path"] for row in exclusions}

    if inventory_paths - cleaned_by_path.keys() != excluded_paths:
        raise ValueError("The log does not match the removed paths.")
    if any(row["kept_path"] not in cleaned_by_path for row in exclusions):
        raise ValueError("A copy marked as retained is missing from the inventory.")


def _verify_manifest_rows(
    manifest_rows: list[dict[str, str]], cleaned_rows: list[dict[str, str]]
) -> None:
    """Check manifest files, text labels, provenance and quotas."""
    cleaned_by_path = {row["path"]: row for row in cleaned_rows}
    labels = {"Human": "0", "Latent Diffusion": "1"}
    quotas = {"train": 210, "validation": 30, "test": 60}
    styles = {row["style"] for row in cleaned_rows if row["source"] in labels}

    if (
        len(manifest_rows) != 6000
        or len({row["path"] for row in manifest_rows}) != 6000
        or len({row["image_id"] for row in manifest_rows}) != 6000
    ):
        raise ValueError("The manifest must have 6,000 unique paths and IDs.")

    counts = Counter()
    for row in manifest_rows:
        path = row["path"]
        original = cleaned_by_path.get(path)
        if original is None or not (DATASET_DIR / path).is_file():
            raise ValueError(f"Missing image : {path}")
        for field in ("source", "style", "original_split"):
            if row[field] != original[field]:
                raise ValueError(f"Inconsistent field ({field}) : {path}")

        expected_origin = "test" if row["split"] == "test" else "train"
        if row["original_split"] != expected_origin:
            raise ValueError(f"Incorrect original split : {path}")
        if row["label"] != labels.get(row["source"]):
            raise ValueError(f"Incorrect label : {path}")
        if original["readable"] != "True" or (original["width"], original["height"]) != ("256", "256"):
            raise ValueError(f"Unreadable image or unexpected dimensions : {path}")

        counts[row["split"], row["source"], row["style"]] += 1

    expected = {
        (split, source, style): quota
        for split, quota in quotas.items()
        for source in labels
        for style in styles
    }
    if len(styles) != 10 or counts != expected:
        raise ValueError("Quotas by split, source and style do not match.")

    print("\nManifest verified: 6,000 images with valid labels, files and quotas.")


def _render_label_review(cleaned_rows: list[dict[str, str]]) -> None:
    """Render the same first examples from official training data, without resampling."""
    preview_styles = ("art_nouveau", "romanticism", "ukiyo_e")
    sources = ("Human", "Latent Diffusion", "Stable Diffusion")
    examples = {}

    for row in cleaned_rows:
        if row["original_split"] == "train" and row["readable"] == "True":
            examples.setdefault((row["style"], row["source"]), row["path"])

    sheet = Image.new("RGB", (720, 660), "white")
    draw = ImageDraw.Draw(sheet)

    for i, style in enumerate(preview_styles):
        for j, source in enumerate(sources):
            x, y = j * 240, i * 220
            draw.text((x + 10, y + 8), f"{source} | {style}", fill="black")
            with Image.open(DATASET_DIR / examples[style, source]) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (184, 184))
                sheet.paste(thumb, (x + (240 - thumb.width) // 2, y + 28))

    sheet.save(figure_path("label_review.png"))
    print("Grid saved : label_review.png")


def verify_preparation() -> None:
    """Audit existing CSVs, then render the label preview."""
    inventory_rows = read_result_rows("image_inventory.csv")
    cleaned_rows = read_result_rows("image_inventory_deduplicated.csv")
    exclusions = read_result_rows("duplicate_exclusions.csv")
    manifest_rows = read_result_rows("dataset_manifest.csv")

    _print_size_statistics(inventory_rows)
    _verify_exact_exclusions(inventory_rows, cleaned_rows, exclusions)
    _verify_manifest_rows(manifest_rows, cleaned_rows)
    _render_label_review(cleaned_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "verify"))
    stage = parser.parse_args().stage
    if stage == "build":
        build_manifest()
    else:
        verify_preparation()


if __name__ == "__main__":
    main()
