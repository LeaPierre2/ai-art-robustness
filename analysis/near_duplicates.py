"""Near-duplicates: detection, review and final-manifest checks."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageDraw, ImageOps

from .paths import DATASET_DIR, figure_path, open_result, result_path


MAX_DISTANCE = 3


def perceptual_hash(path: Path) -> int:
    """Compute 64-bit dHash using grayscale, LANCZOS 9×8 and left > right comparisons."""
    with Image.open(path) as image:
        thumbnail = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(thumbnail.getdata())

    fingerprint = 0
    for row_start in range(0, 72, 9):
        for column in range(8):
            left = pixels[row_start + column]
            right = pixels[row_start + column + 1]
            fingerprint = (fingerprint << 1) | (left > right)

    return fingerprint


def hash_blocks(fingerprint: int) -> Iterator[tuple[int, int]]:
    """Partition 64 bits into MAX_DISTANCE + 1 balanced blocks.

    At most MAX_DISTANCE differences cannot affect every block, so
    every eligible pair shares at least one index key.
    """
    if not isinstance(MAX_DISTANCE, int) or not 0 <= MAX_DISTANCE <= 64:
        raise ValueError("MAX_DISTANCE must be an integer from 0 through 64 for a 64-bit dHash.")
    block_count = MAX_DISTANCE + 1
    base_width, extra_bits = divmod(64, block_count)
    offset = 0
    for block in range(block_count):
        width = base_width + (block < extra_bits)
        # At threshold 64, the last block is empty and makes every pair a candidate.
        yield block, (fingerprint >> offset) & ((1 << width) - 1)
        offset += width


def detect_candidates() -> None:
    """Compare identical dHashes across train/test before selection."""
    inventory_path = result_path("image_inventory_deduplicated.csv")
    output_path = result_path("near_duplicate_candidates.csv")

    rows_by_split = {"train": [], "test": []}

    with inventory_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["source"] in {"Human", "Latent Diffusion"} and row["readable"] == "True":
                rows_by_split[row["original_split"]].append(row)

    test_by_hash = defaultdict(list)

    for row in rows_by_split["test"]:
        fingerprint = perceptual_hash(DATASET_DIR / row["path"])
        test_by_hash[fingerprint].append(row)

    print(f"Test images indexed : {len(rows_by_split['test']):,}", flush=True)

    pair_count = 0

    with open_result(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "train_path", "train_source", "train_style",
            "test_path", "test_source", "test_style",
        ])

        for checked, train_row in enumerate(rows_by_split["train"], start=1):
            fingerprint = perceptual_hash(DATASET_DIR / train_row["path"])

            for test_row in test_by_hash.get(fingerprint, []):
                writer.writerow([
                    train_row["path"], train_row["source"], train_row["style"],
                    test_row["path"], test_row["source"], test_row["style"],
                ])
                pair_count += 1

            if checked % 10000 == 0:
                print(f"Training images checked : {checked:,}", flush=True)

    print(f"Total training images checked : {len(rows_by_split['train']):,}")
    print(f"Candidate pairs between train and test : {pair_count:,}")
    print(f"Results saved to {output_path.name}")


def render_review() -> None:
    """Rebuild the review; without candidates, report this and retain any existing image."""

    with result_path("near_duplicate_candidates.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        pairs = list(csv.DictReader(file))

    if not pairs:
        print(
            "No pairs in near_duplicate_candidates.csv: "
            "no review image to generate. Any existing review image is preserved."
        )
        return

    column_width = 300
    row_height = 300
    sheet = Image.new("RGB", (2 * column_width, len(pairs) * row_height), "white")
    draw = ImageDraw.Draw(sheet)

    for number, pair in enumerate(pairs, start=1):
        y = (number - 1) * row_height

        for column, split in enumerate(("train", "test")):
            x = column * column_width
            draw.text(
                (x + 12, y + 10),
                f"Pair {number} | {split} | {pair[f'{split}_source']}",
                fill="black",
            )

            with Image.open(DATASET_DIR / pair[f"{split}_path"]) as image:
                preview = ImageOps.contain(image.convert("RGB"), (256, 256))
                sheet.paste(preview, (x + (column_width - preview.width) // 2, y + 32))

    sheet.save(figure_path("near_duplicate_review.png"))


def check_manifest() -> None:
    """After selection, check dHash distances across the three splits."""
    with result_path("dataset_manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 6000 or len({row["image_id"] for row in rows}) != 6000:
        raise ValueError("The manifest must contain 6,000 unique images.")

    indexed = defaultdict(list)
    fingerprints = []
    matches = []

    for index, row in enumerate(rows):
        fingerprint = perceptual_hash(DATASET_DIR / row["path"])
        blocks = tuple(hash_blocks(fingerprint))
        candidates = {
            previous
            for block in blocks
            for previous in indexed[block]
            if rows[previous]["split"] != row["split"]
        }

        for previous in sorted(candidates):
            other = rows[previous]
            distance = bin(fingerprint ^ fingerprints[previous]).count("1")
            if distance <= MAX_DISTANCE:
                matches.append([
                    distance,
                    other["image_id"], other["split"], other["source"], other["path"],
                    row["image_id"], row["split"], row["source"], row["path"],
                ])

        for block in blocks:
            indexed[block].append(index)
        fingerprints.append(fingerprint)

        if (index + 1) % 1000 == 0:
            print(f"Images checked : {index + 1}/{len(rows)}", flush=True)

    output = result_path("manifest_near_duplicate_candidates.csv")
    with open_result(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "dhash_distance",
            "image_id_a", "split_a", "source_a", "path_a",
            "image_id_b", "split_b", "source_b", "path_b",
        ])
        writer.writerows(sorted(matches))

    print(f"Candidate pairs across splits : {len(matches)}")
    print(f"Results saved : {output.name}")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inventory", "review", "manifest"))
    stage = parser.parse_args().stage
    if stage == "inventory":
        detect_candidates()
        render_review()
    elif stage == "review":
        render_review()
    else:
        check_manifest()


if __name__ == "__main__":
    main()
