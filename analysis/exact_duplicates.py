"""Detect and handle exact duplicates, then check matching names."""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .paths import DATASET_DIR, open_result, result_path


def detect_duplicates() -> None:
    """Detect exact RGB-pixel duplicates using SHA-256."""
    inventory_path = result_path("image_inventory.csv")
    output_path = result_path("exact_duplicates.csv")

    sources = {"Human", "Latent Diffusion"}
    first_seen = {}
    duplicate_groups = {}
    checked = 0

    with inventory_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["source"] not in sources or row["readable"] != "True":
                continue

            with Image.open(DATASET_DIR / row["path"]) as image:
                rgb = image.convert("RGB")
                fingerprint = (
                    rgb.width,
                    rgb.height,
                    hashlib.sha256(rgb.tobytes()).hexdigest(),
                )

            entry = (
                row["path"],
                row["original_split"],
                row["source"],
                row["style"],
            )

            if fingerprint in duplicate_groups:
                duplicate_groups[fingerprint].append(entry)
            elif fingerprint in first_seen:
                duplicate_groups[fingerprint] = [first_seen[fingerprint], entry]
            else:
                first_seen[fingerprint] = entry

            checked += 1
            if checked % 10000 == 0:
                print(f"Images checked : {checked:,}", flush=True)

    cross_split = 0
    cross_source = 0

    with open_result(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["group_id", "path", "original_split", "source", "style"])

        for group_id, entries in enumerate(duplicate_groups.values(), start=1):
            splits = {entry[1] for entry in entries}
            group_sources = {entry[2] for entry in entries}

            if len(splits) > 1:
                cross_split += 1
            if len(group_sources) > 1:
                cross_source += 1

            for entry in entries:
                writer.writerow([group_id, *entry])

    print(f"\nImages checked : {checked:,}")
    print(f"Exact-duplicate groups : {len(duplicate_groups):,}")
    print(f"Groups present in both train and test : {cross_split:,}")
    print(f"Groups containing Human and Latent Diffusion : {cross_source:,}")
    print(f"Results saved to {output_path.name}")


def exclude_duplicates() -> None:
    """Prefer the test copy, then the minimum path, without deleting images."""
    groups = defaultdict(list)

    with result_path("exact_duplicates.csv").open(
        newline="", encoding="utf-8"
    ) as csv_file:
        for row in csv.DictReader(csv_file):
            groups[row["group_id"]].append(row)

    exclusions = {}

    for group_id, entries in groups.items():
        labels = {(entry["source"], entry["style"]) for entry in entries}
        if len(labels) != 1:
            raise ValueError(f"Different labels in group {group_id}")

        test_entries = [entry for entry in entries if entry["original_split"] == "test"]
        kept = min(test_entries or entries, key=lambda entry: entry["path"])

        for entry in entries:
            if entry["path"] != kept["path"]:
                exclusions[entry["path"]] = {
                    **entry,
                    "kept_path": kept["path"],
                    "reason": "exact_pixel_duplicate",
                }

    kept_count = 0
    removed_count = 0

    with result_path("image_inventory.csv").open(
        newline="", encoding="utf-8"
    ) as source_file, open_result(
        result_path("image_inventory_deduplicated.csv"), "w", newline="", encoding="utf-8"
    ) as output_file:
        reader = csv.DictReader(source_file)
        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            if row["path"] in exclusions:
                removed_count += 1
            else:
                writer.writerow(row)
                kept_count += 1

    with open_result(
        result_path("duplicate_exclusions.csv"), "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "group_id", "path", "original_split", "source", "style",
                "kept_path", "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(exclusions.values())

    print(f"Excluded rows : {removed_count}")
    print(f"Rows retained in the full inventory : {kept_count}")


def check_matching_names() -> None:
    """Check human-artwork names shared by train/test, without exclusions."""
    inventory_path = result_path("image_inventory_deduplicated.csv")
    images_by_name = defaultdict(list)

    with inventory_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["source"] == "Human" and row["readable"] == "True":
                name = Path(row["path"]).stem.casefold()
                images_by_name[name].append(row)

    matches = [
        entries
        for entries in images_by_name.values()
        if {row["original_split"] for row in entries} == {"train", "test"}
    ]

    print(f"Names present in both train and test : {len(matches)}")

    for entries in matches[:5]:
        print()
        for row in entries:
            print(row["path"])


def main() -> None:
    detect_duplicates()
    exclude_duplicates()
    check_matching_names()


if __name__ == "__main__":
    main()
