"""JPEG audit: baselines, observed tables and equivalent quality settings."""

import csv
from collections import defaultdict
from io import BytesIO

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from .paths import (
    DATASET_DIR,
    open_result,
    read_result_rows,
    result_path,
    write_result_rows,
)


def audit_baselines() -> None:
    """JPEG-profile and file-size baselines."""

    inventory = {row["path"]: row for row in read_result_rows("image_inventory.csv")}
    manifest = read_result_rows("dataset_manifest.csv")

    train = [row for row in manifest if row["split"] == "train"]
    test = [row for row in manifest if row["split"] == "test"]

    if len(train) != 4200 or len(test) != 1200:
        raise ValueError("Unexpected counts in the manifest.")

    # Learn the AI-image proportion for each JPEG profile on training data.
    labels_by_profile = defaultdict(list)
    for row in train:
        profile = inventory[row["path"]]["jpeg_profile"]
        if not profile:
            raise ValueError(f"Missing JPEG profile : {row['path']}")
        labels_by_profile[profile].append(int(row["label"]))

    profile_scores = {
        profile: float(np.mean(labels)) for profile, labels in labels_by_profile.items()
    }

    # Choose on training data whether larger or smaller file size predicts AI.
    train_labels = [int(row["label"]) for row in train]
    train_sizes = [int(inventory[row["path"]]["file_size_bytes"]) for row in train]
    size_direction = 1 if roc_auc_score(train_labels, train_sizes) >= 0.5 else -1

    test_labels = [int(row["label"]) for row in test]
    test_metadata = [inventory[row["path"]] for row in test]

    results = [
        {
            "baseline": "jpeg_profile",
            "test_roc_auc": roc_auc_score(
                test_labels,
                [profile_scores[row["jpeg_profile"]] for row in test_metadata],
            ),
        },
        {
            "baseline": "file_size_bytes",
            "test_roc_auc": roc_auc_score(
                test_labels,
                [size_direction * int(row["file_size_bytes"]) for row in test_metadata],
            ),
        },
    ]

    output = result_path("jpeg_metadata_baseline.csv")
    write_result_rows(output, results, fieldnames=["baseline", "test_roc_auc"])

    print(f"Images evaluated : {len(test)}")
    for row in results:
        print(f"{row['baseline']} | ROC-AUC test : {row['test_roc_auc']:.4f}")
    print(f"Results saved : {output.name}")


def export_tables() -> None:
    """Export observed tables for each source and profile."""

    examples = {}
    with result_path("image_inventory.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["readable"] == "True" and row["format"] == "JPEG" and row["jpeg_profile"]:
                examples.setdefault((row["source"], row["jpeg_profile"]), row["path"])

    output_path = result_path("jpeg_quantization_tables.csv")
    with open_result(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source", "jpeg_profile", "example_path", "table_id", "index", "value"])

        for (source, profile), path in sorted(examples.items()):
            with Image.open(DATASET_DIR / path) as image:
                tables = image.quantization
                if not tables:
                    raise ValueError(f"Missing JPEG tables for {path}")

                for table_id, values in sorted(tables.items()):
                    writer.writerows(
                        (source, profile, path, table_id, index, value)
                        for index, value in enumerate(values)
                    )
                    print(
                        f"{source} | profile {profile} | table {table_id} | "
                        f"{len(values)} coefficients | min={min(values)} | max={max(values)}"
                    )

    print(f"Tables saved : {output_path.name}")


def compare_tables() -> None:
    """Exact matches to Pillow quality settings 1 through 100."""
    observed = defaultdict(lambda: defaultdict(dict))
    with result_path("jpeg_quantization_tables.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = (row["source"], row["jpeg_profile"])
            table_id = int(row["table_id"])
            index = int(row["index"])
            observed[key][table_id][index] = int(row["value"])

    if not observed:
        raise ValueError("No JPEG tables found in the input file.")

    sample = Image.new("RGB", (8, 8))
    reference_tables = {}
    for quality in range(1, 101):
        with BytesIO() as buffer:
            sample.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as jpeg:
                reference_tables[quality] = jpeg.quantization

    output = result_path("jpeg_quality_equivalents.csv")
    with open_result(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source", "jpeg_profile", "matching_pillow_qualities"])

        for (source, profile), tables in sorted(observed.items()):
            actual = {}
            for table_id, values in tables.items():
                if set(values) != set(range(64)):
                    raise ValueError(
                        f"Incomplete table : {source}, profile {profile}, table {table_id}"
                    )
                actual[table_id] = [values[index] for index in range(64)]

            matches = [
                str(quality)
                for quality, reference in reference_tables.items()
                if reference == actual
            ]
            result = ", ".join(matches) or "No exact match"
            writer.writerow([source, profile, result])
            print(f"{source} | profile {profile} | equivalent Pillow quality : {result}")

    print(f"Results saved : {output.name}")


def main() -> None:
    audit_baselines()
    export_tables()
    compare_tables()


if __name__ == "__main__":
    main()
