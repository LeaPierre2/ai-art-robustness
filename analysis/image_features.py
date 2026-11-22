"""Visual features and pilot/training-validation feature tables."""

import argparse
import csv
from collections import Counter

import numpy as np
from PIL import Image

from .paths import DATASET_DIR, open_result, result_path, write_result_rows


def extract_features(image: Image.Image) -> dict[str, float]:
    """Compute 55 features in historical order: 48 color, 4 edge statistics,
    then 3 frequency-energy shares after removing the grayscale mean.

    Key order matches the feature table and saved models.
    """
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb)
    gray = np.asarray(rgb.convert("L"), dtype=np.float64) / 255.0
    features = {}

    for index, channel in enumerate(("red", "green", "blue")):
        histogram, _ = np.histogram(pixels[:, :, index], bins=16, range=(0, 256))
        for bin_index, share in enumerate(histogram / histogram.sum()):
            features[f"color_{channel}_{bin_index:02d}"] = float(share)

    for axis, direction in ((1, "horizontal"), (0, "vertical")):
        differences = np.abs(np.diff(gray, axis=axis))
        features[f"edge_{direction}_mean"] = float(differences.mean())
        features[f"edge_{direction}_std"] = float(differences.std())

    # Remove mean brightness before measuring spatial variation.
    spectrum = np.fft.fft2(gray - gray.mean())
    energy = np.abs(spectrum) ** 2
    total_energy = float(energy.sum())

    fy = np.fft.fftfreq(gray.shape[0])[:, None]
    fx = np.fft.fftfreq(gray.shape[1])[None, :]
    radius = np.hypot(fx, fy)

    bands = {
        "low": radius < 0.10,
        "mid": (radius >= 0.10) & (radius < 0.25),
        "high": radius >= 0.25,
    }
    for name, mask in bands.items():
        features[f"freq_{name}"] = float(energy[mask].sum() / total_energy) if total_energy else 0.0

    return features


def preview() -> None:
    counts = Counter()
    examples = []

    with result_path("dataset_manifest.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            source = row["source"]
            if row["split"] != "train" or counts[source] >= 10:
                continue

            with Image.open(DATASET_DIR / row["path"]) as image:
                features = extract_features(image)

            if len(features) != 55 or not np.isfinite(list(features.values())).all():
                raise ValueError(f"Invalid features for {row['image_id']}")

            examples.append(
                {
                    "image_id": row["image_id"],
                    "source": source,
                    "label": row["label"],
                    **features,
                }
            )
            counts[source] += 1

            if len(examples) == 20:
                break

    if counts != {"Human": 10, "Latent Diffusion": 10}:
        raise ValueError(f"Unexpected pilot counts : {dict(counts)}")

    output = result_path("feature_preview.csv")
    write_result_rows(output, examples)

    print(f"Images processed : {len(examples)}")
    print(f"By source : {dict(counts)}")
    print("Features per image: 55 (48 color, 4 edge, 3 frequency)")
    print(f"CSV saved : {output.name}")


def build_table() -> None:
    """Write 55 features for the 4,800 training and validation images."""
    output = result_path("train_validation_features.csv")
    expected_counts = {"train": 4200, "validation": 600}
    metadata_columns = ["image_id", "split", "source", "style", "label"]

    with result_path("dataset_manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] in expected_counts]

    split_counts = Counter(row["split"] for row in rows)
    if split_counts != expected_counts:
        raise ValueError(f"Unexpected counts : {dict(split_counts)}")

    with open_result(output, "w", newline="", encoding="utf-8") as file:
        writer = None

        for index, row in enumerate(rows, start=1):
            with Image.open(DATASET_DIR / row["path"]) as image:
                features = extract_features(image)

            if len(features) != 55 or not np.isfinite(list(features.values())).all():
                raise ValueError(f"Invalid features for {row['image_id']}")

            if writer is None:
                writer = csv.DictWriter(file, fieldnames=metadata_columns + list(features))
                writer.writeheader()

            metadata = {column: row[column] for column in metadata_columns}
            writer.writerow({**metadata, **features})

            if index % 1000 == 0:
                print(f"Images processed : {index}/{len(rows)}", flush=True)

    print(f"Training: {split_counts['train']} images")
    print(f"Validation : {split_counts['validation']} images")
    print(f"Total: {len(rows)} images, 55 features per image")
    print(f"CSV saved : {output.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="55 visual features: preview or training/validation table."
    )
    parser.add_argument(
        "--table", action="store_true", help="Compute the 4,800 training/validation rows."
    )
    args = parser.parse_args()
    if args.table:
        build_table()
    else:
        preview()


if __name__ == "__main__":
    main()
