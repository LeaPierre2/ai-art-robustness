"""Full inventory and count summary."""

import argparse
import csv
from collections import Counter

from PIL import Image

from .paths import DATASET_DIR, open_result, result_path


VALID_STYLES = {
    "art_nouveau",
    "baroque",
    "expressionism",
    "impressionism",
    "post_impressionism",
    "realism",
    "renaissance",
    "romanticism",
    "surrealism",
    "ukiyo_e",
}


def parse_folder_name(folder_name: str) -> tuple[str, str]:
    """Decode source and normalize style, rejecting unknown classes."""
    if folder_name.startswith("AI_LD_"):
        source = "Latent Diffusion"
        style = folder_name.removeprefix("AI_LD_")
    elif folder_name.startswith("AI_SD_"):
        source = "Stable Diffusion"
        style = folder_name.removeprefix("AI_SD_")
    else:
        source = "Human"
        style = folder_name

    if style == "ukiyo-e":
        style = "ukiyo_e"
    if style not in VALID_STYLES:
        raise ValueError(f"Unexpected class directory : {folder_name!r}")

    return source, style


def build_inventory() -> None:
    """Inventory images and validate class directories."""

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    rows = []

    for split in ("train", "test"):
        for folder in sorted((DATASET_DIR / split).iterdir()):
            if not folder.is_dir():
                continue

            source, style = parse_folder_name(folder.name)
            image_paths = sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in image_extensions
            )

            for image_path in image_paths:
                rows.append((image_path.relative_to(DATASET_DIR), split, source, style))

    counts = Counter()
    jpeg_profiles = {}
    jpeg_counts = Counter()
    unreadable = []

    inventory_path = result_path("image_inventory.csv")

    with open_result(inventory_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "path",
                "original_split",
                "source",
                "style",
                "format",
                "width",
                "height",
                "color_mode",
                "file_size_bytes",
                "jpeg_profile",
                "readable",
                "error",
            ]
        )

        for path, split, source, style in rows:
            full_path = DATASET_DIR / path
            file_size = ""
            profile = ""

            try:
                file_size = full_path.stat().st_size
                with Image.open(full_path) as image:
                    image.load()
                    counts[source, image.format, image.width, image.height] += 1

                    if image.format == "JPEG" and image.quantization:
                        tables = tuple(
                            (table_id, tuple(values))
                            for table_id, values in sorted(image.quantization.items())
                        )
                        profile = jpeg_profiles.setdefault(tables, len(jpeg_profiles) + 1)
                        jpeg_counts[source, profile] += 1

                    writer.writerow(
                        [
                            path,
                            split,
                            source,
                            style,
                            image.format,
                            image.width,
                            image.height,
                            image.mode,
                            file_size,
                            profile,
                            True,
                            "",
                        ]
                    )
            except OSError as error:
                unreadable.append((path, error))
                writer.writerow(
                    [
                        path,
                        split,
                        source,
                        style,
                        "",
                        "",
                        "",
                        "",
                        file_size,
                        "",
                        False,
                        str(error),
                    ]
                )

    print("Source | Format | Dimensions | Image count")

    for (source, image_format, width, height), count in sorted(counts.items()):
        print(f"{source} | {image_format} | {width} × {height} | {count}")

    print(f"\nImages examined : {len(rows)}")
    print(f"Unreadable images : {len(unreadable)}")

    for path, error in unreadable:
        print(f"{path} | {error}")

    print("\nSource | JPEG profile | Image count")

    for (source, profile), count in sorted(jpeg_counts.items()):
        print(f"{source} | {profile} | {count}")


def summarize_inventory() -> None:
    """Display counts by split, style and source."""
    inventory_path = result_path("image_inventory.csv")
    counts = Counter()
    styles = set()

    with inventory_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            style = row["style"]
            styles.add(style)
            if row["readable"] == "True":
                counts[row["original_split"], style, row["source"]] += 1

    for split in ("train", "test"):
        print(f"\n{split.upper()}")
        print(f"{'Style':20} {'Human':>8} {'Latent Diffusion':>17} {'Stable Diffusion':>17}")

        for style in sorted(styles):
            human = counts[split, style, "Human"]
            latent = counts[split, style, "Latent Diffusion"]
            stable = counts[split, style, "Stable Diffusion"]
            print(f"{style:20} {human:8} {latent:17} {stable:17}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-only", action="store_true", help="Display counts from the existing inventory."
    )
    args = parser.parse_args()
    if not args.summary_only:
        build_inventory()
    summarize_inventory()


if __name__ == "__main__":
    main()
