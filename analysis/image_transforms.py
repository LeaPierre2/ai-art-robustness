"""Seven deterministic transformations and a training-image preview."""

import csv
from io import BytesIO

from PIL import Image, ImageDraw

from .paths import DATASET_DIR, figure_path, result_path


VARIANT_NAMES = (
    "reference",
    "jpeg_85",
    "jpeg_60",
    "jpeg_35",
    "resize_128",
    "resize_64",
    "resize_128_then_jpeg_60",
)


def recompress_jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Recompress to JPEG with subsampling=2, then return an RGB copy."""
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


def downscale_and_restore(image: Image.Image, side: int) -> Image.Image:
    """Downscale and restore dimensions using LANCZOS in both directions."""
    smaller = image.resize((side, side), Image.Resampling.LANCZOS)
    return smaller.resize(image.size, Image.Resampling.LANCZOS)


def make_variant(image: Image.Image, condition: str) -> Image.Image:
    """Compute only the requested condition using the historical operations.

    This path serves training augmentation. Evaluation keeps make_variants(),
    which shares the 128-pixel resize between two variants.
    """
    reference = image.convert("RGB")
    if condition == "reference":
        return reference
    jpeg_qualities = {"jpeg_85": 85, "jpeg_60": 60, "jpeg_35": 35}
    if condition in jpeg_qualities:
        return recompress_jpeg(reference, jpeg_qualities[condition])
    resize_sides = {"resize_128": 128, "resize_64": 64}
    if condition in resize_sides:
        return downscale_and_restore(reference, resize_sides[condition])
    if condition == "resize_128_then_jpeg_60":
        return recompress_jpeg(downscale_and_restore(reference, 128), 60)
    raise ValueError(f"Unknown transformation condition : {condition!r}")


def make_variants(image: Image.Image) -> dict[str, Image.Image]:
    """Produce seven conditions in prediction order.

    The combined condition downscales/restores before JPEG quality-60 encoding.
    """
    reference = image.convert("RGB")
    resized_128 = downscale_and_restore(reference, 128)

    return {
        "reference": reference,
        "jpeg_85": recompress_jpeg(reference, 85),
        "jpeg_60": recompress_jpeg(reference, 60),
        "jpeg_35": recompress_jpeg(reference, 35),
        "resize_128": resized_128,
        "resize_64": downscale_and_restore(reference, 64),
        "resize_128_then_jpeg_60": recompress_jpeg(resized_128, 60),
    }


def preview() -> None:

    with result_path("dataset_manifest.csv").open(newline="", encoding="utf-8") as file:
        row = next(
            row
            for row in csv.DictReader(file)
            if row["split"] == "train" and row["source"] == "Human"
        )

    with Image.open(DATASET_DIR / row["path"]) as image:
        variants = make_variants(image)

    sheet = Image.new("RGB", (1120, 580), "white")
    draw = ImageDraw.Draw(sheet)
    print(f"Image used : {row['image_id']} | {row['path']}")

    for index, (name, variant) in enumerate(variants.items()):
        if variant.size != (256, 256) or variant.mode != "RGB":
            raise ValueError(f"Unexpected dimensions or mode for {name}")

        x = (index % 4) * 280
        y = (index // 4) * 290
        draw.text((x + 12, y + 8), name, fill="black")
        sheet.paste(variant, (x + 12, y + 28))
        print(f"{name} | {variant.width} × {variant.height} | {variant.mode}")

    sheet.save(figure_path("transform_preview.png"))
    print("Preview saved : transform_preview.png")


if __name__ == "__main__":
    preview()
