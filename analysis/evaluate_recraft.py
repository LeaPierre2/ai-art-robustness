"""Locally evaluate the 20 Recraft images with saved thresholds."""

from PIL import Image

from .detectors import load_detectors, score_image_batch
from .paths import RECRAFT_DIR, result_path, write_result_rows


LOGISTIC_MODELS = ("logistic_full", "logistic_no_frequency")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def main() -> None:
    if not RECRAFT_DIR.is_dir():
        raise SystemExit("The data/recraft directory is missing. Add the images first.")

    paths = sorted(
        path
        for path in RECRAFT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(paths) != 20:
        raise SystemExit(f"Expected 20 images in data/recraft, {len(paths)} found.")

    logistic_models, cnn, device, thresholds = load_detectors()

    results = []

    for path in paths:
        with Image.open(path) as original:
            if original.width != original.height:
                raise ValueError(f"Non-square image : {path.name}")
            image = original.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)

        scores = score_image_batch([image], logistic_models, cnn, device)
        for name, values in scores.items():
            score = float(values[0])
            threshold = thresholds[name]
            results.append(
                {
                    "image": path.name,
                    "model": name,
                    "score_ai": score,
                    "predicted_ai": int(score >= threshold),
                }
            )

    output_path = result_path("recraft_predictions.csv")
    write_result_rows(output_path, results)

    for name in (*LOGISTIC_MODELS, "cnn"):
        detected = sum(row["predicted_ai"] for row in results if row["model"] == name)
        print(f"{name}: {detected}/20 detected ({detected / 20:.0%})")

    print(f"Results saved : {output_path.name}")


if __name__ == "__main__":
    main()
