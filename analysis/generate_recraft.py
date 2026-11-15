"""Generate images through the Recraft API only on explicit command."""

import base64
import csv
import json
from datetime import datetime, timezone
from getpass import getpass
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from .paths import RECRAFT_DIR


MODEL = "recraftv4_1_flash_v2"
SIZE = "1024x1024"
ENDPOINT = "https://external.api.recraft.ai/v1/images/generations"

PROMPTS = (
    ("art_nouveau", "Art Nouveau poster illustration of a woman holding lilies beneath an ornate floral arch"),
    ("baroque", "Baroque oil painting of a still life with a silver bowl, grapes, and a candle"),
    ("expressionism", "Expressionist painting of a crowded city street at dusk"),
    ("impressionism", "Impressionist oil painting of people walking beside a river in spring"),
    ("post_impressionism", "Post-Impressionist painting of a hillside village with cypress trees"),
    ("realism", "Realist oil painting of a tailor working beside a window"),
    ("renaissance", "Renaissance oil portrait of a merchant holding a sealed letter"),
    ("romanticism", "Romantic landscape painting of a ship approaching a rocky coast in a storm"),
    ("surrealism", "Surrealist painting of an empty railway platform floating above a desert"),
    ("ukiyo_e", "Ukiyo-e woodblock print of travelers crossing a bridge in heavy rain"),
)


def generate_pair(token: str, prompt: str) -> list[bytes]:
    """Explicitly request two PNGs from the API and check format and dimensions."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "size": SIZE,
        "n": 2,
        "response_format": "b64_json",
        "image_format": "png",
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=120) as response:
        result = json.load(response)

    images = [base64.b64decode(item["b64_json"]) for item in result["data"]]
    if len(images) != 2:
        raise ValueError("Recraft did not return two images.")

    for content in images:
        with Image.open(BytesIO(content)) as image:
            if image.format != "PNG" or image.size != (1024, 1024):
                raise ValueError("Unexpected format or dimensions.")
            image.load()

    return images


def main() -> None:
    RECRAFT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []

    for index, (style, prompt) in enumerate(PROMPTS, start=1):
        paths = [
            RECRAFT_DIR / f"{index:02d}_{style}_{number:02d}.png"
            for number in (1, 2)
        ]
        existing = sum(path.exists() for path in paths)
        if existing == 1:
            raise SystemExit(f"Incomplete pair for {style}. Check the directory before retrying.")
        if existing == 0:
            jobs.append((style, prompt, paths))

    if not jobs:
        print("All 20 expected images are already present.")
        return

    print(f"Images to generate : {2 * len(jobs)}")
    token = getpass("Paste your Recraft API token, then press Enter: ").strip()
    if not token:
        raise SystemExit("Empty token.")

    log_path = RECRAFT_DIR / "generation_log.csv"
    needs_header = not log_path.exists() or log_path.stat().st_size == 0

    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if needs_header:
            writer.writerow([
                "filename", "style", "prompt", "model",
                "requested_size", "generated_at_utc",
            ])

        for style, prompt, paths in jobs:
            print(f"Generating: {style}", flush=True)
            try:
                images = generate_pair(token, prompt)
            except HTTPError as error:
                raise SystemExit(f"Recraft HTTP error {error.code} for {style}.")
            except (URLError, TimeoutError):
                raise SystemExit(f"Connection interrupted for {style}. Check your credit balance before retrying.")

            generated_at = datetime.now(timezone.utc).isoformat()
            for path, content in zip(paths, images):
                with path.open("xb") as file:
                    file.write(content)
                writer.writerow([path.name, style, prompt, MODEL, SIZE, generated_at])

            log_file.flush()
            print(f"Two PNGs saved for {style}.")

    print("All 20 images are available in data/recraft/.")


if __name__ == "__main__":
    main()
