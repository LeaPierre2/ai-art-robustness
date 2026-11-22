"""Descriptive generalization to Stable Diffusion, without training or recalibration."""

import hashlib
import json
import platform
import random
import time
from collections import Counter, defaultdict
from io import BytesIO
from typing import Any

import numpy as np
import PIL
import sklearn
import torch
from PIL import Image, JpegImagePlugin
from sklearn.metrics import roc_auc_score

from .detectors import load_detectors, score_image_batch
from .evaluate_robustness import BATCH_SIZE
from .image_transforms import VARIANT_NAMES, make_variants
from .inventory import VALID_STYLES, parse_folder_name
from .paths import (
    DATASET_DIR,
    document_path,
    model_path,
    read_result_rows,
    result_path,
    write_result_rows,
)


SEED = 42
SOURCES = ("Human", "Latent Diffusion", "Stable Diffusion")
PREFIX = "stable_diffusion"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reference_rgb(image: Image.Image) -> Image.Image:
    """Decode RGB; resize to 256² with LANCZOS only if needed, without JPEG encoding."""
    rgb = image.convert("RGB")
    if rgb.size != (256, 256):
        rgb = rgb.resize((256, 256), Image.Resampling.LANCZOS)
    return rgb


def image_record(path: str) -> dict[str, Any]:
    """Hash file bytes, native pixels and reference pixels before variants."""
    data = (DATASET_DIR / path).read_bytes()
    with Image.open(BytesIO(data)) as image:
        native = image.convert("RGB")
        reference = reference_rgb(image)
        return dict(
            native_width=image.width,
            native_height=image.height,
            native_format=image.format,
            native_mode=image.mode,
            file_sha256=sha256(data),
            native_rgb_sha256=sha256(native.tobytes()),
            reference_rgb_sha256=sha256(reference.tobytes()),
            resized_to_256=image.size != (256, 256),
        )


def audit_sd_folders() -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Read all 62,923 SD file headers; do not infer dimensions from filenames."""
    candidates = defaultdict(list)
    counts = Counter()
    totals = Counter()
    for split in ("train", "test"):
        folders = sorted((DATASET_DIR / split).glob("AI_SD_*"))
        if len(folders) != 10 or {parse_folder_name(p.name)[1] for p in folders} != VALID_STYLES:
            raise ValueError(f"Expected ten SD style directories in {split}.")
        for folder in folders:
            source, style = parse_folder_name(folder.name)
            if source != "Stable Diffusion":
                raise ValueError("Non-SD directory in the complementary inventory.")
            paths = sorted(
                p
                for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            for path in paths:
                with Image.open(path) as image:
                    table_sha = sha256(
                        json.dumps(getattr(image, "quantization", {}), sort_keys=True).encode()
                    )
                    sampling = JpegImagePlugin.get_sampling(image) if image.format == "JPEG" else -1
                    key = (
                        split,
                        style,
                        image.width,
                        image.height,
                        image.format,
                        image.mode,
                        sampling,
                        table_sha,
                    )
                    counts[key] += 1
                if split == "test":
                    candidates[style].append(str(path.relative_to(DATASET_DIR)))
                totals[split] += 1
            print(f"SD directory checked : {split}/{folder.name}, {len(paths)} files", flush=True)
    if totals != {"train": 52923, "test": 10000} or any(
        len(paths) != 1000 for paths in candidates.values()
    ):
        raise ValueError(
            f"SD counts differ from the expected AI-ArtBench distribution : {dict(totals)}."
        )
    fields = (
        "original_split",
        "style",
        "width",
        "height",
        "format",
        "mode",
        "subsampling",
        "quantization_sha256",
    )
    audit = [dict(zip(fields, key), n=n) for key, n in sorted(counts.items())]
    return dict(candidates), audit


def select_manifest() -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]
):
    """Take a prefix of a permutation per style; exclude exact copies before quotas.

    Hashes are compared to the 4,800 training/validation images and 1,200 test
    references, then to SD images already selected. Copies can match by file
    bytes, native RGB pixels or normalized reference pixels.
    """
    candidates, folder_audit = audit_sd_folders()
    original = read_result_rows("dataset_manifest.csv")
    if (
        len(original) != 6000
        or len({r["image_id"] for r in original}) != 6000
        or Counter(r["split"] for r in original) != {"train": 4200, "validation": 600, "test": 1200}
        or {r["source"] for r in original} != {"Human", "Latent Diffusion"}
    ):
        raise ValueError("Unexpected main manifest or SD already present.")
    file_index = {}
    native_index = {}
    reference_index = {}
    manifest = []
    for i, row in enumerate(original, 1):
        record = image_record(row["path"])
        if (record["native_width"], record["native_height"]) != (256, 256):
            raise ValueError("Main references must already be 256 × 256.")
        if record["native_rgb_sha256"] != record["reference_rgb_sha256"]:
            raise ValueError("Pixels of images already at 256² changed.")
        file_index[record["file_sha256"]] = row
        native_index[
            record["native_width"], record["native_height"], record["native_rgb_sha256"]
        ] = row
        reference_index[record["reference_rgb_sha256"]] = row
        if row["split"] == "test":
            manifest.append(
                {
                    **row,
                    **record,
                    "selection_seed": "",
                    "reference_filter": "RGB; resize only if needed: LANCZOS; no JPEG",
                }
            )
        if i % 1000 == 0:
            print(f"Main duplicate index : {i}/6000", flush=True)
    rng = random.Random(SEED)
    exclusions = []
    for style in sorted(candidates):
        pool = sorted(candidates[style])
        rng.shuffle(pool)
        accepted = 0
        for path in pool:
            try:
                record = image_record(path)
            except OSError as error:
                exclusions.append(
                    dict(
                        path=path,
                        style=style,
                        reason="unreadable",
                        matched_path="",
                        matched_split="",
                        detail=str(error),
                    )
                )
                continue
            match = None
            reason = ""
            for reason, index, key in (
                ("exact_file_bytes", file_index, record["file_sha256"]),
                (
                    "exact_native_rgb",
                    native_index,
                    (record["native_width"], record["native_height"], record["native_rgb_sha256"]),
                ),
                ("exact_reference_256_rgb", reference_index, record["reference_rgb_sha256"]),
            ):
                match = index.get(key)
                if match is not None:
                    break
            if match is not None:
                exclusions.append(
                    dict(
                        path=path,
                        style=style,
                        reason=reason,
                        matched_path=match["path"],
                        matched_split=match["split"],
                        detail="",
                    )
                )
                continue
            selected = dict(
                image_id=f"sd_test_{len(manifest) - 1200 + 1:06d}",
                path=path,
                source="Stable Diffusion",
                style=style,
                label="1",
                original_split="test",
                split="test",
                **record,
                selection_seed=SEED,
                reference_filter="RGB; resize only if needed: LANCZOS; no JPEG",
            )
            manifest.append(selected)
            file_index[record["file_sha256"]] = selected
            native_index[
                record["native_width"], record["native_height"], record["native_rgb_sha256"]
            ] = selected
            reference_index[record["reference_rgb_sha256"]] = selected
            accepted += 1
            if accepted == 60:
                break
        if accepted != 60:
            raise ValueError(f"Fewer than 60 eligible SD images : {style}.")
        print(f"SD selected : {style}, {accepted}", flush=True)
    quotas = Counter((r["source"], r["style"]) for r in manifest)
    if quotas != {(source, style): 60 for source in SOURCES for style in VALID_STYLES}:
        raise ValueError("Quotas of 60 images per source and style are not met.")
    metadata = dict(
        seed=SEED,
        selection="Styles and paths sorted, one random.Random(42); shuffle each style, retain first 60 eligible; no score-based selection",
        checked_train_validation=4800,
        checked_main_test=1200,
        selected_sd=600,
        duplicate_checks=[
            "file bytes",
            "RGB pixels with native dimensions",
            "RGB pixels after common 256x256 normalization",
        ],
        exclusions=len(exclusions),
        exclusion_counts=dict(Counter(r["reason"] for r in exclusions)),
    )
    return manifest, exclusions, folder_audit, metadata


def predict(
    rows: list[dict[str, Any]],
    logistic_models: dict[str, Any],
    cnn: torch.nn.Module,
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize before make_variants; preserve the main test batches.

    The caller evaluates the 1,200 references first, then the 600 SD separately,
    so adding them does not change the last batch of the main reference.
    """
    predictions = []
    pixels = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        variants = []
        for row in batch:
            with Image.open(DATASET_DIR / row["path"]) as image:
                reference = reference_rgb(image)
            if sha256(reference.tobytes()) != row["reference_rgb_sha256"]:
                raise ValueError(f"Reference pixels differ : {row['image_id']}.")
            pair = make_variants(reference)
            if tuple(pair) != VARIANT_NAMES or any(
                im.size != (256, 256) or im.mode != "RGB" for im in pair.values()
            ):
                raise ValueError("All variants must be RGB 256 × 256.")
            if pair["reference"].tobytes() != reference.tobytes():
                raise ValueError("The original condition changed the reference pixels.")
            variants.append(pair)
            for condition, variant in pair.items():
                pixels.append(
                    dict(
                        image_id=row["image_id"],
                        source=row["source"],
                        condition=condition,
                        reference_rgb_sha256=row["reference_rgb_sha256"],
                        variant_rgb_sha256=sha256(variant.tobytes()),
                        width=variant.width,
                        height=variant.height,
                    )
                )
        for condition in VARIANT_NAMES:
            images = [pair[condition] for pair in variants]
            scores_by_model = score_image_batch(images, logistic_models, cnn, torch.device("cpu"))
            for model, values in scores_by_model.items():
                scores = np.asarray(values, dtype=np.float64)
                if not np.isfinite(scores).all():
                    raise ValueError("Non-finite scores.")
                for row, score in zip(batch, scores):
                    predictions.append(
                        dict(
                            image_id=row["image_id"],
                            source=row["source"],
                            style=row["style"],
                            label=row["label"],
                            model=model,
                            condition=condition,
                            score_ai=float(score),
                            threshold=thresholds[model],
                            predicted_ai=int(score >= thresholds[model]),
                            backend="cpu",
                        )
                    )
        print(
            f"Images evaluated ({rows[0]['source']}…) : {start + len(batch)}/{len(rows)}",
            flush=True,
        )
    return predictions, pixels


def summarize(
    predictions: list[dict[str, Any]], manifest: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rates/AUCs and descriptive differences; pairing only within each image."""
    indexed = {(r["model"], r["condition"], r["image_id"]): r for r in predictions}
    if len(indexed) != 1800 * 7 * 3 or len(indexed) != len(predictions):
        raise ValueError("Expected 37,800 unique predictions.")
    groups = {
        source: [r["image_id"] for r in manifest if r["source"] == source] for source in SOURCES
    }
    if any(len(ids) != 600 or len(set(ids)) != 600 for ids in groups.values()):
        raise ValueError("Expected 600 unique images per source.")
    result = []
    for model in ("logistic_full", "logistic_no_frequency", "cnn"):
        summaries = {}
        for condition in VARIANT_NAMES:
            scores = {}
            decisions = {}
            for source, ids in groups.items():
                aligned = [indexed[model, condition, image_id] for image_id in ids]
                if any(r["source"] != source for r in aligned):
                    raise ValueError("Source differs after pairing.")
                scores[source] = np.array([r["score_ai"] for r in aligned])
                decisions[source] = np.array([r["predicted_ai"] for r in aligned])
            row = dict(model=model, condition=condition, human_n=600, ld_n=600, sd_n=600)
            for source, prefix in zip(SOURCES, ("human", "ld", "sd")):
                current = decisions[source]
                original = np.array(
                    [
                        indexed[model, "reference", image_id]["predicted_ai"]
                        for image_id in groups[source]
                    ]
                )
                field = "fpr" if source == "Human" else "detection"
                row[prefix + "_positive_count"] = int(current.sum())
                row[prefix + "_" + field + "_pct"] = float(100 * current.mean())
                row[prefix + "_delta_" + field + "_from_reference_pp"] = float(
                    100 * (current - original).mean()
                )
                row[prefix + "_negative_to_positive"] = int(
                    ((original == 0) & (current == 1)).sum()
                )
                row[prefix + "_positive_to_negative"] = int(
                    ((original == 1) & (current == 0)).sum()
                )
            labels = [0] * 600 + [1] * 600
            for source, prefix in (("Latent Diffusion", "ld"), ("Stable Diffusion", "sd")):
                row["auc_human_vs_" + prefix] = float(
                    roc_auc_score(labels, np.r_[scores["Human"], scores[source]])
                )
            row["sd_minus_ld_detection_pp"] = row["sd_detection_pct"] - row["ld_detection_pct"]
            row["sd_minus_ld_auc"] = row["auc_human_vs_sd"] - row["auc_human_vs_ld"]
            summaries[condition] = row
            result.append(row)
        for row in summaries.values():
            for prefix in ("ld", "sd"):
                row["delta_auc_human_vs_" + prefix + "_from_reference"] = (
                    row["auc_human_vs_" + prefix] - summaries["reference"]["auc_human_vs_" + prefix]
                )
    return result


def main() -> None:
    start = time.perf_counter()
    manifest, exclusions, folder_audit, selection = select_manifest()
    logistic_models, cnn, _, thresholds = load_detectors("cnn_model.pt")
    cnn.to(torch.device("cpu"))
    if any(list(saved["model"].classes_) != [0, 1] for saved in logistic_models.values()):
        raise ValueError("Expected logistic classes in [0, 1] order.")
    predictions, pixels = [], []
    for subset in (manifest[:1200], manifest[1200:]):
        rows, checks = predict(subset, logistic_models, cnn, thresholds)
        predictions.extend(rows)
        pixels.extend(checks)
    summary = summarize(predictions, manifest)
    for name, rows in (
        ("manifest", manifest),
        ("predictions", predictions),
        ("summary", summary),
        ("pixels", pixels),
        ("source_audit", folder_audit),
    ):
        write_result_rows(f"{PREFIX}_{name}.csv", rows)
    write_result_rows(
        f"{PREFIX}_exclusions.csv",
        exclusions,
        fieldnames=["path", "style", "reason", "matched_path", "matched_split", "detail"],
    )
    metadata = dict(
        selection=selection,
        thresholds=thresholds,
        backend="cpu",
        batch_size=BATCH_SIZE,
        reference_rule="RGB decode, LANCZOS resize to 256x256 only if necessary; no JPEG encode before make_variants",
        prediction_rows=len(predictions),
        paired_pixel_checks=len(pixels),
        native_dimensions=dict(
            Counter(f"{r['source']}: {r['native_width']}x{r['native_height']}" for r in manifest)
        ),
        input_sha256={
            "dataset_manifest.csv": sha256(result_path("dataset_manifest.csv").read_bytes())
        },
        model_sha256={
            name: sha256(model_path(name).read_bytes())
            for name in ("cnn_model.pt", "logistic_full.joblib", "logistic_no_frequency.joblib")
        },
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            sklearn=sklearn.__version__,
            pillow=PIL.__version__,
            threads=torch.get_num_threads(),
        ),
        inference="Descriptive only, no hypothesis tests; LD and SD are distinct unpaired groups; Human FPR is shared",
        provenance=dict(
            dataset_url="https://www.kaggle.com/datasets/ravidussilva/real-ai-art",
            primary_paper="https://arxiv.org/html/2412.01512v1",
            archive_version="not recorded",
            sd_model_attribution="Stable Diffusion v2-1 according to dataset authors; not independently recoverable from JPEG pixels",
        ),
        elapsed_seconds=time.perf_counter() - start,
    )
    document_path("STABLE_DIFFUSION_GENERALIZATION.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for row in summary:
        print(
            f"{row['model']} | {row['condition']} | FPR {row['human_fpr_pct']:.2f}% | LD {row['ld_detection_pct']:.2f}% | SD {row['sd_detection_pct']:.2f}% | ΔSD−LD {row['sd_minus_ld_detection_pp']:+.2f} pp"
        )
    print(
        f"Completed in {metadata['elapsed_seconds']:.2f} s ; historical thresholds and models unchanged."
    )


if __name__ == "__main__":
    main()
