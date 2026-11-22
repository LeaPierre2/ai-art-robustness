"""Paired re-encoding experiment using observed Human and LD JPEG tables.

Separate pipeline command: no training or threshold recalibration.
Both JPEGs start from the same decoded RGB, never from the other condition.
"""

import hashlib
import json
import platform
import time
from collections import Counter
from io import BytesIO
from typing import Any

import numpy as np
import PIL
import sklearn
import torch
from PIL import Image, JpegImagePlugin, features as pillow_features
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from .detectors import load_detectors, score_image_batch
from .evaluate_robustness import BATCH_SIZE
from .paths import (
    DATASET_DIR,
    document_path,
    model_path,
    read_result_rows,
    result_path,
    write_result_rows,
)


PROFILES = {"tables_human": "Human", "tables_ld": "Latent Diffusion"}
CONDITIONS = ("reference", *PROFILES)
# For each on-wire zigzag position, the index in the flattened 8 × 8 matrix.
ZIGZAG_TO_NATURAL = (
    0,
    1,
    8,
    16,
    9,
    2,
    3,
    10,
    17,
    24,
    32,
    25,
    18,
    11,
    4,
    5,
    12,
    19,
    26,
    33,
    40,
    48,
    41,
    34,
    27,
    20,
    13,
    6,
    7,
    14,
    21,
    28,
    35,
    42,
    49,
    56,
    57,
    50,
    43,
    36,
    29,
    22,
    15,
    23,
    30,
    37,
    44,
    51,
    58,
    59,
    52,
    45,
    38,
    31,
    39,
    46,
    53,
    60,
    61,
    54,
    47,
    55,
    62,
    63,
)
ENCODER_SETTINGS = {
    "format": "JPEG",
    "quality": -1,
    "subsampling": 2,
    "optimize": False,
    "progressive": False,
    "smooth": 0,
    "keep_rgb": False,
    "exif": b"",
    "icc_profile": None,
    "comment": b"",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jpeg_headers(data: bytes) -> list[tuple[int, bytes]]:
    """Read segments before entropy-coded data, including SOS, without decoding pixels."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("Missing JPEG signature.")
    segments = []
    position = 2
    while position < len(data):
        if data[position] != 0xFF:
            raise ValueError("Expected a JPEG marker before SOS.")
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        length = int.from_bytes(data[position : position + 2], "big")
        if length < 2 or position + length > len(data):
            raise ValueError("Truncated JPEG segment.")
        segments.append((marker, data[position + 2 : position + length]))
        position += length
        if marker == 0xDA:
            return segments
    raise ValueError("Missing SOS segment.")


def raw_quantization_tables(headers: list[tuple[int, bytes]]) -> dict[int, list[int]]:
    """Explicitly convert DQT zigzag order to natural order, as Pillow does."""
    tables = {}
    for marker, payload in headers:
        if marker != 0xDB:
            continue
        position = 0
        while position < len(payload):
            descriptor = payload[position]
            precision = descriptor >> 4
            table_id = descriptor & 15
            if precision not in (0, 1) or table_id in tables:
                raise ValueError("Invalid DQT precision or repeated table.")
            width = precision + 1
            end = position + 1 + 64 * width
            if end > len(payload):
                raise ValueError("Incomplete DQT table.")
            values = [0] * 64
            for wire_index, natural_index in enumerate(ZIGZAG_TO_NATURAL):
                start = position + 1 + wire_index * width
                values[natural_index] = int.from_bytes(payload[start : start + width], "big")
            tables[table_id] = values
            position = end
    return tables


def observed_tables() -> tuple[dict[str, dict[int, list[int]]], dict[str, Any]]:
    """Check the 256 exported coefficients against both source files."""
    exported = read_result_rows("jpeg_quantization_tables.csv")
    profiles = {}
    provenance = {}
    for condition, source in PROFILES.items():
        rows = [row for row in exported if row["source"] == source]
        examples = {(row["jpeg_profile"], row["example_path"]) for row in rows}
        if len(examples) != 1 or len(rows) != 128:
            raise ValueError(f"Expected one profile with two tables of 64 coefficients : {source}.")
        profile, example = examples.pop()
        indexed = {(int(row["table_id"]), int(row["index"])): int(row["value"]) for row in rows}
        if len(indexed) != 128 or set(indexed) != {(t, i) for t in (0, 1) for i in range(64)}:
            raise ValueError(f"Incomplete or repeated DQT indices : {source}.")
        tables = {t: [indexed[t, i] for i in range(64)] for t in (0, 1)}
        encoded = (DATASET_DIR / example).read_bytes()
        with Image.open(BytesIO(encoded)) as image:
            if (
                image.quantization != tables
                or raw_quantization_tables(jpeg_headers(encoded)) != tables
            ):
                raise ValueError(f"Exported tables differ from the source JPEG : {source}.")
        profiles[condition] = tables
        provenance[condition] = dict(
            source=source,
            jpeg_profile=profile,
            example_path=example,
            example_sha256=sha256(encoded),
            tables=tables,
        )
    if profiles["tables_human"] == profiles["tables_ld"]:
        raise ValueError("The two observed profiles must differ.")
    return profiles, provenance


def reencode(
    rgb: Image.Image, tables: dict[int, list[int]]
) -> tuple[Image.Image, bytes, list[tuple[int, bytes]]]:
    """Re-encode without rescaling tables; check DQT, SOF and subsampling.

    quality=-1 prevents scaling by a Pillow quality setting. Verification
    checks the coefficients actually written, not just this option.
    """
    with BytesIO() as buffer:
        rgb.save(buffer, qtables=tables, **ENCODER_SETTINGS)
        encoded = buffer.getvalue()
    headers = jpeg_headers(encoded)
    if raw_quantization_tables(headers) != tables:
        raise ValueError("The written DQT coefficients changed.")
    with Image.open(BytesIO(encoded)) as jpeg:
        if (
            jpeg.quantization != tables
            or jpeg.size != rgb.size
            or jpeg.mode != "RGB"
            or JpegImagePlugin.get_sampling(jpeg) != 2
            or jpeg.layer != [(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)]
            or jpeg.info.get("progressive")
            or jpeg.info.get("progression")
            or jpeg.info.get("exif")
            or jpeg.info.get("icc_profile")
            or not any(marker == 0xC0 for marker, _ in headers)
        ):
            raise ValueError("Output JPEG settings differ from the protocol.")
        return jpeg.convert("RGB"), encoded, headers


def predict(
    rows: list[dict[str, str]], profiles: dict[str, dict[int, list[int]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate all three models at fixed historical thresholds, all conditions on CPU."""
    logistic_models, cnn, _, thresholds = load_detectors("cnn_model.pt")
    device = torch.device("cpu")
    cnn.to(device)
    if any(list(saved["model"].classes_) != [0, 1] for saved in logistic_models.values()):
        raise ValueError("Logistic classes must be ordered [0, 1].")
    predictions = []
    encodings = []
    original_sampling = Counter()
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        variants = []
        for row in batch:
            original_bytes = (DATASET_DIR / row["path"]).read_bytes()
            with Image.open(BytesIO(original_bytes)) as image:
                if (
                    image.quantization
                    != profiles["tables_human" if row["label"] == "0" else "tables_ld"]
                ):
                    raise ValueError(f"Unexpected original JPEG profile : {row['image_id']}.")
                original_sampling[row["source"], JpegImagePlugin.get_sampling(image)] += 1
                rgb = image.convert("RGB")
            input_sha = sha256(rgb.tobytes())
            pair = {"reference": rgb}
            pair_headers = []
            for condition, tables in profiles.items():
                decoded, encoded, headers = reencode(rgb, tables)
                pair[condition] = decoded
                pair_headers.append(
                    [(marker, payload) for marker, payload in headers if marker != 0xDB]
                )
                encodings.append(
                    dict(
                        image_id=row["image_id"],
                        source=row["source"],
                        label=row["label"],
                        condition=condition,
                        input_file_sha256=sha256(original_bytes),
                        input_rgb_sha256=input_sha,
                        encoded_jpeg_sha256=sha256(encoded),
                        decoded_rgb_sha256=sha256(decoded.tobytes()),
                        width=decoded.width,
                        height=decoded.height,
                        subsampling=2,
                        jpeg_bytes=len(encoded),
                        tables_pillow_and_raw_dqt_verified=True,
                    )
                )
            if pair_headers[0] != pair_headers[1] or sha256(rgb.tobytes()) != input_sha:
                raise ValueError("Both encodings must share starting pixels and non-DQT headers.")
            variants.append(pair)
        for condition in CONDITIONS:
            images = [pair[condition] for pair in variants]
            scores_by_model = score_image_batch(images, logistic_models, cnn, device)
            for name, values in scores_by_model.items():
                scores = np.asarray(values, dtype=np.float64)
                if not np.isfinite(scores).all():
                    raise ValueError(f"Non-finite scores : {name}, {condition}.")
                for row, score in zip(batch, scores):
                    predictions.append(
                        dict(
                            image_id=row["image_id"],
                            source=row["source"],
                            style=row["style"],
                            label=row["label"],
                            model=name,
                            condition=condition,
                            score_ai=float(score),
                            predicted_ai=int(score >= thresholds[name]),
                            threshold=thresholds[name],
                            backend="cpu",
                        )
                    )
        print(f"Images evaluated : {start + len(batch)}/{len(rows)}", flush=True)
    metadata = dict(
        thresholds=thresholds,
        cnn_filename="cnn_model.pt",
        backend="cpu",
        batch_size=BATCH_SIZE,
        original_subsampling_counts={
            f"{source}: {sampling}": n for (source, sampling), n in original_sampling.items()
        },
    )
    return predictions, encodings, metadata


def summarize(
    predictions: list[dict[str, Any]], rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Align by image ID; six McNemar tests with a joint Holm correction."""
    indexed = {(r["model"], r["condition"], r["image_id"]): r for r in predictions}
    if len(indexed) != len(predictions) or len(predictions) != 1200 * 3 * 3:
        raise ValueError("Repeated predictions or a count other than 10,800.")
    models = ("logistic_full", "logistic_no_frequency", "cnn")
    labels = np.array([int(row["label"]) for row in rows])
    summary = []
    paired = []
    for model in models:
        scores = {}
        decisions = {}
        for condition in CONDITIONS:
            aligned = [indexed[model, condition, row["image_id"]] for row in rows]
            if any(
                r["label"] != row["label"] or r["source"] != row["source"]
                for r, row in zip(aligned, rows)
            ):
                raise ValueError("Unmatched classes or sources.")
            scores[condition] = np.array([r["score_ai"] for r in aligned])
            thresholds = {r["threshold"] for r in aligned}
            if len(thresholds) != 1:
                raise ValueError("A fixed threshold is required for each model.")
            decisions[condition] = np.array([r["predicted_ai"] for r in aligned])
            if not np.array_equal(
                decisions[condition], scores[condition] >= aligned[0]["threshold"]
            ):
                raise ValueError("Decisions do not match the historical threshold.")
            summary.append(
                dict(
                    model=model,
                    condition=condition,
                    threshold=aligned[0]["threshold"],
                    human_n=600,
                    false_positives=int(decisions[condition][labels == 0].sum()),
                    false_positive_rate_pct=100 * decisions[condition][labels == 0].mean(),
                    ai_n=600,
                    ai_detected=int(decisions[condition][labels == 1].sum()),
                    ai_detection_rate_pct=100 * decisions[condition][labels == 1].mean(),
                    roc_auc=float(roc_auc_score(labels, scores[condition])),
                )
            )
        if len({r["threshold"] for r in summary if r["model"] == model}) != 1:
            raise ValueError("The threshold changed between conditions.")
        for label, source in ((0, "Human"), (1, "Latent Diffusion")):
            mask = labels == label
            delta = scores["tables_ld"][mask] - scores["tables_human"][mask]
            before = decisions["tables_human"][mask]
            after = decisions["tables_ld"][mask]
            gained = int(((before == 0) & (after == 1)).sum())
            lost = int(((before == 1) & (after == 0)).sum())
            p = float(binomtest(gained, gained + lost, p=0.5).pvalue) if gained + lost else 1.0
            paired.append(
                dict(
                    model=model,
                    source=source,
                    label=label,
                    n=600,
                    mean_score_reference=float(scores["reference"][mask].mean()),
                    mean_score_tables_human=float(scores["tables_human"][mask].mean()),
                    mean_score_tables_ld=float(scores["tables_ld"][mask].mean()),
                    mean_delta_score_ld_minus_human=float(delta.mean()),
                    median_delta_score_ld_minus_human=float(np.median(delta)),
                    q25_delta_score=float(np.percentile(delta, 25)),
                    q75_delta_score=float(np.percentile(delta, 75)),
                    min_delta_score=float(delta.min()),
                    max_delta_score=float(delta.max()),
                    score_increased=int((delta > 0).sum()),
                    score_decreased=int((delta < 0).sum()),
                    score_unchanged=int((delta == 0).sum()),
                    human_tables_negative_to_ld_tables_positive=gained,
                    human_tables_positive_to_ld_tables_negative=lost,
                    total_decision_flips=gained + lost,
                    delta_positive_rate_pp=100 * (gained - lost) / 600,
                    mcnemar_exact_two_sided_p=p,
                )
            )
    previous = 0.0
    for rank, row in enumerate(sorted(paired, key=lambda r: r["mcnemar_exact_two_sided_p"])):
        previous = max(previous, min(1.0, (len(paired) - rank) * row["mcnemar_exact_two_sided_p"]))
        row["holm_p"] = previous
    return summary, paired


def main() -> None:
    start = time.perf_counter()
    profiles, provenance = observed_tables()
    rows = [r for r in read_result_rows("dataset_manifest.csv") if r["split"] == "test"]
    if (
        len(rows) != 1200
        or len({r["image_id"] for r in rows}) != 1200
        or Counter((r["source"], r["label"]) for r in rows)
        != Counter({("Human", "0"): 600, ("Latent Diffusion", "1"): 600})
    ):
        raise ValueError("Expected 1,200 unique test images, Human=600 and LD=600.")
    predictions, encodings, metadata = predict(rows, profiles)
    summary, paired = summarize(predictions, rows)
    table_rows = [
        dict(condition=condition, source=PROFILES[condition], table_id=t, index=i, value=v)
        for condition, tables in profiles.items()
        for t, values in tables.items()
        for i, v in enumerate(values)
    ]
    for name, values in (
        ("predictions", predictions),
        ("encodings", encodings),
        ("summary", summary),
        ("paired", paired),
        ("tables", table_rows),
    ):
        write_result_rows(f"jpeg_table_intervention_{name}.csv", values)
    metadata.update(
        manifest_sha256=sha256(result_path("dataset_manifest.csv").read_bytes()),
        exported_tables_sha256=sha256(result_path("jpeg_quantization_tables.csv").read_bytes()),
        model_sha256={
            name: sha256(model_path(name).read_bytes())
            for name in ("cnn_model.pt", "logistic_full.joblib", "logistic_no_frequency.joblib")
        },
        observed_profiles=provenance,
        encoder_settings={
            k: (v.hex() if isinstance(v, bytes) else v) for k, v in ENCODER_SETTINGS.items()
        },
        coefficient_order="Natural row-major 8x8 for Pillow/CSV; raw DQT zigzag independently converted and checked.",
        checks=dict(
            test_images=1200,
            prediction_rows=len(predictions),
            jpeg_pairs=1200,
            encoded_jpegs_checked=len(encodings),
            same_non_dqt_headers_in_each_pair=True,
            same_input_rgb_in_each_pair=True,
            source_tables_checked=True,
        ),
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            sklearn=sklearn.__version__,
            pillow=PIL.__version__,
            jpeg=pillow_features.version("jpg"),
            libjpeg_turbo=pillow_features.version_feature("libjpeg_turbo"),
            threads=torch.get_num_threads(),
        ),
        paired_tests="6 exact two-sided McNemar tests: 3 models x 2 classes, Human-tables vs LD-tables; joint Holm correction.",
        score_changes="Descriptive, LD-tables minus Human-tables; scores in probability units, not percentage points.",
        elapsed_seconds=time.perf_counter() - start,
    )
    document_path("JPEG_TABLE_INTERVENTION.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for row in summary:
        print(
            f"{row['model']} | {row['condition']} | FPR {row['false_positive_rate_pct']:.2f}% | TPR {row['ai_detection_rate_pct']:.2f}% | AUC {row['roc_auc']:.6f}"
        )
    print(
        f"Experiment completed in {metadata['elapsed_seconds']:.2f} s, CPU; no threshold recalibrated."
    )


if __name__ == "__main__":
    main()
