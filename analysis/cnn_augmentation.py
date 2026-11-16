"""Compare two saved CNNs without retraining or changing the main pipeline."""

import argparse
import hashlib
import json
import math
import time
from statistics import mean
from typing import Any

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from .detectors import load_cnn
from .evaluate_robustness import predict_test_rows, summarize_decisions
from .image_transforms import VARIANT_NAMES
from .paths import document_path, model_path, read_result_rows, result_path, write_result_rows


def evaluate_checkpoint(
    variant: str, filename: str, seed: int, test_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a fixed threshold on seven conditions in manifest order."""
    checkpoint = torch.load(model_path(filename), map_location="cpu", weights_only=True)
    augmentation = checkpoint.get("augmentation")
    if variant == "cnn_control" and augmentation:
        raise ValueError("The control checkpoint must not use augmentation.")
    if variant == "cnn_augmented" and (
        not augmentation or augmentation["seed"] != seed
        or augmentation["original_probability"] != 0.5
        or augmentation["transformed_probability_each"] != 1 / 12
    ):
        raise ValueError("The augmented checkpoint does not match the requested protocol.")
    cnn, device, threshold = load_cnn(filename)
    labels = np.array([int(row["label"]) for row in test_rows])
    predictions, decisions = predict_test_rows(
        test_rows, {}, cnn, device, {"cnn": threshold}
    )
    summary = summarize_decisions(decisions, labels == 0, labels == 1)
    by_condition = {
        condition: [row for row in predictions if row["condition"] == condition]
        for condition in VARIANT_NAMES
    }
    aucs = {
        condition: float(roc_auc_score(
            [int(row["label"]) for row in rows],
            [row["score_ai"] for row in rows],
        ))
        for condition, rows in by_condition.items()
    }
    original_detection = next(
        row["ai_detection_rate_pct"] for row in summary if row["condition"] == "reference"
    )
    for row in summary:
        row.update(
            model=variant, seed=seed, threshold=threshold,
            roc_auc=aucs[row["condition"]],
            delta_detection_pp=row["ai_detection_rate_pct"] - original_detection,
            delta_auc=aucs[row["condition"]] - aucs["reference"],
        )
    for row in predictions:
        row.update(model=variant, seed=seed, threshold=threshold)
    metadata = {
        "filename": filename,
        "sha256": hashlib.sha256(model_path(filename).read_bytes()).hexdigest(),
        "threshold": threshold,
        "epoch": checkpoint["epoch"],
        "device": str(device),
        "augmentation": augmentation,
    }
    return predictions, summary, metadata


def compare_models(
    predictions: list[dict[str, Any]], summaries: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Compare the same images, with Holm over 14 tests: 7 conditions × 2 classes.

    AUCs and differences in losses across conditions remain descriptive:
    McNemar tests compare the two models within a given condition.
    """
    indexed = {}
    for row in predictions:
        key = row["model"], row["condition"], row["image_id"]
        if key in indexed:
            raise ValueError(f"Duplicate prediction in the experiment : {key}")
        indexed[key] = row
    summary = {(row["model"], row["condition"]): row for row in summaries}
    comparisons = []
    tests = []
    for condition in VARIANT_NAMES:
        control = summary["cnn_control", condition]
        augmented = summary["cnn_augmented", condition]
        row = {"seed": seed, "condition": condition}
        for name, field in (
            ("fpr_pct", "false_positive_rate_pct"),
            ("detection_pct", "ai_detection_rate_pct"),
            ("auc", "roc_auc"),
            ("delta_fpr_from_original_pp", "delta_fpr_pp"),
            ("delta_detection_from_original_pp", "delta_detection_pp"),
            ("delta_auc_from_original", "delta_auc"),
        ):
            row["control_" + name] = control[field]
            row["augmented_" + name] = augmented[field]
            difference_name = {
                "fpr_pct": "fpr_pp", "detection_pct": "detection_pp"
            }.get(name, name)
            row["difference_" + difference_name] = augmented[field] - control[field]
        for label, endpoint in ((0, "human_fp"), (1, "ai_detection")):
            ids = sorted(
                image_id for model, cond, image_id in indexed
                if model == "cnn_control" and cond == condition
                and int(indexed[model, cond, image_id]["label"]) == label
            )
            augmented_ids = {
                image_id for model, cond, image_id in indexed
                if model == "cnn_augmented" and cond == condition
                and int(indexed[model, cond, image_id]["label"]) == label
            }
            if len(ids) != 600 or set(ids) != augmented_ids:
                raise ValueError(f"Expected 600 paired images : {condition}, label {label}.")
            before = np.array([indexed["cnn_control", condition, k]["predicted_ai"] for k in ids])
            after = np.array([indexed["cnn_augmented", condition, k]["predicted_ai"] for k in ids])
            gained = int(np.sum((before == 0) & (after == 1)))
            lost = int(np.sum((before == 1) & (after == 0)))
            p_value = (
                float(binomtest(gained, gained + lost, p=0.5, alternative="two-sided").pvalue)
                if gained + lost else 1.0
            )
            row[endpoint + "_gained_positive"] = gained
            row[endpoint + "_lost_positive"] = lost
            row[endpoint + "_total_flips"] = gained + lost
            row[endpoint + "_mcnemar_p"] = p_value
            tests.append((row, endpoint, p_value))
        comparisons.append(row)
    previous = 0.0
    for rank, (row, endpoint, p_value) in enumerate(sorted(tests, key=lambda test: test[2])):
        previous = max(previous, min(1.0, (len(tests) - rank) * p_value))
        row[endpoint + "_holm_p"] = previous
    return comparisons


def summarize_seeds() -> None:
    """Rebuild the resize synthesis from saved comparisons for seeds 42, 43, 44.

    Detection gain is augmented minus control. Loss attenuation subtracts the
    gain on originals from the gain on resized images. Rates are in percentage
    points. Means and min–max ranges describe training variation on reused test
    images; they do not provide independent samples or confidence intervals.
    """
    seeds = (42, 43, 44)
    synthesis = []
    for seed in seeds:
        rows = read_result_rows(f"cnn_augmentation_comparison_seed_{seed}.csv")
        comparisons = {row["condition"]: row for row in rows}
        if (
            len(rows) != len(VARIANT_NAMES) or set(comparisons) != set(VARIANT_NAMES)
            or any(int(row["seed"]) != seed for row in rows)
        ):
            raise ValueError(f"Seed {seed}: expected seven unique conditions with matching seed.")
        original_gain = float(comparisons["reference"]["difference_detection_pp"])
        for resolution in (128, 64):
            row = comparisons[f"resize_{resolution}"]
            gain = float(row["difference_detection_pp"])
            attenuation = gain - original_gain
            values = dict(
                seed=seed, resolution_px=resolution,
                control_detection_pct=float(row["control_detection_pct"]),
                augmented_detection_pct=float(row["augmented_detection_pct"]),
                gain_original_detection_pp=original_gain, gain_detection_pp=gain,
                attenuation_loss_pp=attenuation,
                control_fpr_pct=float(row["control_fpr_pct"]),
                augmented_fpr_pct=float(row["augmented_fpr_pct"]),
                difference_fpr_pp=float(row["difference_fpr_pp"]),
            )
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Seed {seed}, resize {resolution}: non-finite value.")
            for actual, expected in (
                (gain, values["augmented_detection_pct"] - values["control_detection_pct"]),
                (values["difference_fpr_pp"], values["augmented_fpr_pct"] - values["control_fpr_pct"]),
                (attenuation, float(row["difference_delta_detection_from_original_pp"])),
            ):
                if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Seed {seed}, resize {resolution}: inconsistent saved differences.")
            synthesis.append(values)
    summary = []
    for resolution in (128, 64):
        for metric in ("gain_detection_pp", "attenuation_loss_pp", "difference_fpr_pp"):
            values = [row[metric] for row in synthesis if row["resolution_px"] == resolution]
            signs = ["positive" if value > 1e-12 else "negative" if value < -1e-12 else "zero"
                     for value in values]
            summary.append(dict(
                resolution_px=resolution, metric=metric, n_seeds=len(seeds),
                mean_pp=mean(values), min_pp=min(values), max_pp=max(values),
                sign_changes="positive" in signs and "negative" in signs,
                signs_seed_42_43_44="/".join(signs), interpretation="descriptive_same_test_images",
            ))
    write_result_rows("cnn_augmentation_seed_synthesis.csv", synthesis)
    write_result_rows("cnn_augmentation_seed_summary.csv", summary)
    print("Rebuilt cnn_augmentation_seed_synthesis.csv and cnn_augmentation_seed_summary.csv "
          "from saved seed comparisons (six rows each); no training or inference.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--seed", type=int, default=42)
    mode.add_argument(
        "--summarize-seeds", action="store_true",
        help="Rebuild the two cross-seed CSVs from saved comparisons for seeds 42, 43 and 44.",
    )
    args = parser.parse_args()
    if args.summarize_seeds:
        summarize_seeds()
        return
    start = time.perf_counter()
    test_rows = [row for row in read_result_rows("dataset_manifest.csv") if row["split"] == "test"]
    if (
        len(test_rows) != 1200 or len({row["image_id"] for row in test_rows}) != 1200
        or sum(row["label"] == "0" for row in test_rows) != 600
        or sum(row["label"] == "1" for row in test_rows) != 600
    ):
        raise ValueError("The test set must contain 1,200 unique images, 600 per class.")
    predictions = []
    summaries = []
    metadata = {}
    for variant in ("control", "augmented"):
        filename = f"cnn_augmentation_{variant}_seed_{args.seed}.pt"
        rows, summary, info = evaluate_checkpoint("cnn_" + variant, filename, args.seed, test_rows)
        predictions.extend(rows)
        summaries.extend(summary)
        metadata[variant] = info
    if metadata["control"]["device"] != metadata["augmented"]["device"]:
        raise ValueError("Both CNNs must be evaluated on the same backend.")
    comparisons = compare_models(predictions, summaries, args.seed)
    prefix = f"cnn_augmentation_{{}}_seed_{args.seed}.csv"
    write_result_rows(prefix.format("predictions"), predictions)
    write_result_rows(prefix.format("summary"), summaries)
    write_result_rows(prefix.format("comparison"), comparisons)
    metadata.update(
        seed=args.seed,
        manifest_sha256=hashlib.sha256(result_path("dataset_manifest.csv").read_bytes()).hexdigest(),
        evaluation_seconds=time.perf_counter() - start,
        # Retain historical serialized wording: exact two-sided McNemar, joint Holm;
        # AUCs and differences of changes are descriptive, without an interaction test.
        paired_tests="Exact two-sided McNemar tests with Holm correction across all 14 tests (7 conditions × FPR/TPR)",
        descriptive="AUC, changes relative to original images, and differences in those changes are descriptive",
    )
    document_path(f"CNN_AUGMENTATION_SEED_{args.seed}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Condition | FPR control → augmented | Detection control → augmented | AUC control → augmented")
    for row in comparisons:
        print(
            f"{row['condition']} | {row['control_fpr_pct']:.2f} → {row['augmented_fpr_pct']:.2f} | "
            f"{row['control_detection_pct']:.2f} → {row['augmented_detection_pct']:.2f} | "
            f"{row['control_auc']:.4f} → {row['augmented_auc']:.4f}"
        )
    print(f"Evaluation and comparison completed in {metadata['evaluation_seconds']:.2f} s.")


if __name__ == "__main__":
    main()
