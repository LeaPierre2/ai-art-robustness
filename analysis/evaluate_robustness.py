"""Predictions and paired metrics from saved detectors."""

import argparse
import csv
from typing import Any

import numpy as np
import torch
from PIL import Image

from .detectors import load_detectors, score_image_batch
from .image_transforms import make_variants
from .paths import DATASET_DIR, result_path, write_result_rows


BATCH_SIZE = 32


def predict_test_rows(
    test_rows: list[dict[str, str]],
    logistic_models: dict[str, dict[str, Any]],
    cnn: torch.nn.Module,
    device: torch.device,
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[np.bool_]]]:
    """Evaluate in batch → condition → model → image-within-batch order.

    Decisions for each (model, condition) follow test_rows exactly, which
    is required for the subsequent paired bootstrap.
    """
    predictions = []
    decisions = {}

    for start in range(0, len(test_rows), BATCH_SIZE):
        batch = test_rows[start:start + BATCH_SIZE]
        batch_variants = []

        for row in batch:
            with Image.open(DATASET_DIR / row["path"]) as image:
                batch_variants.append(make_variants(image))

        for condition in batch_variants[0]:
            images = [variants[condition] for variants in batch_variants]
            scores_by_model = score_image_batch(images, logistic_models, cnn, device)

            for name, scores in scores_by_model.items():
                threshold = thresholds[name]
                # Compare in float64 to preserve the saved threshold precision.
                scores = np.asarray(scores, dtype=np.float64)
                predicted_ai = scores >= threshold
                decisions.setdefault((name, condition), []).extend(predicted_ai)

                for row, score, prediction in zip(batch, scores, predicted_ai):
                    predictions.append({
                        "image_id": row["image_id"],
                        "source": row["source"],
                        "style": row["style"],
                        "label": row["label"],
                        "model": name,
                        "condition": condition,
                        "score_ai": float(score),
                        "predicted_ai": int(prediction),
                    })

        completed = start + len(batch)
        if completed % 160 == 0 or completed == len(test_rows):
            print(f"Images evaluated : {completed}/{len(test_rows)}", flush=True)
    return predictions, decisions


def summarize_decisions(
    decisions: dict[tuple[str, str], list[np.bool_]],
    human: np.ndarray,
    ai: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute rates, flips and paired CIs on the same 600 human artworks.

    The same 2,000 resamples (seed 42) are reused for every condition.
    Masks follow test order, not the order of prediction rows in the CSV.
    """
    # Use the same indices for each artwork's reference and variant.
    rng = np.random.default_rng(42)
    bootstrap_indices = rng.integers(0, 600, size=(2000, 600))
    summary = []

    for (name, condition), values in decisions.items():
        predicted_ai = np.array(values, dtype=bool)
        reference = np.array(decisions[name, "reference"], dtype=bool)[human]
        current = predicted_ai[human]

        change = current.astype(int) - reference.astype(int)
        bootstrap_changes = 100 * change[bootstrap_indices].mean(axis=1)
        low, high = np.percentile(bootstrap_changes, [2.5, 97.5])

        summary.append({
            "model": name,
            "condition": condition,
            "false_positives": int(current.sum()),
            "false_positive_rate_pct": 100 * current.mean(),
            "ai_detected": int(predicted_ai[ai].sum()),
            "ai_detection_rate_pct": 100 * predicted_ai[ai].mean(),
            "delta_fpr_pp": 100 * change.mean(),
            "delta_fpr_ci_low_pp": float(low),
            "delta_fpr_ci_high_pp": float(high),
            "new_false_positives": int((~reference & current).sum()),
            "resolved_false_positives": int((reference & ~current).sum()),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    seed = parser.parse_args().seed

    suffix = "" if seed == 42 else f"_seed_{seed}"
    cnn_filename = f"cnn_model{suffix}.pt"
    predictions_filename = f"test_predictions{suffix}.csv"
    summary_filename = f"robustness_summary{suffix}.csv"

    with result_path("dataset_manifest.csv").open(newline="", encoding="utf-8") as file:
        test_rows = [
            row for row in csv.DictReader(file)
            if row["split"] == "test"
        ]

    labels = np.array([int(row["label"]) for row in test_rows])
    human = labels == 0
    ai = labels == 1
    if (
        len(test_rows) != 1200
        or human.sum() != 600
        or ai.sum() != 600
        or len({row["image_id"] for row in test_rows}) != 1200
    ):
        raise ValueError("The test set must contain 1,200 unique images, 600 per class.")

    logistic_models, cnn, device, thresholds = load_detectors(cnn_filename)
    print(f"CNN evaluated : {cnn_filename} | Device : {device}", flush=True)

    predictions, decisions = predict_test_rows(
        test_rows, logistic_models, cnn, device, thresholds
    )

    expected_predictions = 1200 * 7 * 3
    if len(predictions) != expected_predictions:
        raise ValueError(f"Unexpected prediction count : {len(predictions)}")
    write_result_rows(predictions_filename, predictions)

    summary = summarize_decisions(decisions, human, ai)
    write_result_rows(summary_filename, summary)

    for name in thresholds:
        print(f"\n{name}")
        print(
            f"{'Condition':26} {'FP/600':>7} {'FPR':>8} "
            f"{'AI detection':>13} {'Δ FPR (pp)':>12}"
        )
        for row in summary:
            if row["model"] == name:
                print(
                    f"{row['condition']:26} {row['false_positives']:7d}"
                    f" {row['false_positive_rate_pct']:7.1f}%"
                    f" {row['ai_detection_rate_pct']:12.1f}%"
                    f" {row['delta_fpr_pp']:+12.2f}"
                )

    print(f"\nPredictions saved : {len(predictions):,} in {predictions_filename}")
    print(f"Summary saved : {len(summary)} rows in {summary_filename}")


if __name__ == "__main__":
    main()
