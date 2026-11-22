"""Analyze existing results without retraining."""

import csv
from collections import defaultdict
from statistics import mean

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from .paths import result_path, write_result_rows


FULL_MODEL = "logistic_full"
ABLATED_MODEL = "logistic_no_frequency"
CONDITIONS = (
    "jpeg_85",
    "jpeg_60",
    "jpeg_35",
    "resize_128",
    "resize_64",
    "resize_128_then_jpeg_60",
)


def analyze_scores() -> None:
    """AUC and paired score changes."""
    groups = defaultdict(dict)

    with result_path("test_predictions.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = (row["model"], row["condition"])
            image_id = row["image_id"]
            if image_id in groups[key]:
                raise ValueError(f"Duplicate prediction : {key} / {image_id}")
            groups[key][image_id] = (int(row["label"]), float(row["score_ai"]))

    if len(groups) != 21:
        raise ValueError(f"21 model-condition combinations expected, {len(groups)} found.")

    results = []
    for (model, condition), predictions in groups.items():
        reference = groups[model, "reference"]
        if len(reference) != 1200 or predictions.keys() != reference.keys():
            raise ValueError(f"Unmatched images : {model} / {condition}")

        labels = []
        scores = []
        human_changes = []

        for image_id, (label, score) in predictions.items():
            reference_label, reference_score = reference[image_id]
            if label != reference_label:
                raise ValueError(f"Label differs from the reference : {image_id}")

            labels.append(label)
            scores.append(score)
            if label == 0:
                human_changes.append(score - reference_score)

        if labels.count(0) != 600 or labels.count(1) != 600:
            raise ValueError(f"Expected 600 images per class : {model} / {condition}")

        results.append(
            {
                "model": model,
                "condition": condition,
                "roc_auc": roc_auc_score(labels, scores),
                "mean_human_ai_score_change": mean(human_changes),
            }
        )

    output = result_path("test_score_analysis.csv")
    write_result_rows(output, results)

    print("Model | Condition | ROC-AUC | Mean AI-score change (Human)")
    for row in results:
        print(
            f"{row['model']} | {row['condition']} | "
            f"{row['roc_auc']:.4f} | {row['mean_human_ai_score_change']:+.4f}"
        )
    print(f"Results saved : {output.name}")


def analyze_flips() -> None:
    """Exact McNemar tests and Holm correction."""
    with result_path("robustness_summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    reference_counts = {
        row["model"]: int(row["false_positives"]) for row in rows if row["condition"] == "reference"
    }
    comparisons = [row for row in rows if row["condition"] != "reference"]
    keys = {(row["model"], row["condition"]) for row in comparisons}

    if len(rows) != 21 or len(reference_counts) != 3 or len(comparisons) != 18 or len(keys) != 18:
        raise ValueError("Expected three reference conditions and 18 unique comparisons.")

    results = []
    for row in comparisons:
        model = row["model"]
        condition = row["condition"]
        new = int(row["new_false_positives"])
        resolved = int(row["resolved_false_positives"])
        total_flips = new + resolved

        if min(new, resolved) < 0 or total_flips > 600:
            raise ValueError(f"Invalid flip count : {model} / {condition}")
        if reference_counts[model] + new - resolved != int(row["false_positives"]):
            raise ValueError(f"Inconsistent totals : {model} / {condition}")

        p_value = (
            binomtest(new, total_flips, p=0.5, alternative="two-sided").pvalue
            if total_flips
            else 1.0
        )
        results.append(
            {
                "model": model,
                "condition": condition,
                "new_false_positives": new,
                "resolved_false_positives": resolved,
                "total_flips": total_flips,
                "delta_fpr_pp": 100 * (new - resolved) / 600,
                "mcnemar_p": p_value,
            }
        )

    ranked = sorted(results, key=lambda row: row["mcnemar_p"])
    previous = 0.0
    for rank, row in enumerate(ranked):
        adjusted = min(1.0, (len(ranked) - rank) * row["mcnemar_p"])
        previous = max(previous, adjusted)
        row["holm_adjusted_p"] = previous

    output = result_path("false_positive_flip_tests.csv")
    write_result_rows(output, results)

    print("Model | Condition | New FP | Resolved FP | Δ FPR (pp) | Adjusted p")
    for row in results:
        print(
            f"{row['model']} | {row['condition']} | "
            f"{row['new_false_positives']} | {row['resolved_false_positives']} | "
            f"{row['delta_fpr_pp']:+.2f} | {row['holm_adjusted_p']:.4g}"
        )
    print(f"Results saved : {output.name}")


def analyze_ablation() -> None:
    """Paired contrast of FPR changes with bootstrap intervals."""
    predictions = defaultdict(dict)
    with result_path("test_predictions.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["label"] != "0" or row["model"] not in (FULL_MODEL, ABLATED_MODEL):
                continue

            key = (row["model"], row["condition"])
            image_id = row["image_id"]
            decision = int(row["predicted_ai"])
            if image_id in predictions[key] or decision not in (0, 1):
                raise ValueError(f"Invalid or duplicate prediction : {key} / {image_id}")
            predictions[key][image_id] = decision

    image_ids = sorted(predictions[FULL_MODEL, "reference"])
    if len(image_ids) != 600:
        raise ValueError("Expected 600 human artworks.")
    expected_ids = set(image_ids)

    def decisions(model, condition):
        values = predictions[model, condition]
        if values.keys() != expected_ids:
            raise ValueError(f"Unmatched images : {model} / {condition}")
        return np.array([values[image_id] for image_id in image_ids])

    full_reference = decisions(FULL_MODEL, "reference")
    without_reference = decisions(ABLATED_MODEL, "reference")
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(image_ids), size=(2000, len(image_ids)))

    results = []
    for condition in CONDITIONS:
        full_change = decisions(FULL_MODEL, condition) - full_reference
        without_change = decisions(ABLATED_MODEL, condition) - without_reference
        difference = without_change - full_change

        sampled_differences = 100 * difference[indices].mean(axis=1)
        low, high = np.percentile(sampled_differences, [2.5, 97.5])

        results.append(
            {
                "condition": condition,
                "delta_fpr_full_pp": 100 * full_change.mean(),
                "delta_fpr_without_frequency_pp": 100 * without_change.mean(),
                "difference_without_minus_full_pp": 100 * difference.mean(),
                "ci_low_pp": float(low),
                "ci_high_pp": float(high),
            }
        )

    output = result_path("frequency_ablation_comparison.csv")
    write_result_rows(output, results)

    print("Condition | Without-frequency minus full-model difference | 95% CI")
    for row in results:
        print(
            f"{row['condition']} | {row['difference_without_minus_full_pp']:+.2f} pp | "
            f"[{row['ci_low_pp']:+.2f}, {row['ci_high_pp']:+.2f}] pp"
        )
    print(f"Results saved : {output.name}")


def main() -> None:
    analyze_scores()
    analyze_flips()
    analyze_ablation()


if __name__ == "__main__":
    main()
