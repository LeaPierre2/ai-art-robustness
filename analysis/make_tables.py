"""Markdown tables from existing CSVs, without re-evaluation."""

from .paths import document_path, read_result_rows


MODELS = {
    "logistic_full": "Logistic regression",
    "logistic_no_frequency": "Logistic regression without frequencies",
    "cnn": "CNN",
}

CONDITIONS = {
    "reference": "Reference",
    "jpeg_85": "JPEG 85",
    "jpeg_60": "JPEG 60",
    "jpeg_35": "JPEG 35",
    "resize_128": "Resize via 128 px",
    "resize_64": "Resize via 64 px",
    "resize_128_then_jpeg_60": "Resize via 128 px + JPEG 60",
}


def by_model_and_condition(rows):
    return {(row["model"], row["condition"]): row for row in rows}


def main() -> None:
    metadata = read_result_rows("jpeg_metadata_baseline.csv")
    robustness = by_model_and_condition(read_result_rows("robustness_summary.csv"))
    scores = by_model_and_condition(read_result_rows("test_score_analysis.csv"))
    flip_tests = by_model_and_condition(read_result_rows("false_positive_flip_tests.csv"))
    ablation = read_result_rows("frequency_ablation_comparison.csv")

    lines = [
        "# Report tables",
        "",
        "## JPEG metadata baselines on the reference test set",
        "",
        "| Input | Test ROC-AUC |",
        "|---|---:|",
    ]
    baseline_names = {
        "jpeg_profile": "JPEG quantization profile",
        "file_size_bytes": "File size",
    }
    for row in metadata:
        name = baseline_names[row["baseline"]]
        lines.append(f"| {name} | {float(row['test_roc_auc']):.4f} |")

    lines.extend(
        [
            "",
            "## Detection performance by model and transformation",
            "",
            "The test set contains 600 human artworks and 600 AI-generated images.",
            "Decision thresholds were fixed on the validation set.",
            "",
            "| Model | Condition | FPR (%) | Change in FPR (pp) | "
            "AI detection (%) | ROC-AUC | Holm-adjusted p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, model_name in MODELS.items():
        for condition, condition_name in CONDITIONS.items():
            key = (model, condition)
            row = robustness[key]
            p_value = "—"
            if condition != "reference":
                p_value = f"{float(flip_tests[key]['holm_adjusted_p']):.4g}"
            lines.append(
                f"| {model_name} | {condition_name} | "
                f"{float(row['false_positive_rate_pct']):.2f} | "
                f"{float(row['delta_fpr_pp']):+.2f} | "
                f"{float(row['ai_detection_rate_pct']):.2f} | "
                f"{float(scores[key]['roc_auc']):.4f} | {p_value} |"
            )

    lines.extend(
        [
            "",
            "## Frequency-feature ablation",
            "",
            "The contrast is the change in FPR without frequency features minus "
            "the change in FPR with frequency features. Confidence intervals are "
            "paired bootstrap intervals for each condition, without multiplicity adjustment.",
            "",
            "| Condition | Difference in FPR changes (pp) | 95% CI (pp) |",
            "|---|---:|---:|",
        ]
    )
    for row in ablation:
        name = CONDITIONS[row["condition"]]
        difference = float(row["difference_without_minus_full_pp"])
        low = float(row["ci_low_pp"])
        high = float(row["ci_high_pp"])
        lines.append(f"| {name} | {difference:+.2f} | [{low:+.2f}, {high:+.2f}] |")

    output = document_path("report_tables.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Three tables saved to {output.name}")


if __name__ == "__main__":
    main()
