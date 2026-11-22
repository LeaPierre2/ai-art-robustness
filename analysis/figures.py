"""Robustness and Recraft figures, without evaluation or training."""

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .paths import figure_path, result_path


def plot_robustness() -> None:
    """Plot robustness from the existing summary CSV."""

    models = [
        ("logistic_full", "Logistic regression", "#0072B2", "o"),
        ("logistic_no_frequency", "Logistic regression without frequency features", "#D55E00", "s"),
        ("cnn", "CNN", "#008563", "^"),
    ]
    conditions = [
        ("jpeg_85", "JPEG 85"),
        ("jpeg_60", "JPEG 60"),
        ("jpeg_35", "JPEG 35"),
        ("resize_128", "Resize via 128 px"),
        ("resize_64", "Resize via 64 px"),
        ("resize_128_then_jpeg_60", "Resize via 128 px + JPEG 60"),
    ]

    with result_path("robustness_summary.csv").open(newline="", encoding="utf-8") as file:
        rows = {(row["model"], row["condition"]): row for row in csv.DictReader(file)}

    fig, (ax_fpr, ax_detection) = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    positions = list(range(len(conditions)))
    ax_fpr.axvline(0, color="#555555", linestyle="--", linewidth=1)

    for (model, label, color, marker), offset in zip(models, (-0.20, 0, 0.20)):
        results = [rows[model, condition] for condition, _ in conditions]
        y = [position + offset for position in positions]

        changes = [float(row["delta_fpr_pp"]) for row in results]
        lows = [float(row["delta_fpr_ci_low_pp"]) for row in results]
        highs = [float(row["delta_fpr_ci_high_pp"]) for row in results]
        ax_fpr.hlines(y, lows, highs, color=color, linewidth=1.5)
        ax_fpr.plot(changes, y, marker=marker, linestyle="none", color=color, label=label)

        detection = [float(row["ai_detection_rate_pct"]) for row in results]
        reference = float(rows[model, "reference"]["ai_detection_rate_pct"])
        ax_detection.axvline(reference, color=color, linestyle="--", alpha=0.7)
        ax_detection.plot(detection, y, marker=marker, linestyle="none", color=color)

    ax_fpr.set_title("A. Change in false-positive rate")
    ax_fpr.set_xlabel("Change from reference (percentage points)")
    ax_fpr.set_yticks(positions, [label for _, label in conditions])
    ax_fpr.invert_yaxis()

    ax_detection.set_title("B. AI detection rate")
    ax_detection.set_xlabel("AI images correctly detected (%)")
    ax_detection.set_xlim(left=0)

    for axis in (ax_fpr, ax_detection):
        axis.grid(axis="x", color="#e0e0e0", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    handles, labels = ax_fpr.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    fig.savefig(figure_path("robustness_figure.pdf"), bbox_inches="tight")
    fig.savefig(figure_path("robustness_figure.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Figures saved: robustness_figure.pdf and robustness_figure.png")


def plot_recraft() -> None:
    """Plot Recraft from existing prediction CSVs."""
    models = (
        ("logistic_full", "Logistic regression", "#0072B2"),
        ("logistic_no_frequency", "Logistic regression\nwithout frequency features", "#D55E00"),
        ("cnn", "CNN", "#008563"),
    )

    predictions = defaultdict(dict)
    with result_path("recraft_predictions.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            model, image = row["model"], row["image"]
            if image in predictions[model]:
                raise ValueError(f"Duplicate prediction for {model} : {image}")
            predictions[model][image] = int(row["predicted_ai"])

    image_names = set(predictions["cnn"])
    if len(image_names) != 20 or any(
        set(predictions[model]) != image_names for model, _, _ in models
    ):
        raise ValueError("All three models must have evaluated the same 20 images.")

    detected = [sum(predictions[model].values()) for model, _, _ in models]
    rates = [count / len(image_names) for count in detected]
    positions = list(range(len(models)))

    fig, ax = plt.subplots(figsize=(7.2, 3.3), layout="constrained")
    ax.barh(positions, rates, color=[color for _, _, color in models], height=0.5)

    for position, count, rate in zip(positions, detected, rates):
        ax.text(rate + 0.02, position, f"{count}/20 ({rate:.0%})", va="center")

    ax.set_yticks(positions, [label for _, label, _ in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("AI images correctly detected (%)")
    ax.set_title("External test on Recraft V4.1 Flash V2")
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(figure_path("recraft_figure.pdf"))
    fig.savefig(figure_path("recraft_figure.png"), dpi=300)
    plt.close(fig)

    print("Figures saved: recraft_figure.pdf and recraft_figure.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=("robustness", "recraft"))
    figure = parser.parse_args().figure
    if figure == "robustness":
        plot_robustness()
    else:
        plot_recraft()


if __name__ == "__main__":
    main()
