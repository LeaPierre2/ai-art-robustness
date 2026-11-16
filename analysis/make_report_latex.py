"""Generate two LaTeX tables from the saved complementary results."""

from .paths import document_path, read_result_rows


def write_table(filename, caption, label, columns, header, rows):
    lines = [
        r"\begin{table}[tb]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\hline",
        header + r" \\",
        r"\hline",
        *(row + r" \\" for row in rows),
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    document_path(filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    models = {
        "logistic_full": "Logistic regression (full)",
        "logistic_no_frequency": "Logistic regression (no FFT)",
        "cnn": "CNN",
    }
    jpeg = {
        row["model"]: row
        for row in read_result_rows("jpeg_table_intervention_paired.csv")
        if row["source"] == "Human"
    }
    jpeg_rows = []
    for key, name in models.items():
        row = jpeg[key]
        jpeg_rows.append(
            f"{name} & {row['human_tables_negative_to_ld_tables_positive']} & "
            f"{row['human_tables_positive_to_ld_tables_negative']} & "
            f"{float(row['delta_positive_rate_pp']):+.2f} & {float(row['holm_p']):.2f}"
        )
    write_table(
        "jpeg_intervention_table.tex",
        "Changes in false positives when switching from Human to Latent Diffusion "
        "JPEG tables on the same 600 human artworks. FPR differences are in percentage "
        "points (pp). Adjusted p-values are from the six Holm-corrected McNemar tests.",
        "tab:jpeg-intervention",
        "lrrrr",
        "Model & New FP & Resolved FP & FPR change (pp) & Adjusted $p$",
        jpeg_rows,
    )

    metrics = ("gain_detection_pp", "attenuation_loss_pp", "difference_fpr_pp")
    summary = {
        (int(row["resolution_px"]), row["metric"]): row
        for row in read_result_rows("cnn_augmentation_seed_summary.csv")
    }

    def mean_range(row):
        average, low, high = (float(row[key]) for key in ("mean_pp", "min_pp", "max_pp"))
        return f"${average:+.2f}$ [${low:+.2f}$, ${high:+.2f}$]"

    write_table(
        "cnn_augmentation_table.tex",
        "CNN augmentation effects over seeds 42--44. Values are means and "
        "observed min--max ranges in percentage points. Detection and FPR "
        "changes are augmented minus control. Loss attenuation subtracts "
        "the detection gain on originals from the gain on resized images.",
        "tab:cnn-augmentation",
        "lrrr",
        "Resize & Detection gain & Loss attenuation & FPR change",
        [
            f"{size} px & " + " & ".join(mean_range(summary[size, m]) for m in metrics)
            for size in (128, 64)
        ],
    )


if __name__ == "__main__":
    main()
