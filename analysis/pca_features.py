"""Exploratory PCA of 55 features fitted only on original training images."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import sklearn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .image_features import extract_features
from .image_transforms import VARIANT_NAMES, make_variants
from .paths import (
    DATASET_DIR,
    document_path,
    figure_path,
    model_path,
    read_result_rows,
    result_path,
    write_result_rows,
)
from .train_logistic_models import feature_matrix


SOURCES = (("0", "Human", "#0072B2"), ("1", "Latent Diffusion", "#D55E00"))
PLOT_CONDITIONS = {
    "reference": "Originals",
    "jpeg_60": "JPEG 60",
    "resize_128": "Resize via 128 px",
    "resize_64": "Resize via 64 px",
}
METADATA_COLUMNS = ("image_id", "source", "style", "label")


def input_rows() -> tuple[list[str], list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    """Check model schema and align original training rows with the manifest."""
    saved = joblib.load(model_path("logistic_full.joblib"))
    columns = saved["feature_names"]
    if len(columns) != 55 or len(set(columns)) != 55:
        raise ValueError("The full model must use 55 distinct columns.")
    schema = list(extract_features(Image.new("RGB", (16, 16))))
    if columns != schema:
        raise ValueError("Model feature order differs from the extractor.")
    manifest = read_result_rows("dataset_manifest.csv")
    train = [r for r in manifest if r["split"] == "train"]
    test = [r for r in manifest if r["split"] == "test"]
    for rows, n in ((train, 4200), (test, 1200)):
        if (
            len(rows) != n
            or len({r["image_id"] for r in rows}) != n
            or Counter((r["label"], r["source"]) for r in rows)
            != Counter({("0", "Human"): n // 2, ("1", "Latent Diffusion"): n // 2})
        ):
            raise ValueError("Unexpected counts, sources or IDs in the manifest.")
    if {r["image_id"] for r in train} & {r["image_id"] for r in test}:
        raise ValueError("Training and test sets must be disjoint.")
    original_features = read_result_rows("train_validation_features.csv")
    if not original_features or list(original_features[0]) != [
        "image_id",
        "split",
        "source",
        "style",
        "label",
        *columns,
    ]:
        raise ValueError("Unexpected training/validation table schema.")
    indexed = {r["image_id"]: r for r in original_features}
    if len(indexed) != len(original_features):
        raise ValueError("Repeated IDs in the original features.")
    aligned = []
    for row in train:
        features = indexed[row["image_id"]]
        if any(features[key] != row[key] for key in (*METADATA_COLUMNS, "split")):
            raise ValueError(f"Training metadata differ : {row['image_id']}.")
        aligned.append(features)
    return columns, aligned, test, saved


def extract_test_features(rows: list[dict[str, str]], columns: list[str]) -> list[dict[str, Any]]:
    """Compute seven conditions with the historical transformations and extractor."""
    features = []
    for index, row in enumerate(rows, 1):
        with Image.open(DATASET_DIR / row["path"]) as image:
            variants = make_variants(image)
        if tuple(variants) != VARIANT_NAMES:
            raise ValueError("The seven historical conditions are required.")
        for condition, image in variants.items():
            values = extract_features(image)
            if list(values) != columns or not np.isfinite(list(values.values())).all():
                raise ValueError(f"Invalid features : {row['image_id']}, {condition}.")
            features.append(
                {**{k: row[k] for k in METADATA_COLUMNS}, "condition": condition, **values}
            )
        if index % 200 == 0:
            print(f"Test features : {index}/{len(rows)} images", flush=True)
    return features


def align_test_features(
    features: list[dict[str, Any]], test: list[dict[str, str]], columns: list[str]
) -> list[dict[str, Any]]:
    """Require 1,200 IDs × seven conditions, without duplicates or divergent metadata."""
    indexed = {(r["image_id"], r["condition"]): r for r in features}
    if len(indexed) != len(features) or len(indexed) != len(test) * len(VARIANT_NAMES):
        raise ValueError("Expected 8,400 unique test-feature rows.")
    aligned = []
    for row in test:
        for condition in VARIANT_NAMES:
            features_row = indexed[row["image_id"], condition]
            if list(features_row) != [*METADATA_COLUMNS, "condition", *columns] or any(
                features_row[k] != row[k] for k in METADATA_COLUMNS
            ):
                raise ValueError(f"Test schema or metadata differ : {row['image_id']}.")
            aligned.append(features_row)
    return aligned


def paired_changes(
    rows: list[dict[str, Any]],
    standardized: np.ndarray,
    coordinates: np.ndarray,
    columns: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate PCA displacement from median absolute feature changes.

    Changes are computed per image before aggregation; median |Δz|
    is neither |median(Δz)| nor a contribution to a PCA axis.
    """
    reference_indices = {
        r["image_id"]: i for i, r in enumerate(rows) if r["condition"] == "reference"
    }
    reference = np.array([reference_indices[r["image_id"]] for r in rows])
    delta_z = standardized - standardized[reference]
    delta_pc = coordinates[:, :2] - coordinates[reference, :2]
    distances = np.linalg.norm(delta_pc, axis=1)
    projected = []
    for i, row in enumerate(rows):
        projected.append(
            {
                **{k: row[k] for k in (*METADATA_COLUMNS, "condition")},
                "pc1": float(coordinates[i, 0]),
                "pc2": float(coordinates[i, 1]),
                "original_pc1": float(coordinates[reference[i], 0]),
                "original_pc2": float(coordinates[reference[i], 1]),
                "delta_pc1": float(delta_pc[i, 0]),
                "delta_pc2": float(delta_pc[i, 1]),
                "distance_pc1_pc2": float(distances[i]),
                "distance_all_55_standardized_features": float(np.linalg.norm(delta_z[i])),
            }
        )
    summaries = []
    sensitivities = []
    for condition in VARIANT_NAMES:
        for label, source, _ in SOURCES:
            mask = np.array([r["condition"] == condition and r["label"] == label for r in rows])
            if mask.sum() != 600:
                raise ValueError("Expected 600 pairs per class and condition.")
            delta = delta_pc[mask]
            norms = distances[mask]
            summaries.append(
                dict(
                    condition=condition,
                    source=source,
                    label=label,
                    n=600,
                    mean_delta_pc1=float(delta[:, 0].mean()),
                    mean_delta_pc2=float(delta[:, 1].mean()),
                    median_delta_pc1=float(np.median(delta[:, 0])),
                    median_delta_pc2=float(np.median(delta[:, 1])),
                    mean_distance_pc1_pc2=float(norms.mean()),
                    median_distance_pc1_pc2=float(np.median(norms)),
                    q25_distance_pc1_pc2=float(np.percentile(norms, 25)),
                    q75_distance_pc1_pc2=float(np.percentile(norms, 75)),
                    median_distance_all_55_standardized_features=float(
                        np.median(np.linalg.norm(delta_z[mask], axis=1))
                    ),
                )
            )
            changes = delta_z[mask]
            medians = np.median(np.abs(changes), axis=0)
            ranks = np.argsort(np.argsort(-medians, kind="stable"), kind="stable") + 1
            for j, feature in enumerate(columns):
                sensitivities.append(
                    dict(
                        condition=condition,
                        source=source,
                        label=label,
                        n=600,
                        feature=feature,
                        median_abs_standardized_change=float(medians[j]),
                        median_signed_standardized_change=float(np.median(changes[:, j])),
                        rank_by_median_abs_change=int(ranks[j]),
                    )
                )
    return projected, summaries, sensitivities


def plot() -> None:
    """Recreate the figure from CSVs only, with shared axes across panels."""
    rows = read_result_rows("pca_test_coordinates.csv")
    variance = read_result_rows("pca_explained_variance.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharex=True, sharey=True)
    selected = [r for r in rows if r["condition"] in PLOT_CONDITIONS]
    points = np.array([[float(r["pc1"]), float(r["pc2"])] for r in selected])
    low, high = points.min(axis=0), points.max(axis=0)
    padding = 0.04 * np.maximum(high - low, 1)
    for axis, (condition, title) in zip(axes.flat, PLOT_CONDITIONS.items()):
        for label, source, color in SOURCES:
            group = [r for r in rows if r["condition"] == condition and r["label"] == label]
            if len(group) != 600:
                raise ValueError("Figure: expected 600 points per class and condition.")
            axis.scatter(
                [float(r["pc1"]) for r in group],
                [float(r["pc2"]) for r in group],
                s=8,
                alpha=0.35,
                color=color,
                edgecolors="none",
                label=source,
                rasterized=True,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlim(low[0] - padding[0], high[0] + padding[0])
        axis.set_ylim(low[1] - padding[1], high[1] + padding[1])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#dddddd", linewidth=0.5)
        axis.set_axisbelow(True)
    fig.supxlabel(
        f"PC1 ({100 * float(variance[0]['explained_variance_ratio']):.2f} % of training variance)"
    )
    fig.supylabel(
        f"PC2 ({100 * float(variance[1]['explained_variance_ratio']):.2f} % of training variance)"
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        markerscale=2,
        fontsize=9,
    )
    fig.suptitle(
        "Exploratory PCA — axes fitted on 4,200 original training images", y=0.995, fontsize=11
    )
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.12, top=0.86, hspace=0.23, wspace=0.12)
    fig.savefig(figure_path("pca_feature_transformations.png"), dpi=240, bbox_inches="tight")
    fig.savefig(
        figure_path("pca_feature_transformations.pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure-only", action="store_true", help="Recreate only the figure from PCA CSVs."
    )
    args = parser.parse_args()
    if args.figure_only:
        plot()
        return
    start = time.perf_counter()
    columns, train, test, saved = input_rows()
    x_train = feature_matrix(train, columns)
    if not np.isfinite(x_train).all():
        raise ValueError("Non-finite training features.")
    scaler = StandardScaler().fit(x_train)
    historical_scaler = saved["model"].named_steps["standardscaler"]
    scaler_matches = all(
        np.array_equal(getattr(scaler, name), getattr(historical_scaler, name))
        for name in ("mean_", "var_", "scale_")
    )
    if not scaler_matches:
        raise ValueError("Training standardization does not reproduce the full-model scaler.")
    standardized_train = scaler.transform(x_train)
    pca = PCA(svd_solver="full", whiten=False).fit(standardized_train)
    features = align_test_features(extract_test_features(test, columns), test, columns)
    x_test = feature_matrix(features, columns)
    standardized_test = scaler.transform(x_test)
    coordinates = pca.transform(standardized_test)
    projected, displacements, sensitivities = paired_changes(
        features, standardized_test, coordinates, columns
    )
    scaling = [
        dict(
            feature=name,
            train_mean=float(scaler.mean_[i]),
            train_variance_ddof0=float(scaler.var_[i]),
            train_scale=float(scaler.scale_[i]),
            pca_center_standardized=float(pca.mean_[i]),
        )
        for i, name in enumerate(columns)
    ]
    variance = [
        dict(
            component=f"PC{i + 1}",
            explained_variance=float(value),
            explained_variance_ratio=float(pca.explained_variance_ratio_[i]),
            cumulative_explained_variance_ratio=float(pca.explained_variance_ratio_[: i + 1].sum()),
        )
        for i, value in enumerate(pca.explained_variance_)
    ]
    contributions = []
    for axis in range(2):
        weights = pca.components_[axis]
        ranks = np.argsort(np.argsort(-(weights**2), kind="stable"), kind="stable") + 1
        contributions.extend(
            dict(
                component=f"PC{axis + 1}",
                feature=name,
                axis_coefficient=float(weights[i]),
                axis_contribution_pct=float(100 * weights[i] ** 2),
                rank=int(ranks[i]),
            )
            for i, name in enumerate(columns)
        )
    for filename, rows in (
        ("test_features", features),
        ("test_coordinates", projected),
        ("displacement_summary", displacements),
        ("feature_sensitivity", sensitivities),
        ("scaling", scaling),
        ("explained_variance", variance),
        ("axis_contributions", contributions),
    ):
        write_result_rows(f"pca_{filename}.csv", rows)
    plot()
    metadata = dict(
        fit_split="train",
        fit_condition="reference",
        fit_n=len(train),
        feature_names=columns,
        test_n=len(test),
        test_conditions=list(VARIANT_NAMES),
        scaler="StandardScaler(with_mean=True, with_std=True), population variance ddof=0",
        scaler_matches_saved_logistic_full_exactly=scaler_matches,
        pca="PCA(svd_solver='full', whiten=False), all 55 components; first two plotted",
        variance_pc1_pc2=float(pca.explained_variance_ratio_[:2].sum()),
        axis_contribution="100 * squared unit eigenvector coefficient; sums to 100% per axis",
        sensitivity="median_i(abs((x_variant_i - x_original_i) / train_scale)) per feature and class",
        displacement="Paired PC(variant)-PC(original); summary separates signed means and individual Euclidean distances",
        exploratory="No detector error attribution, class-supervised fitting or hypothesis tests",
        input_sha256={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("dataset_manifest.csv", result_path("dataset_manifest.csv")),
                ("train_validation_features.csv", result_path("train_validation_features.csv")),
                ("logistic_full.joblib", model_path("logistic_full.joblib")),
            )
        },
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            sklearn=sklearn.__version__,
            pillow=PIL.__version__,
        ),
        elapsed_seconds=time.perf_counter() - start,
    )
    document_path("PCA_FEATURES.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"PC1 : {100 * pca.explained_variance_ratio_[0]:.2f} % ; PC2 : {100 * pca.explained_variance_ratio_[1]:.2f} %"
    )
    print(
        f"Analysis completed in {metadata['elapsed_seconds']:.2f} s ; seven dedicated CSVs and two figures."
    )


if __name__ == "__main__":
    main()
