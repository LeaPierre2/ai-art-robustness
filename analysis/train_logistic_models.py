"""Train both regressions and save their validation thresholds."""

import csv

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .detectors import select_threshold
from .paths import model_path, result_path


def feature_matrix(
    selected_rows: list[dict[str, str]], columns: list[str]
) -> np.ndarray:
    """Keep the row and column order defined by the feature CSV."""
    return np.array([
        [float(row[name]) for name in columns]
        for row in selected_rows
    ])


def main() -> None:
    feature_table = result_path("train_validation_features.csv")

    with feature_table.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        all_features = [
            name for name in reader.fieldnames
            if name.startswith(("color_", "edge_", "freq_"))
        ]
        rows = list(reader)

    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]

    y_train = np.array([int(row["label"]) for row in train_rows])
    y_validation = np.array([int(row["label"]) for row in validation_rows])

    feature_sets = {
        "logistic_full": all_features,
        "logistic_no_frequency": [
            name for name in all_features if not name.startswith("freq_")
        ],
    }

    if [len(columns) for columns in feature_sets.values()] != [55, 52]:
        raise ValueError("Unexpected feature count.")

    if len(train_rows) != 4200 or len(validation_rows) != 600:
        raise ValueError("Unexpected training or validation counts.")

    if np.count_nonzero(y_validation == 0) != 300:
        raise ValueError("Validation must contain 300 human artworks.")

    for name, columns in feature_sets.items():
        x_train = feature_matrix(train_rows, columns)
        x_validation = feature_matrix(validation_rows, columns)

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000),
        )
        model.fit(x_train, y_train)

        ai_scores = model.predict_proba(x_validation)[:, 1]
        threshold = select_threshold(ai_scores[y_validation == 0])
        predicted_ai = ai_scores >= threshold

        false_positives = int(predicted_ai[y_validation == 0].sum())
        false_positive_rate = predicted_ai[y_validation == 0].mean()
        detection_rate = predicted_ai[y_validation == 1].mean()

        output_path = model_path(f"{name}.joblib", for_write=True)
        joblib.dump(
            {"model": model, "feature_names": columns, "threshold": threshold},
            output_path,
        )

        print(f"\n{name} | {len(columns)} features")
        print(f"Decision threshold : {threshold:.6f}")
        print(f"Validation false positives : {false_positives}/300 ({false_positive_rate:.1%})")
        print(f"Validation AI detection : {detection_rate:.1%}")
        print(f"Validation ROC-AUC : {roc_auc_score(y_validation, ai_scores):.4f}")
        print(f"Model saved : {output_path.name}")


if __name__ == "__main__":
    main()
