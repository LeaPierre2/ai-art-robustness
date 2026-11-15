"""Architecture, image conversion, thresholds, detector loading and batch scoring."""

from typing import Any, Sequence, Union

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn

from .image_features import extract_features
from .paths import model_path


TARGET_FPR = 0.05


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert RGB to a float32 C×H×W tensor in [0, 1], without resizing."""
    pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def build_cnn() -> nn.Sequential:
    """Build the historical architecture; weights are loaded separately."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(64 * 2 * 2, 1),
    )



def select_threshold(
    human_scores: Union[np.ndarray, Sequence[float]],
) -> float:
    """Limit false positives to 5% using human validation images only.

    The decision rule is score >= threshold. nextafter places the threshold just
    above the first score outside the budget; ties make this conservative.
    Scores must form a finite, nonempty vector.
    """
    scores = np.asarray(human_scores, dtype=np.float64)

    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Human validation scores must be finite and nonempty.")

    descending_scores = np.sort(scores)[::-1]
    max_false_positives = int(TARGET_FPR * scores.size)
    return float(np.nextafter(descending_scores[max_false_positives], np.inf))


def load_cnn(cnn_filename: str) -> tuple[nn.Module, torch.device, float]:
    """Load a CNN and its threshold without requiring logistic models."""
    checkpoint = torch.load(
        model_path(cnn_filename), map_location="cpu", weights_only=True
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cnn = build_cnn().to(device)
    cnn.load_state_dict(checkpoint["model_state"])
    cnn.eval()
    return cnn, device, checkpoint["threshold"]


def load_detectors(cnn_filename: str = "cnn_model.pt"):
    """Load saved models and thresholds without re-estimation.

    The CNN enters evaluation mode on MPS if available, otherwise CPU.
    Logistic scaling parameters remain those of the trained pipeline.
    """
    logistic_models = {
        name: joblib.load(model_path(f"{name}.joblib"))
        for name in ("logistic_full", "logistic_no_frequency")
    }
    cnn, device, cnn_threshold = load_cnn(cnn_filename)

    thresholds = {
        name: saved["threshold"] for name, saved in logistic_models.items()
    }
    thresholds["cnn"] = cnn_threshold

    return logistic_models, cnn, device, thresholds


def score_image_batch(
    images: Sequence[Image.Image],
    logistic_models: dict[str, dict[str, Any]],
    cnn: nn.Module,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Score one nonempty caller-defined batch, without preprocessing or thresholding.

    Callers retain image order, experimental transformations and batch boundaries.
    Logistic columns follow each saved model; CNN probabilities retain float32
    until the caller performs its historical float64 threshold comparison.
    Empty logistic mappings support CNN-only evaluation without feature extraction.
    """
    features = [extract_features(image) for image in images] if logistic_models else []
    scores = {}
    for name, saved in logistic_models.items():
        matrix = np.array([
            [values[column] for column in saved["feature_names"]]
            for values in features
        ])
        scores[name] = saved["model"].predict_proba(matrix)[:, 1]
    tensors = torch.stack([image_to_tensor(image) for image in images])
    with torch.inference_mode():
        logits = cnn(tensors.to(device)).squeeze(1)
        scores["cnn"] = logits.sigmoid().cpu().numpy()
    return scores
