"""CNN training and epoch selection by validation AUC."""

import argparse
import csv
import time
from collections import Counter
from pathlib import Path
from random import Random

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, get_worker_info

from .detectors import build_cnn, image_to_tensor, select_threshold
from .image_transforms import VARIANT_NAMES, make_variant
from .paths import DATASET_DIR, model_path, result_path


SEED = 42
BATCH_SIZE = 32
EPOCHS = 8


class ArtDataset(Dataset):
    """Read images; optional augmentation is training-only with a dedicated RNG.

    The current path uses num_workers=0. The independent Python RNG preserves
    Torch initialization and batch order, even when augmentation is enabled.
    Its sequence advances with each presentation across all eight epochs.
    """

    def __init__(
        self, rows: list[dict[str, str]], *, augment: bool = False, seed: int = SEED
    ) -> None:
        if augment and any(row["split"] != "train" for row in rows):
            raise ValueError("Augmentation is restricted to training images.")
        self.rows = rows
        self.augmentation_rng = Random(seed) if augment else None
        self.augmentation_counts: Counter[tuple[str, str]] = Counter()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        row = self.rows[index]
        with Image.open(DATASET_DIR / row["path"]) as image:
            if self.augmentation_rng is not None:
                if get_worker_info() is not None:
                    raise RuntimeError("Reproducible augmentation requires num_workers=0.")
                condition = draw_training_condition(self.augmentation_rng)
                image = make_variant(image, condition)
                self.augmentation_counts[row["label"], condition] += 1
            pixels = image_to_tensor(image)
        return pixels, float(row["label"])


def draw_training_condition(rng: Random) -> str:
    """Draw independently of class: original 1/2, each variant 1/12."""
    if rng.random() < 0.5:
        return "reference"
    return rng.choice(VARIANT_NAMES[1:])


def validation_scores(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned labels and scores without gradients, as float64 via tolist()."""
    model.eval()
    scores = []
    labels = []

    with torch.inference_mode():
        for images, batch_labels in loader:
            logits = model(images.to(device)).squeeze(1)
            scores.extend(logits.sigmoid().cpu().tolist())
            labels.extend(batch_labels.tolist())

    return np.array(labels), np.array(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment training only: 50 %% originals, 50 %% across six variants.",
    )
    parser.add_argument(
        "--output-model",
        help="Checkpoint .pt filename in models/; keep the historical default.",
    )
    args = parser.parse_args()
    seed = args.seed
    filename = "cnn_model.pt" if seed == SEED else f"cnn_model_seed_{seed}.pt"
    if args.augment:
        filename = (
            "cnn_model_augmented.pt" if seed == SEED else f"cnn_model_augmented_seed_{seed}.pt"
        )
    filename = args.output_model or filename
    if Path(filename).name != filename or Path(filename).suffix != ".pt":
        parser.error("--output-model must be a .pt filename in models/.")
    if args.augment and (filename == "cnn_model.pt" or filename.startswith("cnn_model_seed_")):
        parser.error("The primary model filename is reserved for the unaugmented CNN.")
    experiment_start = time.perf_counter()

    torch.manual_seed(seed)
    print(f"CNN seed : {seed}", flush=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device used : {device}", flush=True)

    with result_path("dataset_manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    if len(train_rows) != 4200 or len(val_rows) != 600:
        raise ValueError("Unexpected training or validation counts.")

    train_dataset = ArtDataset(train_rows, augment=args.augment, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(ArtDataset(val_rows), batch_size=BATCH_SIZE)

    model = build_cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    output_path = model_path(filename, for_write=True)

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        val_labels, scores = validation_scores(model, val_loader, device)
        auc = roc_auc_score(val_labels, scores)
        elapsed = time.perf_counter() - start
        print(
            f"Epoch {epoch}/{EPOCHS} | training loss {total_loss / len(train_rows):.4f}"
            f" | validation AUC {auc:.4f} | {elapsed:.0f} s",
            flush=True,
        )

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_scores = scores.copy()
            best_weights = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }

    human_scores = best_scores[val_labels == 0]
    if len(human_scores) != 300 or np.count_nonzero(val_labels == 1) != 300:
        raise ValueError("Validation must contain 300 images per class.")

    threshold = select_threshold(human_scores)
    predicted_ai = best_scores >= threshold
    false_positives = int(predicted_ai[val_labels == 0].sum())
    detection_rate = predicted_ai[val_labels == 1].mean()

    checkpoint = {"model_state": best_weights, "threshold": threshold, "epoch": best_epoch}
    if args.augment:
        checkpoint["augmentation"] = {
            "seed": seed,
            "original_probability": 0.5,
            "transformed_probability_each": 1 / 12,
            # Historical checkpoint value: RNG independent of Torch, num_workers=0.
            "rng": "random.Random(seed), independent of PyTorch, num_workers=0",
            "epochs_executed": EPOCHS,
            "counts_all_epochs": {
                label: {
                    condition: train_dataset.augmentation_counts[label, condition]
                    for condition in VARIANT_NAMES
                }
                for label in ("0", "1")
            },
        }
    torch.save(checkpoint, output_path)

    print(f"\nBest epoch : {best_epoch}")
    print(f"Decision threshold : {threshold:.6f}")
    print(f"Validation false positives : {false_positives}/300 ({false_positives / 300:.1%})")
    print(f"Validation AI detection : {detection_rate:.1%}")
    print(f"Validation ROC-AUC : {best_auc:.4f}")
    print(f"Model saved : {output_path.name}")
    if args.augment or args.output_model:
        print(
            f"Augmentation : {args.augment} | total time : "
            f"{time.perf_counter() - experiment_start:.2f} s"
        )


if __name__ == "__main__":
    main()
