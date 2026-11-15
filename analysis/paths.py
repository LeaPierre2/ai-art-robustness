"""Shared data and artifact locations relative to the repository root."""

import csv
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TextIO, Union


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "raw" / "Real_AI_SD_LD_Dataset"
RECRAFT_DIR = ROOT / "data" / "recraft"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
DOCS_DIR = ROOT / "docs"


def result_path(filename: Union[str, Path]) -> Path:
    """Accept a CSV filename or an already resolved absolute path, without writing."""
    return RESULTS_DIR / filename


def open_result(
    filename: Union[str, Path], mode: str = "r", **kwargs: Any
) -> TextIO:
    """Create the parent directory only when writing a CSV."""
    path = result_path(filename)
    if any(flag in mode for flag in "wax+"):
        path.parent.mkdir(parents=True, exist_ok=True)
    return path.open(mode, **kwargs)


def model_path(filename: str, *, for_write: bool = False) -> Path:
    """Resolve a model path; create the directory only when saving."""
    path = MODELS_DIR / filename
    if for_write:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure_path(filename: str) -> Path:
    """Prepare an output path for a figure or preview."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename


def document_path(filename: str) -> Path:
    """Prepare an output path for a generated document."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    return DOCS_DIR / filename


def read_result_rows(filename: Union[str, Path]) -> list[dict[str, str]]:
    """Load CSV tables while preserving order and text values.

    Existing streaming paths for large files stay in their modules;
    this helper does not replace them with an in-memory copy.
    """
    with open_result(filename, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_result_rows(
    filename: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """Write rows without sorting or conversion, using the historical CSV format.

    Without explicit columns, use the first row's column order. An empty
    list requires explicit columns so that its header can be written.
    """
    if fieldnames is None:
        if not rows:
            raise ValueError(f"No rows or columns to write {filename}.")
        fieldnames = list(rows[0])
    with open_result(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
