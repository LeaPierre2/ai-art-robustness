"""Reproduce the main analysis from the repository root."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PREPARATION_COMMANDS = (
    ("analysis.inventory",),
    ("analysis.exact_duplicates",),
    ("analysis.near_duplicates", "inventory"),
    ("analysis.manifest", "build"),
    ("analysis.near_duplicates", "manifest"),
    ("analysis.manifest", "verify"),
    ("analysis.image_transforms",),
    ("analysis.image_features", "--table"),
)
TRAINING_COMMANDS = (("analysis.train_logistic_models",), ("analysis.train_cnn",))
RESULT_COMMANDS = (
    ("analysis.audit_jpeg_metadata",),
    ("analysis.analyze_results",),
    ("analysis.figures", "robustness"),
    ("analysis.make_tables",),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--results-only",
        action="store_true",
        help="Reuse existing predictions and CSVs (no training).",
    )
    mode.add_argument(
        "--reuse-models",
        action="store_true",
        help="Prepare data and evaluate the saved models.",
    )
    args = parser.parse_args()
    commands = () if args.results_only else PREPARATION_COMMANDS
    if not args.results_only:
        if not args.reuse_models:
            commands += TRAINING_COMMANDS
        commands += (("analysis.evaluate_robustness",),)
    commands += RESULT_COMMANDS
    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        result = subprocess.run([sys.executable, "-m", *command], cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(f"Failed: {' '.join(command)} (exit code {result.returncode}).")
    print("\nMain analysis completed.")


if __name__ == "__main__":
    main()
