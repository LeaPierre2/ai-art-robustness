# AI-art detection robustness

> How do JPEG compression and image resizing affect AI-art detection performance,
> particularly the false-positive rate on human-created artworks?

The main study compares logistic regression with 55 visual features, an ablation
without its three frequency features, and a CNN. Thresholds target at most 5%
false positives on human validation images and remain fixed across seven test
conditions: original, JPEG 85/60/35, resizing via 128/64 px, and 128 px + JPEG 60.
The main test contains 600 human artworks and 600 Latent Diffusion images.

**The main experiment uses the historical CNN trained on MPS, whereas the
augmentation experiment compares control–augmented CNN pairs trained on CPU.**

## Quick start

Start with [run_all.py](run_all.py), the pipeline entry point. Code is in
[analysis/](analysis/); delivered CSVs and figures are in `results/` and `figures/`.
Generated checkpoints belong in local `models/` and are **not included in the
submission checkout**. The [report tables](docs/report_tables.md) can be inspected without
running code. Shared batch scoring lives in `analysis/detectors.py`; each
evaluator retains its own preprocessing and experiment order. Detailed validation
records are linked below.

Use **Python 3.9.6**. After cloning, run these commands from the repository root.
This first path uses only delivered CSVs: no raw images or checkpoints are needed.

```bash
python3 -m venv .venv_repro
./.venv_repro/bin/python -m pip install -r requirements.txt
./.venv_repro/bin/python -m pip check
./.venv_repro/bin/python -m analysis.analyze_results
./.venv_repro/bin/python -m analysis.make_tables
./.venv_repro/bin/python -m analysis.figures robustness
./.venv_repro/bin/python -m analysis.figures recraft
./.venv_repro/bin/python -m analysis.pca_features --figure-only
./.venv_repro/bin/python -m analysis.cnn_augmentation --summarize-seeds
```

`./.venv_repro/bin/python run_all.py --results-only` reuses saved predictions for the JPEG audit, statistical analyses,
robustness figure and report tables. It still needs the raw inventory examples
for the JPEG audit. A fresh installation from the pinned requirements and a no-checkpoint
reproduction are recorded in the [submission validation](docs/SUBMISSION_VALIDATION.md).

## Data and provenance

Download [AI-ArtBench from Kaggle](https://www.kaggle.com/datasets/ravidussilva/real-ai-art)
and extract it to `data/raw/Real_AI_SD_LD_Dataset/`, keeping the official `train/`
and `test/` folders. The main study uses Human and Latent Diffusion only:
4,200 training, 600 validation and 1,200 test images. Stable Diffusion is reserved
for the separate source-generalization experiment.

**Archive version and download checksum were not recorded.** Local folder counts
and dimensions were checked, but do not identify a specific Kaggle release.
The main selection is identified by `results/dataset_manifest.csv`, SHA-256:
`e04565ff1a9b44a817501680db0dafb316d58e122429bac526b414e5b205a606`.

**License status:** the authors state an MIT license in
[ArtBrain, Section 8](https://arxiv.org/html/2412.01512v1#S8). Earlier project notes
recorded Kaggle's “Other (specified in description)” label; its current terms
could not be independently confirmed, and no accompanying license file was
identified locally. The authors' statement is verifiable, but the terms attached
to the downloaded archive remain unconfirmed. Raw data, Python environments,
Word drafts and local test copies are excluded from Git.

## Reproduce the main analysis

Full runs retrain models and replace primary outputs; use a separate copy to
preserve the delivered historical artifacts.

```bash
OMP_NUM_THREADS=4 PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv_repro/bin/python run_all.py
```

The pipeline inventories data, excludes exact duplicates, reviews near-duplicate
candidates, creates and checks the manifest, extracts features, trains models,
evaluates robustness, audits JPEG metadata, and produces analyses and tables.
Near-duplicate candidates require human review; they are not automatically
excluded. An empty candidate list succeeds without replacing an existing preview.

The full command above creates `models/logistic_full.joblib`,
`models/logistic_no_frequency.joblib` and `models/cnn_model.pt` from scratch.
It runs all eight CNN epochs, selects the checkpoint by original validation AUC
and calibrates its fixed threshold on original human validation images.
No checkpoint download or course workspace is required for a new CPU run.

To evaluate **locally trained** models, or rebuild outputs independently:

```bash
./.venv_repro/bin/python -m analysis.evaluate_robustness
./.venv_repro/bin/python -m analysis.analyze_results
./.venv_repro/bin/python -m analysis.figures robustness
./.venv_repro/bin/python -m analysis.make_tables
```

Evaluation needs the raw images; the last three commands use existing CSVs.
`run_all.py --reuse-models` prepares data and evaluates existing local models;
it does not create missing models. The full shortcut was tested in an isolated copy. Main training and evaluation select
MPS when available, otherwise CPU. CPU retraining is not expected to reproduce
the historical MPS checkpoint. Delivered primary prediction CSVs use the original
MPS-trained CNN (seed 42; additional historical seeds 43/44 are separate).
A prior matched CPU retraining selected epoch 8 but changed the seed-42 threshold
from 0.8154128789901735 to 0.802436649799347. On originals, human false positives
changed from 23/600 to 28/600, AI detections from 298/600 to 302/600 and AUC
from 0.910167 to 0.908064. Across seven conditions, 132/8,400 CNN decisions
changed (52 distinct images); this occurred with reference code too.
These are historical validation observations, not a claim of bitwise equivalence
for arbitrary hardware. Backend and unrecorded historical numerical conditions
cannot be separated retrospectively. See [CPU/MPS comparisons](docs/E2E_VALIDATION.md). Optional `--seed 43` and `--seed 44` on
`analysis.train_cnn` and `analysis.evaluate_robustness` create suffixed artifacts.

Primary outputs are `results/test_predictions.csv`, `robustness_summary.csv`,
`test_score_analysis.csv`, `false_positive_flip_tests.csv`,
`frequency_ablation_comparison.csv` and the three JPEG audit CSVs, all in
`results/`; checkpoints are in `models/`. See the
[robustness figure](figures/robustness_figure.png), [report tables](docs/report_tables.md)
and [end-to-end validation](docs/E2E_VALIDATION.md) for results and comparisons.

## Input requirements at a glance

| Command | Delivered CSVs only | Other inputs | Model output |
|---|---|---|---|
| `-m analysis.analyze_results` | Yes | None | None |
| `-m analysis.figures robustness` / `recraft` | Yes | None | None |
| `-m analysis.make_tables` | Yes | None | None |
| `-m analysis.pca_features --figure-only` | Yes | None | None |
| `-m analysis.cnn_augmentation --summarize-seeds` | Yes | Three delivered per-seed comparisons | None |
| `run_all.py --results-only` | No | Raw JPEG inventory examples for the audit; no checkpoints | None |
| `run_all.py` | No | Raw dataset | Three primary models |
| `-m analysis.train_logistic_models` | No | `train_validation_features.csv` (regenerated by pipeline); no existing models | Two logistic models |
| `-m analysis.train_cnn` | No | Raw images and manifest; no existing models | One CNN |
| `run_all.py --reuse-models` / `-m analysis.evaluate_robustness` | No | Raw data and three primary local models | None |
| `-m analysis.cnn_augmentation --seed 42` | No | Raw test images, manifest and local control/augmented pair | None |
| `-m analysis.jpeg_table_intervention` | No | Raw data, exported JPEG tables and three local models | None |
| `-m analysis.evaluate_stable_diffusion` | No | Raw Human/LD/SD data, manifest and three local models | None |
| `-m analysis.pca_features` | No | Original training feature CSV, raw test images, manifest and local full logistic model (schema check) | None |
| `-m analysis.evaluate_recraft` | No | Delivered Recraft PNGs and three local primary models | None |

The two historical logistic checkpoints and nine CNN checkpoints are excluded by
Git (`*.joblib`, `*.pt`). Environments, caches and editor settings are excluded too.
CSV-only plotting/statistics preserve the delivered historical predictions; a full
run replaces predictions and their derived summaries. Keep historical and newly
trained runs in separate copies, including complementary outputs and JSON notes.

## CNN augmentation: evaluate and rebuild the synthesis

The saved experiment uses seeds 42, 43 and 44. Training keeps an original image
with probability 1/2; otherwise it uniformly draws one of the six existing
variants, computing only that variant. The class-independent RNG is seeded;
validation remains original-only, with eight epochs and the same checkpoint and
threshold rules. See the [three-seed synthesis](docs/CNN_AUGMENTATION_SEEDS_42_44.md).

```bash
# Requires the two locally trained pair checkpoints; see training loop below.
./.venv_repro/bin/python -m analysis.cnn_augmentation --seed 42
# Rebuild both cross-seed CSVs from the three saved per-seed comparison CSVs.
./.venv_repro/bin/python -m analysis.cnn_augmentation --summarize-seeds
```

The summary command performs **no training or image inference**. It writes
[per-seed resize comparisons](results/cnn_augmentation_seed_synthesis.csv) and
[means and min–max ranges](results/cnn_augmentation_seed_summary.csv).
Detection gain is augmented minus control; loss attenuation is resized gain
minus original-image gain; the FPR difference is augmented minus control.
These are descriptive comparisons: all seeds reuse the same test images.

To retrain the pairs, run the following in a separate copy **in a CPU environment**
to match the reported experiment; the trainer selects MPS if it is available:

```bash
export OMP_NUM_THREADS=4 PYTHONHASHSEED=0
for seed in 42 43 44; do
  ./.venv_repro/bin/python -m analysis.train_cnn --seed "$seed" --output-model "cnn_augmentation_control_seed_${seed}.pt"
  ./.venv_repro/bin/python -m analysis.train_cnn --seed "$seed" --augment --output-model "cnn_augmentation_augmented_seed_${seed}.pt"
  ./.venv_repro/bin/python -m analysis.cnn_augmentation --seed "$seed"
done
./.venv_repro/bin/python -m analysis.cnn_augmentation --summarize-seeds
```

Models use `models/cnn_augmentation_{control,augmented}_seed_<seed>.pt`.
Per-seed outputs are `results/cnn_augmentation_{predictions,summary,comparison}_seed_<seed>.csv`
and `docs/CNN_AUGMENTATION_SEED_<seed>.json`. The original timed controls used
isolated copies and default output names before export; dedicated names above
preserve the primary model filenames. The paired training runs were concurrent,
whereas the loop above runs them sequentially.

## Other complementary analyses

These commands are independent of `run_all.py` and use distinct output names:

```bash
OMP_NUM_THREADS=4 PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv_repro/bin/python -m analysis.jpeg_table_intervention
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv_repro/bin/python -m analysis.pca_features
MPLBACKEND=Agg ./.venv_repro/bin/python -m analysis.pca_features --figure-only
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv_repro/bin/python -m analysis.evaluate_stable_diffusion
```

| Analysis | Outputs and interpretation |
|---|---|
| [JPEG table intervention](docs/JPEG_TABLE_INTERVENTION.md) | Five `results/jpeg_table_intervention_*.csv` files and `docs/JPEG_TABLE_INTERVENTION.json`. Paired re-encodings using observed tables; this does not estimate the encoding bias's historical contribution. |
| [PCA](docs/PCA_FEATURES.md) | Seven `results/pca_*.csv` files, `docs/PCA_FEATURES.json`, and [PNG](figures/pca_feature_transformations.png)/[PDF](figures/pca_feature_transformations.pdf). Scaling and axes fit only on original training images; exploratory, not causal error attribution. |
| [Stable Diffusion](docs/STABLE_DIFFUSION_GENERALIZATION.md) | Six `results/stable_diffusion_*.csv` files and `docs/STABLE_DIFFUSION_GENERALIZATION.json`. 600 official-test SD images, 60 per style, seed 42; RGB/LANCZOS to 256 px before the seven conditions. Source, encoding and native resolution are confounded. |

Delivered JPEG intervention and SD evaluation CSVs used the historical checkpoints
and fixed thresholds on CPU, including recalculated Human/LD originals. The commands
require three local primary checkpoints and raw data. With newly trained weights
they produce a **new run**, not the delivered historical intervention/generalization
results. Historical checkpoints remain on the author's Mac; exact re-evaluation needs
those same weights rather than retraining. They are not bundled in the submission
checkout. When re-evaluating an existing checkpoint, record its source and
checksum explicitly.
Recraft evaluation has the same checkpoint requirement; its figure only needs CSVs.

For the 20 existing Recraft images, verify and evaluate locally:

```bash
shasum -a 256 -c data/recraft/SHA256SUMS.txt
./.venv_repro/bin/python -m analysis.evaluate_recraft
./.venv_repro/bin/python -m analysis.figures recraft
```

Outputs are `results/recraft_predictions.csv` and `figures/recraft_figure.{png,pdf}`;
`data/recraft/generation_log.csv` stays with the images. Generating new images is
an explicit, separate API action: `./.venv_repro/bin/python -m analysis.generate_recraft`
requires credentials and credits and is never called by the pipeline.

## Measured runtime and resources

Previously measured wall times include process startup. CPU runs used four
threads; the end-to-end record identifies macOS 15.7.4 arm64 (Apple Silicon),
with eight PyTorch interop threads. The exact Mac/chip was not recorded;
complementary experiment records do not identify additional hardware details.
Common versions: Python 3.9.6, Torch 2.7.1, NumPy 2.0.2, scikit-learn 1.6.1 and
Pillow 11.3.0. These observations are not hardware-independent benchmarks.

| Run | Backend | Measured time |
|---|---|---|
| Full primary pipeline, seed 42 | CPU | 845.13 s; two code versions ran concurrently during validation |
| Fresh no-checkpoint primary pipeline (this review) | CPU | 698.81 s; overlapped the separate scoring comparison during part of the run |
| Historical primary pipeline | MPS | Not measured |
| Augmentation seed 42 | CPU | Control 359.19 s; augmented 359.18 s; evaluation 63.99 s |
| Augmentation seed 43 | CPU | Control 343.96 s; augmented 343.96 s; evaluation 54.51 s |
| Augmentation seed 44 | CPU | Control 344.40 s; augmented 344.40 s; evaluation 54.63 s |
| JPEG table intervention | CPU | 28.34 s |
| PCA | CPU | 48.46 s |
| Stable Diffusion evaluation | CPU | 113.65 s |
| Human/LD reference check with historical checkpoints | CPU | 59.21 s |
| Recraft evaluation / figure | CPU | 2.31 / 0.51 s |

For each augmentation seed, control and augmented training were timed in
parallel, so their durations must not be added to estimate elapsed time.
New clean-copy measurements: `--reuse-models` 450.99 s, `--results-only`
18.79 s, PCA `--figure-only`
4.45 s, seed aggregation 1.36 s, Recraft evaluation 2.03 s and full PCA
39.40 s including process startup (the PCA script itself reported 36.62 s).
The fresh-copy check rebuilt all eleven model files (nine CNN runs, 72 epochs)
and recovered the stored CPU results; historical MPS predictions remain distinct.
Installation and API generation were not timed. Additional command timings and
resource context are in the [submission validation](docs/SUBMISSION_VALIDATION.md).

## Validation records

See [end-to-end comparisons](docs/E2E_VALIDATION.md),
[code quality review](docs/CODE_QUALITY_REVIEW.md),
[Python reorganization](docs/REFACTOR_VALIDATION.md),
[CSV migration](docs/CSV_MIGRATION_VALIDATION.md) and
[repository layout](docs/REPOSITORY_LAYOUT_VALIDATION.md).
The earlier reorganization and experiment notes describe historical
snapshots; their figures, counts and recorded checks belong to those stages.
The submission validation records the new clean-environment runs separately.
Machine-readable JSON records, checkpoint metadata and original Recraft prompts
are retained verbatim. The RNG metadata records independence from Torch with
num_workers=0; paired-test metadata describes exact two-sided McNemar
with joint Holm correction and descriptive AUC/change contrasts. Historical
filenames and placeholders are not renamed. Commands here use the current layout. The final external report is not stored
in this repository.

The historical manifest near-duplicate sheet is delivered as a static artifact;
its full renderer was not retained. Near-duplicate previews can be rebuilt with
`-m analysis.near_duplicates review`. Text embedded in source artworks is retained.

## Attribution

ChatGPT was used for project planning. OpenAI Codex assisted with implementation
and troubleshooting, refactoring, experiment execution and validation, and
documentation. Anthropic Claude audited the Python code and README on
15 November 2026, with recommendations on reproducibility and documentation
(author-reported; exact model not recorded). Recraft generated the 20 images used in the
external evaluation. See the [attribution and report appendix checklist](docs/ATTRIBUTION.md)
for available evidence and provenance that remains to be confirmed.
