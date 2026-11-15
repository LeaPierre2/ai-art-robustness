# Submission preparation and clean reproduction

This record documents checkpoint exclusion, shared scoring and clean-environment
reproduction of the analysis. The scientific artifacts and documented protocol
were preserved. The execution timings below belong to those completed checks;
they are separate from the preparation of the final submission commit.

## Checkpoints and shared scoring

Eleven generated `.pt`/`.joblib` checkpoints remain available locally and are
excluded from the submission tree, together with environments, caches and editor
settings. CSV predictions, figures, historical metadata and Recraft images are
included in the submission.

`analysis.detectors.score_image_batch` now performs identical feature ordering,
logistic `predict_proba` and tensor/CNN probability calculations for four evaluators.
It does not resize images, make experimental variants, choose batches, calibrate
thresholds or change decisions. Main/SD/JPEG keep batch size 32; Recraft keeps
single-image batches. Float32 CNN probabilities and callers' historical float64
comparisons are preserved. CNN-only augmentation evaluation skips feature extraction.

## Clean-copy method

A separate directory was built from the current tracked-file set plus new review
documents, using working-tree contents rather than HEAD. No model file was copied.
A new Python 3.9.6 venv was created with `python3 -m venv` and installed with
`python -m pip install -r requirements.txt`, followed by `python -m pip check`.
Raw data are linked read-only by convention to the existing local dataset; code
writes only into the test copy. The delivered Recraft PNGs are copied unchanged.
No private checkpoint or original environment's site-packages is imported.

The selected backend is CPU: MPS is unavailable in this execution environment.
Recorded platform: macOS 15.7.4 arm64, Python 3.9.6, Torch 2.7.1, NumPy 2.0.2,
scikit-learn 1.6.1, Pillow 11.3.0. CPU runs use four threads. The exact Mac model
is not independently recorded. Pinned dependency installation succeeded after
allowing network access; the first sandbox-restricted attempt could not resolve PyPI.

A fresh no-model copy retains delivered CSVs for the CSV-only checks, then runs
the full unconditional training pipeline. Such a copy tests clone contents;
a new network Git clone or course-workspace submission was not performed.

## Complete scoring comparison (local historical weights, CPU)

| Evaluation | Prediction rows | Maximum absolute score change | Decision changes | CSV bytes |
|---|---:|---:|---:|---|
| Main Human/LD, seven conditions | 25,200 | 0 | 0 | Identical |
| JPEG table intervention | 10,800 | 0 | 0 | Identical |
| Human/LD/SD, seven conditions | 37,800 | 0 | 0 | Identical |
| Recraft, 20 images × three models | 60 | 0 | 0 | Identical |

Before/after copies use the same local checkpoints, environment and CPU backend.
All rows retain their identifiers, order, metadata and fixed thresholds; score and
prediction equality is exact. The original models are used only in these separate
scoring comparisons, never in the no-checkpoint clean reproduction.

## Evidence

Execution results, timings and comparisons are recorded below after the checks.
Full logs and scripts are kept in ignored `local/submission-checks/` in the
project folder. No real Recraft API generation is part of these checks.

## Additional targeted reproducibility checks

Earlier targeted checks supplement the complete clean-environment run:

- All 4,800 PCA scatter coordinates and plotting settings matched the saved data.
  Thumbnail pixels in both near-duplicate sheets were preserved; the historical
  manifest sheet remains a static artifact because its full renderer is unavailable.
- A zero-pair near-duplicate review succeeded and retained an existing preview.
- Paired CPU probes matched exactly for 672 main predictions, 288 JPEG-intervention
  predictions with 64 encoding checks, and 504 source-generalization predictions
  including normalized-pixel hashes. The complete scoring comparisons above cover
  the larger datasets after the shared scoring change.
- A four-image augmentation/training smoke check matched random draws, tensors,
  loss and weight hashes after one Adam update, without saving a checkpoint.
  The full eight-epoch CPU runs are recorded separately below.
- The existing-image guard found all twenty Recraft PNGs and made no API request.
  A separate mocked response verified the same request payload and returned PNG
  bytes. Historical prompts and generator parameters remained unchanged.
- Earlier source-invariant checks confirmed unchanged scientific computations,
  numeric constants, identifiers and command arguments. The subsequent batch
  scoring extraction is covered by the complete prediction comparisons above.

## Limits and attribution

New CPU training is distinct from re-evaluating historical MPS-trained weights on
CPU. The delivered historical prediction CSVs stay intact; exact historical CNN
re-evaluation requires the retained historical weights. CPU/MPS differences already
observed are quantified in the README and `E2E_VALIDATION.md`.

MPS replay, unseen course-workspace rules, archive version/checksum and the
archive-specific license remain unverified. [Attribution](ATTRIBUTION.md) separates
visible execution evidence from author-provided tool provenance and lists report
appendix items. The current review does not retrospectively authenticate an
uninspected external Claude audit or unavailable planning transcript.
## Executed clean-copy results

All commands below exited successfully in the fresh environment. Timings include
process startup; runs used four CPU threads. The main pipeline partially overlapped
the independent historical-weight scoring comparison. Filesystem caches and
concurrent workloads differ from historical runs; these are observations, not
speed benchmarks. Dependency installation was not timed.

| Check | Command (fresh Python) | Wall seconds |
|---|---|---:|
| fresh-pip-check | `python -m pip check` | 1.19 |
| fresh-environment | `python -c 'import torch,platform,numpy,sklearn,PIL; print(platform.platform(),platform.python_version(),torch.__version__,numpy.__version__,sklearn.__version__,PIL.__version__,torch.backends.mps.is_available(),torch.get_num_threads())'` | 19.67 |
| csv-results-only | `python run_all.py --results-only` | 18.79 |
| csv-pca-figure | `python -m analysis.pca_features --figure-only` | 4.45 |
| csv-recraft-figure | `python -m analysis.figures recraft` | 0.45 |
| csv-seed-summary | `python -m analysis.cnn_augmentation --summarize-seeds` | 1.36 |
| clean-full-pipeline | `python run_all.py` | 698.81 |
| clean-recraft-evaluation | `python -m analysis.evaluate_recraft` | 2.03 |
| clean-reuse-models | `python run_all.py --reuse-models` | 450.99 |
| clean-augmentation-control-42 | `python -m analysis.train_cnn --seed 42 --output-model cnn_augmentation_control_seed_42.pt` | 212.19 |
| clean-augmentation-augmented-42 | `python -m analysis.train_cnn --seed 42 --output-model cnn_augmentation_augmented_seed_42.pt --augment` | 232.41 |
| clean-augmentation-evaluate-42 | `python -m analysis.cnn_augmentation --seed 42` | 57.19 |
| clean-augmentation-control-43 | `python -m analysis.train_cnn --seed 43 --output-model cnn_augmentation_control_seed_43.pt` | 215.21 |
| clean-augmentation-augmented-43 | `python -m analysis.train_cnn --seed 43 --output-model cnn_augmentation_augmented_seed_43.pt --augment` | 234.90 |
| clean-augmentation-evaluate-43 | `python -m analysis.cnn_augmentation --seed 43` | 57.89 |
| clean-augmentation-control-44 | `python -m analysis.train_cnn --seed 44 --output-model cnn_augmentation_control_seed_44.pt` | 215.97 |
| clean-augmentation-augmented-44 | `python -m analysis.train_cnn --seed 44 --output-model cnn_augmentation_augmented_seed_44.pt --augment` | 232.13 |
| clean-augmentation-evaluate-44 | `python -m analysis.cnn_augmentation --seed 44` | 57.59 |
| clean-seed-summary | `python -m analysis.cnn_augmentation --summarize-seeds` | 1.46 |
| clean-cnn-43 | `python -m analysis.train_cnn --seed 43` | 219.80 |
| clean-evaluate-43 | `python -m analysis.evaluate_robustness --seed 43` | 56.73 |
| clean-cnn-44 | `python -m analysis.train_cnn --seed 44` | 223.48 |
| clean-evaluate-44 | `python -m analysis.evaluate_robustness --seed 44` | 55.93 |

The separate no-data/no-model fixture passed all six quick-start commands:
`analyze_results`, `make_tables`, `figures robustness`, `figures recraft`,
`pca_features --figure-only`, and `cnn_augmentation --summarize-seeds`.
All 52 CSVs and report tables stayed byte-identical. `run_all.py --results-only`
also passed without checkpoints, using the raw examples needed by its JPEG audit.
The full PCA command used the newly trained logistic model for schema validation;
all seven PCA CSVs matched the delivered files. All ten CLI help entry points and
Python 3.9.6 compilation passed. No API call was made.

Eight regenerated preparation CSVs are byte-identical, including the manifest,
features and near-duplicate candidates. Manifest SHA-256:

```text
e04565ff1a9b44a817501680db0dafb316d58e122429bac526b414e5b205a606
```

### Newly trained models and prior CPU runs

The two logistic checkpoints match the delivered files byte-for-byte. Plain CNN
checkpoints are compared against the previous isolated CPU validation, not MPS
weights. Augmentation pairs are compared against the delivered CPU experiment.

| Checkpoint | Max weight change | Other checkpoint fields | File bytes |
|---|---:|---|---|
| `cnn_augmentation_augmented_seed_42.pt` | 0 | Identical | Identical |
| `cnn_augmentation_augmented_seed_43.pt` | 0 | Identical | Identical |
| `cnn_augmentation_augmented_seed_44.pt` | 0 | Identical | Identical |
| `cnn_augmentation_control_seed_42.pt` | 0 | Identical | Different archive member prefix (output filename) |
| `cnn_augmentation_control_seed_43.pt` | 0 | Identical | Different archive member prefix (output filename) |
| `cnn_augmentation_control_seed_44.pt` | 0 | Identical | Different archive member prefix (output filename) |
| `cnn_model.pt` | 0 | Identical | Identical |
| `cnn_model_seed_43.pt` | 0 | Identical | Identical |
| `cnn_model_seed_44.pt` | 0 | Identical | Identical |
| `logistic_full.joblib` | 0 | Identical | Identical |
| `logistic_no_frequency.joblib` | 0 | Identical | Identical |

An altered Torch archive prefix does not alter tensors, selected epoch, threshold
or augmentation metadata. All scientific checkpoint fields are compared exactly.
The nine primary/Recraft CSVs checked against the preceding CPU run are identical:
`test_predictions.csv`, `robustness_summary.csv`, `test_score_analysis.csv`, `false_positive_flip_tests.csv`, `frequency_ablation_comparison.csv`, `jpeg_metadata_baseline.csv`, `jpeg_quantization_tables.csv`, `jpeg_quality_equivalents.csv`, `recraft_predictions.csv`.

### Augmentation and optional seeds

All eleven augmentation CSVs (nine per-seed predictions/summary/comparison tables
and two cross-seed synthesis tables) are byte-identical to the delivered CPU
experiment. The four additional primary prediction/summary CSVs for seeds 43/44
match the earlier isolated CPU validation. All eleven models were rebuilt from
local raw data and the documented tables, with no preinstalled checkpoint.
Nine CNN training runs executed all eight epochs each (72 epochs total).

### Newly trained CPU versus delivered historical MPS predictions

Both logistic models have maximum score difference 0 and no decision changes.
For the seed-42 CNN, the maximum absolute score difference is
0.1766042709350586 across 8,400 test predictions, with 132 changed decisions.
The same difference was already observed with pre-reorganization reference code.
This is separate from the exact before/after scoring-refactor comparison above.

| Condition | Human FP historical → CPU /600 | AI detections historical → CPU /600 | AUC historical → CPU |
|---|---:|---:|---:|
| `reference` | 23 → 28 | 298 → 302 | 0.910167 → 0.908064 |
| `jpeg_85` | 23 → 28 | 295 → 299 | 0.907372 → 0.905189 |
| `jpeg_60` | 23 → 28 | 295 → 298 | 0.905978 → 0.903783 |
| `jpeg_35` | 24 → 28 | 296 → 297 | 0.906583 → 0.904414 |
| `resize_128` | 21 → 26 | 252 → 260 | 0.903361 → 0.900822 |
| `resize_64` | 20 → 23 | 179 → 188 | 0.878258 → 0.874239 |
| `resize_128_then_jpeg_60` | 21 → 26 | 254 → 261 | 0.902314 → 0.899719 |

Seed-42 threshold changes from 0.8154128789901735 (historical MPS) to
0.802436649799347 (new CPU), with epoch 8 selected. Recraft's CNN maximum
score change versus delivered results is 0.06654608249664307, without decision
changes; the two logistic Recraft scores are identical. No historical output was
replaced in the project outputs. Main scientific conclusions retain the limits
explained in `E2E_VALIDATION.md`.

### Artifact integrity

All 102 protected artifacts preserve their start-of-pass SHA-256: 52 results CSVs,
11 local models, 10 figures, 22 Recraft assets and seven historical JSON records.
All eleven saved checkpoints remain on the Mac and are excluded from the
submission tree.
`git diff --check` and `git diff --cached --check` pass. The prospective tracked
file set contains no checkpoint, raw image dataset, virtual environment, cache
or editor settings. The Word draft paths remain ignored.
