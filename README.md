# AI-art detection robustness

How do JPEG compression and image resizing affect AI-art detection performance, particularly the false-positive rate on human-created artworks?

The study compares two logistic regression models and a CNN across seven test conditions. Decision thresholds are calibrated on validation images and held fixed across conditions. The main test set contains 600 human artworks and 600 Latent Diffusion images.

Start with the [report tables](docs/report_tables.md) and the [robustness figure](figures/robustness_figure.png). Source code is in [`analysis/`](analysis/), CSVs in [`results/`](results/), and figures in [`figures/`](figures/).

## Setup and quick start

From the repository root, use Python 3.9.6 to install the pinned dependencies.

```bash
python3 -m venv .venv_repro
./.venv_repro/bin/python -m pip install -r requirements.txt
./.venv_repro/bin/python -m pip check
```

The following commands use the CSVs included in the repository and require neither raw images nor trained models.

```bash
./.venv_repro/bin/python -m analysis.analyze_results
./.venv_repro/bin/python -m analysis.make_tables
./.venv_repro/bin/python -m analysis.make_report_latex
./.venv_repro/bin/python -m analysis.figures robustness
./.venv_repro/bin/python -m analysis.figures recraft
./.venv_repro/bin/python -m analysis.pca_features --figure-only
./.venv_repro/bin/python -m analysis.cnn_augmentation --summarize-seeds
```

## Data and full reproduction

Download [AI-ArtBench from Kaggle](https://www.kaggle.com/datasets/ravidussilva/real-ai-art). Extract it to `data/raw/Real_AI_SD_LD_Dataset/`, preserving the official `train/` and `test/` folders. The [dataset manifest](results/dataset_manifest.csv) records the study splits. The [dataset authors](https://arxiv.org/html/2412.01512v1#S8) report an MIT license. The exact archive version and the terms attached to the downloaded copy were not independently confirmed.

```bash
OMP_NUM_THREADS=4 PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv_repro/bin/python run_all.py
```

The pipeline inventories and prepares the data, trains the three models, evaluates them, and generates the main tables and figures. Model files are generated locally in `models/`. Run the full pipeline in a separate copy if you want to preserve the delivered prediction files. `run_all.py --reuse-models` requires locally trained models. `run_all.py --results-only` uses saved predictions and still needs raw JPEG examples for the metadata audit.

A fresh CPU run without pretrained checkpoints took about 12 minutes on an Apple Silicon Mac with four CPU threads, excluding installation and dataset download. The main results use the original MPS-trained CNN. CPU retraining can produce different predictions. The augmentation experiment compares models trained on CPU. See the [submission validation](docs/SUBMISSION_VALIDATION.md) and [CPU/MPS comparison](docs/E2E_VALIDATION.md) for the measured runtimes, environment, and reproduction checks.

## Complementary experiments

The linked notes provide commands, inputs, and outputs for each experiment.

| Experiment | Documentation and results |
|---|---|
| CNN augmentation | [Three-seed protocol and results](docs/CNN_AUGMENTATION_SEEDS_42_44.md) · [Seed summary](results/cnn_augmentation_seed_summary.csv) |
| JPEG table intervention | [Protocol and results](docs/JPEG_TABLE_INTERVENTION.md) |
| PCA | [Protocol and results](docs/PCA_FEATURES.md) · [Figure](figures/pca_feature_transformations.pdf) |
| Stable Diffusion | [Protocol and results](docs/STABLE_DIFFUSION_GENERALIZATION.md) |
| Recraft | [Predictions](results/recraft_predictions.csv) · [Figure](figures/recraft_figure.pdf) · [Image provenance](data/recraft/generation_log.csv) |

To evaluate the included Recraft images with locally trained models, run `./.venv_repro/bin/python -m analysis.evaluate_recraft`. Generating new images is a separate step that requires Recraft API credentials and credits.

## Attribution

OpenAI Codex helped debug, improve the code and check reproducibility. Anthropic Claude conducted a final audit of the code, README and checked the outputs reproducible in its environment. Recraft generated the 20 images used in the external evaluation. See [tool attribution](docs/ATTRIBUTION.md) for details.