# Detector generalization to Stable Diffusion in AI-ArtBench

## Protocol and provenance

Complementary evaluation of three distinct groups: unchanged **600 Human and 600 Latent Diffusion (LD)** from the main test, and **600 Stable Diffusion (SD)** from the official test, a source absent from local training. No retraining or recalibration. Use both historical regressions and historical seed-42 CNN, byte-identical to `20291cb`. CPU-retrained/augmented models are not used.

The [authors' AI-ArtBench paper, ArtBrain](https://arxiv.org/html/2412.01512v1#S4.SS1) describes SD v2-1, JPEG 768 × 768 and 52 923 train / 10 000 test counts (Table 1, Section 4.1), linking to [AI-ArtBench on Kaggle](https://www.kaggle.com/datasets/ravidussilva/real-ai-art). Local `data/raw/Real_AI_SD_LD_Dataset/{train,test}/AI_SD_<style>/` structure/counts agree. Generator attribution relies on these sources/data organization, not pixels alone. Downloaded archive version/hash were not recorded; this does not authenticate the original archive.

All **62 923 SD file headers on disk** were reread: JPEG RGB **768 × 768**, 4:2:0. Official test has 1 000 images per ten styles. Selected images are fully decoded; unselected SD checked at header level only. SD training directories never fit a detector or select images.

## Selection, duplicates and manifest

- Single `random.Random(42)`, sorted styles/paths. Shuffle each style's 1 000 official test paths; retain first 60 eligible, independently of scores/performance.
- Compare candidates to **4 800 actual training/validation images** (4 200 + 600), 1 200 main-test references and already selected SD.
- SHA-256 duplicate checks: file bytes, native RGB pixels with dimensions, normalized 256 × 256 RGB. The last covers actual post-reduction model inputs.
- **No exclusions**: quotas of 60 images per style met without exact duplicates. Exclusion CSV contains header only. These checks do not exclude perceptual near-duplicates/imitations.
- Complementary manifest has **1 800 images**, 60 per style/source, preserving main Human/LD IDs/paths. Distinct SD IDs `sd_test_000001` through `sd_test_000600`. Native dimensions, file/pixel hashes and 256-px reference hashes are saved.

## Common preparation and thresholds

For every image: **decode RGB → if needed resize to 256 × 256 with `Image.Resampling.LANCZOS` → call `make_variants`**. No added JPEG recompression, EXIF correction or ICC normalization at this 256-px step. Human/LD already at 256 × 256 keep exact pixels. SD's 768 → 256 reduction precedes experimental transformations.

All seven historical conditions start from that reference: original 256, JPEG 85/60/35 at 4:2:0, reduce to 128/64 then restore 256 with LANCZOS, and 128 then JPEG 60. Every variant stays RGB 256 × 256. Reference/variant hashes are verified/exported. SD reference is not native 768 px; it already includes reduction.

All groups/conditions share environment and **CPU** CNN: Python 3.9.6, NumPy 2.0.2, scikit-learn 1.6.1, Torch 2.7.1, Pillow 11.3.0, four threads, batch size 32. Evaluate 1 200 Human/LD first, then 600 SD in separate batches to preserve even the final main-reference batch.

Saved thresholds applied unchanged to every source/condition (score ≥ threshold in float64):

- Full logistic regression: `0.7903385087201857`.
- Without-frequency logistic regression: `0.7582680947254987`.
- Historical CNN 42: `0.8154128789901735`.

## Commands actually executed

Reproduction command from the root:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv_repro/bin/python -m analysis.evaluate_stable_diffusion
```

Actual execution in isolated `local/stable-diffusion-generalization-2026-11-15/evaluation/`, same interpreter and read-only raw data. Process time: **113.65 s**, including 111.15 s after imports. Outputs are dedicated to this experiment.

Reference control ran **in another isolated copy**, `local/stable-diffusion-generalization-2026-11-15/baseline/`, with identical historical models and CPU:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=0 MPLBACKEND=Agg "$PY" -m analysis.evaluate_robustness
```

`PY` is the absolute `.venv_repro/bin/python` path. Control took 59.21 s, writing only in that copy, not replacing historical delivered predictions.

## Results: distinct sources, shared human FPR

All rates are percentages; each source/condition has **n = 600**. Each AUC compares the same 600 Human to the indicated AI group, totaling 1 200 images. Human FPR is **one shared rate**, not two independent estimates. SD − LD is descriptive in percentage points, without pairing the two sources' images.

| Model | Condition | FPR Human (%) | LD detection (%) | SD detection (%) | SD − LD (pp) | AUC H/LD | AUC H/SD |
|---|---|---:|---:|---:|---:|---:|---:|
| Full logistic regression | 256-px reference | 5.00 | 31.00 | 3.33 | -27.67 | 0.809553 | 0.495608 |
| Full logistic regression | JPEG 85 | 5.17 | 32.83 | 3.83 | -29.00 | 0.815456 | 0.503247 |
| Full logistic regression | JPEG 60 | 5.17 | 38.00 | 3.83 | -34.17 | 0.836364 | 0.490086 |
| Full logistic regression | JPEG 35 | 5.00 | 28.50 | 3.50 | -25.00 | 0.804944 | 0.480872 |
| Full logistic regression | 128 px | 1.83 | 10.50 | 0.83 | -9.67 | 0.775081 | 0.447567 |
| Full logistic regression | 64 px | 3.17 | 11.50 | 0.83 | -10.67 | 0.722089 | 0.495239 |
| Full logistic regression | 128 px then JPEG 60 | 2.17 | 10.67 | 0.83 | -9.83 | 0.769711 | 0.456608 |
| Without-frequency logistic regression | 256-px reference | 6.17 | 30.00 | 4.83 | -25.17 | 0.768639 | 0.451289 |
| Without-frequency logistic regression | JPEG 85 | 6.17 | 29.67 | 5.17 | -24.50 | 0.768228 | 0.455867 |
| Without-frequency logistic regression | JPEG 60 | 6.67 | 32.50 | 5.33 | -27.17 | 0.776378 | 0.448264 |
| Without-frequency logistic regression | JPEG 35 | 6.50 | 30.33 | 5.00 | -25.33 | 0.764925 | 0.442228 |
| Without-frequency logistic regression | 128 px | 3.67 | 15.83 | 0.67 | -15.17 | 0.744672 | 0.423467 |
| Without-frequency logistic regression | 64 px | 2.17 | 9.00 | 0.00 | -9.00 | 0.718408 | 0.441633 |
| Without-frequency logistic regression | 128 px then JPEG 60 | 3.83 | 16.00 | 0.50 | -15.50 | 0.738067 | 0.431258 |
| Historical CNN 42 | 256-px reference | 3.83 | 49.67 | 45.50 | -4.17 | 0.910167 | 0.851539 |
| Historical CNN 42 | JPEG 85 | 3.83 | 49.17 | 44.33 | -4.83 | 0.907372 | 0.848331 |
| Historical CNN 42 | JPEG 60 | 3.83 | 49.17 | 44.33 | -4.83 | 0.905978 | 0.849208 |
| Historical CNN 42 | JPEG 35 | 4.00 | 49.33 | 44.33 | -5.00 | 0.906583 | 0.849578 |
| Historical CNN 42 | 128 px | 3.50 | 42.00 | 46.50 | +4.50 | 0.903361 | 0.861586 |
| Historical CNN 42 | 64 px | 3.33 | 29.83 | 48.00 | +18.17 | 0.878258 | 0.872883 |
| Historical CNN 42 | 128 px then JPEG 60 | 3.50 | 42.33 | 44.67 | +2.33 | 0.902314 | 0.858761 |

## Changes from each source's own reference

Subtract each transformed image's decision from its **own 256-px reference** in the direction variant minus reference. Rate differences below are percentage points, not LD/SD pairs. CSV also contains AUC differences and 0 → 1 / 1 → 0 counts per source. Reference differences are zero by construction.

| Model | Condition | Δ FPR Human (pp) | Δ detection LD (pp) | Δ detection SD (pp) |
|---|---|---:|---:|---:|
| Full logistic regression | JPEG 85 | +0.17 | +1.83 | +0.50 |
| Full logistic regression | JPEG 60 | +0.17 | +7.00 | +0.50 |
| Full logistic regression | JPEG 35 | +0.00 | -2.50 | +0.17 |
| Full logistic regression | 128 px | -3.17 | -20.50 | -2.50 |
| Full logistic regression | 64 px | -1.83 | -19.50 | -2.50 |
| Full logistic regression | 128 px then JPEG 60 | -2.83 | -20.33 | -2.50 |
| Without-frequency logistic regression | JPEG 85 | +0.00 | -0.33 | +0.33 |
| Without-frequency logistic regression | JPEG 60 | +0.50 | +2.50 | +0.50 |
| Without-frequency logistic regression | JPEG 35 | +0.33 | +0.33 | +0.17 |
| Without-frequency logistic regression | 128 px | -2.50 | -14.17 | -4.17 |
| Without-frequency logistic regression | 64 px | -4.00 | -21.00 | -4.83 |
| Without-frequency logistic regression | 128 px then JPEG 60 | -2.33 | -14.00 | -4.33 |
| Historical CNN 42 | JPEG 85 | +0.00 | -0.50 | -1.17 |
| Historical CNN 42 | JPEG 60 | +0.00 | -0.50 | -1.17 |
| Historical CNN 42 | JPEG 35 | +0.17 | -0.33 | -1.17 |
| Historical CNN 42 | 128 px | -0.33 | -7.67 | +1.00 |
| Historical CNN 42 | 64 px | -0.50 | -19.83 | +2.50 |
| Historical CNN 42 | 128 px then JPEG 60 | -0.33 | -7.33 | -0.83 |

No hypothesis test added. Pairing is used only for an image versus its variants and reference checks. The 600 LD and 600 SD are distinct unpaired groups despite equal style quotas. Conditions reuse identical images.

## Summary for the report

At 256 px, detection goes **31.00% LD → 3.33% SD** for full logistic regression (**−27.67 points**), **30.00% → 4.83%** without frequency (**−25.17 points**), **49.67% → 45.50%** for CNN (**−4.17 points**). Human/SD AUCs are **0.496**, **0.451**, **0.852**, versus **0.810**, **0.769**, **0.910** for Human/LD. Score direction/thresholds stay fixed even with SD AUC < 0.5. Shared human FPRs: **5.00%, 6.17%, 3.83%**.

Regressions generalize poorly to this SD sample under this preparation. CNN retains stronger discrimination but detects only 273/600 SD at its historical threshold, versus 298/600 LD. This measures source transfer, not an isolated causal generator-architecture effect.

Within-source effects differ: at **64 px**, CNN detection decreases **19.83 points** for LD but increases **2.50 points** for SD (45.50 → 48.00%). At 64 px, SD logistic changes are **−2.50 and −4.83 points**, ending at **0.83% and 0.00%**; small absolute losses also reflect already-low baselines. Interpret between-source gaps separately from within-source changes.

## Reference verification and integrity

All **25 200 Human/LD predictions** reproduce **exact scores and decisions** from the main evaluator rerun on CPU with identical checkpoints in all seven conditions. Unresized 256-px references and their variants match.

Separate comparison with delivered historical artifacts, whose CNN predictions used MPS:

| Model | Predictions compared | Maximum absolute score difference | Different decisions |
|---|---:|---:|---:|
| Full logistic regression | 8400 | 0 | 0 |
| Without-frequency logistic regression | 8400 | 0 | 0 |
| Historical CNN 42 | 8400 | 5.66244125366e-07 | 0 |

CPU/MPS CNN differences do not change the method: identical weights/threshold retained. Main source/condition comparisons use CPU only.

Other actual checks: independently replay selection; quotas/test provenance; independent file/native-/normalized-pixel checks for 4 800 train/validation and 600 SD; recompute hashes of **12 600 variants** to confirm normalization before transformations; independently recompute pairing, thresholds, decisions, rates, differences, AUCs and flips; models checked against `20291cb`; Python 3.9 syntax; `git diff --check`.

All **132 protected preexisting files** retain SHA-256: main models/results, augmentation, JPEG intervention, PCA, figures/documents. Main manifest unchanged: `e04565ff1a9b44a817501680db0dafb316d58e122429bac526b414e5b205a606`. No preexisting repository file modified by this experiment.

## Outputs and limitations

- [`analysis/evaluate_stable_diffusion.py`](../analysis/evaluate_stable_diffusion.py): standalone command outside `run_all.py`.
- [`stable_diffusion_manifest.csv`](../results/stable_diffusion_manifest.csv): 1 800 images, paths, source/style, provenance, dimensions/hashes.
- [`stable_diffusion_predictions.csv`](../results/stable_diffusion_predictions.csv): 37 800 scores, decisions, thresholds/backend by image/model/condition.
- [`stable_diffusion_summary.csv`](../results/stable_diffusion_summary.csv): 21 rows of counts, shared FPR, LD/SD detection, two AUCs, source gaps and changes from each reference.
- [`stable_diffusion_pixels.csv`](../results/stable_diffusion_pixels.csv): 12 600 paired references/variants with dimensions/pixel SHA-256.
- [`stable_diffusion_source_audit.csv`](../results/stable_diffusion_source_audit.csv): aggregated inventory of on-disk SD dimensions, formats, styles/encodings.
- [`stable_diffusion_exclusions.csv`](../results/stable_diffusion_exclusions.csv): exclusion log (none here).
- [`STABLE_DIFFUSION_GENERALIZATION.json`](STABLE_DIFFUSION_GENERALIZATION.json): thresholds, models, settings, environment, commands, timings, provenance/audit.

One style-balanced selection of 600 SD; no uncertainty interval or between-source test estimated. Sources differ in generation, encoding and native resolution. SD undergoes initial 768 → 256 reduction absent for others. Original JPEG traces persist without extra compression at normalization. All may contribute to gaps; none is isolated. Checks exclude local-training exact duplicates, not near-duplicates or generator-pretraining overlap.

Logs, copies, scripts and integrity hashes remain in Git-ignored `local/stable-diffusion-generalization-2026-11-15/`. No API generation, training, recalibration, commit or push.