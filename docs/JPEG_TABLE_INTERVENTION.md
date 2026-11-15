# Sensitivity to observed JPEG quantization tables

## Question and main contrast

For each existing test image, compare predictions after two independent re-encodings of the same decoded RGB pixels, using observed Human tables or observed Latent Diffusion (LD) tables. The main contrast is **LD tables minus Human tables**, paired by image ID. Originals provide a descriptive reference, not the main control: they have not undergone the same extra encoding step.

Reuse the main experiment's three models without training: `logistic_full.joblib`, `logistic_no_frequency.joblib` and historical seed-42 CNN `cnn_model.pt`. Their bytes exactly match commit `20291cb`. No augmentation-experiment model is used.

## Reproduction

From the repository root with `.venv_repro`:

```bash
OMP_NUM_THREADS=4 PYTHONHASHSEED=0 ./.venv_repro/bin/python -m analysis.jpeg_table_intervention
```

The command actually ran in isolated copy `local/jpeg-table-intervention-2026-11-15/evaluation/`, using the same interpreter and links to raw data. Code: [`analysis/jpeg_table_intervention.py`](../analysis/jpeg_table_intervention.py). Only the dedicated outputs listed below are written. JPEGs are generated in memory, not retained, but their SHA-256 and decoded-pixel hashes are recorded for identical regeneration.

Environment: Python 3.9.6, NumPy 2.0.2, scikit-learn 1.6.1, Torch 2.7.1, Pillow 11.3.0, libjpeg-turbo 3.1.1 (JPEG API 6.2), four threads. All three models and conditions run on **CPU**, batch size 32. The historical CNN keeps its MPS-trained weights and historical threshold; original images are reevaluated on CPU with that checkpoint, without recalibration.

Measured process time: **28.34 s**; internal time after imports: 24.10 s.

## Tables and encoding

Coefficients come from `results/jpeg_quantization_tables.csv`, checked against the source JPEGs:

- **Human tables**, profile 2: `train/art_nouveau/a-y-jackson_algoma-in-november-1935.jpg`; SHA-256 `a04d4136f3b1ecc945768f629f4b8aac885d835bde7e06de88a89cf164263229`.
- **LD tables**, profile 1: `train/AI_LD_art_nouveau/0-239013729-197363.jpg`; SHA-256 `c28bddb090ef114d9b63d99abbc7ec4a0acf14f7810bf1af4dc580de06c931a6`.

Each profile contains tables 0 (luminance) and 1 (chrominance), each with 64 coefficients. Actual observed coefficients are passed directly as `qtables`, not reconstructed from equivalent quality settings. The existing audit finds exact Pillow quality-95 and quality-75 matches; those quality values are not used in this intervention.

CSV and `Image.quantization` use natural row-major 8 × 8 order. DQT segments store zigzag order: an explicit segment reader converts and checks both source JPEGs and every output JPEG. Two asymmetric tables with distinct coefficients also test against transposition/permutation.

Common settings for both variants:

- Original dimensions preserved (256 × 256 for all 1 200 images), RGB decoded once per image, without extra resizing, ICC color normalization or EXIF correction.
- Identical starting RGB for both encodings, checked by SHA-256 and unchanged pixels between calls. The second variant never starts from the first.
- Sequential baseline JPEG, 8-bit YCbCr; subsampling `2` = **4:2:0**, table 0 for Y and table 1 for Cb/Cr. All 600 Human and 600 LD originals are also 4:2:0.
- Observed `qtables`, `quality=-1` to avoid rescaling, `optimize=False`, `progressive=False`, `smooth=0`, `keep_rgb=False`; no EXIF, ICC or comment forwarded.
- Identical JPEG headers within each pair except DQT: dimensions, components, subsampling, Huffman tables and scan header checked equal. File size and entropy-coded data may vary with tables.
- Coefficients reread with Pillow and from DQT segments exactly match the requested tables for all 2 400 encodings.

Fixed historical thresholds:

- Full logistic regression: `0.7903385087201857` (AI if score ≥ threshold, float64 comparison).
- Without-frequency logistic regression: `0.7582680947254987` (AI if score ≥ threshold, float64 comparison).
- Historical CNN 42: `0.8154128789901735` (AI if score ≥ threshold, float64 comparison).

## Rates and AUC

Each condition contains exactly **600 human artworks and 600 LD images**, aligned on identical IDs.

| Model | Condition | FP / 600 | Human FPR (%) | AI detected / 600 | AI detection (%) | AUC |
|---|---|---:|---:|---:|---:|---:|
| Full logistic regression | Original (descriptive) | 30 | 5.00 | 186 | 31.00 | 0.809553 |
| Full logistic regression | Human tables | 32 | 5.33 | 186 | 31.00 | 0.807708 |
| Full logistic regression | LD tables | 31 | 5.17 | 185 | 30.83 | 0.808572 |
| Without-frequency logistic regression | Original (descriptive) | 37 | 6.17 | 180 | 30.00 | 0.768639 |
| Without-frequency logistic regression | Human tables | 37 | 6.17 | 176 | 29.33 | 0.765819 |
| Without-frequency logistic regression | LD tables | 41 | 6.83 | 177 | 29.50 | 0.763914 |
| Historical CNN 42 | Original (descriptive) | 23 | 3.83 | 298 | 49.67 | 0.910167 |
| Historical CNN 42 | Human tables | 23 | 3.83 | 296 | 49.33 | 0.909344 |
| Historical CNN 42 | LD tables | 23 | 3.83 | 297 | 49.50 | 0.908306 |

## Paired scores and decisions: LD minus Human

Scores are AI-class probabilities; differences below are in **score units [0, 1]**, not percentage points. Each row covers 600 images. Arrows show decision changes from Human tables to LD tables, regardless of true class.

| Model | True class | Mean Δ score | Median Δ score | Scores ↑ / ↓ / = | Decision 0 → 1 | Decision 1 → 0 | Δ positive rate (pp) | p McNemar | p Holm |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| Full logistic regression | Human | -0.00217382 | -0.00154668 | 231 / 369 / 0 | 1 | 2 | -0.17 | 1.000000 | 1.000000 |
| Full logistic regression | Latent Diffusion | +0.00009446 | +0.00018353 | 336 / 264 / 0 | 1 | 2 | -0.17 | 1.000000 | 1.000000 |
| Without-frequency logistic regression | Human | +0.00076640 | -0.00039447 | 290 / 310 / 0 | 6 | 2 | +0.67 | 0.289062 | 1.000000 |
| Without-frequency logistic regression | Latent Diffusion | +0.00058746 | +0.00058783 | 402 / 198 / 0 | 2 | 1 | +0.17 | 1.000000 | 1.000000 |
| Historical CNN 42 | Human | +0.00348931 | +0.00098972 | 432 / 168 / 0 | 0 | 0 | +0.00 | 1.000000 | 1.000000 |
| Historical CNN 42 | Latent Diffusion | +0.00187381 | +0.00110977 | 457 / 143 / 0 | 2 | 1 | +0.17 | 1.000000 | 1.000000 |

Quartiles, minima and maxima of paired differences are also in the `paired` CSV. Score changes remain descriptive; no score tests are added.

Decision tests: **six exact two-sided McNemar tests**, one per model × true class, comparing only the two re-encoded versions of the same images. A **joint Holm correction over six tests** controls this family; originals and AUCs add no tests. Every adjusted p-value is 1. This does not prove no effect: scores change and discordant decisions are few (0 to 8 out of 600).

## Direction of shifts and conclusion

- **CNN**: LD tables increase AI scores in both classes, in mean and median (432/600 human artworks and 457/600 LD images have higher scores). No human false-positive decisions change; among LD images, two change 0 → 1 and one 1 → 0. FPR stays at 3.83%, detection rises 49.33% → 49.50%, and AUC drops 0.909344 → 0.908306.
- **Full logistic regression**: mean/median shifts have opposite signs by class: negative for Human, slightly positive for LD. Decisions nevertheless change both ways in both classes (one 0 → 1 and two 1 → 0). FPR and detection each decrease 0.17 point.
- **Without-frequency logistic regression**: the **mean** shift is positive in both classes, but the Human median is negative and 310/600 scores decrease. It is not a uniform shift toward AI. FPR increases 0.67 point (six new positives, two resolved), detection 0.17 point.

**Conclusion for the report:** with identical starting RGB pixels, observed quantization tables cause small score changes and a few individual decision flips, with model- and class-dependent responses. The CNN shifts toward AI on average under LD tables in both classes, without changing FPR in this test. The six corrected paired tests do not establish an imbalance in decision changes. This measures sensitivity to controlled re-encoding, not the share of historical performance attributable to JPEG bias.

## Limitations and the original encoding bias

Starting RGB pixels have already been decoded from JPEGs: they retain source-correlated quantization, subsampling and compression traces. Human-table re-encoding does not make an LD image equivalent to an original Human image, or vice versa. Classes have different compression histories, including double compression.

Models receive RGB pixels or features computed from them, never DQT headers. Observed differences therefore pass through encoded pixels, not demonstrated metadata reading by detectors. The audit's separation of sources by original tables establishes a potential confounder, neither its use by a particular model nor its causal contribution to historical AUC.

The intervention keeps the test set and models trained under the original bias; it does not replace retraining on data with encoding controlled from the start or decorrelated from class. Small differences here do not rule out a historical effect of that bias. No new seed, split or external evaluation.

The two logistic models' CPU originals match historical predictions exactly. For the CNN, historical MPS → CPU yields maximum absolute score difference `3.5762786865234375e-07` and **no changed decisions** on 1 200 originals. The main comparison does not mix backends: all CNN predictions use CPU.

## Files and checks

- [`jpeg_table_intervention_predictions.csv`](../results/jpeg_table_intervention_predictions.csv): 10 800 scores/decisions, ID, class, model, condition, threshold and backend.
- [`jpeg_table_intervention_summary.csv`](../results/jpeg_table_intervention_summary.csv): nine FPR/detection/AUC rows.
- [`jpeg_table_intervention_paired.csv`](../results/jpeg_table_intervention_paired.csv): six score-change/flip/test/Holm rows.
- [`jpeg_table_intervention_encodings.csv`](../results/jpeg_table_intervention_encodings.csv): 2 400 rows with input, output JPEG and decoded RGB SHA-256, dimensions and checked settings.
- [`jpeg_table_intervention_tables.csv`](../results/jpeg_table_intervention_tables.csv): 256 coefficients actually used, in natural order.
- [`JPEG_TABLE_INTERVENTION.json`](JPEG_TABLE_INTERVENTION.json): tables, sources, model/input hashes, environment, thresholds, timing and independent audit.

Checks actually performed: counts, uniqueness and pairing of 1 200 IDs; thresholds matching checkpoints; models matching `20291cb`; independent recomputation of decisions, rates, AUCs, score statistics, McNemar and Holm; independent regeneration of **2 400 JPEGs** from the coefficient CSV with matching SHA-256; coefficient order with asymmetric tables and odd dimensions; original scores on one batch checked through the existing evaluator; `git diff --check`.

All **112 protected preexisting files** (code, models, historical/augmentation results, figures, documents, Recraft) have identical before/after SHA-256. Manifest unchanged: `e04565ff1a9b44a817501680db0dafb316d58e122429bac526b414e5b205a606`. Test copies, logs, verification scripts and hashes remain in Git-ignored `local/jpeg-table-intervention-2026-11-15/`. Main pipeline unchanged and not retrained. No API call, commit or push.