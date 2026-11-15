# Complementary experiment: CNN augmentation, seed 42

## Result

In this CPU training pair, augmentation modestly attenuates resizing-related detection losses by **2.17 points at 128 px** and **1.67 points at 64 px**, relative to each CNN's original performance. The 64-px loss remains **17.33 points**. Absolute detection gains are 3.67 and 3.17 points. At each model's threshold, observed FPRs are slightly lower with augmentation, without a difference established by corrected tests.

Original-image detection increases 1.50 points and FPR decreases 0.50 point, but AUC drops 0.007867. Augmentation therefore does not dominate every measure. After a joint Holm correction over 14 tests, detection gains at 128 and the combined condition are supported; the 64-px gain is not. Loss attenuation is descriptive, without a formal model × transformation interaction test.

## Preserved protocol and augmentation

- CNN architecture, splits, batch size 32, Adam `lr=0.001`, eight epochs and selection rule unchanged: checkpoint with highest **original-validation** AUC.
- Validation/test are never randomly augmented. Each CNN has its own threshold on 300 original human validation artworks using historical `nextafter`, with a budget of 15 false positives (5%).
- Training only: retain original with probability 1/2; otherwise uniformly choose among six existing variants, probability 1/12 each.
- Same law for both classes: drawing receives neither label nor source. Realized counts may differ by class; no forced balancing of draws.
- Dedicated `random.Random(seed)`, independent of Torch states for initial weights and batch shuffling. Sequence advances per image and epoch, without resetting each epoch. `num_workers=0` is required for this reproducible path.
- Only the drawn variant is computed through `make_variant()`, reusing existing JPEG/LANCZOS primitives. No `make_variants()` call during augmentation. Evaluation still uses seven historical variants.
- Seed **42**, **CPU**, four threads, Python **3.9.6**, PyTorch 2.7.1, NumPy 2.0.2, scikit-learn 1.6.1, Pillow 11.3.0. MPS unavailable: historical MPS control is not a training comparator. Both CNNs were actually retrained in the same environment.

Manifest unchanged:
`e04565ff1a9b44a817501680db0dafb316d58e122429bac526b414e5b205a606`.

| CNN | Selected epoch | Threshold | Validation FP / 300 | Validation AI detected / 300 | Validation AUC |
|---|---|---|---|---|---|
| control | 8 | 0.802436649799347 | 15 | 151 | 0.899956 |
| augmented | 8 | 0.8537070155143739 | 15 | 147 | 0.890156 |

Thresholds are not comparable as confidence levels between networks: each is independently calibrated to the same validation budget. Validation scores were recomputed after checkpoint loading, exactly recovering both thresholds on original images.

## Results on 1 200 test images

FPR uses 600 humans; detection rates use 600 AI images. “Difference” means augmented minus control, in percentage points for rates; AUCs use 0–1 units. Originals are already JPEG; resizing reduces then restores to 256 pixels.

| Condition | Control FPR % | Augmented FPR % | FPR difference pp | Control detection % | Augmented detection % | Detection difference pp | Control AUC | Augmented AUC | AUC difference |
|---|---|---|---|---|---|---|---|---|---|
| Originals | 4.67 | 4.17 | -0.50 | 50.33 | 51.83 | 1.50 | 0.908064 | 0.900197 | -0.007867 |
| JPEG 85 | 4.67 | 4.17 | -0.50 | 49.83 | 50.67 | 0.83 | 0.905189 | 0.898844 | -0.006344 |
| JPEG 60 | 4.67 | 4.17 | -0.50 | 49.67 | 51.67 | 2.00 | 0.903783 | 0.900181 | -0.003603 |
| JPEG 35 | 4.67 | 4.17 | -0.50 | 49.50 | 51.17 | 1.67 | 0.904414 | 0.898347 | -0.006067 |
| 128 → 256 | 4.33 | 4.00 | -0.33 | 43.33 | 47.00 | 3.67 | 0.900822 | 0.896728 | -0.004094 |
| 64 → 256 | 3.83 | 3.00 | -0.83 | 31.33 | 34.50 | 3.17 | 0.874239 | 0.883911 | 0.009672 |
| 128 → 256 then JPEG 60 | 4.33 | 4.17 | -0.17 | 43.50 | 47.50 | 4.00 | 0.899719 | 0.897661 | -0.002058 |

## Changes from each CNN's originals

These differences separate original performance from degradation under transformation. Smaller loss alone would not imply a gain if the model were simply worse on originals; both are shown here. Positive detection “difference of changes” means attenuated loss for the augmented CNN, not an interaction test.

| Condition | Δ Control FPR pp | Δ Augmented FPR pp | Control Δ detection pp | Augmented Δ detection pp | Detection difference of changes pp | Δ Control AUC | Δ Augmented AUC |
|---|---|---|---|---|---|---|---|
| Originals | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000000 | 0.000000 |
| JPEG 85 | 0.00 | 0.00 | -0.50 | -1.17 | -0.67 | -0.002875 | -0.001353 |
| JPEG 60 | 0.00 | 0.00 | -0.67 | -0.17 | 0.50 | -0.004281 | -0.000017 |
| JPEG 35 | 0.00 | 0.00 | -0.83 | -0.67 | 0.17 | -0.003650 | -0.001850 |
| 128 → 256 | -0.33 | -0.17 | -7.00 | -4.83 | 2.17 | -0.007242 | -0.003469 |
| 64 → 256 | -0.83 | -1.17 | -19.00 | -17.33 | 1.67 | -0.033825 | -0.016286 |
| 128 → 256 then JPEG 60 | -0.33 | 0.00 | -6.83 | -4.33 | 2.50 | -0.008344 | -0.002536 |

## Paired CNN tests and individual changes

Within each condition, identical images are paired by ID. Exact two-sided McNemar compares binary decisions separately for humans (event = false positive) and AI (event = detection). **One Holm correction covers all 14 tests: seven conditions × two classes.** AUCs and differences of changes remain descriptive. Tests are conditional on these trained models, not estimates of learning-algorithm variation across seeds.

“Gained/lost” describes an augmented-CNN positive prediction relative to control. For a human, gained positive is a new false positive; for AI, a new detection.

| Condition | Human new FP | Human resolved FP | Raw p FP | p Holm FP | AI gained detections | AI lost detections | Raw p detection | Holm p detection |
|---|---|---|---|---|---|---|---|---|
| Originals | 4 | 7 | 0.548828 | 1.000000 | 22 | 13 | 0.175465 | 1.000000 |
| JPEG 85 | 4 | 7 | 0.548828 | 1.000000 | 21 | 16 | 0.511376 | 1.000000 |
| JPEG 60 | 4 | 7 | 0.548828 | 1.000000 | 26 | 14 | 0.080690 | 0.887595 |
| JPEG 35 | 4 | 7 | 0.548828 | 1.000000 | 22 | 12 | 0.121449 | 1.000000 |
| 128 → 256 | 5 | 7 | 0.774414 | 1.000000 | 37 | 15 | 0.003185 | 0.041408 |
| 64 → 256 | 3 | 8 | 0.226562 | 1.000000 | 34 | 15 | 0.009399 | 0.112791 |
| 128 → 256 then JPEG 60 | 6 | 7 | 1.000000 | 1.000000 | 39 | 15 | 0.001496 | 0.020948 |

At 128 px: 37 gained detections and 15 lost, net 22/600 (Holm p = 0.041408). Combined condition: 39 gained and 15 lost, net 24/600 (Holm p = 0.020948). At 64: 34 gained and 15 lost, net 19/600, but Holm p = 0.112791. All seven FPR tests have Holm p = 1.

## Realized draws

33 600 image presentations (4 200 × 8), including 16 800 in each class. Counts are not forced; intended probabilities are identical. In total, 16 807 originals were drawn, or 50.0208%.

| Class | Original | JPEG85 | JPEG60 | JPEG35 | 128 | 64 | 128 then JPEG60 | % originals |
|---|---|---|---|---|---|---|---|---|
| 0 | 8303 | 1386 | 1389 | 1443 | 1401 | 1417 | 1461 | 49.4226 |
| 1 | 8504 | 1401 | 1385 | 1324 | 1473 | 1410 | 1303 | 50.6190 |

Condition totals exactly match a `Random(42)` sequence replayed for 33 600 draws. Validation consumes none.

## Measured times

Both training runs launched concurrently with four threads per process. These are observed wall times, not a benchmark isolating PIL overhead. Command times include Python startup.

| Stage | Seconds | Minutes |
|---|---|---|
| Control, eight epochs | 359.19 | 5.99 |
| Augmented, eight epochs | 359.18 | 5.99 |
| Both CNNs evaluation and comparison | 63.99 | 1.07 |
| Parallel training then evaluation | 423.18 | 7.05 |

## Reproduction from the root

The experiment stays separate from `run_all.py`. Raw data and the validated manifest are required; filenames below preserve primary models.

```bash
OMP_NUM_THREADS=4 PYTHONHASHSEED=0 ./.venv_repro/bin/python -m analysis.train_cnn --seed 42 --output-model cnn_augmentation_control_seed_42.pt
OMP_NUM_THREADS=4 PYTHONHASHSEED=0 ./.venv_repro/bin/python -m analysis.train_cnn --seed 42 --augment --output-model cnn_augmentation_augmented_seed_42.pt
OMP_NUM_THREADS=4 PYTHONHASHSEED=0 ./.venv_repro/bin/python -m analysis.cnn_augmentation --seed 42
```

Backend choice remains MPS if available, otherwise CPU. Keep environment/backend identical for both training runs and evaluation. These figures come from CPU; MPS reproduction requires retraining both variants and is not expected to be bit-identical.

For the original-behavior check, the actual control ran without options in an isolated copy:

```bash
$PY -m analysis.train_cnn
$PY -m analysis.train_cnn --seed 42 --augment --output-model cnn_augmentation_augmented_seed_42.pt
$PY -m analysis.cnn_augmentation --seed 42
```

The first checkpoint is exported under the separate control filename. Reproduction's `--output-model` only changes destination. The no-option control is byte-identical to the seed-42 CPU CNN from earlier end-to-end validation; its 8 400 scores and predictions also match.

Outputs, without replacing historical artifacts:

- `models/cnn_augmentation_control_seed_42.pt`
- `models/cnn_augmentation_augmented_seed_42.pt`
- `results/cnn_augmentation_predictions_seed_42.csv`: 16 800 predictions;
- `results/cnn_augmentation_summary_seed_42.csv`: 14 metric/change rows;
- `results/cnn_augmentation_comparison_seed_42.csv`: seven comparisons/tests;
- `docs/CNN_AUGMENTATION_SEED_42.json`: thresholds, counts, environment, timings and audits;
- this note.

## Checks and preservation

- Original unaugmented command: eight-epoch training; control matches the previously validated CPU rerun.
- Seven lazy variants exactly match historical pixels on RGB, L and RGBA (21 cases). Call counters verify that only the drawn variant is computed.
- Same-seed RNG replay produces identical sequences; validation/test augmentation is rejected, as are multiple workers.
- Original validation reread with a mock forbidding transformations: 15/300 FP and exact recalibrated thresholds for both checkpoints.
- Predictions, metrics, AUCs, changes and all 14 McNemar/Holm tests independently recomputed from CSVs, aligned by image ID.
- Main inference compared before/after on 16 balanced images, seven conditions and three models: same scores/decisions. CNN-only path matches and skips logistic features.
- `run_all.py --results-only` in a copy: all main CSVs/tables match delivered artifacts.
- Primary checkpoint names protected against `--augment`; `--output-model` cannot escape `models/`.
- Manifest/historical artifact SHA-256 checked; architecture, optimizer, eight epochs and selection rule preserved.

## Limitations

One pair, seed 42: tests do not cover between-training variation. Loss attenuation is not tested as an interaction. AUCs remain descriptive and decrease on originals and most conditions despite gains at the selected operating point. The 64-px detection loss remains large. Training/test transformations share the same six-setting family despite disjoint images; no guarantee for new quality levels, sizes or operation orders.

Augmentation does not prove elimination of original JPEG bias: 50% of presentations stay original and recompression does not erase different class encoding histories. No additional Recraft evaluation/API generation was performed. Seeds 43/44 remained untested for this experiment at the time of this note. The full main pipeline was not rerun; affected paths and the results command were checked as above.

Changes remain local, without commit or push.