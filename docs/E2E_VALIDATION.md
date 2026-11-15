# End-to-end validation and answer to the research question

## Verdict

**The current local version works and exactly reproduces commit `20291cb` when both versions are fully retrained and evaluated on identical data and CPU.** Differences from delivered historical artifacts involve the retrained CNN and also occur with reference code. They are not a regression introduced by reorganization.

For these models, dataset and transformations, additional JPEG barely changes aggregate FPR; resizing often lowers FPR but substantially degrades AI detection; the combination mostly follows resizing. Original JPEG separation between classes limits generalization to intrinsic AI-art detection.

## Validation protocol

Three objects are distinguished:

- **A**: code extracted from `20291cb7d40d6df8d5d3c2d232c969dad7562e66`;
- **B**: actual current local files, including uncommitted code/other changes, copied before validation;
- **H**: delivered historical artifacts, separately preserved before execution in either layout. The historical README reports MPS training.

In scientific tables, “original image” means CSV condition `reference`, not the reference Git commit.

Generated models, CSVs and figures were removed from A/B before execution, preventing silent cache reuse. Only the manual-review CSV, a documentary input, was kept. Both copies use linked shared raw data and identical 20 Recraft PNGs verified with `SHA256SUMS.txt`. No API call. README/entry points in both versions were read to identify equivalent commands, including the final manifest check: a separate command in A, now part of B's pipeline.

Local date: 15/11/2026 17:33 CET. Both suites ran concurrently for 28.7 minutes total, excluding copy preparation and extra audits. The main pipeline took 845.1 seconds in each copy.

| Setting | Value |
|---|---|
| Python | 3.9.6 (default, Apr 30 2025, 02:07:17)  |
| Backend | CPU; MPS unavailable in this environment |
| Threads PyTorch | 4 ; interop : 8 |
| PyTorch / NumPy / scikit-learn / Pillow | 2.7.1 / 2.0.2 / 1.6.1 / 11.3.0 |
| Environment | Same .venv_repro executable; installed versions match identical requirements |
| Shared variables | OMP_NUM_THREADS=4 ; PYTHONHASHSEED=0 ; MPLBACKEND=Agg |
| Sampling | Seed 42; 4 200 train, 600 validation, 1 200 test |
| CNN | Seeds 42/43/44; 8 epochs each; batch 32; Adam lr=0.001 |
| Thresholds | Chosen on 300 human validation artworks; budget 15 FP |
| Bootstrap | 2 000 paired resamples, seed 42, 600 human test artworks |
| Tests FP | Exact two-sided McNemar; Holm over 18 comparisons for main seed |

Hash of the 19 local Python sources: `eef78688321953f2c75e64287371baa8bd97b6252094df983fb3d32db161b0d5`.
Per-file hashes are in [E2E_COMPARISON.json](E2E_COMPARISON.json).

## A versus B comparison

Predictions were aligned by image ID, model, seed and condition, not simply row number. Recraft keys are PNG filename × model. Labels, styles and sources are also checked.

| Item | A/B result |
|---|---|
| Manifest and splits | 6 000 unique paths/IDs; same 60 source × style × split strata |
| Manifest SHA-256 | e04565ff1a9b44a817501680db0dafb316d58e122429bac526b414e5b205a606 |
| CSV | 23/23 byte-identical: 22 regenerated, manual review retained |
| Models | 5/5 files identical; all coefficients, weights and thresholds identical |
| CNN | 48 epochs total, 8 per version × seed; epoch 8 selected for all three seeds in A/B |
| Test predictions | 75 600 rows per version; max score difference 0; no decision changes |
| Recraft | 60 predictions per version; identical scores/decisions |
| Thresholds/validation | 5/5 thresholds exactly recovered from scores; 15/300 FP per model |
| AUC and metrics | Identical across 63 model × condition × seed groups |
| Bootstrap CI | 63/63 intervals exactly reconstructed from aligned predictions |
| Paired tests | Identical tests/adjusted p; seed-43/44 AUC/tests also audited from predictions |
| Tables | report_tables.md identical |
| Plotted data | Same FPR deltas, CI bounds, detection rates and references; same Recraft counts |
| Regenerated figures | 5 PNG and 2 PDF byte-identical; PDF-date-neutral comparison also verified |
| Regressions | None detected; no code fix required |

## Validation calibration and thresholds

Scores were recomputed from newly saved models on 600 validation images. Logistic matrices follow saved feature names; the CNN reads images in the same batch size 32. Complete score tables match between versions. Recomputed thresholds exactly match saved thresholds, including `nextafter`.

| Model | Seed | Threshold A = B | FP / 300 humans | AI detected / 300 | Validation AUC |
|---|---|---|---|---|---|
| LR 55 features | — | 0.7903385087201857 | 15 | 95 | 0.828022 |
| LR without 3 FFT features | — | 0.7582680947254987 | 15 | 94 | 0.776367 |
| CNN | 42 | 0.802436649799347 | 15 | 151 | 0.899956 |
| CNN | 43 | 0.6667082309722902 | 15 | 122 | 0.883478 |
| CNN | 44 | 0.7040196657180787 | 15 | 151 | 0.893144 |

The 5% target is empirical on validation, not a guarantee of 5% FPR on new test data. The 600 human and 600 AI test images are strictly separate from calibration.

## Comparison with historical artifacts H

Logistic and JPEG results reproduce, including models/thresholds. The CPU-retrained CNN differs from historical MPS models reported in README. The same difference occurs with A, so reorganization is not its cause. Historical MPS execution was not replayed; each difference cannot be uniquely attributed to backend rather than the complete numerical conditions of that earlier run. Distinguish reevaluating existing MPS weights on CPU (often tiny score differences) from eight-epoch CPU retraining, where differences can amplify.

| CNN seed | Changed scores / 8 400 | Max absolute Δ score | Changed decisions / 8 400 | Distinct images / 1 200 | Original-condition FP H → B | Original AI detection H → B | Original AUC H → B |
|---|---|---|---|---|---|---|---|
| 42 | 8400 | 0.176604 | 132 | 52 | 23 → 28 | 298 → 302 | 0.910167 → 0.908064 |
| 43 | 8400 | 0.570022 | 437 | 127 | 22 → 28 | 268 → 257 | 0.898050 → 0.896658 |
| 44 | 8400 | 0.147252 | 129 | 47 | 29 → 34 | 322 → 317 | 0.895472 → 0.891264 |

One image can change in several conditions: image × condition decisions are not independent subjects. Historical/retrained CNN thresholds are below; selected epoch stays 8 throughout.

| Seed | Threshold H | Threshold A = B | Δ threshold |
|---|---|---|---|
| 42 | 0.8154128789901735 | 0.802436649799347 | -0.012976229 |
| 43 | 0.7227054238319398 | 0.6667082309722902 | -0.055997193 |
| 44 | 0.6970588564872743 | 0.7040196657180787 | +0.006960809 |

Main-seed differences by condition:

| Condition | FP H → B | AI detected H → B | Δ FPR pp | Δ detection pp | Δ AUC | Changed human decisions | Changed AI decisions |
|---|---|---|---|---|---|---|---|
| Original image | 23 → 28 | 298 → 302 | +0.833 | +0.667 | -0.002103 | 5 | 16 |
| JPEG 85 | 23 → 28 | 295 → 299 | +0.833 | +0.667 | -0.002183 | 5 | 12 |
| JPEG 60 | 23 → 28 | 295 → 298 | +0.833 | +0.500 | -0.002194 | 5 | 11 |
| JPEG 35 | 24 → 28 | 296 → 297 | +0.667 | +0.167 | -0.002169 | 6 | 11 |
| 128 → 256 | 21 → 26 | 252 → 260 | +0.833 | +1.333 | -0.002539 | 5 | 20 |
| 64 → 256 | 20 → 23 | 179 → 188 | +0.500 | +1.500 | -0.004019 | 3 | 13 |
| 128 → 256 then JPEG 60 | 21 → 26 | 254 → 261 | +0.833 | +1.167 | -0.002594 | 5 | 15 |

Main conclusions persist: no established FPR increase with JPEG alone, reduced detection under resizing, and a combination close to resizing. Exact values and some CNN error identities do not reproduce H. Even the sign of a one-false-positive change can reverse; avoid fine-grained claims about a particular image or exact invariance. Recraft decisions stay 2/20, 1/20 and 0/20 despite changed CNN scores.

| CSV | A = B in bytes | B = H in bytes |
|---|---|---|
| dataset_manifest.csv | yes | yes |
| duplicate_exclusions.csv | yes | yes |
| exact_duplicates.csv | yes | yes |
| false_positive_flip_tests.csv | yes | no |
| feature_preview.csv | yes | yes |
| frequency_ablation_comparison.csv | yes | yes |
| image_inventory.csv | yes | yes |
| image_inventory_deduplicated.csv | yes | yes |
| jpeg_metadata_baseline.csv | yes | yes |
| jpeg_quality_equivalents.csv | yes | yes |
| jpeg_quantization_tables.csv | yes | yes |
| manifest_near_duplicate_candidates.csv | yes | yes |
| near_duplicate_candidates.csv | yes | yes |
| near_duplicate_review.csv | yes | yes |
| recraft_predictions.csv | yes | no |
| robustness_summary.csv | yes | no |
| robustness_summary_seed_43.csv | yes | no |
| robustness_summary_seed_44.csv | yes | no |
| test_predictions.csv | yes | no |
| test_predictions_seed_43.csv | yes | no |
| test_predictions_seed_44.csv | yes | no |
| test_score_analysis.csv | yes | no |
| train_validation_features.csv | yes | yes |

## Quantified answer to the research question

> How do JPEG compression and image resizing affect AI-art detection performance,
> particularly the false-positive rate on human-created artworks?

Results below: **validated CPU version, main seed 42**, with identical validation-fixed thresholds across conditions. FPR denominators are 600 human artworks; detection denominators are 600 Latent Diffusion images. Originals are already JPEG, so JPEG conditions add recompression. Resizing downscales to 128 or 64 then restores to 256 with LANCZOS. Only one combination/order is tested: 128 → 256, then JPEG 60.

| Model | Condition | FP / 600 | FPR % | AI detected / 600 | Detection % | AUC |
|---|---|---|---|---|---|---|
| LR 55 features | Original image | 30 | 5.00 | 186 | 31.00 | 0.809553 |
| LR 55 features | JPEG 85 | 31 | 5.17 | 197 | 32.83 | 0.815456 |
| LR 55 features | JPEG 60 | 31 | 5.17 | 228 | 38.00 | 0.836364 |
| LR 55 features | JPEG 35 | 30 | 5.00 | 171 | 28.50 | 0.804944 |
| LR 55 features | 128 → 256 | 11 | 1.83 | 63 | 10.50 | 0.775081 |
| LR 55 features | 64 → 256 | 19 | 3.17 | 69 | 11.50 | 0.722089 |
| LR 55 features | 128 → 256 then JPEG 60 | 13 | 2.17 | 64 | 10.67 | 0.769711 |
| LR without 3 FFT features | Original image | 37 | 6.17 | 180 | 30.00 | 0.768639 |
| LR without 3 FFT features | JPEG 85 | 37 | 6.17 | 178 | 29.67 | 0.768228 |
| LR without 3 FFT features | JPEG 60 | 40 | 6.67 | 195 | 32.50 | 0.776378 |
| LR without 3 FFT features | JPEG 35 | 39 | 6.50 | 182 | 30.33 | 0.764925 |
| LR without 3 FFT features | 128 → 256 | 22 | 3.67 | 95 | 15.83 | 0.744672 |
| LR without 3 FFT features | 64 → 256 | 13 | 2.17 | 54 | 9.00 | 0.718408 |
| LR without 3 FFT features | 128 → 256 then JPEG 60 | 23 | 3.83 | 96 | 16.00 | 0.738067 |
| CNN | Original image | 28 | 4.67 | 302 | 50.33 | 0.908064 |
| CNN | JPEG 85 | 28 | 4.67 | 299 | 49.83 | 0.905189 |
| CNN | JPEG 60 | 28 | 4.67 | 298 | 49.67 | 0.903783 |
| CNN | JPEG 35 | 28 | 4.67 | 297 | 49.50 | 0.904414 |
| CNN | 128 → 256 | 26 | 4.33 | 260 | 43.33 | 0.900822 |
| CNN | 64 → 256 | 23 | 3.83 | 188 | 31.33 | 0.874239 |
| CNN | 128 → 256 then JPEG 60 | 26 | 4.33 | 261 | 43.50 | 0.899719 |

### JPEG alone

- **LR 55**: original FPR 5.00%, then 5.17 / 5.17 / 5.00% at qualities 85 / 60 / 35. AI detection 31.00%, then 32.83 / 38.00 / 28.50%. Detection is therefore not monotonic in quality.
- **LR without 3 FFT features**: FPR 6.17%, then 6.17 / 6.67 / 6.50%; detection 30.00%, then 29.67 / 32.50 / 30.33%.
- **CNN**: FPR 4.67% across all four conditions; detection 50.33%, then 49.83 / 49.67 / 49.50%.

All seed-42 JPEG FPR tests have Holm-adjusted p = 1. This proves neither formal equivalence nor per-image decision invariance. At JPEG 35, LR 55 retains 30 false positives but gains four and resolves four: eight human artworks switch class. LR without FFT changes twelve (7 new FP, 5 resolved); CNN changes two (1 new, 1 resolved).

### Resizing alone

At 128 then 64 px, LR 55 FPR becomes 1.83 / 3.17%, but AI detection only 10.50 / 11.50%. LR without FFT reaches 3.67 / 2.17% FPR and 15.83 / 9.00% detection. CNN reaches 4.33 / 3.83% FPR and 43.33 / 31.33% detection.

FPR decreases are supported after Holm for LR 55 at 128 (p = 0.000356674) and LR without FFT at 128 (p = 0.00411987) and 64 (p = 0.0000144839). Not for LR 55 at 64 (p = 0.679107) or CNN (p = 1 at 128, 0.75 at 64). Detection losses are descriptive: supplied tests concern human FP, not AI detection.

Lower FPR does not mean overall improvement: many more AI images pass undetected at the original threshold. AUCs also fall, especially at 64: 0.809553 → 0.722089 for LR 55, 0.768639 → 0.718408 without FFT and 0.908064 → 0.874239 for CNN.

### Resizing then JPEG combination

FPR/detection: LR 55 **2.17 / 10.67%**, LR without FFT **3.83 / 16.00%**, CNN **4.33 / 43.50%**. Both logistic FPR reductions are supported (Holm 0.00122070 and 0.00726318), CNN's is not (p = 1).

Relative to resizing at 128 alone, adding JPEG 60 increases FPR by 0.33 point for LR 55, 0.17 without FFT and 0 for CNN. Detection increases only 0.17 point for each. For CNN, that small net increase masks 3 gained and 2 lost detections. The combination is close to resizing in this protocol, without proving no interaction at other orders or intensities.

### All seed-42 decision flips

AI columns are descriptive, not extra confirmatory paired tests. Each row compares a variant to the same model's original-image condition.

| Model | Condition | New FP | Resolved FP | Humans changed | AI gained detections | AI lost detections | Δ FPR pp | Bootstrap CI 95% pp | p Holm FP |
|---|---|---|---|---|---|---|---|---|---|
| LR 55 features | JPEG 85 | 3 | 2 | 5 | 11 | 0 | +0.167 | [-0.500 ; 0.837] | 1 |
| LR 55 features | JPEG 60 | 3 | 2 | 5 | 43 | 1 | +0.167 | [-0.500 ; 1.000] | 1 |
| LR 55 features | JPEG 35 | 4 | 4 | 8 | 6 | 21 | +0.000 | [-1.000 ; 1.000] | 1 |
| LR 55 features | 128 → 256 | 1 | 20 | 21 | 0 | 123 | -3.167 | [-4.667 ; -1.833] | 0.000356674 |
| LR 55 features | 64 → 256 | 8 | 19 | 27 | 8 | 125 | -1.833 | [-3.500 ; 0.000] | 0.679107 |
| LR 55 features | 128 → 256 then JPEG 60 | 1 | 18 | 19 | 0 | 122 | -2.833 | [-4.167 ; -1.500] | 0.0012207 |
| LR without 3 FFT features | JPEG 85 | 2 | 2 | 4 | 4 | 6 | +0.000 | [-0.667 ; 0.667] | 1 |
| LR without 3 FFT features | JPEG 60 | 5 | 2 | 7 | 19 | 4 | +0.500 | [-0.333 ; 1.333] | 1 |
| LR without 3 FFT features | JPEG 35 | 7 | 5 | 12 | 9 | 7 | +0.333 | [-0.833 ; 1.500] | 1 |
| LR without 3 FFT features | 128 → 256 | 1 | 16 | 17 | 0 | 85 | -2.500 | [-4.000 ; -1.333] | 0.00411987 |
| LR without 3 FFT features | 64 → 256 | 1 | 25 | 26 | 0 | 126 | -4.000 | [-5.667 ; -2.500] | 1.44839e-05 |
| LR without 3 FFT features | 128 → 256 then JPEG 60 | 1 | 15 | 16 | 0 | 84 | -2.333 | [-3.667 ; -1.167] | 0.00726318 |
| CNN | JPEG 85 | 0 | 0 | 0 | 2 | 5 | +0.000 | [0.000 ; 0.000] | 1 |
| CNN | JPEG 60 | 0 | 0 | 0 | 2 | 6 | +0.000 | [0.000 ; 0.000] | 1 |
| CNN | JPEG 35 | 1 | 1 | 2 | 2 | 7 | +0.000 | [-0.500 ; 0.500] | 1 |
| CNN | 128 → 256 | 0 | 2 | 2 | 1 | 43 | -0.333 | [-0.833 ; 0.000] | 1 |
| CNN | 64 → 256 | 0 | 5 | 5 | 6 | 120 | -0.833 | [-1.667 ; -0.167] | 0.75 |
| CNN | 128 → 256 then JPEG 60 | 0 | 2 | 2 | 1 | 42 | -0.333 | [-0.833 ; 0.000] | 1 |

### CNN seed sensitivity

Regressions are identical in all three seed files; only CNNs are retrained. The three CNNs evaluate the same 1 200 images, so repetitions do not multiply independent subjects.

| Seed | Condition | FPR % | Detection % | AUC | Humans changed vs original | p Holm FP (audit) |
|---|---|---|---|---|---|---|
| 42 | Original image | 4.67 | 50.33 | 0.908064 | 0 | — |
| 42 | JPEG 85 | 4.67 | 49.83 | 0.905189 | 0 | 1 |
| 42 | JPEG 60 | 4.67 | 49.67 | 0.903783 | 0 | 1 |
| 42 | JPEG 35 | 4.67 | 49.50 | 0.904414 | 2 | 1 |
| 42 | 128 → 256 | 4.33 | 43.33 | 0.900822 | 2 | 1 |
| 42 | 64 → 256 | 3.83 | 31.33 | 0.874239 | 5 | 0.75 |
| 42 | 128 → 256 then JPEG 60 | 4.33 | 43.50 | 0.899719 | 2 | 1 |
| 43 | Original image | 4.67 | 42.83 | 0.896658 | 0 | — |
| 43 | JPEG 85 | 4.67 | 42.00 | 0.893767 | 0 | 1 |
| 43 | JPEG 60 | 4.50 | 41.50 | 0.890850 | 1 | 1 |
| 43 | JPEG 35 | 4.50 | 42.33 | 0.895614 | 1 | 1 |
| 43 | 128 → 256 | 4.17 | 34.50 | 0.884867 | 3 | 1 |
| 43 | 64 → 256 | 3.83 | 26.00 | 0.855658 | 5 | 0.75 |
| 43 | 128 → 256 then JPEG 60 | 4.33 | 34.50 | 0.886417 | 2 | 1 |
| 44 | Original image | 5.67 | 52.83 | 0.891264 | 0 | — |
| 44 | JPEG 85 | 5.00 | 52.17 | 0.889931 | 4 | 1 |
| 44 | JPEG 60 | 5.33 | 52.33 | 0.890806 | 2 | 1 |
| 44 | JPEG 35 | 5.17 | 52.17 | 0.889525 | 3 | 1 |
| 44 | 128 → 256 | 5.33 | 46.00 | 0.881742 | 4 | 1 |
| 44 | 64 → 256 | 4.83 | 33.83 | 0.858122 | 7 | 1 |
| 44 | 128 → 256 then JPEG 60 | 5.00 | 46.50 | 0.882628 | 6 | 1 |

Across seeds 42/43/44, original detection ranges 42.83–52.83%, versus 26.00–33.83% at 64 px. Small JPEG FPR changes and resizing detection losses persist. Audited CNN tests remain nonsignificant after Holm. Seed analyses describe sensitivity conditional on these three training runs, not a general guarantee. For 43/44, AUCs/tests were reconstructed from predictions with identical formulas; original scripts supplied only predictions/summaries.

### Frequency-feature ablation

Ablation removes only three FFT-energy shares, not all pixel frequency information: color/edges remain transformation-sensitive. At 64 px, the FPR-change contrast “without FFT minus full” is −2.17 points, bootstrap CI [−4.17; −0.33]. These six intervals have no multiplicity correction. Ablation neither removes resizing effects nor proves that three FFT features cause JPEG bias.

### Existing Recraft images, without API generation

| Model | Detected / 20 | Rate |
|---|---|---|
| LR 55 features | 2/20 | 10.00 % |
| LR without 3 FFT features | 1/20 | 5.00 % |
| CNN | 0/20 | 0.00 % |

Counts match in A, B and H. All 20 images are AI, initially PNG 1 024 × 1 024 then resized to 256, so they cannot estimate external human FPR. Low detection shows limited generalization to this small set. Generator, content, format and processing history change together; no cause is isolated.

## Methodological limitations already present in 20291cb

### Original JPEG encoding and causal attribution

The audit recovers AUC **1.0000** for quantization profile and **0.9837** for file size on the selected test. Profile scores are learned on train; file-size direction is chosen on train. Observed tables exactly match Pillow quality 95 for Human and 75 for LD; profile IDs 1/2 are identifiers, not quality levels. Table equivalence does not prove which encoder/setting originally produced a JPEG.

| Split | Source | Profile | Images |
|---|---|---|---|
| test | Human | 2 | 600 |
| test | Latent Diffusion | 1 | 600 |
| train | Human | 2 | 2100 |
| train | Latent Diffusion | 1 | 2100 |
| validation | Human | 2 | 300 |
| validation | Latent Diffusion | 1 | 300 |

This demonstrates nearly perfect association of class with available encoding cues, not that these detectors read metadata. Models receive only RGB or RGB-derived color/edges/FFT. They **could** exploit pixel encoding traces; the audit does not measure their causal contribution. File size also reflects content/compressibility, not just JPEG quality.

Common recompression does not erase previous compression. Applying JPEG 85 to Human originals near Q95 and AI originals near Q75 does not create identical histories. Results measure robustness to extra processing, conditional on these data/trained models, without isolating intrinsic AI detection.

Proposed future control, separate from validation: balance compression histories across classes, start from precompression sources if possible, then retrain/evaluate varied profiles and external datasets/generators. None of these methodological changes was introduced in this comparison.

### Bootstrap CIs and small counts

Percentile-bootstrap CIs are neither adjusted for the 18-test family nor inversions of exact McNemar tests. Current example, also obtained with A: CNN at 64 has zero new FP and five resolved, Δ FPR −0.833 point, CI [−1.667; −0.167], but exact McNemar p = 0.0625 and Holm p = 0.75. The empirical bootstrap distribution has no observed new FP to draw in the opposite direction. Do not infer an established CNN decrease from that CI alone. Significance conclusions follow corrected exact tests.

Proposed future improvement: prespecify paired tests/intervals suitable for small counts and a consistent multiplicity strategy. Historical procedure is retained here and its limitation stated.

### Scope of the question

The study balances ten styles but main AI images come from LD, not every generator. Thresholds/tests condition on a training run; three seeds do not cover all variation. No significant JPEG difference is not equivalence. Exact/dHash checks reduce duplication risk without proving absence of cropped/retouched variants or artist dependencies.

## Executed commands and traceability

`$PY` is the single absolute `.venv_repro/bin/python` executable in the source working directory. Commands ran in the indicated copies, with Python/Matplotlib caches in the test directory. Reference comes from `git archive 20291cb`; B from actual local files without copying environments, `.git` or the local application.

| Copy | Stage | Command | Time s |
|---|---|---|---|
| reference | integrity | `shasum -a 256 -c data/recraft/SHA256SUMS.txt` | 0.2 |
| current | integrity | `shasum -a 256 -c data/recraft/SHA256SUMS.txt` | 0.2 |
| current | main | `$PY run_all.py` | 845.1 |
| reference | main | `$PY run_all.py` | 845.1 |
| reference | manifest-near | `$PY check_manifest_near_duplicates.py` | 5.0 |
| current | feature-preview | `$PY -m analysis.image_features` | 0.2 |
| current | train-43 | `$PY -m analysis.train_cnn --seed 43` | 351.3 |
| reference | feature-preview | `$PY image_features.py` | 0.2 |
| reference | train-43 | `$PY train_cnn.py --seed 43` | 346.3 |
| current | evaluate-43 | `$PY -m analysis.evaluate_robustness --seed 43` | 79.6 |
| reference | evaluate-43 | `$PY evaluate_robustness.py --seed 43` | 79.6 |
| current | train-44 | `$PY -m analysis.train_cnn --seed 44` | 359.8 |
| reference | train-44 | `$PY train_cnn.py --seed 44` | 359.8 |
| reference | evaluate-44 | `$PY evaluate_robustness.py --seed 44` | 81.9 |
| current | evaluate-44 | `$PY -m analysis.evaluate_robustness --seed 44` | 81.9 |
| reference | recraft-evaluation | `$PY evaluate_recraft.py` | 2.3 |
| current | recraft-evaluation | `$PY -m analysis.evaluate_recraft` | 2.3 |
| reference | recraft-figure | `$PY plot_recraft.py` | 0.5 |
| current | recraft-figure | `$PY -m analysis.figures recraft` | 0.5 |

Additional audits: `audit_validation.py reference/current` (validation scores/thresholds), CSV/weight alignment and comparison, independent AUC/flip/exact-test/bootstrap reconstruction, `pip check`, all pinned requirements versions, protected-file SHA-256 and `git diff --check`.

Logs certify eight epoch lines per version × seed. Audit scripts and full logs are retained with the copies.

## Preservation and unchecked items

All 87 protected existing files retain SHA-256: code, README, CSVs, models, figures, documents and Recraft files. The size/mtime index of 185,016 shared raw files is unchanged; it is not presented as a complete historical pixel hash. Copies/logs are in Git-ignored `local/e2e-validation-2026-11-15/`. New CPU models/results replace no delivered artifacts.

Unchecked: replay on unavailable historical MPS, exact provenance/version of the unrecorded Kaggle archive, original human visual review (not repeated), and direct causality between JPEG traces/model decisions. The historical manifest near-duplicate sheet has no generator in the reference; it is retained while candidate lists are recomputed. API generation is intentionally excluded.

No regression required a fix. No scientific calculation or existing code changed during this validation. No commit, staging or push. Only this note and [E2E_COMPARISON.json](E2E_COMPARISON.json) are added to files proposed for review; the validation directory stays local/ignored.