# CNN augmentation: seeds 42, 43 and 44

Descriptive comparison on the same 600 human artworks and 600 AI images. Each seed has a control and an augmented CNN trained on CPU in the same Python 3.9.6 / Torch 2.7.1 environment with four threads. Data, splits, architecture, Adam (0.001), batch size 32 and eight epochs are unchanged. Checkpoint selection strictly maximizes original-validation AUC; each model's threshold is calibrated on its 300 original human validation images, then fixed for all seven test conditions.

Augmentation applies only to training with the same law for both classes: original probability 1/2, each of the six variants probability 1/12. A dedicated `random.Random(seed)` independent of Torch advances through eight epochs with `num_workers=0`. Only the drawn variant is computed.

## Results

All three measures are percentage points: gain = augmented detection − control; attenuation = resized gain − original gain; FPR difference = augmented FPR − control. These conditions are resizing alone, without added JPEG.

| Seed | Resolution | Detection gain | Loss attenuation | Human FPR difference |
|---|---|---:|---:|---:|
| 42 | 128 px | +3,67 | +2,17 | -0,33 |
| 42 | 64 px | +3,17 | +1,67 | -0,83 |
| 43 | 128 px | +7,67 | +3,83 | -0,50 |
| 43 | 64 px | +5,67 | +1,83 | -0,33 |
| 44 | 128 px | -3,83 | +0,83 | -1,00 |
| 44 | 64 px | -3,83 | +0,83 | -1,17 |

| Resolution | Gain: mean [min ; max] | Attenuation: mean [min ; max] | FPR: mean [min ; max] |
|---|---:|---:|---:|
| 128 px | +2,50 [-3,83 ; +7,67] | +2,28 [+0,83 ; +3,83] | -0,61 [-1,00 ; -0,33] |
| 64 px | +1,67 [-3,83 ; +5,67] | +1,44 [+0,83 ; +1,83] | -0,78 [-1,17 ; -0,33] |

These ranges are min–max, not confidence intervals. Seeds represent training variation; test images are reused, not pooled into 1 800 independent observations. Cross-seed comparisons and differences of changes remain descriptive. Paired tests with Holm correction in each individual CSV are not tests between seeds.

## Direction of effects

- 128 px, `gain_detection_pp`: signs in 42/43/44 order = positive/positive/negative. Sign changes: yes.
- 128 px, `attenuation_loss_pp`: signs in 42/43/44 order = positive/positive/positive. Sign changes: no.
- 128 px, `difference_fpr_pp`: signs in 42/43/44 order = negative/negative/negative. Sign changes: no.
- 64 px, `gain_detection_pp`: signs in 42/43/44 order = positive/positive/negative. Sign changes: yes.
- 64 px, `attenuation_loss_pp`: signs in 42/43/44 order = positive/positive/positive. Sign changes: no.
- 64 px, `difference_fpr_pp`: signs in 42/43/44 order = negative/negative/negative. Sign changes: no.

## Actual executions and timing

`PY` is the absolute path to `.venv_repro/bin/python`. Shared variables: `OMP_NUM_THREADS=4 PYTHONHASHSEED=0 MPLBACKEND=Agg`; Matplotlib/Python caches are in the test directory. MPS is unavailable here, so training and evaluation use CPU. Control and augmented models train concurrently per pair; seeds run successively. Times below use `time.perf_counter`, including process startup.

The control commands below ran in isolated copies `local/cnn-augmentation-seeds43-44-2026-11-15/seed-{43,44}/`, never at the root: default filenames therefore overwrite no historical model. Before evaluation, each control is copied to `cnn_augmentation_control_seed_<graine>.pt`. Here the historical placeholder `<graine>` means the seed.

```bash
$PY -m analysis.train_cnn --seed 43
$PY -m analysis.train_cnn --seed 43 --augment --output-model cnn_augmentation_augmented_seed_43.pt
$PY -m analysis.cnn_augmentation --seed 43
$PY -m analysis.train_cnn --seed 44
$PY -m analysis.train_cnn --seed 44 --augment --output-model cnn_augmentation_augmented_seed_44.pt
$PY -m analysis.cnn_augmentation --seed 44
```

To reproduce directly in the repository without touching historical names, add `--output-model cnn_augmentation_control_seed_43.pt` to control 43 (respectively 44). The other two commands are unchanged.

| Seed | Control (s) | Augmented (s) | Evaluation of both (s) |
|---|---:|---:|---:|
| 43 | 343.96 | 343.96 | 54.51 |
| 44 | 344.40 | 344.40 | 54.63 |

## Checks and files

- SHA-256 hashes of all preexisting files in `models/`, `results/`, `figures/`, `docs/`, `analysis/`, README and `run_all.py` match before export. Historical MPS artifacts and every seed-42 artifact remain unchanged.
- Manifest identical in both copies and repository: `e04565ff1a9b44a817501680db0dafb316d58e122429bac526b414e5b205a606`.
- Per seed: 16 800 predictions, 14 summaries, seven comparisons; 1 200 IDs aligned per condition, fixed model thresholds, decisions recomputed from scores, counts and AUCs recomputed.
- Calibration reevaluated without augmentation on original validation images: exact threshold and 15/300 false positives for each CNN. All eight epochs and 33 600 augmentation draws verified; RNG sequence replayed.
- CPU controls 43/44 exactly reproduce checkpoints and 8 400 CNN predictions from previous CPU validation runs. Historical MPS models remain separate artifacts.
- Synthesis independently recomputed from all three seeds' decisions; identical 1 200 IDs. `git diff --check` passed.
- Exact McNemar and Holm (14 tests per seed) independently recomputed. No extra cross-seed test.
- New models: `models/cnn_augmentation_{control,augmented}_seed_{43,44}.pt`; detailed CSVs: `results/cnn_augmentation_{predictions,summary,comparison}_seed_{43,44}.csv`; metadata, calibration and timing: `docs/CNN_AUGMENTATION_SEED_{43,44}.json`.
- Synthesis: `results/cnn_augmentation_seed_synthesis.csv` (six rows) and `results/cnn_augmentation_seed_summary.csv` (mean/min–max/signs). Logs, copies and audits in Git-ignored `local/cnn-augmentation-seeds43-44-2026-11-15/`.

## Conclusion for the report

Across three seeds, augmentation attenuates the detection loss from resizing at 128 px (mean +2.28 points) and 64 px (+1.44 points), with positive attenuation for each seed. Human FPR falls for all three. Absolute detection gain nevertheless changes sign: positive for 42/43, negative for 44. For 44, detection drops 4.67 points on originals and 3.83 after resizing: a smaller resizing loss does not guarantee better absolute detection. These remain descriptive observations on the same test set.

## Limitations

Three seeds, one split and reused test images: no statistical generalization to other images, architectures or augmentation laws. No new MPS training or Recraft API generation. Main pipeline not rerun: its code/artifacts are unchanged; affected commands belong only to this experiment. No commit or push.