# Moving CSV files into results/

> Archived record of an earlier stage. Code and artifact paths have since
> changed; see [REPOSITORY_LAYOUT_VALIDATION.md](REPOSITORY_LAYOUT_VALIDATION.md)
> and the [current README](../README.md) for commands to use.

Local branch: `refactor/reproducible-analysis`. The previous Python reorganization
and its uncommitted changes are preserved. Files present before this move and
their SHA-256 hashes were copied to a temporary directory before any edits.
That local state, rather than the scripts predating the mergers, is the reference
for this operation.

The 23 root-level CSVs move into a single `results/` directory without changing
any bytes. No CSV remains at the root. The feature pilot, seeds 43/44 and Recraft
predictions follow the main results. The manual review stays with the candidates
it documents; it remains manual and is never overwritten by the pipeline.
`data/recraft/generation_log.csv` stays unchanged with the Recraft images.
Models, images, figures and `report_tables.md` are not moved at this stage.

## Producers and readers

All CSV filenames below are relative to `results/`.

| Moved CSV | Producer | Readers |
|---|---|---|
| `image_inventory.csv` | `inventory.py` | summary in `inventory.py`, `exact_duplicates.py`, `verify_data_preparation.py`, `audit_jpeg_metadata.py` |
| `exact_duplicates.csv` | `exact_duplicates.py::detect_duplicates()` | `exact_duplicates.py::exclude_duplicates()` |
| `duplicate_exclusions.csv` | `exact_duplicates.py::exclude_duplicates()` | `verify_data_preparation.py` |
| `image_inventory_deduplicated.csv` | `exact_duplicates.py::exclude_duplicates()` | name checks, `near_duplicates.py inventory`, `build_manifest.py`, `verify_data_preparation.py` |
| `near_duplicate_candidates.csv` | `near_duplicates.py inventory` | `near_duplicates.py::render_review()` |
| `near_duplicate_review.csv` | historical human review | manual inspection; no Python reader |
| `dataset_manifest.csv` | `build_manifest.py` | `near_duplicates.py manifest`, `verify_data_preparation.py`, `image_features.py`, `image_transforms.py`, `train_cnn.py`, `evaluate_robustness.py`, `audit_jpeg_metadata.py` |
| `manifest_near_duplicate_candidates.csv` | `near_duplicates.py manifest` | manual inspection |
| `feature_preview.csv` | `image_features.py` without options | pilot inspection |
| `train_validation_features.csv` | `image_features.py --table` | `train_logistic_models.py` |
| `test_predictions.csv` | `evaluate_robustness.py` | `analyze_results.py` |
| `robustness_summary.csv` | `evaluate_robustness.py` | `analyze_results.py`, `plot_robustness.py`, `make_tables.py` |
| `test_predictions_seed_43.csv` | `evaluate_robustness.py --seed 43` | sensitivity-check inspection |
| `test_predictions_seed_44.csv` | `evaluate_robustness.py --seed 44` | sensitivity-check inspection |
| `robustness_summary_seed_43.csv` | `evaluate_robustness.py --seed 43` | manual seed comparison |
| `robustness_summary_seed_44.csv` | `evaluate_robustness.py --seed 44` | manual seed comparison |
| `jpeg_metadata_baseline.csv` | `audit_jpeg_metadata.py::audit_baselines()` | `make_tables.py` |
| `jpeg_quantization_tables.csv` | `audit_jpeg_metadata.py::export_tables()` | `audit_jpeg_metadata.py::compare_tables()` |
| `jpeg_quality_equivalents.csv` | `audit_jpeg_metadata.py::compare_tables()` | audit inspection |
| `test_score_analysis.csv` | `analyze_results.py::analyze_scores()` | `make_tables.py` |
| `false_positive_flip_tests.csv` | `analyze_results.py::analyze_flips()` | `make_tables.py` |
| `frequency_ablation_comparison.csv` | `analyze_results.py::analyze_ablation()` | `make_tables.py` |
| `recraft_predictions.csv` | `evaluate_recraft.py` | `plot_recraft.py` |

`result_paths.py` centralizes filename resolution relative to the repository,
independently of the launch directory. `result_path()` creates no directory when
reading. `open_result()` creates the parent before writing. All CSV producers
use it and readers use the same paths. Recraft's generation log keeps its handling
in `generate_recraft.py`. Manifest and inventory rows still contain paths relative
to raw images, without a `results/` prefix.

## Checks

Commands use `.venv_repro/bin/python`, with Python and Matplotlib caches in
temporary directories. Comparisons use bytes and SHA-256 hashes, not just parsed
CSV values.

Successful checks:

- Move all 23 files and compare again after execution: every file's contents
  and SHA-256 remain unchanged.
- Actually execute `run_all.py --results-only`: JPEG audit, three analyses,
  figure and tables read/write the new paths. No root CSV is recreated;
  `report_tables.md` matches the initial copy.
- Run targeted commands on complete inputs: `inventory.py --summary-only`,
  `build_manifest.py`, `verify_data_preparation.py`, `near_duplicates.py manifest`,
  `near_duplicates.py review`, `image_features.py`, `image_features.py --table`,
  `image_transforms.py`, `plot_recraft.py`, then
  `exact_duplicates.exclude_duplicates()` and `check_matching_names()`.
- In two isolated copies, process a small real corpus of three JPEGs with the
  versions before/after migration: `inventory.py`, `exact_duplicates.py`,
  `near_duplicates.py inventory` and `review`. Initially there is no `results/`
  directory; the inventory creates it on write. All CSVs and the review sheet
  match the old version's outputs.
- Call the real `evaluate_robustness.write_csv()` writer with a nonexistent
  directory for the six prediction/summary filenames for seeds 42, 43 and 44:
  directory creation and correct paths succeed. Reading a missing input does
  not create a directory.
- Run `evaluate_recraft.py` before/after migration in isolated copies with the
  same images and saved models on CPU. The new version creates its missing
  directory and produces exactly the same bytes in `results/recraft_predictions.csv`.
  Delivered predictions are retained.
- Compile all active Python files, run `pip check` and `git diff --check`
  successfully. Search active Python and README paths: no root-level CSV access
  remains.

`run_all.py`, algorithms, saved models, CNN architecture and Recraft generation
are preserved. Figures changed by checks were restored to their original bytes,
avoiding PDF-date or rendering changes unrelated to CSV paths. The historical
Python-reorganization report is explicitly marked as preceding the migration.

The training pipeline and full robustness inference were not rerun for this
migration: existing predictions were reused, and the writer and reader paths
were checked. No API generation, commit or GitHub publication.

## SHA-256 before and after

Each row has the same SHA-256 before migration and after all checks.
Filenames are relative to `results/`.

| CSV | Unchanged SHA-256 |
|---|---|
| `dataset_manifest.csv` | `e04565ff1a9b44a817501680db0dafb316d58e122429bac526b414e5b205a606` |
| `duplicate_exclusions.csv` | `b8988ef28bf2124230766da5051cfad436f90cd6b7a10e7abdfcf7231df2c534` |
| `exact_duplicates.csv` | `e4bc07963098ca0e03f120e7f8c56add88045f9616bb222d942fcacb4c167741` |
| `false_positive_flip_tests.csv` | `3c7e7f1fc8fb69d1d3a6a358d6d59bdcbf4a374b41aa24ca032575079797b3e9` |
| `feature_preview.csv` | `ca25b05fb5c3c115bed7101f29667658fd50287200f31a56d619dbecaf64e5a5` |
| `frequency_ablation_comparison.csv` | `3776031da6ac786c4c645631d02a71880c7ae267513c9ea3564e69af525e30dd` |
| `image_inventory.csv` | `6227ad1a9be0b805d719b96872dddcce3b460b19edf7e83bfe287be67474a680` |
| `image_inventory_deduplicated.csv` | `2fe03ceaf18e14ea7a96bc9ecee1feabb64aa14f89af4483a226deb4c502b93d` |
| `jpeg_metadata_baseline.csv` | `26c15919e04a004b3553cf46b93d22363d3a08a2e4688f9793958ca7638c317a` |
| `jpeg_quality_equivalents.csv` | `bb32745185e0bd46e64e219ed7928899bd8eb96a6c3d8cb25c531ad86425929b` |
| `jpeg_quantization_tables.csv` | `70d53a1cb3faa5eceab9fb4ef696b9338c1e67fc4fc8b7ac1759b138338e26c6` |
| `manifest_near_duplicate_candidates.csv` | `0b00a25195cb9670361d37da0844dcc3cada1dc4a6337dfbaa95d0a12ccabf91` |
| `near_duplicate_candidates.csv` | `beb48d6d31a68ec2060f03605ad94de65922970c85a5ffdc5d5c9aa638026541` |
| `near_duplicate_review.csv` | `611afc7be1ebc3acc136ae49e7b524ec54b26542ec1e39e8cacf3304d264d2e9` |
| `recraft_predictions.csv` | `749a3455885d44a13ac2d100a29bdc9d5d4bbe7ee726c57d52b115bb7b4502d4` |
| `robustness_summary.csv` | `16827cf38831808d276a3a22f55eaf45aa1f199752018c0f9d815c3d5cdb5f8f` |
| `robustness_summary_seed_43.csv` | `3e835d4a76660d88ff56c3b58694c2ee064c007994710440cf672eff9b807940` |
| `robustness_summary_seed_44.csv` | `86998c69aa5b127668d548fcf65becc2b536156fc7bcfe5453d2f04a3ba7e622` |
| `test_predictions.csv` | `daa9935dcd0ec8b069036973fb27a9698b03819048b8ea8bbeb0c649db1b025f` |
| `test_predictions_seed_43.csv` | `dae6e9472f03d4ffd50b2936f6923f79222e0ead07164f017534483f0ed61d92` |
| `test_predictions_seed_44.csv` | `f2f80ad1d4bde637af77ef17ea87545a5a4553c4d05f9ee8632e120a8b8dd632` |
| `test_score_analysis.csv` | `f5b554e58c1d3bb8f9c997fa60e76f3b7ea4e0d417e779efefa6e9c0127304c7` |
| `train_validation_features.csv` | `6e600ee0b465568c4286212c91cd16fbf95c44f392239a9462f9dab888759b3e` |

Existing scripts whose CSV paths were updated: `analyze_results.py`, `audit_jpeg_metadata.py`, `build_manifest.py`, `evaluate_recraft.py`, `evaluate_robustness.py`, `exact_duplicates.py`, `image_features.py`, `image_transforms.py`, `inventory.py`, `make_tables.py`, `near_duplicates.py`, `plot_recraft.py`, `plot_robustness.py`, `train_cnn.py`, `train_logistic_models.py`, `verify_data_preparation.py`.