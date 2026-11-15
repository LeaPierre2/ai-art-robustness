# Local reorganization: verification

> Archived record of an earlier stage. Code/artifact paths have since changed;
> see [REPOSITORY_LAYOUT_VALIDATION.md](REPOSITORY_LAYOUT_VALIDATION.md) and the
> [current README](../README.md) for commands.

> Historical Python reorganization record, before CSV migration. CSV names
> and commands below describe that earlier state. CSVs are now in `results/`;
> see [CSV_MIGRATION_VALIDATION.md](CSV_MIGRATION_VALIDATION.md).

Reference: `origin/main`, commit `20291cb7d40d6df8d5d3c2d232c969dad7562e66`.
Initially `main` pointed there and `git status --short` was empty. Working branch:
`refactor/reproducible-analysis`. No push or PR. Reference is the locally recorded
remote branch, without `git fetch`.

## Mergers

| Destination | Former scripts | Reason |
|---|---|---|
| `analyze_results.py` | `analyze_test_scores.py`, `analyze_false_positive_flips.py`, `analyze_frequency_ablation.py` | One existing-results stage with three named functions, without loading images or training models. |
| `audit_jpeg_metadata.py` | `audit_jpeg_metadata.py`, `export_jpeg_tables.py`, `compare_jpeg_tables.py` | Sequences baseline/export/comparison; avoids comparing a stale table from a previous run. |
| `inventory.py` | `check_labels.py`, `summarize_inventory.py` | Build and summarize the same inventory. |
| `exact_duplicates.py` | `check_duplicates.py`, `clean_duplicates.py`, `check_matching_names.py` | Detection/exclusions share one policy; name checks follow on cleaned inventory and change no selection. |
| `near_duplicates.py` | `check_near_duplicates.py`, `view_near_duplicates.py`, `check_manifest_near_duplicates.py` | Shared dHash, explicit `inventory`, `review`, `manifest` commands; final check waits for the final manifest. |
| `image_features.py` | `image_features.py`, `build_feature_table.py` | Feature definition, pilot and full table without self-import. `--table` builds the table; no option retains the pilot. |

`check_image.py` was an exploratory three-file check with no callers or persistent
outputs; the inventory now covers all images. Replaced scripts are removed only
after comparing outputs and checking imports, pipeline and documentation uses.

Review covered every root Python file: imports, entry points, reads/writes and
README/pipeline calls. Transformation, architecture and shared utility modules
stay separate. Manifest construction/verification remain independently useful:
audit an existing manifest without resampling it. Training commands stay separate
(different dependencies/options), as do evaluation, figures and tables. Recraft
retains three actions: explicit API generation, local evaluation and plotting
from predictions. No API call in the pipeline.

## Protocol and commands

Environment: `.venv_repro/bin/python`, Python 3.9.6. Isolated `/tmp/ai-art-reference`
comes from `git archive origin/main`, linked to the same raw images. No retraining
for merger comparisons; saved predictions/models retained. Matplotlib/compilation
caches are in `/tmp`.

Git commands: `git status --short`, `git branch --show-current`,
`git log -1 --oneline`, `git rev-parse origin/main`, `git diff origin/main --stat`,
`git switch -c refactor/reproducible-analysis`, `git archive --output=… origin/main`.
Branch creation required access to sandbox-protected `.git`.

Historical scripts run in the copy (Python prefix above):

```text
check_labels.py
summarize_inventory.py
check_duplicates.py
clean_duplicates.py
check_matching_names.py
check_near_duplicates.py
view_near_duplicates.py
build_manifest.py
check_manifest_near_duplicates.py
verify_data_preparation.py
image_features.py
build_feature_table.py
audit_jpeg_metadata.py
export_jpeg_tables.py
compare_jpeg_tables.py
analyze_test_scores.py
analyze_false_positive_flips.py
analyze_frequency_ablation.py
make_tables.py
plot_robustness.py
plot_recraft.py
```

New commands run and compared with historical outputs:

```text
inventory.py
exact_duplicates.py
near_duplicates.py inventory
near_duplicates.py review
build_manifest.py
near_duplicates.py manifest
verify_data_preparation.py
image_features.py
image_features.py --table
audit_jpeg_metadata.py
analyze_results.py
make_tables.py
plot_robustness.py
plot_recraft.py
```

Additional checks: `pip check`, Python compilation, `run_all.py --help`,
`near_duplicates.py --help`, imports/uses, `shasum -a 256 -c data/recraft/SHA256SUMS.txt`
(20 files OK), `git diff --check`. Initial compilation could not write the macOS
cache; retry with `PYTHONPYCACHEPREFIX=/tmp/ai-art-pycache` succeeded.

## Output comparisons

Comparisons are byte for byte (including columns, row order, values and formats),
with SHA-256. Tracked CSVs are also checked against regenerated historical outputs
to identify stale snapshots.

## Check limitations

CNN training (seeds 42, 43, 44) and logistic training are not rerun: code,
hyperparameters, thresholds and serialization remain unchanged. Normal and
`--reuse-models` order is checked with mocked subprocesses, not an integral
retraining run. Preparation/merged-stage outputs are actually recomputed.

Recraft generation is not run; it remains an explicit API command. Environment
creation/installation is not replayed because `.venv_repro` exists; `pip check`
validates dependencies. Historical human visual review is not repeated; candidate
lists/review sheets are compared. The manifest check does not generate
`manifest_near_duplicate_review.png`: that historical image is not a reproducible
output of the reference version.

## Independent evaluation with saved models

`evaluate_robustness.py` and `evaluate_recraft.py` ran in `/tmp/ai-art-saved-evaluation`
with reorganized code and existing models on CPU. Working predictions were not
replaced. Compared to tracked CSVs:

- test: 25 200 rows, 4 955 changed `score_ai` cells, CNN only; maximum absolute
  difference `5.662441253662109e-07`; no decisions changed;
- summary: 21 rows, no changed cells;
- Recraft: 60 rows, 14 changed CNN scores, maximum `8.940696716308594e-08`;
  no decisions changed.

These precision differences are controlled by also running `origin/main` scripts
on CPU in `/tmp/ai-art-original-evaluation`, with exactly the same images/models.

Result: old/new CPU evaluation outputs are **byte-identical** for all three files.
Small differences from tracked scores already occur with historical code on this
backend; they do not result from reorganization. Logistic scores, decisions and
summary metrics are unchanged. Tracked CSVs remain inputs to statistical analyses.

## Merger comparison results

All outputs below are byte-identical to historical scripts on identical inputs.
All CSVs also match tracked files on `origin/main`.

| File | SHA-256 after reorganization | Matches tracked file |
|---|---|---|
| `image_inventory.csv` | `6227ad1a9be0b805d719b96872dddcce3b460b19edf7e83bfe287be67474a680` | yes |
| `exact_duplicates.csv` | `e4bc07963098ca0e03f120e7f8c56add88045f9616bb222d942fcacb4c167741` | yes |
| `duplicate_exclusions.csv` | `b8988ef28bf2124230766da5051cfad436f90cd6b7a10e7abdfcf7231df2c534` | yes |
| `image_inventory_deduplicated.csv` | `2fe03ceaf18e14ea7a96bc9ecee1feabb64aa14f89af4483a226deb4c502b93d` | yes |
| `near_duplicate_candidates.csv` | `beb48d6d31a68ec2060f03605ad94de65922970c85a5ffdc5d5c9aa638026541` | yes |
| `near_duplicate_review.png` | `7043cc4885079f05d2a722459821a6f53d1227c79b661b0a778dd2ad68d3205c` | yes |
| `dataset_manifest.csv` | `e04565ff1a9b44a817501680db0dafb316d58e122429bac526b414e5b205a606` | yes |
| `manifest_near_duplicate_candidates.csv` | `0b00a25195cb9670361d37da0844dcc3cada1dc4a6337dfbaa95d0a12ccabf91` | yes |
| `label_review.png` | `eaddd3e7e2580a614cf861f3a4f32e25965b097b2856efee0ab2702dbeb50799` | yes |
| `feature_preview.csv` | `ca25b05fb5c3c115bed7101f29667658fd50287200f31a56d619dbecaf64e5a5` | yes |
| `train_validation_features.csv` | `6e600ee0b465568c4286212c91cd16fbf95c44f392239a9462f9dab888759b3e` | yes |
| `jpeg_metadata_baseline.csv` | `26c15919e04a004b3553cf46b93d22363d3a08a2e4688f9793958ca7638c317a` | yes |
| `jpeg_quantization_tables.csv` | `70d53a1cb3faa5eceab9fb4ef696b9338c1e67fc4fc8b7ac1759b138338e26c6` | yes |
| `jpeg_quality_equivalents.csv` | `bb32745185e0bd46e64e219ed7928899bd8eb96a6c3d8cb25c531ad86425929b` | yes |
| `test_score_analysis.csv` | `f5b554e58c1d3bb8f9c997fa60e76f3b7ea4e0d417e779efefa6e9c0127304c7` | yes |
| `false_positive_flip_tests.csv` | `3c7e7f1fc8fb69d1d3a6a358d6d59bdcbf4a374b41aa24ca032575079797b3e9` | yes |
| `frequency_ablation_comparison.csv` | `3776031da6ac786c4c645631d02a71880c7ae267513c9ea3564e69af525e30dd` | yes |
| `report_tables.md` | `fbbf65d76d92793fd1c669727eb219c95e881737af1de0d8f6129766d496d44f` | yes |
| `robustness_figure.png` | `6b88b53e5180e6c95743f358929d90244b92d5980e3ef2d3f1c69774e2b24415` | no, see note |
| `recraft_figure.png` | `c9a3d03753131b399e6f4213af452d3d9bd220d06c26d77e5c20e83160ca456c` | yes |

Both PDFs also match after neutralizing their creation dates. Tracked PDFs are
retained to avoid metadata-only changes. No tracked CSV changed.

Regenerated robustness PNG is 3 270 × 1 616, versus tracked 3 269 × 1 616.
This rendering discrepancy already occurs with historical scripts/tracked CSVs;
its precise cause is not established. Old/new regenerated renders match. The
tracked PNG was kept to avoid an unrelated figure change during reorganization.

`image_features.py --table`, `inventory.py --summary-only` and
`run_all.py --results-only` subsequently ran successfully and outputs were
recompared. Text summary exactly matches former `summarize_inventory.py`.
Full retraining pipeline was not executed.

## Removed files

- `analyze_test_scores.py`
- `analyze_false_positive_flips.py`
- `analyze_frequency_ablation.py`
- `export_jpeg_tables.py`
- `compare_jpeg_tables.py`
- `check_labels.py`
- `summarize_inventory.py`
- `check_duplicates.py`
- `clean_duplicates.py`
- `check_matching_names.py`
- `check_near_duplicates.py`
- `view_near_duplicates.py`
- `check_manifest_near_duplicates.py`
- `build_feature_table.py`
- `check_image.py`

There are 20 Python files remaining at the root at this stage.

After removal, compilation/import of seven merged/pipeline modules succeeds;
no old script-name references remain in active Python or README. All three modes
pass mocked ordering checks. `image_transforms.py` also ran: `transform_preview.png`
exactly matches the tracked image. `git diff --check` passes. Final Git state
contains code/documentation changes only; tracked data, CSVs, models and figures
are preserved. Changes remain uncommitted on the dedicated branch for local review.