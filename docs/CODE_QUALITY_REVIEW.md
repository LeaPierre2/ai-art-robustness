# Python code quality review

## Reference and scope

Branch: `refactor/reproducible-analysis`. At the start of this review, `git log`
contained only `20291cb` (“Save project baseline before refactoring”). There was
therefore no reorganization commit yet. The uncommitted reorganized local state
was copied before editing, with sources, 23 CSVs, models, figures and SHA-256
hashes. That copy is the before/after reference; the historical Git version was
not restored.

All 19 Python files were read: `run_all.py`, 17 functional modules and the
initializer in `analysis/`. Reviewed paths cover inventory, exact/perceptual
duplicates, manifest selection/validation, features, training, saved-model
evaluation, analyses, JPEG, tables, figures and Recraft. Structure, dependencies
and commands are preserved.

## Main improvements

| Files | Finding and improvement |
|---|---|
| `analysis/paths.py` and callers | Repeated image roots and small CSV readers/writers. `DATASET_DIR` and `RECRAFT_DIR` become shared references. `read_result_rows` and `write_result_rows` preserve values, column/row order, UTF-8 and `newline=""`. Existing streaming paths stay streaming. Specialized `csv.writer` calls remain explicit. |
| `analysis/manifest.py` | `verify_preparation` mixed statistics, integrity, quotas and rendering. It now orchestrates `_print_size_statistics`, `_verify_exact_exclusions`, `_verify_manifest_rows` and `_render_label_review` in the original order. `inventory_rows`, `cleaned_rows` and `manifest_rows` clarify provenance. Selection and RNG calls are unchanged. |
| `analysis/evaluate_robustness.py` | `main` mixed inference, bootstrap, saving and display. `predict_test_rows` and `summarize_decisions` isolate the calculations. Batch/condition/model/image order and decision alignment with human/AI masks are documented. Predictions are still written before summarization, as in the reference. |
| `analysis/train_logistic_models.py` | Row-to-matrix conversion is a named module-level function with an explicit row/column order contract. Hyperparameters, fitting, thresholds and serialization do not change. |
| `analysis/detectors.py`, `image_features.py`, `image_transforms.py`, `train_cnn.py`, `near_duplicates.py` | Types/docstrings clarify RGB float32 tensors without resizing, historical 55-feature order, seven conditions, conservative `nextafter` threshold, float64 validation scores via `tolist`, 64-bit dHash and a four-block index valid at distance ≤ 3. |
| `analysis/inventory.py` | Folder decoding extracted into pure `parse_folder_name`, allowed styles at module level; same `ukiyo-e` normalization and rejection of unknown classes. |
| `analysis/analyze_results.py` | Clearer `FULL_MODEL` and `ABLATED_MODEL` constants; calculations and saved CSV names unchanged. |
| All modules, `run_all.py` | Imports grouped, deduplicated and moved to the top; unused imports/root aliases removed after checking uses. Command return types, module documentation, `PREPARATION_COMMANDS`, `TRAINING_COMMANDS`, `RESULT_COMMANDS` and consistent formatting. |
| `near_duplicates.render_review` and `paths.write_result_rows` | Explicit errors for a review without candidates or writing without rows/columns. Invalid writing fails before opening, protecting an existing file. No valid result changes. |

Small helpers stay in the existing paths module: no new runtime package,
dependency or code layout. Removed readers/writers had only internal callers,
now replaced by shared functions; README commands and scientific functions used
by other modules remain available.

## Scientific issues and possible bugs

No confirmed bug requiring current results to change was identified. Seeds,
splits, thresholds, metrics, bootstrap settings, Holm correction, transformations
and output formats are unchanged. Evaluation retains float64 comparisons of
scores and saved thresholds.

The two clarified errors involve empty inputs and do not affect existing
results. No fallback scores, automatic exclusions, new selection criteria or
statistical changes were introduced.

Backend precision differences and small figure-rendering discrepancies were
already documented. This review isolates them by comparing old/new code on the
same inputs and CPU.

## Checks actually performed

Environment: `.venv_repro`, Python **3.9.6**. Python/Matplotlib caches use temporary
directories. All comparisons use isolated before/after package copies, the same
CSVs/models and links to the same images. Repository models/results are not used
as outputs of comparative training or evaluation.

### Pipeline and affected commands

`run_all.py --results-only` succeeded in the repository and in both copies.
It covers JPEG, three analyses, robustness figure and tables.

Commands run in both versions (prefix `.venv_repro/bin/python`):

```text
run_all.py --results-only
-m analysis.manifest build
-m analysis.manifest verify
-m analysis.near_duplicates manifest
-m analysis.near_duplicates review
-m analysis.image_transforms
-m analysis.image_features
-m analysis.image_features --table
-m analysis.figures recraft
-m analysis.inventory --summary-only
-m analysis.evaluate_recraft
-m analysis.train_logistic_models
-m analysis.evaluate_robustness
-m analysis.analyze_results
```

`exact_duplicates.exclude_duplicates()` and `exact_duplicates.check_matching_names()`
were also run on complete CSVs. A real three-JPEG fixture passes through
`inventory`, `exact_duplicates`, `near_duplicates inventory` and
`near_duplicates review` before/after. It also checks shared paths and creation
of missing directories.

### Comparison results

- All 23 CSVs in the copies match after the initial checks, including the three
  analysis CSVs, three JPEG CSVs, features, manifest, exclusions/candidates and
  Recraft predictions recalculated on CPU.
- The rebuilt manifest still has SHA-256:
  `e04565ff1a9b44a817501680db0dafb316d58e122429bac526b414e5b205a606`.
- Both `.joblib` files from comparative training match byte for byte, including
  thresholds.
- Full CPU evaluation of 1,200 images, seven conditions and three models yields
  the same 25,200 predictions and summary in both versions. Recomputing the
  three analyses on those new predictions also gives identical CSVs.
- Five regenerated PNGs (`label_review`, `near_duplicate_review`,
  `transform_preview`, `robustness_figure`, `recraft_figure`) match across versions.
  Both PDFs match after neutralizing their creation dates.
- The three tables in `docs/report_tables.md` match.
- Console outputs for manifest construction/verification, inventory summary and
  exclusions also match.

### Separate CNN check

The AST body of `train_cnn.main()` is unchanged. The only path changes are
imports, annotations/docstrings and the shared image directory replacing
`IMAGE_DIR`.

An isolated before/after test reads four balanced real training images, uses
seed 42 and the real `DataLoader`/architecture, and executes an actual Adam step
with `lr=0.001` and `BCEWithLogitsLoss`. It then validates on four balanced real
images, selects the threshold and saves/reloads a temporary checkpoint. Loss,
labels, scores, threshold and every weight-tensor SHA-256 match exactly. No
reference checkpoint is replaced.

### Contracts, errors and orchestration

- CSV containing accented characters, commas, quotes, line breaks and floats:
  identical bytes to the historical `csv.DictWriter`, preserved column order and
  CRLF, and verified text reading;
- reading a missing CSV creates no directory; writing creates the directory;
- writing without rows/columns fails before opening, leaving an existing file
  intact; an empty table with explicit columns is accepted;
- review without candidates raises an explicit error; unknown class directory
  is explicitly rejected; `ukiyo-e` normalization is preserved;
- thresholds on constant/tied/short/sorted scores match before/after exactly and
  respect the false-positive budget; empty, nonfinite or nonvector scores fail;
- `generate_pair()` tested with a simulated API response containing two real
  1 024 × 1 024 PNGs: payload, images and behavior match; no network or API credits;
- full command sequences for all three `run_all.py` modes match under mocked
  subprocesses; stopping at the first failure is verified;
- syntax with `feature_version=9`, Python 3.9.6 compilation, unused-import search
  and `git diff --check` succeed.

## Differences and limitations

No scientific difference on the tested inputs. PDF dates and the one-pixel
historical robustness-PNG width discrepancy remain known rendering differences.
Delivered artifacts are restored to their initial bytes rather than replaced
with new renders or inference from another backend. The 23 CSVs and five
models remain unchanged; existing notes/drafts and `.gitignore` are preserved.

The CNN was not retrained on 4,200 images for eight epochs during this review;
its path is checked by the separate test above, unchanged main-function AST and
full saved-weight inference. Seeds 43/44 were not retrained. Full raw-image
preparation was not rerun on the whole corpus: inventory/detection use a fixture
and existing intermediates feed later stages. Real API generation and human
visual review were not repeated.

The layout retains 19 Python files. This review note is the only new repository
file; no new dependency or executable file. No commit, staging or push. Earlier
local changes remain on `refactor/reproducible-analysis`.

Improved Python files:

- `run_all.py`
- `analysis/analyze_results.py`
- `analysis/audit_jpeg_metadata.py`
- `analysis/detectors.py`
- `analysis/evaluate_recraft.py`
- `analysis/evaluate_robustness.py`
- `analysis/exact_duplicates.py`
- `analysis/figures.py`
- `analysis/generate_recraft.py`
- `analysis/image_features.py`
- `analysis/image_transforms.py`
- `analysis/inventory.py`
- `analysis/make_tables.py`
- `analysis/manifest.py`
- `analysis/near_duplicates.py`
- `analysis/paths.py`
- `analysis/train_cnn.py`
- `analysis/train_logistic_models.py`
