# Repository layout: inventory and validation

Branch: `refactor/reproducible-analysis`. This operation's reference is the initial local state, including Python mergers and 23 CSVs already in `results/`. Sources, CSVs, models, figures, notes and SHA-256 were copied before editing. Previous local changes are preserved; no old path was restored from Git.

## Actual inventory before editing

There were **21 project Python files actually present**, all at the root. Virtual environments, VS Code and third-party data are excluded. The 38 tracked paths already missing are cross-checked against `git status --short`: 15 former Python files and 23 root CSVs. They still appear in `git ls-files` because deletions/moves are uncommitted. Inventory below reflects disk contents.

Actual top-level tree before:

```text
ai-art-robustness/
├── .DS_Store
├── .git/
├── .gitignore
├── .venv/
├── .venv_repro/
├── CSV_MIGRATION_VALIDATION.md
├── Methodology for the report.docx
├── README.md
├── REFACTOR_VALIDATION.md
├── TO do list.docx
├── Visual Studio Code.app/
├── analyze_results.py
├── audit_jpeg_metadata.py
├── build_manifest.py
├── cnn_model.pt
├── cnn_model.py
├── cnn_model_seed_43.pt
├── cnn_model_seed_44.pt
├── common.py
├── data/
├── evaluate_recraft.py
├── evaluate_robustness.py
├── exact_duplicates.py
├── generate_recraft.py
├── image_features.py
├── image_transforms.py
├── inventory.py
├── label_review.png
├── logistic_full.joblib
├── logistic_no_frequency.joblib
├── make_tables.py
├── manifest_near_duplicate_review.png
├── near_duplicate_review.png
├── near_duplicates.py
├── plot_recraft.py
├── plot_robustness.py
├── recraft_figure.pdf
├── recraft_figure.png
├── report_tables.md
├── requirements.txt
├── result_paths.py
├── results/
├── robustness_figure.pdf
├── robustness_figure.png
├── run_all.py
├── train_cnn.py
├── train_logistic_models.py
├── transform_preview.png
├── verify_data_preparation.py
```

## Every Python file inspected before reorganization

Physical line counts, imports, roles, I/O and uses. “Pipeline” means an explicit `run_all.py` call; shared modules are also imported indirectly. No file was merely a launcher to merge automatically based on size.

| File | Lines | Imports | Inputs / outputs and role | Pipeline | README |
|---|---:|---|---|---|---|
| `analyze_results.py` | 231 | `from result_paths import result_path, open_result`<br>`import csv`<br>`from collections import defaultdict`<br>`from pathlib import Path`<br>`from statistics import mean`<br>`from sklearn.metrics import roc_auc_score`<br>`from scipy.stats import binomtest`<br>`import numpy as np` | Reads predictions/summary; writes three analysis CSVs. | yes | yes |
| `audit_jpeg_metadata.py` | 174 | `from result_paths import result_path, open_result`<br>`import csv`<br>`from collections import defaultdict`<br>`from pathlib import Path`<br>`import numpy as np`<br>`from sklearn.metrics import roc_auc_score`<br>`from PIL import Image`<br>`from io import BytesIO` | Reads inventory, manifest and representative images; writes three JPEG CSVs. | yes | yes |
| `build_manifest.py` | 92 | `from result_paths import result_path, open_result`<br>`import csv`<br>`import random`<br>`from collections import Counter, defaultdict`<br>`from pathlib import Path` | Reads cleaned inventory; writes dataset_manifest.csv. | yes | yes |
| `cnn_model.py` | 25 | `import numpy as np`<br>`import torch`<br>`from torch import nn` | Architecture and tensor conversion, no I/O. | no | yes |
| `common.py` | 44 | `from pathlib import Path`<br>`import joblib`<br>`import numpy as np`<br>`import torch`<br>`from cnn_model import build_cnn` | Validation threshold; reads .pt/.joblib models. | no | yes |
| `evaluate_recraft.py` | 80 | `from result_paths import result_path, open_result`<br>`import csv`<br>`from pathlib import Path`<br>`import numpy as np`<br>`import torch`<br>`from PIL import Image`<br>`from cnn_model import image_to_tensor`<br>`from common import load_detectors`<br>`from image_features import extract_features` | Reads Recraft images/models; writes recraft_predictions.csv. | no | yes |
| `evaluate_robustness.py` | 163 | `from result_paths import result_path, open_result`<br>`import argparse`<br>`import csv`<br>`from pathlib import Path`<br>`import numpy as np`<br>`import torch`<br>`from PIL import Image`<br>`from cnn_model import image_to_tensor`<br>`from common import load_detectors`<br>`from image_features import extract_features`<br>`from image_transforms import make_variants` | Reads manifest/images/models; writes predictions/summaries, seeds 42/43/44. | yes | yes |
| `exact_duplicates.py` | 178 | `from result_paths import result_path, open_result`<br>`import csv`<br>`import hashlib`<br>`from pathlib import Path`<br>`from PIL import Image`<br>`from collections import defaultdict` | Reads inventory/pixels; writes groups, exclusions and cleaned inventory; name checks. | yes | yes |
| `generate_recraft.py` | 124 | `import base64`<br>`import csv`<br>`import json`<br>`from datetime import datetime, timezone`<br>`from getpass import getpass`<br>`from io import BytesIO`<br>`from pathlib import Path`<br>`from urllib.error import HTTPError, URLError`<br>`from urllib.request import Request, urlopen`<br>`from PIL import Image` | Explicit API; writes images and data/recraft/generation_log.csv. | no | yes |
| `image_features.py` | 157 | `from result_paths import result_path, open_result`<br>`import csv`<br>`from collections import Counter`<br>`from pathlib import Path`<br>`import numpy as np`<br>`from PIL import Image`<br>`import argparse` | Reads manifest/images; writes pilot or training/validation table; shared extract_features. | yes | yes |
| `image_transforms.py` | 71 | `from result_paths import result_path`<br>`import csv`<br>`from io import BytesIO`<br>`from pathlib import Path`<br>`from PIL import Image, ImageDraw` | Reads one manifest image; writes preview; shared make_variants. | yes | yes |
| `inventory.py` | 161 | `from result_paths import result_path, open_result`<br>`import csv`<br>`from collections import Counter`<br>`from pathlib import Path`<br>`from PIL import Image`<br>`import argparse` | Reads raw images; writes inventory and displays summary. | yes | yes |
| `make_tables.py` | 106 | `from result_paths import result_path`<br>`import csv`<br>`from pathlib import Path` | Reads JPEG/summary/analysis CSVs; writes report_tables.md. | yes | yes |
| `near_duplicates.py` | 187 | `from result_paths import result_path, open_result`<br>`import csv`<br>`from collections import defaultdict`<br>`from pathlib import Path`<br>`from PIL import Image, ImageDraw, ImageOps`<br>`import argparse` | Reads cleaned inventory/manifest/images; writes candidate CSVs and PNG review. | yes | yes |
| `plot_recraft.py` | 55 | `from result_paths import result_path`<br>`import csv`<br>`from collections import defaultdict`<br>`from pathlib import Path`<br>`import matplotlib.pyplot as plt`<br>`from matplotlib.ticker import PercentFormatter` | Reads recraft_predictions.csv; writes Recraft PNG/PDF. | no | yes |
| `plot_robustness.py` | 72 | `from result_paths import result_path`<br>`import csv`<br>`from pathlib import Path`<br>`import matplotlib.pyplot as plt` | Reads robustness_summary.csv; writes robustness PNG/PDF. | yes | yes |
| `result_paths.py` | 17 | `from pathlib import Path` | Resolves/opens CSVs; not a launcher. | no | yes |
| `run_all.py` | 51 | `import argparse`<br>`import subprocess`<br>`import sys`<br>`from pathlib import Path` | Orchestrates commands: normal, existing-model and existing-results modes. | no | yes |
| `train_cnn.py` | 146 | `from result_paths import result_path`<br>`from common import select_threshold`<br>`import csv`<br>`import argparse`<br>`import time`<br>`from pathlib import Path`<br>`import numpy as np`<br>`import torch`<br>`from PIL import Image`<br>`from sklearn.metrics import roc_auc_score`<br>`from torch import nn`<br>`from torch.utils.data import DataLoader, Dataset`<br>`from cnn_model import build_cnn, image_to_tensor` | Reads manifest/images; writes CNN with threshold/best epoch. | yes | yes |
| `train_logistic_models.py` | 83 | `from result_paths import result_path`<br>`from common import select_threshold`<br>`import csv`<br>`from pathlib import Path`<br>`import joblib`<br>`import numpy as np`<br>`from sklearn.linear_model import LogisticRegression`<br>`from sklearn.metrics import roc_auc_score`<br>`from sklearn.pipeline import make_pipeline`<br>`from sklearn.preprocessing import StandardScaler` | Reads training/validation table; writes both regressions and thresholds. | yes | yes |
| `verify_data_preparation.py` | 119 | `from result_paths import result_path`<br>`import csv`<br>`from collections import Counter, defaultdict`<br>`from pathlib import Path`<br>`from statistics import median, quantiles`<br>`from PIL import Image, ImageDraw, ImageOps` | Reads inventories/exclusions/manifest/images; checks consistency and writes label preview. | yes | yes |

## Decisions and mergers

- `cnn_model.py` (25 lines) + `common.py` (44) → `analysis/detectors.py`. Architecture, tensor conversion, threshold selection and loading form a shared detector interface. Named `build_cnn`, `image_to_tensor`, `select_threshold`, `load_detectors` remain independently usable. The small file was not a launcher.
- `build_manifest.py` (92) + `verify_data_preparation.py` (119) → `analysis/manifest.py`, functions `build_manifest`/`verify_preparation`, subcommands `build`/`verify`. An existing manifest can still be checked without rebuilding. Order remains build → manifest near-duplicate check → verify.
- `plot_robustness.py` (72) + `plot_recraft.py` (55) → `analysis/figures.py`, functions `plot_robustness`/`plot_recraft`, subcommands `robustness`/`recraft`. Both only read results to create figures; no training/evaluation.
- `result_paths.py` (17) becomes `analysis/paths.py`, covering CSV/model/figure/document directories. It stays lightweight without ML imports; merging with detectors would unnecessarily burden data preparation. Root-level `run_all.py` remains the visible orchestrator.

Other stages stay separate because their data, costs or control roles justify independent commands. Both training and evaluation pairs retain commands. Recraft generation stays explicit/separate. Logistic training moves into `main()` so importing the package does not train models.

## Directories and paths

- `analysis/`: all stage/shared code; package-relative imports and root-level `python -m analysis.…` commands.
- `results/`: same 23 CSVs, same paths as immediately before this stage.
- `models/`: five saved models, readers/writers updated.
- `figures/`: eight PNG/PDF results/previews; diagnostic and plotting writers use these paths.
- `docs/`: Markdown tables, historical validations, this validation, two retained Git-ignored DOCX drafts.
- `data/recraft/`: images, log and checksums stay in place.
- `local/`: local VS Code application renamed/moved, not deleted, and Git-ignored. `.venv*` environments stay in place to preserve executables and path references.

Historical notes are marked as archives pointing to current README. Output directories are created on write; CSV/model readers do not require duplicate root artifacts.

## Actual tree after

```text
ai-art-robustness/
├── .DS_Store
├── .git/
├── .gitignore
├── .venv/
├── .venv_repro/
├── README.md
├── analysis/
├── data/
├── docs/
├── figures/
├── local/
├── models/
├── requirements.txt
├── results/
├── run_all.py
```

There are **19 project Python files**: root `run_all.py` and 18 in `analysis/`, including `__init__.py` (1 line). Excluding initializer: 18 versus 21 before. Root visible entries fall from 44 to 10.

| Current file | Lines |
|---|---:|
| `run_all.py` | 51 |
| `analysis/__init__.py` | 1 |
| `analysis/analyze_results.py` | 229 |
| `analysis/audit_jpeg_metadata.py` | 170 |
| `analysis/detectors.py` | 68 |
| `analysis/evaluate_recraft.py` | 78 |
| `analysis/evaluate_robustness.py` | 161 |
| `analysis/exact_duplicates.py` | 178 |
| `analysis/figures.py` | 140 |
| `analysis/generate_recraft.py` | 125 |
| `analysis/image_features.py` | 156 |
| `analysis/image_transforms.py` | 70 |
| `analysis/inventory.py` | 160 |
| `analysis/make_tables.py` | 104 |
| `analysis/manifest.py` | 227 |
| `analysis/near_duplicates.py` | 185 |
| `analysis/paths.py` | 38 |
| `analysis/train_cnn.py` | 144 |
| `analysis/train_logistic_models.py` | 88 |

## Executed checks and comparisons

Environment: `.venv_repro/bin/python`, Python 3.9.6; temporary Python/Matplotlib caches. Isolated before/after copies access identical raw images, Recraft images, CSVs and saved models.

- Actually run `run_all.py --results-only`, then rerun final code via its absolute file from `/tmp`. The orchestrator launches modules from the repository root.
- Compare before → after commands on identical inputs:

| Before | After (Python prefix, from root) |
|---|---|
| `build_manifest.py` | `-m analysis.manifest build` |
| `verify_data_preparation.py` | `-m analysis.manifest verify` |
| `plot_robustness.py` | `-m analysis.figures robustness` |
| `plot_recraft.py` | `-m analysis.figures recraft` |
| `image_transforms.py` | `-m analysis.image_transforms` |
| `image_features.py` | `-m analysis.image_features` |
| `image_features.py --table` | `-m analysis.image_features --table` |
| `near_duplicates.py manifest` | `-m analysis.near_duplicates manifest` |
| `near_duplicates.py review` | `-m analysis.near_duplicates review` |
| `evaluate_recraft.py` | `-m analysis.evaluate_recraft` |
| `audit_jpeg_metadata.py` | `-m analysis.audit_jpeg_metadata` |
| `analyze_results.py` | `-m analysis.analyze_results` |
| `make_tables.py` | `-m analysis.make_tables` |
| `evaluate_robustness.py` | `-m analysis.evaluate_robustness` |
| `train_logistic_models.py` | `-m analysis.train_logistic_models` |

Regenerated before/after CSVs are byte-identical, including three analyses, three JPEG CSVs, manifest, features, final candidates and Recraft predictions. Full CPU evaluation of 1 200 test images (25 200 predictions) and summaries also matches old/new scripts. Both logistic training runs produce byte-identical `.joblib`, including thresholds. Working models were not replaced by isolated training outputs.

Five tested regenerable PNGs (`label_review`, `near_duplicate_review`, `transform_preview`, `robustness_figure`, `recraft_figure`) match old/new commands. Both PDFs match after neutralizing creation dates. `docs/report_tables.md` matches the former root table output.

Additional checks:

- Load both regressions and all three CNNs with old loader and `analysis.detectors.load_detectors()`;
- three-real-JPEG minimal corpus in isolated copies: inventory, exact detection/exclusions, near-duplicate detection/review; all CSVs/review sheets match before/after;
- automatic missing-directory creation: inventory `results/`, review `figures/`, logistic training `models/`, table generation `docs/`;
- import all package modules without training, reading results or calling API;
- mocked subprocess checks of all three `run_all.py` modes: existing modules, build → manifest near-duplicate check → verify, no API;
- compile all project Python and run `git diff --check`;
- check README local links and documented command modules; old paths remain only in archives;
- verify `.gitignore` excludes none of the code/CSV/model/figure files intended for review.

## Integrity of files that were only moved

All **23 CSVs in `results/`** retain exact original bytes and SHA-256 after working-copy executions. Manifest SHA-256:

`e04565ff1a9b44a817501680db0dafb316d58e122429bac526b414e5b205a606`.

Five models, eight figures/previews and two moved DOCX files match initial contents/SHA-256. The two historical validations receive archive banners only; their bodies are verified unchanged. VS Code moves by rename into `local/`, without editing/deleting its contents. Recraft images/log remain in place.

## Explained differences and limitations

- PDF creation dates vary; compare old/new after neutralizing that metadata.
- As in previous validation, regenerated robustness PNG is 3 270 × 1 616 versus delivered 3 269 × 1 616. Historical code gives the same render in this environment; exact cause remains unknown. Old/new regenerated renders match; delivered figures are restored to original contents after verification.
- Small CPU versus historical CNN-score differences also occur with old code (previous validations). Comparing both versions on identical CPU gives no difference; historical predictions are not replaced.
- CNN not retrained; seeds 43/44 loaded, but full sensitivity evaluation not rerun. Architecture, conversion and thresholds checked through saved-model inference.
- Full CNN-retraining pipeline not executed. Ordering is mocked; moved stages checked separately with existing artifacts.
- Recraft API generation not run; command remains separate.
- Historical human visual review not repeated; lists and regenerable sheets compared. `figures/manifest_near_duplicate_review.png` is historical without a generator in reference scripts, retained with this limitation stated in README.

Changes remain uncommitted on the local branch, without push or PR.