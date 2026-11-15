# Report tables

## JPEG metadata baselines on the reference test set

| Input | Test ROC-AUC |
|---|---:|
| JPEG quantization profile | 1.0000 |
| File size | 0.9837 |

## Detection performance by model and transformation

The test set contains 600 human artworks and 600 AI-generated images.
Decision thresholds were fixed on the validation set.

| Model | Condition | FPR (%) | Change in FPR (pp) | AI detection (%) | ROC-AUC | Holm-adjusted p |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression | Reference | 5.00 | +0.00 | 31.00 | 0.8096 | — |
| Logistic regression | JPEG 85 | 5.17 | +0.17 | 32.83 | 0.8155 | 1 |
| Logistic regression | JPEG 60 | 5.17 | +0.17 | 38.00 | 0.8364 | 1 |
| Logistic regression | JPEG 35 | 5.00 | +0.00 | 28.50 | 0.8049 | 1 |
| Logistic regression | Resize via 128 px | 1.83 | -3.17 | 10.50 | 0.7751 | 0.0003567 |
| Logistic regression | Resize via 64 px | 3.17 | -1.83 | 11.50 | 0.7221 | 0.6791 |
| Logistic regression | Resize via 128 px + JPEG 60 | 2.17 | -2.83 | 10.67 | 0.7697 | 0.001221 |
| Logistic regression without frequencies | Reference | 6.17 | +0.00 | 30.00 | 0.7686 | — |
| Logistic regression without frequencies | JPEG 85 | 6.17 | +0.00 | 29.67 | 0.7682 | 1 |
| Logistic regression without frequencies | JPEG 60 | 6.67 | +0.50 | 32.50 | 0.7764 | 1 |
| Logistic regression without frequencies | JPEG 35 | 6.50 | +0.33 | 30.33 | 0.7649 | 1 |
| Logistic regression without frequencies | Resize via 128 px | 3.67 | -2.50 | 15.83 | 0.7447 | 0.00412 |
| Logistic regression without frequencies | Resize via 64 px | 2.17 | -4.00 | 9.00 | 0.7184 | 1.448e-05 |
| Logistic regression without frequencies | Resize via 128 px + JPEG 60 | 3.83 | -2.33 | 16.00 | 0.7381 | 0.007263 |
| CNN | Reference | 3.83 | +0.00 | 49.67 | 0.9102 | — |
| CNN | JPEG 85 | 3.83 | +0.00 | 49.17 | 0.9074 | 1 |
| CNN | JPEG 60 | 3.83 | +0.00 | 49.17 | 0.9060 | 1 |
| CNN | JPEG 35 | 4.00 | +0.17 | 49.33 | 0.9066 | 1 |
| CNN | Resize via 128 px | 3.50 | -0.33 | 42.00 | 0.9034 | 1 |
| CNN | Resize via 64 px | 3.33 | -0.50 | 29.83 | 0.8783 | 1 |
| CNN | Resize via 128 px + JPEG 60 | 3.50 | -0.33 | 42.33 | 0.9023 | 1 |

## Frequency-feature ablation

The contrast is the change in FPR without frequency features minus the change in FPR with frequency features. Confidence intervals are paired bootstrap intervals for each condition, without multiplicity adjustment.

| Condition | Difference in FPR changes (pp) | 95% CI (pp) |
|---|---:|---:|
| JPEG 85 | -0.17 | [-1.17, +0.67] |
| JPEG 60 | +0.33 | [-0.67, +1.33] |
| JPEG 35 | +0.33 | [-1.00, +1.67] |
| Resize via 128 px | +0.67 | [-0.83, +2.17] |
| Resize via 64 px | -2.17 | [-4.17, -0.33] |
| Resize via 128 px + JPEG 60 | +0.50 | [-1.00, +2.00] |
