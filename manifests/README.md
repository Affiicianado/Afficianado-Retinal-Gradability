# Manifests

Frozen once and reused, never recomputed. Every later notebook reads these rather than
re-splitting, so every result is measured on the same images.

| File | From | What it is |
|---|---|---|
| `manifest_split.csv` | `retina-results` | all 28,792 EyeQ images with their train / val / test split (patient-disjoint) |
| `manifest_external_512.csv` | `retina-external-512` | the 4,556 external images with raw and harmonised labels, sub-scores, disease grades |
| `manifest_degraded_split.csv` | `retina-multihead` | the 33,990 synthetic images, their four severities and their split |
| `manifest_cam_eval.csv` | `retina-multihead` | the 340 synthetic images used for the localisation tests |

Image paths in these files point at Kaggle mounts. They contain labels and identifiers
from public research datasets, and no images.
