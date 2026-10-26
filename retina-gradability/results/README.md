# Result files

One JSON per experiment, written by `record()` (`src/eval/record.py`) at the moment the
measurement was made. Each holds the experiment name, when it ran, its parameters and
its metrics. Every headline number in the report traces to one of these; `docs/results.md`
says which.

| File | Notebook | What it measured |
|---|---|---|
| `w5-majority.json` | retina-baseline | always-Good floor on the EyeQ test partition |
| `w5-classical-lgbm.json` | retina-baseline | 8 features + LightGBM, with bootstrap CI and per-class scores |
| `w6-effnet-b0-224.json` | retina-04-cnn | the gradability CNN at 224 px |
| `w6-effnet-b0-384.json` | retina-04-cnn | the same at 384 px (resolution ablation) |
| `w10-defect-heads.json` | retina-06-multihead | defect heads on the synthetic test split; probe vs MLP |
| `w10-deepdrid-transfer.json` | retina-06-multihead | defect heads against DeepDRiD's human sub-scores |
| `w10-idrid-pathology-control.json` | retina-06-multihead | Reject rate by disease grade on IDRiD |
| `w10-pathology-diagnostic.json` | retina-06-multihead | defect trends with grade; good-quality reject rates per dataset |
| `w10-cam-localisation.json` | retina-06-multihead | Grad-CAM top-third mass per defect |
| `w11-generalisation.json` | retina-07-generalisation | per-dataset metrics, confusion matrices, safety |
| `w11-defect-transfer-external.json` | retina-07-generalisation | sub-score correlations on every dataset that has them |
| `w11-localisation-logit-test.json` | retina-07-generalisation | post-sigmoid vs pre-activation Grad-CAM |
| `w12-safety-under-compression.json` | retina-08-compression | reject-as-good per dataset per TFLite variant |
| `w12-pathology-under-compression.json` | retina-08-compression | disease-grade correlations per variant |
| `w12-compression-decision.json` | retina-08-compression | size, latency, agreement for every variant (first rule) |
| `w12-compression-decision-final.json` | retina-08-compression | the corrected rule and its choice (float16); supersedes the file above |
| `predictions_external.csv` | retina-07-generalisation | every external image: verdict, class probabilities, defect severities, human labels |

The `w5`–`w12` prefixes are the experiment identifiers the notebooks used, kept so each
file matches the name printed in its notebook.
