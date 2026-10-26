"""[RUN in Kaggle - CPU is fine, about 2 minutes] Does infer_offline.py give the same
answers the model was validated with?

Attach three datasets: retina-deployable, retina-external-512, retina-generalisation.
Paste this whole file into one cell and run it.

It runs infer_offline.py's own prepare() + model on 400 external images (100 per
dataset) and compares the verdicts with retina-07-generalisation's
predictions_external.csv, which came from the uncompressed Keras model. Expect
agreement close to 100%: float16 matched the original on 100% of confident
predictions in retina-08, and the preprocessing now repeats training.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def find(name):
    hits = sorted(INPUT.rglob(name))
    assert hits, f"{name} not found - is the right dataset attached?"
    return hits[0]


# infer_offline.py: use the corrected copy from the repo. Paste deploy/infer_offline.py
# into /kaggle/working first (a %%writefile cell is easiest), or attach it.
script = WORK / "infer_offline.py"
assert script.exists(), ("put the repo's deploy/infer_offline.py in /kaggle/working "
                         "first (%%writefile /kaggle/working/infer_offline.py)")
model = find("retina_multihead_float16.tflite")
ref = pd.read_csv(find("predictions_external.csv"))
images = find("manifest_external_512.csv").parent / "external_512"

sys.path.insert(0, str(WORK))
import infer_offline as inf                                      # noqa: E402

assessor = inf.Assessor(model)
sample = ref.groupby("dataset").sample(n=100, random_state=42)   # every set has >= 194

rows = []
for r in sample.itertuples():
    verdict, conf, defects = assessor.assess(images / r.filename)
    rows.append({"dataset": r.dataset,
                 "same_verdict": verdict == inf.CLASSES[int(r.pred)],
                 "p_diff": abs(conf - max(r.p_good, r.p_usable, r.p_reject))})
res = pd.DataFrame(rows)

print(res.groupby("dataset").same_verdict.mean().map("{:.1%}".format).to_string())
print(f"\noverall verdict agreement: {res.same_verdict.mean():.1%}  "
      f"(n={len(res)})   median |confidence diff| {res.p_diff.median():.4f}")
print("""
Expect ~99-100%. The few that differ should be images the original model was
already unsure about. If agreement is clearly lower, the preprocessing in
infer_offline.py does not match training - do not demo it until it does.""")
