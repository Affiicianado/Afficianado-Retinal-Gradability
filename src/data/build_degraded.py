"""Generate the synthetic training set published as `retina-degraded`.

Copied from retina-05-degradation (generation cell), wrapped in a function.

For every Good image in the EyeQ training split (6,798 images, 4,365 patients): one
clean copy with every severity at 0, so the model sees genuine zeros, plus N_VARIANTS
degraded copies, each with 1-3 defects at severities drawn from [0.25, 1.0].
6,798 x 5 = 33,990 images, 5.8 GB.

Expected label statistics, both of which the notebook measured and matched:
  each defect non-zero in 40% of rows  (4/5 rows degraded x on average 2 of 4 defects)
  mean severity when present 0.625     (the mean of uniform(0.25, 1.0))

Reproducibility: the published dataset is the frozen record. Re-running this produces
statistically equivalent images, not byte-identical ones, because in the notebook the
figure and probe cells drew from the same random generator before generation began.
"""
import cv2
import pandas as pd
from tqdm.auto import tqdm

from . import degrade


def build_degraded(source, eyeq_dir, out_dir, n_variants=degrade.N_VARIANTS):
    """source : split-manifest rows with split == 'train' and quality == 0.
    eyeq_dir  : folder of 512 px EyeQ PNGs (eyeq-512-full/eyeq_512).
    out_dir   : where the degraded PNGs are written.
    Returns the manifest that was published as manifest_degraded.csv.
    """
    patient = dict(zip(source.stem, source.patient_id))
    records = []

    for stem_ in tqdm(source.stem, desc="generating"):
        base = cv2.imread(str(eyeq_dir / f"{stem_}.png"))      # BGR, 512x512
        if base is None:
            raise IOError(f"cannot read {stem_}")

        clean_name = f"{stem_}__clean.png"
        cv2.imwrite(str(out_dir / clean_name), base)
        records.append({"filename": clean_name, "source_stem": stem_,
                        "patient_id": patient[stem_],
                        **{d: 0.0 for d in degrade.DEFECTS}})

        for v in range(n_variants):
            sev = degrade.sample_severities(degrade.rng)
            out = degrade.apply_all(base, sev)
            name = f"{stem_}__v{v}.png"
            cv2.imwrite(str(out_dir / name), out)
            records.append({"filename": name, "source_stem": stem_,
                            "patient_id": patient[stem_], **sev})

    return pd.DataFrame(records)
