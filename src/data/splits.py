"""Patient-level splitting - the most consequential code in the project.

EyeQ has up to two images per patient (left and right eye) and 56% of its images sit
in a correlated pair. Split by image and one eye lands in training while the other
lands in test, which inflates every number reported afterwards. Every split in the
project is grouped by patient_id, and verify_splits() runs on every split that is
loaded.

split_official, verify_splits : copied from retina-baseline.
group_split, assert_disjoint  : copied from retina-06-multihead (synthetic split cell).
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SEED = 42


def split_official(df, val_frac=0.18, seed=SEED, drop_overlap=False):
    """Honour EyeQ's published train/test partition, then carve a validation set
    out of train - grouped by patient.

    val_frac=0.18 leaves 10,264 training images against the 12,543 that published
    methods trained on. Results are still comparable on the test set, which is
    untouched.

    drop_overlap: only needed if patients spanned EyeQ's own train/test boundary.
    retina-01b-extend-eyeq checked: none do.
    """
    df = df.copy().reset_index(drop=True)
    df["split"] = np.where(df.eyeq_split == "test", "test", "train")

    if drop_overlap:
        spread = df.groupby("patient_id")["eyeq_split"].nunique()
        bad = set(spread[spread > 1].index)
        if bad:
            n = df[(df.split == "test") & (df.patient_id.isin(bad))].shape[0]
            df = df[~((df.split == "test") & (df.patient_id.isin(bad)))]
            print(f"dropped {n} test images from {len(bad)} overlapping patients")

    tr = df[df.split == "train"]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    _, val_i = next(gss.split(tr, groups=tr.patient_id))
    df.loc[tr.index[val_i], "split"] = "val"
    return df.reset_index(drop=True)


def verify_splits(df):
    """Raise if any patient spans splits. Call this every single time."""
    spread = df.groupby("patient_id")["split"].nunique()
    bad = spread[spread > 1]
    if not bad.empty:
        raise AssertionError(
            f"{len(bad)} patients in more than one split, "
            f"e.g. {list(bad.index[:5])}")

    missing = {"train", "val", "test"} - set(df["split"].unique())
    if missing:
        raise AssertionError(f"missing splits: {missing}")

    print(f"verified: {df.patient_id.nunique():,} patients, no leakage")
    return True


def group_split(df, frac, seed=SEED):
    """Two-way split grouped by patient_id. Used twice for the synthetic set:
    70 / 30, then the 30 halved into validation and test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    a, b = next(gss.split(df, groups=df.patient_id))
    return df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)


def assert_disjoint(parts):
    """parts: {"train": df, "val": df, "test": df}. Five degraded copies of one
    source image are five correlated images, so one patient on both sides of a
    boundary inflates every number that follows."""
    names = list(parts)
    ids = {n: set(parts[n].patient_id) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = ids[a] & ids[b]
            print(f"  {a} n {b}: {len(overlap)} shared patients")
            if overlap:
                raise AssertionError(f"PATIENT LEAK between {a} and {b}")
    return True
