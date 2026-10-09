from sklearn.model_selection import GroupShuffleSplit


def split_official(df, val_frac=0.18, seed=SEED, drop_overlap=False):
    """Honour EyeQ's published train/test partition, then carve a validation
    set out of train - grouped by patient.

    val_frac=0.18 leaves ~10,285 training images against the 12,543 that
    published methods used. Note the difference in the report.

    drop_overlap: only needed if cell A6 found patients spanning EyeQ's
    train/test boundary. Removes them from test to give a leak-free variant.
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