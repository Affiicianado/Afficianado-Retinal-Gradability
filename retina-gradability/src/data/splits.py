from sklearn.model_selection import GroupShuffleSplit


def make_splits(df, val_frac=0.15, test_frac=0.15, seed=SEED):
    """Assign train / val / test, grouped by patient.

    Two passes: pull out test first, then split the remainder. GroupShuffleSplit
    guarantees no patient spans a boundary.
    """
    df = df.copy().reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    rest_i, test_i = next(gss.split(df, groups=df.patient_id))

    rest = df.iloc[rest_i]
    gss2 = GroupShuffleSplit(n_splits=1,
                             test_size=val_frac / (1 - test_frac),
                             random_state=seed)
    _, val_i = next(gss2.split(rest, groups=rest.patient_id))

    df["split"] = "train"
    df.loc[df.index[test_i], "split"] = "test"
    df.loc[rest.index[val_i], "split"] = "val"
    return df


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