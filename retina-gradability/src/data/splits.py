# FILE: src/data/splits.py  (the check half - paste into notebooks)

def verify_splits(df):
    """Raise if any patient spans more than one split.

    Call this immediately after make_splits(), every single time.
    """
    spread = df.groupby("patient_id")["split"].nunique()
    bad = spread[spread > 1]
    if not bad.empty:
        raise AssertionError(
            f"{len(bad)} patients appear in more than one split, "
            f"e.g. {list(bad.index[:5])}")

    missing = {"train", "val", "test"} - set(df["split"].unique())
    if missing:
        raise AssertionError(f"missing splits: {missing}")

    print(f"splits verified: {df.patient_id.nunique():,} patients, "
          f"no leakage")
    return True