"""Patient-level splitting: no patient may appear on both sides of a boundary."""
import numpy as np
import pandas as pd
import pytest

from src.data.splits import assert_disjoint, group_split, split_official, verify_splits


def fake_eyeq(n_patients=400, seed=0):
    """One or two eyes per patient, split by patient like EyeQ's own partition."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        part = "test" if p % 2 else "train"
        eyes = ["left", "right"] if rng.random() < 0.4 else ["left"]
        for eye in eyes:
            rows.append({"patient_id": str(p), "eye": eye, "eyeq_split": part,
                         "quality": int(rng.integers(0, 3))})
    return pd.DataFrame(rows)


def test_official_split_is_patient_disjoint_and_keeps_eyeq_test():
    m = split_official(fake_eyeq())
    assert verify_splits(m)
    assert set(m.split) == {"train", "val", "test"}
    assert (m[m.eyeq_split == "test"].split == "test").all()


def test_verify_splits_catches_a_leak():
    m = split_official(fake_eyeq())
    leak = m[m.split == "train"].iloc[[0]].copy()
    leak["split"] = "test"
    with pytest.raises(AssertionError):
        verify_splits(pd.concat([m, leak]))


def test_synthetic_split_keeps_all_copies_of_a_patient_together():
    deg = pd.DataFrame({"patient_id": np.repeat(np.arange(200), 5).astype(str)})
    train, temp = group_split(deg, 0.30)
    val, test = group_split(temp, 0.50)
    assert assert_disjoint({"train": train, "val": val, "test": test})
    assert len(train) + len(val) + len(test) == len(deg)
