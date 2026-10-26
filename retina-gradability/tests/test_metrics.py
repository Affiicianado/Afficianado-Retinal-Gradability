"""Metrics, including the safety metric and how binary datasets are scored."""
import numpy as np
from sklearn.metrics import accuracy_score

from src.eval.metrics import (bootstrap_ci, gradable_binary, safe_spearman,
                              safety_numbers, three_class_report)


def test_perfect_predictions_score_one():
    y = [0, 1, 2] * 20
    rep = three_class_report(y, y)
    assert rep["accuracy"] == rep["macro_f1"] == rep["kappa"] == 1.0


def test_reject_as_good_counts_only_true_rejects():
    y_true = [2, 2, 2, 2, 0, 1]
    y_pred = [0, 1, 2, 2, 0, 0]
    s = safety_numbers(y_true, y_pred)
    assert s["n_reject"] == 4
    assert s["reject_as_good"] == 0.25          # one of four rejects called Good
    assert s["reject_as_gradable"] == 0.5       # Good or Usable


def test_usable_counts_as_gradable_on_binary_datasets():
    assert gradable_binary([0, 1, 2]).tolist() == [0, 0, 1]


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 500)
    p = np.where(rng.random(500) < 0.8, y, rng.integers(0, 3, 500))
    est, lo, hi = bootstrap_ci(y, p, accuracy_score, n_boot=200)
    assert lo <= est <= hi


def test_constant_input_is_reported_not_hidden():
    assert np.isnan(safe_spearman([1, 1, 1], [1, 2, 3]))
