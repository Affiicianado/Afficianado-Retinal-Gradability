"""Evaluation metrics for the gradability task.

Written before any model exists, so that metric selection cannot be
influenced by observed results.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)

CLASSES = ["Good", "Usable", "Reject"]

def three_class_report(y_true, y_pred):
    """Everything we report for three-class gradability.

    Returns a plain dict so it can be JSON-serialised straight into results/.
    """
    p, r, f1, n = precision_recall_fscore_support(
        y_true, y_pred, labels=range(3), zero_division=0)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        # quadratic weighting penalises Good -> Reject far harder than 
        # Good -> Usable, matching how a clinician sees an ordered scale
        "kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "n": int(n[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=range(3)).tolist()
            }

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=0):
    """Resample with replacement n_boot times; report the middle 95%.

    Essential on the small external sets. DRIMDB has under 200 images,
    where the difference between 0.91 and 0.94 could easily be noise.

    Returns (point_estimate, lower, upper).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng, n = np.random.default_rng(seed), len(y_true)

    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue        # degenarate resample, skip
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(metric_fn(y_true, y_pred)), float(lo), float(hi)