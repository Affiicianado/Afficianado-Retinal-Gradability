"""Every metric the project reports.

three_class_report, bootstrap_ci     : copied from retina-baseline. Written before
                                       any model existed, so the choice of metric
                                       could not be influenced by results.
gradable_binary, safety_numbers      : copied from retina-07-generalisation.
safe_spearman                        : copied from retina-08-compression.

The metric that decides deployment is reject_as_good: of the photographs a human
called ungradable, the fraction the model passes as Good. That is the failure the
project exists to prevent - an unreadable image sent on as fine, discovered after the
patient has gone.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

CLASSES = ["Good", "Usable", "Reject"]
GOOD, USABLE, REJECT = 0, 1, 2


def three_class_report(y_true, y_pred):
    """Everything reported for three-class gradability. JSON-serialisable."""
    p, r, f1, n = precision_recall_fscore_support(
        y_true, y_pred, labels=range(3), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        # quadratic weighting penalises Good->Reject far harder than Good->Usable,
        # matching how a clinician reads an ordered scale
        "kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "n": int(n[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=range(3)).tolist(),
    }


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=0):
    """Resample with replacement; report the point estimate and the middle 95%.

    On the 16,249-image EyeQ test set the interval is about +/- 0.5 points. On the
    194-image DRIMDB set it is about +/- 4.5, which is why every external result is
    reported with one.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng, n = np.random.default_rng(seed), len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    if not scores:
        return float(metric_fn(y_true, y_pred)), float("nan"), float("nan")
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(metric_fn(y_true, y_pred)), float(lo), float(hi)


def gradable_binary(labels):
    """Collapse to gradable (Good or Usable) vs ungradable (Reject). 1 = ungradable.

    A Usable image CAN be graded, so counting a predicted Usable as ungradable would
    punish the model for a clinically correct answer. This is how a 3-class model is
    scored against a binary dataset (DRIMDB, DeepDRiD).
    """
    return (np.asarray(labels) == REJECT).astype(int)


def safety_numbers(y_true3, y_pred3):
    """The metric that actually matters: ungradable images waved through."""
    t, p = np.asarray(y_true3), np.asarray(y_pred3)
    truly_bad = t == REJECT
    if truly_bad.sum() == 0:
        return {"n_reject": 0, "reject_as_good": None, "reject_as_gradable": None}
    return {"n_reject": int(truly_bad.sum()),
            "reject_as_good": float((p[truly_bad] == GOOD).mean()),
            "reject_as_gradable": float((p[truly_bad] != REJECT).mean())}


def safe_spearman(a, b):
    """Spearman is undefined if either side is constant - return NaN explicitly
    rather than let it pass silently. A model that predicts one value for every
    image would do this, and that is itself a finding."""
    from scipy.stats import spearmanr
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)
