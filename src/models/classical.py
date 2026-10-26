"""Classical baseline: eight hand-crafted image measurements + gradient boosting.

Copied from retina-baseline. On EyeQ's test partition it scored 0.710 accuracy,
0.657 macro-F1 and 0.674 quadratic kappa, against a majority-class floor of 0.521 -
and it let 13.1% of ungradable images through as Good. That last number became the
project's safety metric.

Each feature measures a physical property one of the defects changes. All intensity
statistics are taken inside the retina mask only, so a camera that leaves more black
border does not look darker.
"""
import cv2
import numpy as np

FEATURE_NAMES = ["lap_var", "tenengrad", "mean", "std",
                 "clip_low", "clip_high", "illum_cv", "green_std"]


def features(img):
    """img: RGB uint8, 512x512 from the cached dataset. Returns 8 scalars."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > 7                      # retina only, not the black padding
    f = {}

    # ---- SHARPNESS ----------------------------------------------------
    # Laplacian variance: high on crisp edges, near zero on blur.
    f["lap_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Tenengrad: gradient energy. A second opinion on focus that responds
    # differently to sensor noise than the Laplacian does.
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    f["tenengrad"] = float(np.mean(gx ** 2 + gy ** 2))

    # ---- EXPOSURE -----------------------------------------------------
    v = gray[mask]
    f["mean"] = float(v.mean())
    f["std"] = float(v.std())
    f["clip_low"] = float((v < 15).mean())       # crushed shadows
    f["clip_high"] = float((v > 240).mean())     # blown highlights

    # ---- ILLUMINATION EVENNESS ----------------------------------------
    # An eyelid or bad flash angle darkens one region. Compare quadrant means;
    # uneven lighting shows as high relative spread.
    h, w = gray.shape
    quads = [gray[:h // 2, :w // 2], gray[:h // 2, w // 2:],
             gray[h // 2:, :w // 2], gray[h // 2:, w // 2:]]
    qm = [q[q > 7].mean() if (q > 7).any() else 0.0 for q in quads]
    f["illum_cv"] = float(np.std(qm) / (np.mean(qm) + 1e-6))

    # ---- COLOUR -------------------------------------------------------
    # The green channel carries most retinal vessel contrast, so its spread is a
    # cheap proxy for structural visibility.
    f["green_std"] = float(img[:, :, 1][mask].std())

    return f


def make_classifier(seed=42):
    """The LightGBM configuration the baseline was trained with."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              class_weight="balanced", random_state=seed, verbose=-1)
