"""Synthetic degradation: break good photographs on purpose, so severity IS the label.

Copied from retina-05-degradation (the degradation and sampling cells). Logic is
unchanged; reseed() is new.

No public dataset labels which defect made a photograph ungradable, or how badly.
So we take EyeQ images graded Good, apply a known defect at a known severity, and the
severity we passed in becomes an exact label. Four defects:

    underexposed   small pupil or weak flash         gamma + scale
    defocus        out of focus                      Gaussian blur
    obstruction    eyelid and lashes from the top    dark ellipse + lines
    off_centre     disc and macula drifting out      whole-pixel translation

Three bugs were found in this file by measuring its output, and each fix is
explained where it lives. None of them raised an error.

The functions share one module-level generator, as they did in the notebook. That
matters for reproducibility: see build_degraded.py.
"""
import cv2
import numpy as np

DEFECTS = ["underexposed", "defocus", "obstruction", "off_centre"]
SEED = 42
N_VARIANTS = 4              # degraded copies per source image
MAX_DEFECTS = 3

rng = np.random.default_rng(SEED)


def reseed(seed=SEED):
    """Reset the shared generator (tests use this)."""
    global rng
    rng = np.random.default_rng(seed)
    return rng


def _fov_mask(img, tol=7):
    """Boolean mask of the circular retina, excluding black padding."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > tol


def underexpose(img, sev):
    """Reduced illumination. Mimics a small pupil or weak flash.

    Gamma > 1 darkens midtones more than shadows, which is how real underexposure
    behaves - a linear scale looks obviously fake.

    The ranges are bounded so the retina stays above the field-of-view threshold even
    at severity 1.0. A harsher first version pushed pixels below it; they dropped out
    of the mask, the mean was taken over the brightest survivors, and measured
    brightness went 91 -> 47 -> 25 -> 14 -> 40. The label stopped being monotonic.
    """
    if sev <= 0:
        return img
    gamma = 1.0 + 1.6 * sev                  # 1.0 -> 2.6
    scale = 1.0 - 0.45 * sev                 # 1.0 -> 0.55
    lut = np.array([((i / 255.0) ** gamma) * 255.0 * scale
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def defocus(img, sev):
    """Out-of-focus blur. Gaussian is a fair approximation at this scale.

    The curve is tuned for the resolution the MODEL sees, not the 512 px file on
    disk. Downscaling 512 -> 224 is itself a low-pass filter: it strips about as much
    fine detail on its own as a sigma 1.0 blur. Any sigma much below 1.0 applied here
    is erased by the resize, and what reaches the model is indistinguishable from a
    clean image while still labelled "defocus, severity 0.25". That is a label-noise
    generator, not a hard example.

    Hence the 0.60 floor: the mildest severity actually sampled (0.25) has to survive
    the resize. The 1.7 exponent spreads the rest of the range evenly. The first
    version (a linear ramp to sigma 8) failed the other way: it destroyed all detail
    by severity 0.25, so every level above that was identical.
    """
    if sev <= 0:
        return img
    sigma = 0.60 + 3.6 * (sev ** 1.7)        # 0.25 -> 0.94,  1.0 -> 4.20 px
    k = int(2 * round(3 * sigma) + 1)        # kernel must be odd
    return cv2.GaussianBlur(img, (k, k), sigma)


def obstruction(img, sev):
    """Eyelid intruding from the top edge, plus lashes at higher severity.

    Clipped to the field of view so it does not spill onto the black padding,
    which would be an obvious giveaway rather than a defect.
    """
    if sev <= 0:
        return img
    out = img.copy()
    h, w = img.shape[:2]
    mask = _fov_mask(img)

    # dark ellipse descending from the top
    depth = int(h * (0.12 + 0.48 * sev))
    lid = np.zeros((h, w), np.uint8)
    cv2.ellipse(lid, (w // 2, depth - int(h * 0.35)),
                (int(w * 0.75), int(h * 0.45)), 0, 0, 360, 255, -1)
    lid = (lid > 0) & mask
    out[lid] = (out[lid] * (1.0 - 0.85 * sev)).astype(np.uint8)

    # lashes appear once the lid is well into frame
    if sev > 0.45:
        for _ in range(int(14 * sev)):
            x = int(rng.uniform(w * 0.15, w * 0.85))
            length = int(h * rng.uniform(0.05, 0.16) * sev)
            cv2.line(out, (x, max(depth - 12, 0)), (x + int(rng.uniform(-14, 14)),
                     max(depth - 12, 0) + length), (12, 12, 12),
                     thickness=max(1, int(3 * sev)))
    return out


def off_centre(img, sev):
    """Translate the field of view so the disc and macula drift off frame.

    The shift is rounded to whole pixels, and this is not cosmetic. warpAffine
    resamples, and a fractional offset makes it interpolate between neighbouring
    pixels - bilinear interpolation is a low-pass filter, so a sub-pixel shift
    silently blurs the image. The amount depended on the fractional part of a RANDOM
    angle, so every off-centre image carried a different, meaningless amount of blur,
    contaminating the defocus label as well as this one. Measured: severity 0.25 kept
    41% of clean detail before the fix and 84% after. An integer shift moves pixels
    without touching their values.
    """
    if sev <= 0:
        return img
    h, w = img.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    dist = 0.42 * sev * w
    M = np.float32([[1, 0, round(dist * np.cos(angle))],
                    [0, 1, round(dist * np.sin(angle))]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderValue=(0, 0, 0))


DEGRADE_FN = {
    "underexposed": underexpose,
    "defocus": defocus,
    "obstruction": obstruction,
    "off_centre": off_centre,
}


def sample_severities(rng, defects=DEFECTS, max_defects=MAX_DEFECTS):
    """Pick 1-3 defects and a severity for each. Returns a full dict with zeros for
    the defects not chosen - that is the multi-label target."""
    k = rng.integers(1, max_defects + 1)
    chosen = rng.choice(defects, size=k, replace=False)
    sev = {d: 0.0 for d in defects}
    for d in chosen:
        # weight away from the very low end - a severity of 0.02 is
        # indistinguishable from clean and just adds label noise
        sev[d] = float(rng.uniform(0.25, 1.0))
    return sev


def apply_all(img, sev):
    """Apply in a fixed order. Obstruction last so the lid sits on top of whatever
    else happened, as it would in a real capture."""
    for name in ["off_centre", "underexposed", "defocus", "obstruction"]:
        if sev[name] > 0:
            img = DEGRADE_FN[name](img, sev[name])
    return img
