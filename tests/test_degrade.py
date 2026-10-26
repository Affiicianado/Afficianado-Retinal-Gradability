"""Regression tests for the three silent bugs found in the degradation code.

The notebook caught each of these by printing a measurement and noticing it was
impossible. These tests make the same measurements on a synthetic retina, so the
bugs cannot quietly come back.
"""
import cv2
import numpy as np
import pytest

from src.data import degrade as D

LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.fixture
def retina():
    """512 px textured disc on black, brightness in the range real retinas have."""
    rng = np.random.default_rng(0)
    img = np.zeros((512, 512, 3), np.uint8)
    yy, xx = np.mgrid[:512, :512]
    disc = (yy - 256) ** 2 + (xx - 256) ** 2 <= 250 ** 2
    tex = cv2.GaussianBlur(rng.integers(90, 220, (512, 512, 3)).astype(np.uint8),
                           (3, 3), 0.8)
    img[disc] = tex[disc]
    return img


def mean_in_fov(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return g[g > 7].mean()


def lap_var_at_224(img):
    """Sharpness at the resolution the model sees, not the one stored on disk."""
    small = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def illum_cv(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    q = [g[:256, :256], g[:256, 256:], g[256:, :256], g[256:, 256:]]
    m = [x[x > 7].mean() for x in q]
    return np.std(m) / np.mean(m)


def test_underexposure_darkens_at_every_step(retina):
    # the first version measured 91 -> 47 -> 25 -> 14 -> 40
    m = [mean_in_fov(D.underexpose(retina, s)) for s in LEVELS]
    assert all(a > b for a, b in zip(m, m[1:])), m


def test_underexposure_keeps_the_retina_inside_the_mask(retina):
    before = (cv2.cvtColor(retina, cv2.COLOR_BGR2GRAY) > 7).sum()
    after = (cv2.cvtColor(D.underexpose(retina, 1.0), cv2.COLOR_BGR2GRAY) > 7).sum()
    assert after / before > 0.9


def test_defocus_levels_stay_distinct_at_224(retina):
    clean = lap_var_at_224(retina)
    kept = [lap_var_at_224(D.defocus(retina, s)) / clean for s in LEVELS[1:]]
    assert kept[0] < 0.8, "mildest severity is erased by the 512 -> 224 resize"
    assert all(a > b for a, b in zip(kept, kept[1:])), kept


def test_off_centre_moves_pixels_without_resampling(retina):
    D.reseed(0)
    out = D.off_centre(retina, 0.5)
    before = set(map(tuple, retina.reshape(-1, 3)))
    after = set(map(tuple, out.reshape(-1, 3)))
    assert after <= before | {(0, 0, 0)}, "shift created new pixel values (= blur)"


def test_a_fractional_shift_would_have_blurred(retina):
    """Documents the original bug: a sub-pixel warpAffine interpolates."""
    M = np.float32([[1, 0, 10.37], [0, 1, -4.61]])
    out = cv2.warpAffine(retina, M, (512, 512), borderValue=(0, 0, 0))
    assert lap_var_at_224(out) < 0.9 * lap_var_at_224(retina)


def test_obstruction_darkens_the_top(retina):
    D.reseed(0)
    v = [illum_cv(D.obstruction(retina, s)) for s in (0.0, 0.5, 1.0)]
    assert v[2] > v[1] > v[0]


def test_sampled_labels_follow_the_contract():
    rng = np.random.default_rng(0)
    for _ in range(500):
        sev = D.sample_severities(rng)
        present = [v for v in sev.values() if v > 0]
        assert set(sev) == set(D.DEFECTS)
        assert 1 <= len(present) <= 3
        assert all(0.25 <= v <= 1.0 for v in present)
