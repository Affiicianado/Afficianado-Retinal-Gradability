"""The offline script must prepare images exactly as training did."""
import cv2
import numpy as np
import pytest

import infer_offline as inf
from src.data.preprocess import preprocess


def camera_frame(h=2592, w=3888, r=1150):
    """A rectangular camera frame with a textured disc, written to disk as PNG."""
    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.ogrid[:h, :w]
    disc = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= r ** 2
    tex = cv2.GaussianBlur(rng.integers(60, 200, (h, w, 3)).astype(np.uint8), (5, 5), 1.5)
    img[disc] = tex[disc]
    return img


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), camera_frame())
    return p


def test_prepare_shape_and_range(photo):
    x = inf.prepare(photo)
    assert x.shape == (1, 224, 224, 3) and x.dtype == np.float32
    assert 0 <= x.min() and x.max() <= 255 and x.max() > 1     # not normalised


def test_prepare_matches_the_training_pipeline(photo):
    """Training: preprocess() -> stored 512 PNG -> tf.image.resize(method='area')."""
    tf = pytest.importorskip("tensorflow")
    stored = preprocess(cv2.imread(str(photo)))
    rgb = cv2.cvtColor(stored, cv2.COLOR_BGR2RGB)
    want = tf.image.resize(rgb, (224, 224), method="area").numpy()
    got = inf.prepare(photo)[0]
    assert np.abs(got - want).max() < 0.5      # within uint8 rounding


def test_the_old_direct_resize_was_different(photo):
    """The original script resized the raw frame straight to 224 (bilinear),
    squashing a 3:2 frame into a square."""
    from PIL import Image
    old = np.asarray(Image.open(photo).convert("RGB").resize((224, 224), Image.BILINEAR),
                     dtype=np.float32)
    new = inf.prepare(photo)[0]
    assert np.abs(old - new).mean() > 10


def test_advice_names_the_defect_or_admits_there_is_none():
    assert inf.advice("Good", {d: 0.0 for d in inf.DEFECTS}).startswith("Gradable")
    blur = {"underexposed": 0.1, "defocus": 0.7, "obstruction": 0.0, "off_centre": 0.0}
    assert "Blurred" in inf.advice("Reject", blur)
    faint = {d: 0.05 for d in inf.DEFECTS}
    assert "eye itself" in inf.advice("Reject", faint)
