"""The shared preprocessing: crop to the retina, pad to square, resize to 512."""
import numpy as np

from src.data.preprocess import crop_fov, pad_square, preprocess, stem


def camera_frame(h=300, w=400, r=120, shift=30):
    """A bright disc on black, off-centre in a rectangular frame (BGR)."""
    img = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    disc = (yy - h // 2) ** 2 + (xx - w // 2 - shift) ** 2 <= r ** 2
    img[disc] = (40, 90, 160)
    return img


def test_stem_strips_folder_and_extension():
    assert stem("train/10003_left.jpeg") == "10003_left"
    assert stem("00000.PNG") == "00000"


def test_crop_fov_trims_to_the_disc():
    out = crop_fov(camera_frame())
    assert out.shape[:2] == (241, 241)          # diameter 2r + 1


def test_crop_fov_leaves_a_blank_image_alone():
    blank = np.zeros((50, 80, 3), np.uint8)
    assert crop_fov(blank) is blank


def test_pad_square_centres_the_content():
    img = np.full((100, 60, 3), 255, np.uint8)
    out = pad_square(img)
    assert out.shape[:2] == (100, 100)
    assert out[:, 20:80].min() == 255 and out[:, :20].max() == 0 and out[:, 80:].max() == 0


def test_preprocess_gives_a_512_square():
    out = preprocess(camera_frame())
    assert out.shape == (512, 512, 3) and out.dtype == np.uint8
