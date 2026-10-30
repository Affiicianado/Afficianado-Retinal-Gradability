"""The shared preprocessing every image in the project went through.

Copied from week-3-retina, retina-01b-extend-eyeq and acquire-external-datasets, which
all carry byte-identical copies. EyeQ and the four external datasets were processed
by exactly these four functions.

That sameness is the point. If one dataset had been given special treatment, the
generalisation study would have measured preprocessing differences instead of camera
and population differences, and nothing would have raised an error to say so.

Images are OpenCV BGR uint8 throughout.
"""
import os

import cv2
import numpy as np

TARGET = 512


def stem(s):
    """'train/10003_left.jpeg' -> '10003_left'"""
    return os.path.splitext(os.path.basename(str(s)))[0]


def crop_fov(img, tol=7):
    """Trim the black surround around the circular field of view."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def pad_square(img):
    """Pad the short side so resizing does not distort the retina."""
    h, w = img.shape[:2]
    s = max(h, w)
    t, l = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, t, s - h - t, l, s - w - l,
                              cv2.BORDER_CONSTANT, value=[0, 0, 0])


def preprocess(img, size=TARGET):
    """Full pipeline: crop, pad, resize. BGR in, BGR out.

    INTER_AREA because it averages rather than samples, so downscaling a 3888 px
    photograph does not alias. It is still a low-pass filter, which is why the
    defocus curve in degrade.py is tuned at the resolution the model sees.
    """
    return cv2.resize(pad_square(crop_fov(img)), (size, size),
                      interpolation=cv2.INTER_AREA)
