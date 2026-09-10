import os
import cv2
import numpy as np

TARGET = 512


def stem(s):
    """'train/10003_left.jpeg' -> '10003_left'"""
    return os.path.splitext(os.path.basename(str(s)))[0]


def crop_fov(img, tol=7):
    """Trim the black surround around the circular field of view.
       tol is the brightness below which a pixel counts as background."""
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
    """Full pipeline: crop, pad, resize. BGR in, BGR out."""
    return cv2.resize(pad_square(crop_fov(img)), (size, size),
                      interpolation=cv2.INTER_AREA)