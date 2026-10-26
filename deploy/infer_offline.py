"""Offline gradability check for retinal photographs. No network, no GPU.

    pip install -r requirements-deploy.txt
    python infer_offline.py photo1.jpg [photo2.jpg ...]

Needs retina_multihead_float16.tflite next to this file (download it from the
`retina-deployable` Kaggle dataset, tflite/ folder), or pass --model PATH.

For each photograph it prints a verdict (Good / Usable / Reject), the model's
confidence, the most likely defect when the verdict is not Good, and a one-line
retake instruction for the operator.

How this differs from the file retina-08-compression wrote
-----------------------------------------------------------
That version resized the photograph straight to 224 x 224 with bilinear
interpolation. The model was trained and validated on images that had first been
cropped to the retina, padded to a square and area-resized to 512 px, and then
area-resized to 224. Skipping those steps squashes a rectangular camera frame into a
square and changes the resampling - and this model is, among other things, a blur
detector. prepare() below repeats the training pipeline step for step, using the same
OpenCV calls as src/data/preprocess.py. This was found when the code was archived;
deploy/check_against_reference.py measures the agreement in Kaggle.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

STORE_SIZE = 512      # every training image was stored at 512 x 512
SIZE = 224            # what the model sees
CLASSES = ["Good", "Usable", "Reject"]
DEFECTS = ["underexposed", "defocus", "obstruction", "off_centre"]
DEFAULT_MODEL = Path(__file__).resolve().parent / "retina_multihead_float16.tflite"

# The defect head was trained on severities of 0 (absent) or 0.25-1.0 (present),
# so 0.25 is the smallest value that means "this defect is there".
PRESENT = 0.25

RETAKE = {
    "underexposed": "Too dark. Check pupil size and flash, then retake.",
    "defocus": "Blurred. Refocus and ask the patient to keep still, then retake.",
    "obstruction": "Eyelid or lashes in the way. Ask the patient to open wide, then retake.",
    "off_centre": "Off centre. Re-align on the optic disc and macula, then retake.",
}
NO_CAMERA_CAUSE = ("No camera defect stands out, so the eye itself may be the cause "
                   "(for example a cataract). Retake once; if it is still rejected, "
                   "send it on for the grader to judge.")


# ---- preprocessing: identical to src/data/preprocess.py ------------------------
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


def prepare(path):
    """Photograph on disk -> model input, exactly as in training.

    1. crop to the retina, pad to square, area-resize to 512 (as every dataset was)
    2. area-resize 512 -> 224 in float32 (training used tf.image.resize "area")
    Returns float32 [1, 224, 224, 3] in [0, 255] - NOT normalised; EfficientNet
    rescales internally.
    """
    img = cv2.imread(str(path))                      # BGR uint8, like every notebook
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    stored = cv2.resize(pad_square(crop_fov(img)), (STORE_SIZE, STORE_SIZE),
                        interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(stored, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return x[None]


# ---- the model -----------------------------------------------------------------
def _interpreter_class():
    """LiteRT if installed, then the older tflite-runtime, then full TensorFlow.

    The TensorFlow fallback must be attribute access. The version retina-08 wrote
    used `from tensorflow.lite import Interpreter`, which raises ImportError on
    TensorFlow 2.20 because tensorflow.lite is loaded lazily - and tflite-runtime has
    no builds for Python 3.12+, so on a current laptop that script could not start.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    return Interpreter


class Assessor:
    def __init__(self, model_path=DEFAULT_MODEL):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found - download it from the retina-deployable "
                f"Kaggle dataset (tflite/ folder) and put it next to this script")
        self.it = _interpreter_class()(model_path=str(model_path))
        self.it.resize_tensor_input(self.it.get_input_details()[0]["index"],
                                    [1, SIZE, SIZE, 3])
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        # Identify the two heads by SHAPE. TFLite output names are opaque and their
        # order is not guaranteed; reading them by position would silently swap
        # gradability for defects.
        by_units = {int(d["shape"][-1]): d["index"]
                    for d in self.it.get_output_details()}
        if 3 not in by_units or len(DEFECTS) not in by_units:
            raise RuntimeError(f"unexpected model outputs: {by_units}")
        self.grad, self.defects = by_units[3], by_units[len(DEFECTS)]

    def assess(self, path):
        """Returns (verdict, confidence, {defect: severity})."""
        self.it.set_tensor(self.inp, prepare(path))
        self.it.invoke()
        g = self.it.get_tensor(self.grad)[0]
        d = self.it.get_tensor(self.defects)[0]
        return CLASSES[int(g.argmax())], float(g.max()), dict(zip(DEFECTS, d.tolist()))


def advice(verdict, defects):
    """One line for the operator."""
    if verdict == "Good":
        return "Gradable. No retake needed."
    worst = max(defects, key=defects.get)
    cause = RETAKE[worst] if defects[worst] >= PRESENT else NO_CAMERA_CAUSE
    if verdict == "Usable":
        return "Gradable, but borderline. If the patient is still here: " + cause
    return "Not gradable. " + cause


def main(argv=None):
    ap = argparse.ArgumentParser(description="Offline retinal gradability check")
    ap.add_argument("images", nargs="+", help="photograph(s) to check")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="path to the .tflite")
    args = ap.parse_args(argv)

    assessor = Assessor(args.model)
    for p in args.images:
        try:
            verdict, conf, defects = assessor.assess(p)
        except ValueError as e:
            print(f"{p}: {e}")
            continue
        print(f"{p}: {verdict}  ({conf:.0%} confident)")
        if verdict != "Good":
            worst = max(defects, key=defects.get)
            print(f"  most likely cause: {worst}  (severity {defects[worst]:.2f})")
        print(f"  {advice(verdict, defects)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
