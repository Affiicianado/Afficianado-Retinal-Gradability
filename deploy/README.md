# Running the model on the screening laptop

Everything here runs on a CPU with no network connection.

## One-off setup (needs internet once)

1. Install Python 3.10 or later.
2. From this folder:
   ```bash
   pip install -r ../requirements-deploy.txt
   ```
   If `ai-edge-litert` will not install on your machine, run `pip install tensorflow`
   instead. The script tries LiteRT first, then the older `tflite-runtime`, then
   TensorFlow, and works with any of them.
3. Download `retina_multihead_float16.tflite` from the `retina-deployable` Kaggle
   dataset (`tflite/` folder) and put it in this folder.

After that the laptop can go offline.

## Use

```bash
python infer_offline.py photo.jpg
python infer_offline.py photo1.jpg photo2.jpg       # several at once
python infer_offline.py photo.jpg --model other.tflite
```

On macOS and Linux, `python infer_offline.py visit_photos/*.jpg` checks a whole folder.
The Windows shells pass `*` through unexpanded, so list the files there.

The current LiteRT release installs on Windows, Linux and Apple-silicon Macs, but not on
Intel Macs.

For each photograph you get:

- **a verdict** - Good (gradable), Usable (gradable, but borderline) or Reject (not
  gradable) - and the model's confidence;
- **the most likely cause** when the verdict is not Good: whichever of underexposure,
  defocus, obstruction or off-centre framing has the highest predicted severity;
- **a one-line instruction** for the operator, e.g. *"Blurred. Refocus and ask the
  patient to keep still, then retake."*

If the verdict is Reject but no defect reaches a severity of 0.25 (the smallest
severity the defect head was trained on), the script says no camera defect stands out
and the eye itself may be the cause, and suggests retaking once before sending the
image on for the grader. The operator always has the final say; nothing is deleted or
blocked.

## Before relying on it

- Run the `retina-09-demo-pack` notebook in Kaggle once, or paste
  `check_against_reference.py` into a Kaggle cell (instructions at the top of the file).
  Either confirms this script gives the same verdicts the model was validated with; the
  notebook also builds the demo pack.
- On a new camera the Reject threshold needs recalibrating. On DeepDRiD, an unseen
  camera, 8.6% of ungradable photographs were passed as Good.
- This is a research prototype, not a medical device. See `../docs/model-card.md`.

## What changed from the notebook's version

`retina-08-compression` wrote the first version of this script. Two problems were
found when the code was archived, and both are fixed here:

1. It resized the raw photograph straight to 224 x 224 with bilinear interpolation,
   skipping the crop-to-retina, pad-to-square and 512 px steps every training image
   went through. `prepare()` now repeats the training pipeline; a test confirms it
   matches TensorFlow's area resize to within half a grey level.
2. Its last-resort import, `from tensorflow.lite import Interpreter`, fails on
   TensorFlow 2.20, and `tflite-runtime` does not install on Python 3.12 or later. On a
   current laptop the original could not start. The fallback now uses
   `tf.lite.Interpreter`, and LiteRT is tried first.
