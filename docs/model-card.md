# Model card: retina_multihead (float16 TFLite)

Following the structure of Mitchell et al. (2019).

## What it is

One EfficientNet-B0 backbone (ImageNet initialisation) with two heads:

- **gradability** - Good / Usable / Reject, softmax;
- **defects** - severity in [0, 1] for underexposure, defocus, obstruction and
  off-centre framing, independent sigmoids.

Input: one colour fundus photograph, cropped to the retina, padded square and resized
to 224 x 224 (`deploy/infer_offline.py` does this). Shipped as
`retina_multihead_float16.tflite`, 8.74 MB, about 18 ms per image on a CPU.

## Intended use

Feedback to a screening operator at the moment of capture: is this photograph likely
to be gradable, and if not, what is the probable cause. It is advice; the operator
decides, and the photograph is never deleted or blocked.

## Not for

- Diagnosing or grading diabetic retinopathy. It judges the photograph, not the eye.
- Use without human oversight, or as a medical device. It has no clinical or regulatory
  validation.
- New cameras without recalibrating the Reject threshold first.

## Training data

- Gradability head: 10,264 EyeQ training images (EyePACS), patient-disjoint from test.
- Defect head: 23,850 synthetically degraded copies of EyeQ Good training images, with
  the applied severity as the label.

## Evaluation

| Test | Result |
|---|---|
| EyeQ test, 16,249 images | accuracy 0.884, macro-F1 0.867, quadratic κ 0.900; 0.6% of ungradable passed as Good |
| Synthetic test, 5,100 images | defect detection AUC 0.990 / 0.999 / 0.995 / 0.996 |
| DeepDRiD sub-scores | defocus vs Clarity r = −0.469 (classical baseline 0.120) |
| Unseen cameras, ungradable passed as Good | DRIMDB 0.0%, FQS 2.1%, DeepDRiD 8.6% |
| IDRiD, Reject rate by disease grade | rises (ρ = 0.481); defocus head rises with it (ρ = 0.516) |
| float16 vs original | same verdict on 100% of confident predictions |

Full tables in `docs/results.md`.

## Known limits

- **Population.** Trained and tested on public datasets from the US, India, China and
  elsewhere. No Western Australian, remote-community or Aboriginal and Torres Strait
  Islander images, and no handheld camera.
- **Calibration.** On DeepDRiD, 8.6% of ungradable images passed as Good, against 0.6%
  on EyeQ.
- **Disease.** The Reject rate rises with disease grade. Our evidence says this is
  real blur in advanced disease, not lesion confusion, but an operator should not take
  repeated Rejects of a diseased eye as a reason to stop trying.
- **Location.** Not established. The localisation test was mis-specified (it looked
  above the eyelid margin the heat maps follow), so no heat map is shown to operators.
- **Coverage.** Four defect types. Overexposure, motion blur and media opacity were
  not modelled.
- **Input.** Nothing checks that the input is a retinal photograph: any image gets a
  verdict. A file that cannot be read is reported, not guessed at.

## Ethical considerations

The error that matters is an ungradable photograph passed as fine: the patient leaves,
and the miss surfaces weeks later. The model was selected and compressed against that
error rather than against accuracy. The opposite error costs a retake. Automation bias
is the main operational risk: an operator who trusts a tool that is usually right may
stop checking. The verdict is therefore advisory and always overridable.
