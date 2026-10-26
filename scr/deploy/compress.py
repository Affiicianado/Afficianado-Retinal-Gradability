"""Compression to TensorFlow Lite, and the rule that picks what ships.

Copied from retina-08-compression (cells 3, 4, 5 and the corrected cell 10).

Four variants of the 18.37 MB Keras model were built and re-tested on all 4,556
external images against the uncompressed model's own predictions:

    variant   size     CPU ms/img  confident agreement  worst reject-as-good
    float32   17.36    17.8        100.0%               8.6%
    float16    8.74    18.1        100.0%               8.4%   <- shipped
    dynamic    4.88    32.6         94.9%               9.1%
    int8       5.25    18.7         60.2%               3.3%

int8 looks safest on that last column and is the wrong model: it rejects three times
as many good-quality IDRiD images (19.0% vs 6.2%) and its defocus head loses most of
its link to disease grade (0.167 vs 0.516). A smaller model with a better-looking
score had become a different model. choose_variant() encodes the rule that rejects it.

TensorFlow is imported inside the functions that need it, so choose_variant() can be
tested without it.
"""
import time

import numpy as np

VARIANTS = ("float32", "float16", "dynamic", "int8")


def convert(model, name, path, calibration=None):
    """Write one TFLite variant to `path`; return its size in MB.

    int8 calibration MUST use real images (we used 200 external images). Random
    noise gives wrong activation ranges and a model that is quietly inaccurate
    rather than broken.
    """
    import tensorflow as tf
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    if name == "float16":
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec = tf.lite.TargetSpec(supported_types=[tf.float16])
    elif name == "dynamic":
        c.optimizations = [tf.lite.Optimize.DEFAULT]
    elif name == "int8":
        if calibration is None:
            raise ValueError("int8 needs calibration images")
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = lambda: ([calibration[i:i + 1]]
                                            for i in range(len(calibration)))
    elif name != "float32":
        raise ValueError(f"unknown variant {name}")
    blob = c.convert()
    path.write_bytes(blob)
    return len(blob) / 1e6


def open_interpreter(path, size=224, n_defects=4):
    """Open a TFLite model and find its two outputs BY SHAPE, not by order.

    TFLite names outputs 'StatefulPartitionedCall_1:0' and so on, with no promise
    about which is which. Reading them positionally would silently swap gradability
    for defects. The defect head has 4 units and the gradability head 3, so shape
    identifies them unambiguously.
    Returns (interpreter, input_index, gradability_index, defects_index).
    """
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(path))
    it.resize_tensor_input(it.get_input_details()[0]["index"], [1, size, size, 3])
    it.allocate_tensors()
    inp = it.get_input_details()[0]["index"]
    by_units = {int(d["shape"][-1]): d["index"] for d in it.get_output_details()}
    assert 3 in by_units and n_defects in by_units, (
        f"cannot identify outputs by shape: {by_units}")
    return it, inp, by_units[3], by_units[n_defects]


def latency_ms(path, probe, warmup=10, timed=60):
    """Batch-1 CPU latency. Batch 1 on purpose: in the van the operator photographs
    one eye and waits for one answer, so throughput over a big batch is not the
    number that matters. Treat the ranking as the result; absolute milliseconds
    depend on the machine."""
    it, inp, _, _ = open_interpreter(path, size=probe.shape[1])
    for j in range(warmup):
        it.set_tensor(inp, probe[j:j + 1])
        it.invoke()
    t0 = time.perf_counter()
    for j in range(warmup, warmup + timed):
        it.set_tensor(inp, probe[j:j + 1])
        it.invoke()
    return (time.perf_counter() - t0) / timed * 1000


def choose_variant(stats, reference, latency_budget_ms=200.0, agreement_floor=0.99,
                   safety_tolerance=0.01, pathology_tol=0.10):
    """The deployment rule, as run in the corrected cell 10 of retina-08.

    stats[name] needs: size_mb, ms, confident_agreement, worst_reject_as_good,
                       grade_vs_reject, grade_vs_defocus
    reference needs:   worst_reject_as_good, grade_vs_reject, grade_vs_defocus
                       (all measured on the uncompressed model)

    A variant is eligible only if it is still the same model:
      - it agrees with the original on >= 99% of confident predictions,
      - its worst reject-as-good rate rises by no more than 1 point,
      - both disease-grade correlations stay within 0.10 of the original,
      - it answers one photograph inside the latency budget.
    Among eligible variants the SMALLEST wins. Latency is a budget, not a score:
    the constraint is shipping to offline laptops, and anything well inside an
    operator's patience is equally good. (The first version of this rule sorted on
    latency first and shipped float32 - barely compressed, 1.1x - to save less than
    a millisecond per image.)

    Returns (chosen, eligible, exclusions).
    """
    eligible, why = [], {}
    for name, s in stats.items():
        fails = []
        if s["confident_agreement"] < agreement_floor:
            fails.append(f"agreement {s['confident_agreement']:.1%}")
        if s["worst_reject_as_good"] > reference["worst_reject_as_good"] + safety_tolerance:
            fails.append("reject-as-good +"
                         f"{100 * (s['worst_reject_as_good'] - reference['worst_reject_as_good']):.1f}pt")
        if abs(s["grade_vs_reject"] - reference["grade_vs_reject"]) > pathology_tol:
            fails.append(f"grade-vs-reject {s['grade_vs_reject']:+.3f}")
        if abs(s["grade_vs_defocus"] - reference["grade_vs_defocus"]) > pathology_tol:
            fails.append(f"grade-vs-defocus {s['grade_vs_defocus']:+.3f}")
        if s["ms"] > latency_budget_ms:
            fails.append(f"latency {s['ms']:.0f} ms")
        why[name] = fails
        if not fails:
            eligible.append(name)
    if not eligible:
        raise ValueError("no variant met the bar - ship the uncompressed model and say so")
    chosen = min(eligible, key=lambda n: stats[n]["size_mb"])
    return chosen, eligible, why
