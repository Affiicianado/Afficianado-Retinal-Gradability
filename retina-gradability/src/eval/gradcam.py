"""Grad-CAM for the defect head, and the test that checks whether it localises.

Copied from retina-06-multihead (cells 11 and 12) and retina-07-generalisation
(cell 8, the pre-sigmoid comparison).

What was measured: the share of heat-map mass in the top three rows of the 7 x 7
grid was 0.409 / 0.426 / 0.412 / 0.447 for the four defects, against 0.429 for no
spatial preference. That test was mis-specified for obstruction. The plotted maps sit
on the eyelid MARGIN, not the dark lid, and at the severities tested (0.6-1.0) the
margin crosses the image centre at 51-70% of its height - grid rows 3 and 4 counting
from 0, outside the three rows top_mass() measures. top_mass() is kept as it ran; a test of mass around each
image's known margin is what should replace it.
Switching to the pre-sigmoid score gave identical numbers, as it must: the sigmoid's
derivative is a single positive number per image, and it cancels when each map is
normalised by its maximum. At 224 px each grid cell covers 32 x 32 pixels.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

CAM_BATCH = 16      # GradientTape keeps every intermediate activation for the backward
                    # pass, so this is far hungrier than predict(). 120 images at once
                    # ran a T4 out of memory on a (120, 14, 14, 480) tensor; chunks of
                    # 16 give numerically identical maps (checked against chunks of 8).
NULL_TOP3 = 3 / 7   # top-third share expected with no spatial preference


def make_cam_model(inputs, feature_map, target_out):
    """inputs: w6.inputs. feature_map: gap.input. target_out: the defect_out handle
    returned by graft() - NOT multihead.outputs[1] (see the ordering note there)."""
    return keras.Model(inputs, [feature_map, target_out], name="cam_model")


def _cam_chunk(cam_model, images, index):
    x = tf.convert_to_tensor(images, dtype=tf.float32)
    with tf.GradientTape() as tape:
        fmap, pred = cam_model(x, training=False)
        tape.watch(fmap)            # fmap is an intermediate tensor, not a variable.
        target = pred[:, index]     # Without watch(), gradient() returns None and the
                                    # einsum fails far from the real cause.
    grads = tape.gradient(target, fmap)
    if grads is None:
        raise RuntimeError("gradient is None - tape.watch(fmap) is missing or misplaced")
    weights = tf.reduce_mean(grads, axis=(1, 2))                 # [B, C]
    cam = tf.nn.relu(tf.einsum("bhwc,bc->bhw", fmap, weights))
    return (cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)).numpy()


def grad_cam(cam_model, images, index, size=224, upsample=True, batch=CAM_BATCH):
    """images: float32 [B, size, size, 3] in [0, 255]. Chunked to fit GPU memory.
    upsample=False returns the raw 7 x 7 maps, which is what top_mass() measures."""
    cams = np.concatenate([_cam_chunk(cam_model, images[i:i + batch], index)
                           for i in range(0, len(images), batch)], axis=0)
    if not upsample:
        return cams
    return tf.image.resize(cams[..., None], (size, size),
                           method="bilinear")[..., 0].numpy()


def top_mass(cams_7x7, rows=3):
    """Share of each map's mass in its top `rows` rows. NULL_TOP3 means no preference."""
    tot = cams_7x7.sum(axis=(1, 2)) + 1e-8
    return cams_7x7[:, :rows, :].sum(axis=(1, 2)) / tot


def logit_clone(defects_layer):
    """A linear copy of the sigmoid defect layer, giving the pre-activation score.
    Check it: sigmoid(clone) must reproduce the model output (it did, to 5.96e-08).

    Take `defects_layer` from the multi-head model loaded from disk, as retina-07 did.
    On the in-memory model straight after graft(), the layer has two call sites (the
    head trained on embeddings, then the graft) and `.input` refers to the first.
    """
    lin = keras.layers.Dense(defects_layer.units, activation=None, name="defects_logit")
    lin.build(defects_layer.input.shape)
    lin.set_weights(defects_layer.get_weights())
    return lin(defects_layer.input)
