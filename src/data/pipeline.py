"""tf.data input pipeline for the gradability CNN and everything built on it.

decode, augment, make_ds : copied from retina-04-cnn.
decode_image, image_ds   : the path-only variants used by retina-06, -07 and -08.

Pixels stay float32 in [0, 255]. Keras' EfficientNet does its own rescaling as its
first layer, so dividing by 255 here would hand it values near zero. It would still
train, just badly, and nothing would warn you.
"""
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def decode(path, label, size):
    """Read a PNG and resize. Returns float32 in [0, 255] - NOT normalised."""
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (size, size), method="area")
    return tf.cast(img, tf.float32), label


def decode_image(path, size=224):
    """Same as decode(), without a label."""
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (size, size), method="area")
    return tf.cast(img, tf.float32)


def augment(img, label):
    """Geometric augmentation only.

    The usual advice is to jitter brightness and contrast and add noise. Here that
    would destroy the label: exposure and blur ARE what the model must learn to see.
    Flipping or rotating a fundus photograph does not change whether it can be read.
    (The defect heads in retina-06 use no augmentation at all: obstruction always
    enters from the top, and rotating would teach the model it has no direction.)
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    return img, label


def make_ds(sub, images_dir, size, batch=32, training=False, seed=42):
    """sub: rows of the split manifest. images_dir: the eyeq_512 folder."""
    paths = np.array([str(images_dir / f"{s}.png") for s in sub.stem])
    labels = sub.quality.values.astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(sub), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: decode(p, y, size), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


def image_ds(filenames, image_dir, size=224, batch=64):
    """Images only, in the SAME ORDER as `filenames`.

    deterministic=True is load-bearing: a parallel map without it may reorder
    elements, which would silently decouple every prediction from its label.
    """
    paths = np.array([str(image_dir / f) for f in filenames])
    return (tf.data.Dataset.from_tensor_slices(paths)
            .map(lambda p: decode_image(p, size), num_parallel_calls=AUTOTUNE,
                 deterministic=True)
            .batch(batch).prefetch(AUTOTUNE))
