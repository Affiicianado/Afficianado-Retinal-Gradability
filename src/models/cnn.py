"""EfficientNet-B0 gradability classifier, trained by two-phase transfer learning.

Copied from retina-04-cnn.

Phase 1: backbone frozen; only the new head trains (3,843 parameters), lr 1e-3.
Phase 2: unfreeze the backbone from layer 100 on (3,844,191 trainable), lr 1e-5.
Both phases: batch 32, up to 20 epochs, balanced class weights, early stopping on
validation macro-F1 (see src/train/callbacks.py for why not val_loss).

Result on EyeQ's test partition: 0.884 accuracy, 0.867 macro-F1, 0.900 quadratic
kappa, and 0.6% of ungradable images passed as Good (18 of 3,220).
"""
from tensorflow import keras

N_CLASSES = 3
FREEZE_UNTIL = 100       # the earliest blocks learn generic edges and colour; leave them
PHASE1_LR, PHASE2_LR = 1e-3, 1e-5
EPOCHS_PER_PHASE = 20
BATCH = 32


def build_model(size, n_classes=N_CLASSES, dropout=0.3):
    """EfficientNet-B0, frozen backbone, fresh head.

    Returns (model, base) so the caller can unfreeze for phase 2.
    """
    inputs = keras.Input(shape=(size, size, 3))

    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(size, size, 3))
    base.trainable = False

    # training=False keeps BatchNorm in inference mode permanently, using the
    # pretrained moving statistics. Without this, fine-tuning with batches of 32
    # overwrites those statistics and accuracy collapses for no visible reason.
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs), base


def unfreeze(base, freeze_until=FREEZE_UNTIL):
    """Phase 2: train everything from layer `freeze_until` onwards."""
    base.trainable = True
    for layer in base.layers[:freeze_until]:
        layer.trainable = False


def compile_phase(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
