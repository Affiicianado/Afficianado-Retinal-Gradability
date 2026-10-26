"""Callbacks that monitor the metric we actually care about.

Copied from retina-04-cnn.

Keras applies class_weight to the TRAINING loss only. Validation loss stays
unweighted, so it is dominated by the Good class (66% of training data), and early
stopping on val_loss stops at the epoch that is best for Good rather than best
overall. MacroF1 writes val_macro_f1 into the logs so that EarlyStopping,
ReduceLROnPlateau and ModelCheckpoint can all monitor it instead.

Order matters: MacroF1 must come first in the list, because it writes the value the
others read.
"""
import numpy as np
from sklearn.metrics import f1_score
from tensorflow import keras


class MacroF1(keras.callbacks.Callback):
    """Adds val_macro_f1 to logs so EarlyStopping can monitor it.

    Labels are collected once at construction. Predictions come from a single
    predict() call over the whole validation set - calling predict() per batch
    re-enters Keras 72 times an epoch and is several times slower.

    Relies on val_ds being unshuffled (make_ds with training=False), so that
    prediction order matches label order.
    """

    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds
        self.y_true = np.concatenate([y.numpy() for _, y in val_ds])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        y_pred = self.model.predict(self.val_ds, verbose=0).argmax(axis=1)
        f1 = f1_score(self.y_true, y_pred, average="macro")
        logs["val_macro_f1"] = f1
        print(f"  - val_macro_f1: {f1:.4f}")


def callbacks_for(val_ds, ckpt, patience=5):
    return [
        MacroF1(val_ds),
        keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max",
            patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1", mode="max",
            factor=0.3, patience=2, min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_macro_f1", mode="max",
            save_best_only=True, verbose=0),
    ]
