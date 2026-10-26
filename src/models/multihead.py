"""The multi-head model: one backbone, a gradability head and a defect head.

Copied from retina-06-multihead (cells 2, 5, 6 and 8).

The trained gradability CNN from retina-04-cnn is kept exactly as it is. Its pooled
1280-number embedding is cached for the 33,990 synthetic images, a small defect head
is trained on those cached vectors, and the trained head layers are then attached
("grafted") to the same pooled output. Image in, two answers out:

    gradability  3-way softmax  (Good / Usable / Reject)   unchanged from retina-04
    defects      4 sigmoids     severity in [0, 1] for each of the four defects

The heads were trained with MSE on severities, not as classifiers: a bounded
regression output reads directly as a severity, and thresholding it also gives
presence or absence, so one head serves both uses.
"""
from tensorflow import keras

SEED = 42
N_DEFECTS = 4


def build_embedder(w6):
    """Wrap the retina-04 model so it outputs its pooled embedding.

    Returns (embedder, gap, feature_map). The pooling layer is found by TYPE, not
    by name, because Keras auto-names it and the suffix varies. gap.input is the
    last convolutional feature map (7 x 7 x 1280 at 224 px), already wired into the
    graph, which is what Grad-CAM needs.
    """
    try:
        gap = next(l for l in w6.layers
                   if isinstance(l, keras.layers.GlobalAveragePooling2D))
    except StopIteration:
        raise RuntimeError("No GlobalAveragePooling2D layer - probably the wrong "
                           "checkpoint. You want effnet_b0_best.keras.")
    embedder = keras.Model(w6.inputs, gap.output, name="embedder")
    return embedder, gap, gap.input


def make_heads(emb_dim, n_defects=N_DEFECTS, hidden=0, dropout=0.3, seed=SEED):
    """Returns (head_model, layers).

    hidden=0 gives a linear probe; hidden=256 the MLP that was kept. `layers` is
    reused by graft(): calling a Keras layer on a second input makes a new graph
    node but keeps the SAME weights, so grafting needs no retraining.
    """
    keras.utils.set_random_seed(seed)
    layers = [keras.layers.Dropout(dropout, name="defect_drop")]
    if hidden:
        layers += [keras.layers.Dense(hidden, activation="relu", name="defect_hidden"),
                   keras.layers.Dropout(dropout, name="defect_drop2")]
    layers.append(keras.layers.Dense(n_defects, activation="sigmoid", name="defects"))

    emb_in = keras.Input(shape=(emb_dim,), name="emb")
    x = emb_in
    for l in layers:
        x = l(x)
    m = keras.Model(emb_in, x, name="defect_head")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m, layers


def choose_head(mse_probe, mse_mlp, min_gain=0.05):
    """The rule written before training: keep the linear probe unless the MLP cuts
    validation MSE by at least 5%. Measured: 0.01141 vs 0.00623, a 45.4% gain, so
    the MLP was kept - the defect information is in the embedding but needs a
    non-linear read-out."""
    rel = (mse_probe - mse_mlp) / mse_probe
    return ("linear" if rel < min_gain else "mlp256"), rel


def graft(w6, gap, head_layers):
    """Attach the trained defect head to the retina-04 model.

    Returns (multihead, defect_out). Keep and use the `defect_out` handle.

    KERAS SORTS DICT OUTPUTS ALPHABETICALLY AND DOES NOT WARN YOU:
        multihead.outputs[0] -> defects      (4 units)
        multihead.outputs[1] -> gradability  (3 units)
    the reverse of how they are declared below. Indexing by position fails
    silently: pred[:, 1] is a valid column of the 3-unit softmax, every shape checks
    out, and you would draw a Grad-CAM of the "Usable" class and label it "defocus".
    Index by key, or by the handle returned here. Never by position.

    retina-06 verified the graft: on 64 test images the grafted model and the
    head-on-cached-embeddings path agreed to 1.19e-07.
    """
    x = gap.output
    for l in head_layers:
        x = l(x)                   # SAME layer objects -> already trained
    defect_out = x
    grad_out = w6.outputs[0]
    multihead = keras.Model(inputs=w6.inputs,
                            outputs={"gradability": grad_out, "defects": defect_out},
                            name="retina_multihead")
    return multihead, defect_out


def split_outputs(out, n_defects=N_DEFECTS):
    """Return (gradability, defects) from a predict() result, identified by shape
    rather than by position, whatever form Keras returned it in."""
    if isinstance(out, dict):
        return out["gradability"], out["defects"]
    g = next(o for o in out if o.shape[-1] == 3)
    d = next(o for o in out if o.shape[-1] == n_defects)
    return g, d
