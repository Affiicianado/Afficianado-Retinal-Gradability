"""The model-building code, run end to end on a tiny stand-in for the real network.

The real backbone is EfficientNet-B0 and needs the trained checkpoint, so these tests
build a small model with the same shape of graph (input -> conv feature map -> global
average pooling -> dropout -> 3-way softmax) and push it through the same functions the
notebooks used. They check structure, not accuracy.
"""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from tensorflow import keras  # noqa: E402

from src.deploy.compress import convert, open_interpreter  # noqa: E402
from src.eval.gradcam import grad_cam, logit_clone, make_cam_model, top_mass  # noqa: E402
from src.models.multihead import build_embedder, graft, make_heads, split_outputs  # noqa: E402

SIZE = 64


@pytest.fixture(scope="module")
def w6():
    inp = keras.Input((SIZE, SIZE, 3))
    fmap = keras.layers.Conv2D(16, 3, strides=8, padding="same")(inp / 255.0)
    x = keras.layers.GlobalAveragePooling2D()(fmap)
    x = keras.layers.Dropout(0.3)(x)
    return keras.Model(inp, keras.layers.Dense(3, activation="softmax")(x))


@pytest.fixture(scope="module")
def images():
    return np.random.default_rng(0).uniform(0, 255, (6, SIZE, SIZE, 3)).astype("float32")


@pytest.fixture(scope="module")
def grafted(w6):
    embedder, gap, feature_map = build_embedder(w6)
    head, layers = make_heads(int(embedder.output.shape[-1]), hidden=8)
    multihead, defect_out = graft(w6, gap, layers)
    return embedder, head, multihead, defect_out, feature_map


def test_graft_gives_the_same_answer_as_head_on_embeddings(grafted, images):
    embedder, head, multihead, _, _ = grafted
    a = head.predict(embedder.predict(images, verbose=0), verbose=0)
    _, b = split_outputs(multihead.predict(images, verbose=0))
    assert np.abs(a - b).max() < 1e-5


def test_keras_sorts_dict_outputs_alphabetically(grafted):
    """The trap documented in graft(): 'defects' comes before 'gradability'."""
    _, _, multihead, _, _ = grafted
    assert [int(t.shape[-1]) for t in multihead.outputs] == [4, 3]


def test_grad_cam_shapes_and_normalisation(w6, grafted, images):
    _, _, _, defect_out, feature_map = grafted
    cam_model = make_cam_model(w6.inputs, feature_map, defect_out)
    raw = grad_cam(cam_model, images, index=1, size=SIZE, upsample=False, batch=4)
    assert raw.shape == (6, 8, 8)
    assert raw.max() <= 1.0 + 1e-6 and raw.min() >= 0.0
    assert top_mass(raw).shape == (6,)
    up = grad_cam(cam_model, images, index=1, size=SIZE, upsample=True, batch=4)
    assert up.shape == (6, SIZE, SIZE)


def test_logit_clone_reproduces_the_sigmoid(grafted, images, tmp_path):
    """Done on the SAVED model, as retina-07 did: in memory, the shared 'defects'
    layer has two call sites and layer.input refers to the first one."""
    _, _, multihead, _, _ = grafted
    multihead.save(tmp_path / "mh.keras")
    mh = keras.models.load_model(tmp_path / "mh.keras", compile=False)
    layer = mh.get_layer("defects")
    pre = keras.Model(mh.inputs, logit_clone(layer)).predict(images, verbose=0)
    _, post = split_outputs(mh.predict(images, verbose=0))
    assert np.abs(1 / (1 + np.exp(-pre)) - post).max() < 1e-5


def test_tflite_outputs_are_found_by_shape(grafted, images, tmp_path):
    _, _, multihead, _, _ = grafted
    path = tmp_path / "m_float16.tflite"
    mb = convert(multihead, "float16", path)
    assert mb > 0
    it, inp, gi, di = open_interpreter(path, size=SIZE)
    it.set_tensor(inp, images[:1])
    it.invoke()
    assert it.get_tensor(gi).shape == (1, 3)
    assert it.get_tensor(di).shape == (1, 4)
