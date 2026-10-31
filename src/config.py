"""Project constants, plus the two path helpers every notebook starts with.

Copied from the setup cells of the notebooks. The values here are the ones the
published results were produced with; change them and you are running a different
experiment.
"""
from pathlib import Path

SEED = 42

# Every dataset is preprocessed once to 512 x 512 and stored that way. The model
# itself sees 224 x 224: retina-04-cnn ran the full training at 384 px as well and
# it scored the same (0.884 accuracy at both sizes), so 224 was kept.
STORE_SIZE = 512
SIZE = 224

# Gradability classes, using EyeQ's own encoding.
CLASSES = ["Good", "Usable", "Reject"]
GOOD, USABLE, REJECT = 0, 1, 2

# The four synthetic defects. This is also the column order of every manifest and of
# the model's defect output.
DEFECTS = ["underexposed", "defocus", "obstruction", "off_centre"]

# The bar the learned defocus head had to beat on real data: a plain Laplacian
# variance correlates with DeepDRiD's human Clarity score at +0.120
# (retina-05-degradation, last comparison cell). Written down before training.
LAPVAR_CLARITY_BASELINE = 0.120

# Private Kaggle datasets, in the order the pipeline produces them, with the notebook
# that publishes each one.
KAGGLE_DATASETS = {
    "eyeq-512-preprocessed": "week-3-retina",
    "eyeq-512-full": "retina-01b-extend-eyeq",
    "retina-external-512": "acquire-external-datasets",
    "retina-results": "retina-baseline",
    "retina-models": "retina-04-cnn",
    "retina-degraded": "retina-05-degradation",
    "retina-multihead": "retina-06-multihead",
    "retina-generalisation": "retina-07-generalisation",
    "retina-deployable": "retina-08-compression",
}


def resolve(slug, root=Path("/kaggle/input"), max_depth=3, required=True):
    """Find an attached Kaggle dataset folder by its slug.

    Kaggle mounts your own datasets under /kaggle/input/datasets/<owner>/<slug> and
    competitions under a different prefix. Hardcoded paths broke twice during the
    project, so every notebook searches instead. The search is depth-limited and
    never walks into the image folders.
    """
    for depth in range(1, max_depth + 1):
        pattern = "/".join(["*"] * (depth - 1) + [slug])
        hits = [p for p in root.glob(pattern) if p.is_dir()]
        if hits:
            return hits[0]
    if required:
        raise FileNotFoundError(
            f"'{slug}' is not attached - Add Data in the right-hand panel")
    return None


def find_file(name, root=Path("/kaggle/input"), required=True):
    """Locate one file anywhere under /kaggle/input, whichever dataset holds it."""
    hits = sorted(root.rglob(name))
    if not hits:
        if required:
            raise FileNotFoundError(f"'{name}' not found under {root}")
        return None
    if len(hits) > 1:
        print(f"   note: {len(hits)} copies of {name}, using {hits[0]}")
    return hits[0]
