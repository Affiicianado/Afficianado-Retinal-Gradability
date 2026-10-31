"""Loaders for the four external datasets, and the label crosswalk they apply.

Copied from acquire-external-datasets. The only change: each loader now takes its
dataset folder as an argument instead of reading a notebook global.

Every loader returns the same columns, so everything downstream is source-agnostic:
    quality_raw  exactly as the source gives it. Never overwritten.
    quality      harmonised 0 / 1 / 2 (Good / Usable / Reject) via the crosswalk.
    role         'external' for generalisation tests, 'control' for IDRiD.
    dr_grade, mos, clarity, field_def, artifact
                 extra signal where a source provides it, null elsewhere.

The crosswalk, with reasons, is written up in docs/crosswalk.md.
"""
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from .preprocess import stem

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
GOOD, USABLE, REJECT = 0, 1, 2
SCHEMA = ["dataset", "image_id", "src_path", "patient_id",
          "quality_raw", "quality", "role", "dr_grade", "mos",
          "clarity", "field_def", "artifact"]

# ---------------------------------------------------------------------------
# CROSSWALK - keep in sync with docs/crosswalk.md
#
# DRIMDB   Good -> 0, Bad -> 2, Outlier dropped (not fundus images). Binary
#          source: the Usable middle class cannot be recovered, so DRIMDB is
#          scored gradable-vs-ungradable only.
# DeepDRiD Binary overall quality. Direction verified against the sub-scores
#          rather than assumed: quality 1 has higher clarity (8.66 vs 7.11) and
#          3.2x fewer artifacts (1.27 vs 4.11). So 1 = good.
# IDRiD    All high quality -> 0. Control only, never scored for generalisation.
# FQS      Three-level, same encoding as EyeQ (0 Good, 1 Usable, 2 Reject).
#          The only external source that can test the Good/Usable boundary.
# ---------------------------------------------------------------------------
DRIMDB_MAP = {"good": GOOD, "bad": REJECT}       # outlier deliberately absent
DEEPDRID_MAP = {1: GOOD, 0: REJECT}
FQS_MAP = {"good": GOOD, "usable": USABLE, "reject": REJECT}

IDRID_NATIVE = (4288, 2848)
NULLS = {"dr_grade": None, "mos": None,
         "clarity": None, "field_def": None, "artifact": None}


def _check(df, name):
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if df.quality.isna().any():
        raise ValueError(f"{name}: {df.quality.isna().sum()} unmapped quality")
    if df.image_id.duplicated().any():
        raise ValueError(f"{name}: {df.image_id.duplicated().sum()} duplicate ids")
    return df[SCHEMA].reset_index(drop=True)


def load_drimdb(root):
    """Good 125 + Bad 69 = 194.

    Only images whose IMMEDIATE parent is the class folder. The Kaggle upload nests
    iWeb gallery exports under <class>/drimdb_<class>/iwebalbumfiles/. They are
    excluded by depth, not by size, because some of them are full resolution. An
    earlier "robust" recursive search found 651 files instead of 216.
    """
    rows = []
    for cls, q in DRIMDB_MAP.items():
        folder = next((d for d in Path(root).rglob("*")
                       if d.is_dir() and d.name.lower() == cls), None)
        if folder is None:
            print(f"  WARNING: no '{cls}' folder")
            continue
        for f in sorted(folder.iterdir()):        # iterdir, NOT rglob
            if f.suffix.lower() not in IMG_EXTS:
                continue
            sid = re.sub(r"[^A-Za-z0-9]+", "_", stem(f)).strip("_")
            sid = sid[7:] if sid.lower().startswith("drimdb_") else sid
            rows.append({"dataset": "drimdb", "image_id": f"drimdb_{sid}",
                         "src_path": str(f), "patient_id": f"drimdb_{sid}",
                         "quality_raw": cls, "quality": q,
                         "role": "external", **NULLS})

    df = pd.DataFrame(rows)
    print(f"drimdb   : {len(df):4d}  {df.quality_raw.value_counts().to_dict()}")
    assert len(df) == 194, f"expected 194, got {len(df)}"
    return _check(df, "drimdb")


def load_idrid(root, native=IDRID_NATIVE):
    """Genuine IDRiD only, identified by native resolution.

    The Kaggle upload merges two sources and renumbers each split from 0001:
        516 at 4288x2848  <- IDRiD native, matches the published count
       1032 at 4880x3862  <- different aspect ratio, not IDRiD
    Train/IDRiD_0001.jpg and Test/IDRiD_0001.jpg are DIFFERENT images, so image_id
    carries the split prefix (csv.parent.name, not csv.parent.parent.name - the
    latter collided across splits and produced duplicate ids).

    dr_grade is kept for the pathology control: does the Reject rate climb with
    disease grade?
    """
    rows = []
    for csv in sorted(Path(root).rglob("annotations.csv")):
        split = csv.parent.name.lower()           # Train / Test
        img_dir = csv.parent / "images"
        ann = pd.read_csv(csv).set_index("Image name")
        kept = skipped = 0

        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            with Image.open(p) as im:
                if im.size != native:
                    skipped += 1
                    continue
            rows.append({"dataset": "idrid",
                         "image_id": f"idrid_{split}_{stem(p)}",
                         "src_path": str(p),
                         "patient_id": f"idrid_{split}_{stem(p)}",
                         "quality_raw": "high", "quality": GOOD,
                         "role": "control",       # never a generalisation test
                         **{**NULLS,
                            "dr_grade": ann["Retinopathy grade"].get(p.name)}})
            kept += 1

        print(f"  {split:6s}: kept {kept:4d}, skipped {skipped:4d}")

    df = pd.DataFrame(rows)
    print(f"idrid    : {len(df):4d} at native resolution")
    print(f"           grades {df.dr_grade.value_counts().sort_index().to_dict()}")
    assert len(df) == 516, f"expected 516, got {len(df)}"
    return _check(df, "idrid")


def load_deepdrid(root):
    """Regular fundus images only. Ultra-widefield is a different imaging modality.

    The label CSVs are named explicitly. regular-fundus-source-*.csv also exist and
    carry no quality column; an earlier wildcard picked them up and silently nulled
    the quality label on all 1,600 images.
    """
    reg = Path(root) / "regular_fundus_images"
    want = {"regular-fundus-training.csv", "regular-fundus-validation.csv"}
    csvs = sorted(p for p in reg.rglob("*.csv") if p.name.lower() in want)
    print("label files:", [p.name for p in csvs])

    lab = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    print(f"{len(lab)} labelled rows")
    print("Overall quality:", lab["Overall quality"].value_counts().to_dict())

    index = {stem(p): str(p) for p in reg.rglob("*")
             if p.suffix.lower() in IMG_EXTS}
    lab["stem"] = lab["image_id"].map(stem)
    lab["src_path"] = lab["stem"].map(index)
    miss = lab.src_path.isna().sum()
    if miss:
        print(f"WARNING: {miss} labelled images not on disk")
    lab = lab.dropna(subset=["src_path", "Overall quality"]).drop_duplicates("stem")

    df = pd.DataFrame({
        "dataset": "deepdrid",
        "image_id": "deepdrid_" + lab["stem"],
        "src_path": lab["src_path"],
        "patient_id": "deepdrid_" + lab["patient_id"].astype(str),
        "quality_raw": lab["Overall quality"].astype(int),
        "quality": lab["Overall quality"].astype(int).map(DEEPDRID_MAP),
        "role": "external",
        "dr_grade": lab["patient_DR_Level"],
        "mos": None,
        "clarity": lab["Clarity"],
        "field_def": lab["Field definition"],
        "artifact": lab["Artifact"],
    })

    print(f"deepdrid : {len(df):4d}  {df.quality_raw.value_counts().to_dict()}")
    print(f"           {df.patient_id.nunique()} patients "
          f"({len(df) / df.patient_id.nunique():.1f} images each)")
    return _check(df, "deepdrid")


def load_fqs(root, fold="00"):
    """2,246 images, three-level labels plus a continuous opinion score (MOS 0-100).

    Fold 00's train + val + test files together cover every image exactly once
    (1572 + 225 + 449 = 2246). The other nine folds are alternative random splits of
    the same images and are irrelevant here: FQS is test-only for us.

    The label files say '01625.jpeg' while the images on disk are '00000.PNG'-style.
    stem() strips the extension from both sides so the join works; a raw filename
    match would have found nothing and failed silently.
    """
    div = Path(root) / "divisions"
    parts = []
    for split in ("train", "val", "test"):
        f = div / f"cross_validation-{fold}-{split}.xlsx"
        parts.append(pd.read_excel(f))
        print(f"  {f.name}: {len(parts[-1])} rows")

    lab = pd.concat(parts, ignore_index=True)
    print(f"columns: {list(lab.columns)}")

    ID_COL = "file_name"              # e.g. '01625.jpeg'
    LABEL_COL = "qualityLevel"        # numeric 0/1/2, same encoding as EyeQ
    MOS_COL = "meanOpinionScore"      # continuous 0-100

    index = {stem(p): str(p) for p in (Path(root) / "images").iterdir()
             if p.suffix.lower() in IMG_EXTS}
    print(f"{len(index)} images on disk")

    lab["stem"] = lab[ID_COL].map(stem)
    lab["src_path"] = lab["stem"].map(index)
    miss = lab.src_path.isna().sum()
    if miss:
        print(f"WARNING: {miss} labelled images not on disk")
    lab = lab.dropna(subset=["src_path"]).drop_duplicates("stem")

    rawlab = lab[LABEL_COL]
    mapped = (rawlab.str.strip().str.lower().map(FQS_MAP)
              if rawlab.dtype == object else rawlab.astype(int))

    df = pd.DataFrame({
        "dataset": "fqs",
        "image_id": "fqs_" + lab["stem"],
        "src_path": lab["src_path"],
        "patient_id": "fqs_" + lab["stem"],   # EXIF stripped by the authors, no patient ids
        "quality_raw": rawlab,
        "quality": mapped,
        "role": "external",
        "dr_grade": None,
        "mos": lab[MOS_COL],
        "clarity": None, "field_def": None, "artifact": None,
    })

    counts = df.quality.value_counts().sort_index().to_dict()
    print(f"fqs      : {len(df):4d}  {counts}   (expect {{0: 514, 1: 794, 2: 938}})")
    print(f"           MOS range {df.mos.min():.1f} - {df.mos.max():.1f}")
    assert len(df) == 2246, f"expected 2246, got {len(df)}"
    return _check(df, "fqs")
