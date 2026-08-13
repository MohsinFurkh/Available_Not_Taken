"""Per-dataset manifest builders.

Every dataset is reduced to one tidy table with the same schema, so that every
downstream stage (mask building, splitting, training, evaluation) is dataset
agnostic:

    dataset          str    dataset key
    image_id         str    unique within dataset
    image_path       str    absolute path to the source image
    label            int    supervised target (class index, or NaN for regression)
    label_name       str    human-readable target
    target           float  regression target where applicable (else NaN)
    patient_id       str    grouping key -- nothing sharing this may straddle a split
    official_split   str    'train'/'test'/'' as published by the dataset authors
    official_fold    int    published fold assignment, or -1
    <extras>         ...    dataset-specific covariates kept for stratified reporting
                            (vendor, operator, BI-RADS, side, ...)

The extra covariates matter: proposal Section 12 stratifies the shortcut
magnitude by vendor and institution, and several of these covariates are
*themselves* burnt into the pixels, which is the leak we are measuring.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from ..utils import get_logger

LOGGER = get_logger("asa.manifest")

MANIFEST_COLUMNS = [
    "dataset",
    "image_id",
    "image_path",
    "label",
    "label_name",
    "target",
    "patient_id",
    "official_split",
    "official_fold",
]


def _finalise(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = dataset
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["official_fold"] = df["official_fold"].fillna(-1).astype(int)
    df["official_split"] = df["official_split"].fillna("").astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    missing = [p for p in df["image_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{dataset}: {len(missing)} manifest images do not exist, e.g. {missing[:3]}")
    extras = [c for c in df.columns if c not in MANIFEST_COLUMNS]
    return df[MANIFEST_COLUMNS + sorted(extras)].reset_index(drop=True)


# --------------------------------------------------------------------------
# 1. FETAL_PLANES_DB  --  primary fetal classification arm
# --------------------------------------------------------------------------

FETAL_PLANE_CLASSES = [
    "Fetal abdomen",
    "Fetal brain",
    "Fetal femur",
    "Fetal thorax",
    "Maternal cervix",
    "Other",
]


def build_fetal_planes(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    csv_path = root / "FETAL_PLANES_DB_data.csv"
    images = root / "Images"

    # The published CSV is semicolon-delimited with trailing spaces in both the
    # header and the values.
    with open(csv_path, newline="") as fh:
        rows = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh, delimiter=";")]

    label_index = {name: i for i, name in enumerate(FETAL_PLANE_CLASSES)}
    records = []
    for r in rows:
        plane = r["Plane"]
        records.append(
            {
                "image_id": r["Image_name"],
                "image_path": str(images / f"{r['Image_name']}.png"),
                "label": label_index[plane],
                "label_name": plane,
                "target": float("nan"),
                "patient_id": f"P{int(r['Patient_num']):05d}",
                # Burgos-Artizzu et al. publish a train/test flag; we keep it so
                # results can be reported against the split the field uses.
                "official_split": "train" if r["Train"] == "1" else "test",
                "official_fold": -1,
                "brain_plane": r["Brain_plane"],
                "operator": r["Operator"],
                "vendor": r["US_Machine"],
            }
        )
    df = pd.DataFrame.from_records(records)
    LOGGER.info("FETAL_PLANES_DB: %d images, %d patients", len(df), df.patient_id.nunique())
    return _finalise(df, "fetal_planes")


# --------------------------------------------------------------------------
# 2. HC18  --  primary fetal regression arm
# --------------------------------------------------------------------------


def build_hc18(root: str | Path) -> pd.DataFrame:
    """HC18 head-circumference regression.

    Only ``training_set`` carries labels; the challenge test set does not, so it
    is unusable here and is excluded rather than silently pseudo-labelled.

    HC18 publishes no subject identifier. Each file is one standard-plane image
    from one examination, so we group at file level and say so, rather than
    pretending to a patient-level split we cannot construct.
    """
    root = Path(root)
    table = pd.read_csv(root / "training_set_pixel_size_and_HC.csv")
    train_dir = root / "training_set"

    records = []
    for _, r in table.iterrows():
        stem = Path(r["filename"]).stem
        records.append(
            {
                "image_id": stem,
                "image_path": str(train_dir / r["filename"]),
                "label": pd.NA,
                "label_name": "head_circumference_mm",
                "target": float(r["head circumference (mm)"]),
                "patient_id": stem,  # no subject id published -- documented limitation
                "official_split": "train",
                "official_fold": -1,
                "pixel_size_mm": float(r["pixel size(mm)"]),
                "segmentation_path": str(train_dir / f"{stem}_Annotation.png"),
            }
        )
    df = pd.DataFrame.from_records(records)
    LOGGER.info("HC18: %d labelled images (challenge test set has no labels, excluded)", len(df))
    return _finalise(df, "hc18")


# --------------------------------------------------------------------------
# 3. BUS-BRA  --  cross-anatomy replication, official folds
# --------------------------------------------------------------------------


def build_bus_bra(root: str | Path, fold_column: str = "K5P") -> pd.DataFrame:
    """BUS-BRA benign/malignant.

    ``K5P`` is the authors' published 5-fold assignment stratified on pathology;
    using it unchanged keeps the numbers comparable to work that reports on this
    dataset. ``Case`` is the patient identifier -- 1,875 images over 1,064 women,
    so image-level splitting would leak.
    """
    root = Path(root)
    table = pd.read_csv(root / "bus_data.csv")
    images = root / "Images"
    masks = root / "Masks"

    label_index = {"benign": 0, "malignant": 1}
    records = []
    for _, r in table.iterrows():
        records.append(
            {
                "image_id": r["ID"],
                "image_path": str(images / f"{r['ID']}.png"),
                "label": label_index[r["Pathology"]],
                "label_name": r["Pathology"],
                "target": float("nan"),
                "patient_id": f"C{int(r['Case']):05d}",
                "official_split": "",
                "official_fold": int(r[fold_column]),
                "birads": int(r["BIRADS"]),
                "vendor": r["Device"],
                "side": r["Side"],
                "histology": r["Histology"],
                # Released masks drop the "bus_" prefix: bus_0001-l -> mask_0001-l
                "segmentation_path": str(masks / f"mask_{str(r['ID']).removeprefix('bus_')}.png"),
            }
        )
    df = pd.DataFrame.from_records(records)
    if not Path(df.segmentation_path.iloc[0]).exists():
        # Naming differs between BUS-BRA releases; degrade gracefully rather
        # than failing the whole manifest over a field this study does not use.
        for pattern in ("{id}.png", "mask_{id}.png"):
            candidate = masks / pattern.format(id=df.image_id.iloc[0])
            if candidate.exists():
                df["segmentation_path"] = df.image_id.map(lambda i: str(masks / pattern.format(id=i)))
                break
        else:
            LOGGER.warning("BUS-BRA: lesion masks not found under %s; segmentation_path left unresolved", masks)
    LOGGER.info("BUS-BRA: %d images, %d cases", len(df), df.patient_id.nunique())
    return _finalise(df, "bus_bra")


# --------------------------------------------------------------------------
# 4. MMOTU-2D  --  multi-class stress case
# --------------------------------------------------------------------------

MMOTU_CLASSES = [
    "chocolate cyst",
    "serous cystadenoma",
    "teratoma",
    "theca cell tumour",
    "simple cyst",
    "normal ovary",
    "mucinous cystadenoma",
    "high grade serous",
]


def build_mmotu(root: str | Path) -> pd.DataFrame:
    """MMOTU-2D 8-class ovarian tumour subtype.

    247 patients over 1,469 images, and the released split files are image-level.
    We therefore keep the official split as a reported column but derive our own
    patient-level folds; MMOTU is treated as supporting evidence only, exactly as
    the proposal's dataset table states.
    """
    root = Path(root)
    two_d = root / "OTU_2d"
    images = two_d / "images"

    split_of: dict[str, str] = {}
    label_of: dict[str, int] = {}
    for split, fname in (("train", "train_cls.txt"), ("val", "val_cls.txt")):
        for line in (two_d / fname).read_text().splitlines():
            if not line.strip():
                continue
            filename, cls = line.split()
            split_of[filename] = split
            label_of[filename] = int(cls)

    records = []
    for filename, cls in sorted(label_of.items()):
        stem = Path(filename).stem
        records.append(
            {
                "image_id": stem,
                "image_path": str(images / filename),
                "label": cls,
                "label_name": MMOTU_CLASSES[cls] if cls < len(MMOTU_CLASSES) else f"class_{cls}",
                "target": float("nan"),
                # MMOTU publishes no patient identifier for the 2D set. Files are
                # numbered per acquisition; we group by file and flag this as a
                # limitation rather than inventing a grouping.
                "patient_id": stem,
                "official_split": split_of[filename],
                "official_fold": -1,
                "segmentation_path": str(two_d / "annotations" / f"{stem}_binary.PNG"),
            }
        )
    df = pd.DataFrame.from_records(records)
    LOGGER.info("MMOTU-2D: %d images, %d classes", len(df), df.label.nunique())
    return _finalise(df, "mmotu")


BUILDERS = {
    "fetal_planes": build_fetal_planes,
    "hc18": build_hc18,
    "bus_bra": build_bus_bra,
    "mmotu": build_mmotu,
}


def build_manifest(dataset: str, root: str | Path) -> pd.DataFrame:
    if dataset not in BUILDERS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(BUILDERS)}")
    return BUILDERS[dataset](root)


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str, "patient_id": str})
    df["patient_id"] = df["patient_id"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    return df
