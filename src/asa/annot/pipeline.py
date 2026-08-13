"""Stage A: run detection + parsing over a whole manifest.

Split of labour, chosen for the Colab/Kaggle target:

    Stage A (this module, local CPU)   detect masks at native resolution, parse
                                       content, write masks as bilevel PNGs plus
                                       one features table.
    Stage B (asa.data.packaging)       resize image + mask, build all four
                                       variants, write compact uint8 shards.
    Stage C (asa.train, Colab GPU)     train and evaluate from the shards.

Masks are detected at native resolution and never at the training resolution:
glyph strokes are 2-3 px wide, so detecting on a 224 px downsample would destroy
the very structures being measured.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from ..config import AnnotationConfig
from ..utils import get_logger, write_json, provenance
from .detect import detect, masks_to_uint8, overlay_visualisation, summarise, to_bgr
from .parse import parse_annotation, ocr_available

LOGGER = get_logger("asa.annot.pipeline")

MASK_LAYERS = ("annotation", "fan", "text", "calipers", "ticks", "overlay", "redaction")


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return to_bgr(image)


def mask_path(mask_dir: Path, image_id: str, layer: str) -> Path:
    return mask_dir / layer / f"{image_id}.png"


def write_masks(mask_dir: Path, image_id: str, det) -> None:
    for layer, array in masks_to_uint8(det).items():
        out = mask_path(mask_dir, image_id, layer)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), array)


def load_mask(mask_dir: Path, image_id: str, layer: str = "annotation") -> np.ndarray:
    array = cv2.imread(str(mask_path(mask_dir, image_id, layer)), cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise FileNotFoundError(f"missing {layer} mask for {image_id}")
    return array > 127


def _process_one(args: tuple[str, str, dict[str, Any], str, bool]) -> dict[str, Any]:
    image_id, image_path, cfg_dict, mask_dir, use_ocr = args
    cfg = AnnotationConfig(**cfg_dict)
    image = read_image(image_path)
    det = detect(image, cfg)
    write_masks(Path(mask_dir), image_id, det)
    parsed = parse_annotation(image, det, cfg, use_ocr=use_ocr)
    row: dict[str, Any] = {"image_id": image_id}
    row.update(det.features)
    row.update(parsed.to_row())
    return row


def run(
    manifest: pd.DataFrame,
    out_dir: str | Path,
    cfg: AnnotationConfig | None = None,
    use_ocr: bool = True,
    workers: int | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Detect and parse every image in ``manifest``.

    Returns the features table; also writes it, the config, and a provenance
    stamp under ``out_dir``.
    """
    cfg = cfg or AnnotationConfig()
    out_dir = Path(out_dir)
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    if use_ocr and not ocr_available():
        LOGGER.warning(
            "Tesseract not available: masks, the audit (H1/H2) and the geometric "
            "caliper features are unaffected; only the text-content fields for the "
            "H3 supervision arm will be empty. Install with: apt-get install tesseract-ocr"
        )
        use_ocr = False

    rows_in = manifest if limit is None else manifest.head(limit)
    jobs = [
        (str(r.image_id), str(r.image_path), asdict(cfg), str(mask_dir), use_ocr)
        for r in rows_in.itertuples()
    ]

    workers = workers if workers is not None else max(1, (mp.cpu_count() or 2) - 1)
    LOGGER.info("detecting annotations on %d images with %d workers", len(jobs), workers)

    if workers <= 1:
        records = [_report(_process_one(job), i, len(jobs)) for i, job in enumerate(jobs)]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            records = [
                _report(row, i, len(jobs))
                for i, row in enumerate(pool.imap_unordered(_process_one, jobs, chunksize=8))
            ]

    features = pd.DataFrame.from_records(records)
    features.to_csv(out_dir / "annotation_features.csv", index=False)
    write_json(out_dir / "annotation_config.json", asdict(cfg))
    write_json(out_dir / "provenance.json", provenance({"stage": "A", "n_images": len(features), "ocr": use_ocr}))
    LOGGER.info("wrote %s", out_dir / "annotation_features.csv")
    return features


def _report(row: dict[str, Any], index: int, total: int) -> dict[str, Any]:
    if index % 500 == 0:
        LOGGER.info("  %d / %d", index, total)
    return row


def prevalence_report(features: pd.DataFrame) -> dict[str, Any]:
    """Dataset-level annotation prevalence.

    This is the screening layer of Section C.1 and the input to H4: shortcut
    magnitude is hypothesised to be predictable from annotation prevalence and
    annotation-label mutual information, so both must be measured before any
    model is trained.
    """
    area = features["annotation_area_frac"]
    return {
        "n_images": int(len(features)),
        "annotation_area_frac_mean": float(area.mean()),
        "annotation_area_frac_median": float(area.median()),
        "annotation_area_frac_p90": float(area.quantile(0.90)),
        "frac_images_with_any_annotation": float((area > 0.0005).mean()),
        "frac_images_with_calipers": float((features["calipers_n_components"] > 0).mean()),
        "frac_images_with_text": float((features["text_n_components"] > 0).mean()),
        "frac_images_with_overlay": float((features["overlay_area_frac"] > 1e-5).mean()),
        "frac_images_with_redaction": float((features["redaction_area_frac"] > 1e-4).mean()),
        "mean_n_calipers": float(features["calipers_n_components"].mean()),
        # Masks this large are a detector failure, not a heavily annotated
        # frame. Reported so the rate is visible before anything is trained.
        "frac_masks_suspect": float(features.get("mask_suspect", pd.Series(0.0, index=features.index)).mean()),
        "frac_images_with_doppler": float(
            (features.get("doppler_area_frac", pd.Series(0.0, index=features.index)) > 1e-4).mean()
        ),
    }


def export_verification_sample(
    manifest: pd.DataFrame,
    mask_dir: str | Path,
    out_dir: str | Path,
    fraction: float = 0.05,
    seed: int = 20260805,
    cfg: AnnotationConfig | None = None,
) -> pd.DataFrame:
    """Write the human-verification sample (proposal Section 9).

    Two independent verifiers score these overlays and inter-verifier agreement
    is reported, because "the entire study rests on mask quality". The scoring
    sheet is emitted pre-filled with image ids and empty verdict columns.
    """
    cfg = cfg or AnnotationConfig()
    out_dir = Path(out_dir)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    sample = manifest.sample(frac=fraction, random_state=seed).reset_index(drop=True)

    for row in sample.itertuples():
        image = read_image(row.image_path)
        det = detect(image, cfg)
        cv2.imwrite(str(out_dir / "overlays" / f"{row.image_id}.png"), overlay_visualisation(image, det))

    sheet = sample[["image_id", "image_path"]].copy()
    # Verifier columns, left blank on purpose.
    for verifier in ("v1", "v2"):
        sheet[f"{verifier}_missed_annotation"] = ""   # annotation present but unmasked
        sheet[f"{verifier}_false_positive"] = ""      # anatomy masked as annotation
        sheet[f"{verifier}_mask_usable"] = ""         # yes / no
        sheet[f"{verifier}_notes"] = ""
    sheet.to_csv(out_dir / "verification_sheet.csv", index=False)
    write_json(
        out_dir / "verification_instructions.json",
        {
            "task": "Score each overlay in overlays/.",
            "colour_key": {
                "red": "text", "yellow": "calipers", "orange": "ruler ticks",
                "green": "chromatic overlay", "magenta": "curator redaction box",
                "grey outline": "detected imaging sector",
            },
            "columns": {
                "missed_annotation": "1 if any burnt-in annotation is left unmasked, else 0",
                "false_positive": "1 if any anatomy is masked as annotation, else 0",
                "mask_usable": "yes/no overall verdict",
            },
            "gate": "A dataset whose mask sensitivity falls below the configured floor is excluded (Step 3).",
        },
    )
    LOGGER.info("verification sample of %d images -> %s", len(sample), out_dir)
    return sheet


def score_verification(sheet_path: str | Path) -> dict[str, Any]:
    """Sensitivity / specificity and inter-verifier agreement from the returned sheet."""
    sheet = pd.read_csv(sheet_path)
    result: dict[str, Any] = {}
    per_verifier = {}
    for verifier in ("v1", "v2"):
        missed = pd.to_numeric(sheet.get(f"{verifier}_missed_annotation"), errors="coerce").dropna()
        false_pos = pd.to_numeric(sheet.get(f"{verifier}_false_positive"), errors="coerce").dropna()
        if missed.empty:
            continue
        per_verifier[verifier] = {
            "n_scored": int(len(missed)),
            "image_level_sensitivity": float(1.0 - missed.mean()),
            "image_level_specificity": float(1.0 - false_pos.mean()) if len(false_pos) else float("nan"),
        }
    result["per_verifier"] = per_verifier

    both = sheet.dropna(subset=["v1_missed_annotation", "v2_missed_annotation"])
    if len(both):
        a = pd.to_numeric(both["v1_missed_annotation"], errors="coerce")
        b = pd.to_numeric(both["v2_missed_annotation"], errors="coerce")
        agreement = float((a == b).mean())
        # Cohen's kappa on the binary "missed annotation" verdict
        pa = agreement
        pe = float((a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean())))
        result["inter_verifier"] = {
            "n": int(len(both)),
            "raw_agreement": pa,
            "cohens_kappa": float((pa - pe) / (1 - pe)) if pe < 1 else float("nan"),
        }
    return result
