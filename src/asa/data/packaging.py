"""Stage B: turn images + masks into compact variant shards.

Rationale for the split: the four study datasets total ~2.5 GB on disk, which is
painful to move into Colab and slower still to decode 12,400 PNGs per epoch from
a mounted Drive. Packaging once, locally, at the training resolution gives a
single memory-mappable uint8 array per variant -- roughly 150 MB per dataset at
224 px -- which uploads in minutes and trains without an IO bottleneck.

The masks are still *detected* at native resolution (Stage A); only the packaged
copies are downsampled, and the mask is downsampled with nearest-neighbour so a
2-px glyph stroke survives as a mask pixel rather than being averaged away.

Layout written to ``out_dir``:

    a_full.npy        uint8  (N, S, S, C)   memory-mappable
    a_removed.npy     uint8
    a_only.npy        uint8
    ctrl_random.npy   uint8
    annotation.npy    uint8  (N, S, S)      packed mask, for saliency-mass scoring
    index.csv         the manifest rows, in array order, joined to the features
    meta.json         resolution, channel count, config, provenance
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ..config import AnnotationConfig, VARIANTS
from ..utils import get_logger, provenance, write_json
from ..annot.detect import Detection, to_bgr
from ..annot.pipeline import load_mask, read_image
from ..annot.variants import build_variants

LOGGER = get_logger("asa.packaging")


def _resize(image: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(image, (size, size), interpolation=interp)


def _detection_from_masks(mask_dir: Path, image_id: str, shape: tuple[int, int]) -> Detection:
    """Rebuild a Detection from the PNGs written by Stage A.

    Variants are built from the stored masks rather than by re-running detection,
    so that the masks a human verified are exactly the masks the models are
    trained on. Re-detecting here would silently decouple the two.
    """
    layers = {}
    for layer in ("annotation", "fan", "text", "calipers", "ticks", "overlay", "redaction"):
        try:
            layers[layer] = load_mask(mask_dir, image_id, layer)
        except FileNotFoundError:
            layers[layer] = np.zeros(shape, dtype=bool)
    det = Detection(
        fan=layers["fan"],
        text=layers["text"],
        calipers=layers["calipers"],
        ticks=layers["ticks"],
        overlay=layers["overlay"],
        redaction=layers["redaction"],
        annotation=layers["annotation"],
    )
    return det


def _pack_one(args: tuple[int, str, str, str, dict[str, Any], int, bool, int]) -> tuple[int, dict[str, np.ndarray]]:
    row_index, image_id, image_path, mask_dir, cfg_dict, size, grayscale, seed = args
    cfg = AnnotationConfig(**cfg_dict)
    image = read_image(image_path)
    det = _detection_from_masks(Path(mask_dir), image_id, image.shape[:2])
    variants = build_variants(image, det, cfg, seed=seed + row_index)

    packed: dict[str, np.ndarray] = {}
    for name, array in variants.items():
        array = to_bgr(array)
        if grayscale:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        packed[name] = _resize(array, size)
    packed["annotation"] = _resize(det.annotation.astype(np.uint8) * 255, size, nearest=True)
    return row_index, packed


def package(
    manifest: pd.DataFrame,
    mask_dir: str | Path,
    out_dir: str | Path,
    cfg: AnnotationConfig | None = None,
    size: int = 224,
    grayscale: bool = False,
    features: pd.DataFrame | None = None,
    workers: int | None = None,
    seed: int = 20260805,
) -> Path:
    cfg = cfg or AnnotationConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(manifest)
    channels = 1 if grayscale else 3

    arrays: dict[str, np.ndarray] = {
        name: np.lib.format.open_memmap(
            out_dir / f"{name}.npy", mode="w+", dtype=np.uint8,
            shape=(n, size, size) if grayscale else (n, size, size, channels),
        )
        for name in VARIANTS
    }
    arrays["annotation"] = np.lib.format.open_memmap(
        out_dir / "annotation.npy", mode="w+", dtype=np.uint8, shape=(n, size, size)
    )

    jobs = [
        (i, str(r.image_id), str(r.image_path), str(mask_dir), asdict(cfg), size, grayscale, seed)
        for i, r in enumerate(manifest.itertuples())
    ]
    workers = workers if workers is not None else max(1, (mp.cpu_count() or 2) - 1)
    LOGGER.info("packaging %d images at %dpx (%s) with %d workers", n, size, "gray" if grayscale else "rgb", workers)

    def _store(result: tuple[int, dict[str, np.ndarray]]) -> None:
        index, packed = result
        for name, array in packed.items():
            arrays[name][index] = array

    if workers <= 1:
        for i, job in enumerate(jobs):
            _store(_pack_one(job))
            if i % 500 == 0:
                LOGGER.info("  %d / %d", i, n)
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_pack_one, jobs, chunksize=8)):
                _store(result)
                if i % 500 == 0:
                    LOGGER.info("  %d / %d", i, n)

    for array in arrays.values():
        array.flush()

    index = manifest.reset_index(drop=True).copy()
    index["array_index"] = np.arange(n)
    if features is not None:
        index = index.merge(features, on="image_id", how="left", suffixes=("", "_feat"))
    index.to_csv(out_dir / "index.csv", index=False)

    write_json(
        out_dir / "meta.json",
        {
            "n": n,
            "size": size,
            "grayscale": grayscale,
            "variants": list(VARIANTS),
            "annotation_config": asdict(cfg),
            "provenance": provenance({"stage": "B"}),
        },
    )
    LOGGER.info("packaged -> %s", out_dir)
    return out_dir


def load_package(pack_dir: str | Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Memory-map a packaged dataset. Safe to call from a Colab notebook."""
    from ..utils import read_json

    pack_dir = Path(pack_dir)
    meta = read_json(pack_dir / "meta.json")
    index = pd.read_csv(pack_dir / "index.csv", dtype={"image_id": str})
    index["patient_id"] = index["patient_id"].astype(str)
    arrays = {
        name: np.load(pack_dir / f"{name}.npy", mmap_mode="r")
        for name in [*meta["variants"], "annotation"]
        if (pack_dir / f"{name}.npy").exists()
    }
    return index, arrays, meta
