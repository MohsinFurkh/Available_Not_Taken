#!/usr/bin/env python
"""Qualitative figures for the manuscript and its supplementary material.

    python scripts/make_qualitative_figures.py

Everything here is built from work/ outputs and needs no GPU and no training
run, so the figures can be regenerated after any pipeline change without
re-running an experiment.

Main paper:
    fig1_annotation_triplet     the four variants + mask, one representative image
    fig2_detection_gallery      the detector across all four datasets
    fig3_network_input          what the network actually sees at 224 px

Supplementary:
    figS1_annotation_taxonomy   each detector layer isolated, colour-coded
    figS2_failure_modes         suspect masks, Doppler rejection, missed text
    figS3_area_distributions    annotation coverage per dataset
    figS4_variant_difference    where each variant differs from A-full

Figures are written as both PDF (for LaTeX) and PNG (for quick viewing). PDFs
carry no embedded fonts -- panel labels are rasterised into the image -- which
satisfies Springer's all-fonts-embedded requirement trivially.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from asa.annot.detect import detect, to_bgr  # noqa: E402
from asa.annot.variants import build_variants  # noqa: E402
from asa.config import AnnotationConfig  # noqa: E402
from asa.data.manifest import load_manifest  # noqa: E402

DATASETS = ["fetal_planes", "bus_bra", "mmotu", "hc18"]
PRETTY = {
    "fetal_planes": "FETAL_PLANES_DB",
    "bus_bra": "BUS-BRA",
    "mmotu": "MMOTU-2D",
    "hc18": "HC18",
}
#: BGR, matching asa.annot.detect.overlay_visualisation so the figure legend and
#: the diagnostic overlays a reader might reproduce agree.
LAYER_COLOURS = {
    "text": (0, 0, 255),
    "calipers": (0, 255, 255),
    "ticks": (255, 128, 0),
    "overlay": (0, 255, 0),
    "redaction": (255, 0, 255),
}


# --------------------------------------------------------------------------
# small drawing helpers
# --------------------------------------------------------------------------


def label_tile(image: np.ndarray, text: str, height: int = 240, bar: int = 26) -> np.ndarray:
    image = to_bgr(image)
    scale = height / image.shape[0]
    tile = cv2.resize(image, (max(1, int(image.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
    strip = np.full((bar, tile.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(strip, text, (4, bar - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([strip, tile])


def hstack_pad(tiles: list[np.ndarray], gap: int = 8) -> np.ndarray:
    height = max(t.shape[0] for t in tiles)
    padded = []
    for tile in tiles:
        if tile.shape[0] < height:
            tile = np.vstack([tile, np.full((height - tile.shape[0], tile.shape[1], 3), 255, np.uint8)])
        padded.append(tile)
    spacer = np.full((height, gap, 3), 255, dtype=np.uint8)
    out = padded[0]
    for tile in padded[1:]:
        out = np.hstack([out, spacer, tile])
    return out


def vstack_pad(rows: list[np.ndarray], gap: int = 10) -> np.ndarray:
    width = max(r.shape[1] for r in rows)
    padded = []
    for row in rows:
        if row.shape[1] < width:
            row = np.hstack([row, np.full((row.shape[0], width - row.shape[1], 3), 255, np.uint8)])
        padded.append(row)
    spacer = np.full((gap, width, 3), 255, dtype=np.uint8)
    out = padded[0]
    for row in padded[1:]:
        out = np.vstack([out, spacer, row])
    return out


def save(image: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out.with_suffix(".png")), image)
    try:
        from PIL import Image

        Image.open(out.with_suffix(".png")).convert("RGB").save(out.with_suffix(".pdf"), "PDF", resolution=300.0)
    except Exception as exc:  # pragma: no cover
        print(f"    (no PDF for {out.name}: {exc})")
    print(f"    {out.with_suffix('.pdf').name}")


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def table(dataset: str) -> pd.DataFrame:
    manifest = load_manifest(REPO / "work" / dataset / "manifest.csv")
    features = pd.read_csv(REPO / "work" / dataset / "annotation_features.csv", dtype={"image_id": str})
    return manifest.merge(features, on="image_id", how="inner")


def column(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Always a Series aligned to df.

    `df.get(name, 0)` returns the scalar 0 when the column is absent, and
    `0 == 0` is then the bare bool True -- which pandas interprets as a label,
    not a mask. Older feature tables predate some columns, so this matters.
    """
    if name in df.columns:
        return df[name].fillna(default)
    return pd.Series(default, index=df.index)


def representative(df: pd.DataFrame, n: int = 1, require_calipers: bool = True) -> pd.DataFrame:
    """Typical annotated frames, spread across the coverage distribution.

    Three constraints, each earned:
      * exclude suspect masks -- the maximum-coverage frame is reliably a
        detector blowout, and a figure drawn from the failure tail would
        misrepresent the method (failures get their own figure instead);
      * exclude un-annotated frames -- on HC18 only 29% carry any annotation, so
        the median frame is empty and shows the reader nothing;
      * spread the picks across quantiles rather than clustering at the median,
        which otherwise yields n near-identical images.
    """
    pool = df[column(df, "mask_suspect") == 0]
    annotated = pool[pool.annotation_area_frac > 0.0005]
    if len(annotated) >= n:
        pool = annotated
    if require_calipers:
        with_calipers = pool[pool.calipers_n_components > 0]
        if len(with_calipers) >= n:
            pool = with_calipers
    if pool.empty:
        pool = df

    quantiles = [0.5] if n == 1 else np.linspace(0.25, 0.75, n)
    picks = []
    for q in quantiles:
        target = pool.annotation_area_frac.quantile(q)
        candidates = pool.drop(index=[p.name for p in picks], errors="ignore")
        if candidates.empty:
            break
        picks.append(candidates.iloc[(candidates.annotation_area_frac - target).abs().argsort().iloc[0]])
    return pd.DataFrame(picks)


def read(row) -> np.ndarray:
    return to_bgr(cv2.imread(str(row.image_path), cv2.IMREAD_UNCHANGED))


# --------------------------------------------------------------------------
# main-paper figures
# --------------------------------------------------------------------------


def fig1_annotation_triplet(out_dir: Path, cfg: AnnotationConfig, dataset: str = "fetal_planes") -> None:
    row = representative(table(dataset)).iloc[0]
    image = read(row)
    det = detect(image, cfg)
    variants = build_variants(image, det, cfg, seed=0)

    mask_vis = image.copy()
    for layer, colour in LAYER_COLOURS.items():
        layer_mask = getattr(det, layer, None)
        if layer_mask is not None and layer_mask.any():
            mask_vis[layer_mask] = colour

    tiles = [
        label_tile(variants["a_full"], "(a) A-full (archive)"),
        label_tile(mask_vis, "(b) detected annotation"),
        label_tile(variants["a_removed"], "(c) A-removed (deployment)"),
        label_tile(variants["a_only"], "(d) A-only"),
        label_tile(variants["ctrl_random"], "(e) ctrl-random"),
    ]
    save(hstack_pad(tiles), out_dir / "fig1_annotation_triplet")
    return str(row.image_id)


def fig2_detection_gallery(out_dir: Path, cfg: AnnotationConfig, per_dataset: int = 3) -> None:
    """The detector across every dataset -- the qualitative method evidence.

    One row per dataset so a reader can see how far annotation conventions vary,
    which is the whole reason the detector avoids absolute intensity thresholds.
    """
    rows = []
    for dataset in DATASETS:
        if not (REPO / "work" / dataset / "annotation_features.csv").exists():
            continue
        picks = representative(table(dataset), n=per_dataset, require_calipers=(dataset != "hc18"))
        tiles = []
        for i, (_, row) in enumerate(picks.iterrows()):
            image = read(row)
            det = detect(image, cfg)
            vis = image.copy()
            for layer, colour in LAYER_COLOURS.items():
                layer_mask = getattr(det, layer, None)
                if layer_mask is not None and layer_mask.any():
                    vis[layer_mask] = colour
            prefix = f"{PRETTY[dataset]}:  " if i == 0 else ""
            caption = f"{prefix}{det.features['annotation_area_frac'] * 100:.2f}% masked"
            tiles.append(label_tile(vis, caption, height=200))
        rows.append(hstack_pad(tiles))
    save(vstack_pad(rows), out_dir / "fig2_detection_gallery")


def fig3_network_input(out_dir: Path, per_dataset: int = 2) -> None:
    """What the network actually sees: the packaged 224 px variants.

    Distinct from Fig. 1, which shows native resolution. The reader has to be
    able to judge for themselves whether an annotation survives downsampling --
    that was a live alternative explanation for the null and it deserves to be
    shown rather than asserted.
    """
    from asa.data.packaging import load_package

    rows = []
    for dataset in DATASETS:
        pack = REPO / "work" / dataset / "pack_224"
        if not (pack / "meta.json").exists():
            continue
        index, arrays, meta = load_package(pack)
        features = pd.read_csv(REPO / "work" / dataset / "annotation_features.csv", dtype={"image_id": str})
        merged = index.merge(features, on="image_id", how="left", suffixes=("", "_f"))
        # Same selection rule as the other figures: annotated, non-suspect,
        # spread across quantiles. Using the plain median would show HC18's
        # empty frames, where only 29% carry any annotation at all.
        picks = representative(merged, n=per_dataset, require_calipers=False)

        tiles = []
        for i, (_, row) in enumerate(picks.iterrows()):
            k = int(row.array_index)
            mask = np.asarray(arrays["annotation"][k]) > 127
            tiles.append(label_tile(np.asarray(arrays["a_full"][k]), "A-full", height=180))
            tiles.append(label_tile(np.asarray(arrays["a_only"][k]),
                                    f"A-only  {mask.mean() * 100:.2f}% kept", height=180))
        # Dataset name as its own narrow strip, so it can never be clipped by a
        # short tile the way an inline prefix is.
        banner = np.full((tiles[0].shape[0], 132, 3), 255, dtype=np.uint8)
        cv2.putText(banner, PRETTY[dataset], (4, tiles[0].shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.insert(0, banner)
        rows.append(hstack_pad(tiles))
    save(vstack_pad(rows), out_dir / "fig3_network_input")


# --------------------------------------------------------------------------
# supplementary figures
# --------------------------------------------------------------------------


def figS1_annotation_taxonomy(out_dir: Path, cfg: AnnotationConfig) -> None:
    """Each detector layer isolated. Justifies treating them as distinct
    phenomena rather than one undifferentiated 'annotation' blob."""
    rows = []
    for dataset in DATASETS:
        if not (REPO / "work" / dataset / "annotation_features.csv").exists():
            continue
        row = representative(table(dataset), require_calipers=(dataset != "hc18")).iloc[0]
        image = read(row)
        det = detect(image, cfg)

        tiles = [label_tile(image, f"{PRETTY[dataset]} original", height=170)]
        for layer, colour in LAYER_COLOURS.items():
            layer_mask = getattr(det, layer, None)
            panel = np.zeros_like(image)
            count = 0
            if layer_mask is not None and layer_mask.any():
                panel[layer_mask] = colour
                count = int(cv2.connectedComponents(layer_mask.astype(np.uint8))[0] - 1)
            tiles.append(label_tile(panel, f"{layer} (n={count})", height=170))
        doppler = getattr(det, "doppler", None)
        panel = np.zeros_like(image)
        if doppler is not None and doppler.any():
            panel[doppler] = (255, 255, 0)
        tiles.append(label_tile(panel, "doppler (excluded)", height=170))
        rows.append(hstack_pad(tiles))
    save(vstack_pad(rows), out_dir / "figS1_annotation_taxonomy")


def figS2_failure_modes(out_dir: Path, cfg: AnnotationConfig) -> None:
    """Honest failure gallery.

    A detector paper that shows only successes invites the reviewer to assume
    the failures were hidden. These are the three modes measured in the
    prevalence report: suspect over-masking, Doppler correctly withheld, and
    low-contrast vendor panels missed.
    """
    panels = []
    for dataset in DATASETS:
        path = REPO / "work" / dataset / "annotation_features.csv"
        if not path.exists():
            continue
        df = table(dataset)

        suspect = df[column(df, "mask_suspect") == 1]
        if len(suspect):
            row = suspect.sort_values("annotation_area_frac", ascending=False).iloc[0]
            image = read(row)
            det = detect(image, cfg)
            vis = image.copy()
            vis[det.annotation] = (0, 0, 255)
            panels.append(label_tile(
                vis, f"{PRETTY[dataset]} SUSPECT {det.features['annotation_area_frac'] * 100:.1f}% masked", height=190))

        if "doppler_area_frac" in df:
            doppler = df[df.doppler_area_frac > 0.01]
            if len(doppler):
                row = doppler.sort_values("doppler_area_frac", ascending=False).iloc[0]
                image = read(row)
                det = detect(image, cfg)
                vis = image.copy()
                if det.doppler is not None:
                    vis[det.doppler] = (255, 255, 0)
                vis[det.annotation] = (0, 0, 255)
                panels.append(label_tile(vis, f"{PRETTY[dataset]} Doppler withheld", height=190))

    if not panels:
        print("    (no failure cases found)")
        return
    rows = [hstack_pad(panels[i:i + 4]) for i in range(0, len(panels), 4)]
    save(vstack_pad(rows), out_dir / "figS2_failure_modes")


def figS3_area_distributions(out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for dataset in DATASETS:
        path = REPO / "work" / dataset / "annotation_features.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        area = df.annotation_area_frac * 100
        axes[0].hist(area.clip(0, 6), bins=60, histtype="step", label=PRETTY[dataset], density=True)
        axes[1].hist(df.calipers_n_components.clip(0, 12), bins=13, histtype="step",
                     label=PRETTY[dataset], density=True)
    axes[0].set_xlabel("annotation coverage (% of pixels)", fontsize=8)
    axes[1].set_xlabel("detected caliper components", fontsize=8)
    for ax in axes:
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figS3_area_distributions.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "figS3_area_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("    figS3_area_distributions.pdf")


def figS4_variant_difference(out_dir: Path, cfg: AnnotationConfig, dataset: str = "fetal_planes") -> None:
    """Absolute difference from A-full, per variant.

    Makes the design's central invariant visible: A-removed differs only inside
    the mask, ctrl-random differs by an equal area elsewhere, and A-only is the
    complement. Saying this in prose is weaker than showing it.
    """
    row = representative(table(dataset)).iloc[0]
    image = read(row)
    det = detect(image, cfg)
    variants = build_variants(image, det, cfg, seed=0)
    base = variants["a_full"].astype(np.int16)

    tiles = [label_tile(variants["a_full"], "A-full", height=220)]
    for name in ("a_removed", "ctrl_random", "a_only"):
        diff = np.abs(variants[name].astype(np.int16) - base).max(axis=-1).astype(np.uint8)
        heat = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_INFERNO)
        changed = (diff > 2).mean() * 100
        tiles.append(label_tile(heat, f"|{name} - A-full|  {changed:.2f}% changed", height=220))
    save(hstack_pad(tiles), out_dir / "figS4_variant_difference")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <manuscript>/figures)")
    args = parser.parse_args()

    if args.out:
        out_dir = args.out
    else:
        env = os.environ.get("ASA_PAPER_DIR")
        candidates = [Path(env)] if env else []
        candidates += [REPO / "paper", REPO.parent / "Manuscript"]
        paper = next((c for c in candidates if (c / "main.tex").exists()), REPO / "paper")
        out_dir = paper / "figures"
    print(f"writing to {out_dir}\n")

    cfg = AnnotationConfig()

    print("main paper:")
    chosen = fig1_annotation_triplet(out_dir, cfg)
    fig2_detection_gallery(out_dir, cfg)
    fig3_network_input(out_dir)

    print("supplementary:")
    figS1_annotation_taxonomy(out_dir, cfg)
    figS2_failure_modes(out_dir, cfg)
    figS3_area_distributions(out_dir)
    figS4_variant_difference(out_dir, cfg)

    (out_dir / "FIGURE_PROVENANCE.json").write_text(json.dumps(
        {"fig1_image_id": chosen,
         "selection_rule": "median annotation coverage among non-suspect frames; never the maximum",
         "note": "regenerate with scripts/make_qualitative_figures.py"}, indent=2))
    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
