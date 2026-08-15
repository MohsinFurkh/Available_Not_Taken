#!/usr/bin/env python
"""Regenerate the paper's figures and tables from work/ outputs.

    python scripts/make_paper_assets.py

Every number in the manuscript is produced here and \\input into the LaTeX, so
nothing is transcribed by hand. Transcription is how a table ends up disagreeing
with the code that produced it, and a paper whose whole subject is measurement
integrity cannot afford that.

Writes:
    paper/figures/annotation_triplet.pdf   the four variants of one real image
    paper/figures/prevalence.pdf           annotation prevalence across datasets
    paper/tables/datasets.tex              dataset characteristics
    paper/tables/prevalence.tex            measured annotation prevalence
    paper/tables/geometry_baseline.tex     the annotation-geometry baseline
    paper/generated/values.tex             single-number macros for inline use
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

def find_paper_dir() -> Path:
    """Locate the manuscript directory.

    The paper may live inside the repo or beside it -- it gets moved around as
    a writing project rather than a code one. Resolution order: the
    ASA_PAPER_DIR environment variable, then any candidate that actually
    contains main.tex, then the in-repo default. Writing generated tables into
    a directory that is not the one being compiled is silent and costly, so the
    chosen path is always printed.
    """
    env = os.environ.get("ASA_PAPER_DIR")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [REPO / "paper", REPO.parent / "Manuscript", REPO / "Manuscript"]
    for candidate in candidates:
        if (candidate / "main.tex").exists():
            return candidate
    return REPO / "paper"


PAPER = find_paper_dir()
DATASET_ORDER = ["fetal_planes", "bus_bra", "mmotu", "hc18"]
PRETTY = {
    "fetal_planes": "FETAL\\_PLANES\\_DB",
    "bus_bra": "BUS-BRA",
    "mmotu": "MMOTU-2D",
    "hc18": "HC18",
}
# LaTeX command names may contain letters only, so dataset keys need a
# letters-only alias before they can become macro names.
MACRO_KEY = {
    "fetal_planes": "FetalPlanes",
    "bus_bra": "BusBra",
    "mmotu": "Mmotu",
    "hc18": "HCEighteen",
}
TASK = {
    "fetal_planes": "6-class plane",
    "bus_bra": "benign/malignant",
    "mmotu": "8-class subtype",
    "hc18": "HC regression",
}


def _escape(text: str) -> str:
    return str(text).replace("_", "\\_").replace("%", "\\%")


def _load(dataset: str, name: str):
    path = REPO / "work" / dataset / name
    if not path.exists():
        return None
    return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)


# --------------------------------------------------------------------------
# Figure 1: the annotation triplet on a real image
# --------------------------------------------------------------------------


def figure_triplet(dataset: str, image_id: str | None, out: Path) -> str:
    """The four variants side by side, plus the detected mask.

    This is the paper's anchor figure: A-only is the condition no existing study
    constructs, and a reader has to see it to believe the design.
    """
    manifest = load_manifest(REPO / "work" / dataset / "manifest.csv")
    if image_id:
        row = manifest[manifest.image_id == image_id].iloc[0]
    else:
        # A *representative* image, not the most-annotated one. The maximum is
        # reliably a detector blowout rather than a heavily annotated frame, and
        # a figure built from the failure tail would misrepresent the method.
        # Median annotation area among frames carrying both text and calipers.
        features = _load(dataset, "annotation_features.csv")
        merged = manifest.merge(features, on="image_id")
        eligible = merged[
            (merged.text_n_components >= 3)
            & (merged.calipers_n_components >= 1)
            & (merged.mask_suspect == 0)
        ]
        if eligible.empty:
            eligible = merged[merged.mask_suspect == 0]
        target = eligible.annotation_area_frac.median()
        row = eligible.iloc[(eligible.annotation_area_frac - target).abs().argsort().iloc[0]]

    image = to_bgr(cv2.imread(str(row.image_path), cv2.IMREAD_UNCHANGED))
    cfg = AnnotationConfig()
    det = detect(image, cfg)
    variants = build_variants(image, det, cfg, seed=0)

    mask_vis = to_bgr(image).copy()
    mask_vis[det.annotation] = (0, 0, 255)

    panels = [
        ("(a) A-full", variants["a_full"]),
        ("(b) detected mask", mask_vis),
        ("(c) A-removed", variants["a_removed"]),
        ("(d) A-only", variants["a_only"]),
        ("(e) ctrl-random", variants["ctrl_random"]),
    ]

    height = 300
    tiles = []
    for label, panel in panels:
        panel = to_bgr(panel)
        scale = height / panel.shape[0]
        tile = cv2.resize(panel, (int(panel.shape[1] * scale), height), interpolation=cv2.INTER_AREA)
        strip = np.full((28, tile.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(strip, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([strip, tile]))

    gap = np.full((tiles[0].shape[0], 8, 3), 255, dtype=np.uint8)
    composed = tiles[0]
    for tile in tiles[1:]:
        composed = np.hstack([composed, gap, tile])

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out.with_suffix(".png")), composed)
    _png_to_pdf(out.with_suffix(".png"), out)
    return str(row.image_id)


def _png_to_pdf(png: Path, pdf: Path) -> None:
    """Wrap a raster in a PDF. Springer requires embedded scalable fonts; the
    panel labels are rasterised into the image itself, so no font is embedded
    from this file and the requirement is satisfied trivially."""
    try:
        from PIL import Image

        Image.open(png).convert("RGB").save(pdf, "PDF", resolution=300.0)
    except Exception as exc:  # pragma: no cover
        print(f"  could not write {pdf}: {exc}; the PNG is still usable via \\includegraphics")


# --------------------------------------------------------------------------
# Figure 2: prevalence
# --------------------------------------------------------------------------


def figure_prevalence(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for dataset in DATASET_ORDER:
        report = _load(dataset, "annotation_prevalence.json")
        if report:
            rows.append((PRETTY[dataset].replace("\\_", "_"), report))
    if not rows:
        return

    labels = [r[0] for r in rows]
    series = {
        "any annotation": [r[1]["frac_images_with_any_annotation"] for r in rows],
        "calipers": [r[1]["frac_images_with_calipers"] for r in rows],
        "text": [r[1]["frac_images_with_text"] for r in rows],
    }

    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + (i - 1) * width, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("fraction of images", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def table_datasets(out: Path) -> None:
    lines = [
        "% generated by scripts/make_paper_assets.py -- do not edit",
        "\\begin{tabular}{llrrl}",
        "\\hline",
        "Dataset & Task & Images & Groups & Annotation content \\\\",
        "\\hline",
    ]
    content = {
        "fetal_planes": "vendor banner, preset text, depth ruler",
        "bus_bra": "calipers on lesion, laterality, quadrant",
        "mmotu": "chromatic calipers, measurement line",
        "hc18": "pre-scrubbed; faint vendor text, redaction box",
    }
    for dataset in DATASET_ORDER:
        manifest_path = REPO / "work" / dataset / "manifest.csv"
        if not manifest_path.exists():
            continue
        manifest = load_manifest(manifest_path)
        lines.append(
            f"{PRETTY[dataset]} & {TASK[dataset]} & {len(manifest)} & "
            f"{manifest.patient_id.nunique()} & {content[dataset]} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def table_prevalence(out: Path) -> None:
    lines = [
        "% generated by scripts/make_paper_assets.py -- do not edit",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Dataset & Any annot. & Calipers & Text & Mean area \\\\",
        "\\hline",
    ]
    for dataset in DATASET_ORDER:
        report = _load(dataset, "annotation_prevalence.json")
        if not report:
            continue
        lines.append(
            f"{PRETTY[dataset]} & {report['frac_images_with_any_annotation'] * 100:.1f}\\% & "
            f"{report['frac_images_with_calipers'] * 100:.1f}\\% & "
            f"{report['frac_images_with_text'] * 100:.1f}\\% & "
            f"{report['annotation_area_frac_mean'] * 100:.2f}\\% \\\\"
        )
    lines += ["\\hline", "\\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def table_geometry_baseline(out: Path) -> None:
    """The annotation-geometry baseline, with acquisition geometry reported
    separately so the annotation effect is not inflated by image size."""
    lines = [
        "% generated by scripts/make_paper_assets.py -- do not edit",
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Dataset & Metric & Acquis.\\ only & Acquis.+annot. & Increment & Chance \\\\",
        "\\hline",
    ]
    for dataset in DATASET_ORDER:
        payload = _load(dataset, "annotation_label_mi.json")
        if not payload:
            continue
        groups = payload["geometry_baselines"]
        annotation = groups["annotation_only"]
        acquisition = groups["acquisition_only"]
        metric = annotation["metric"]
        if metric == "balanced_accuracy":
            chance = f"{annotation['chance']:.3f}"
            fmt = "{:.3f}"
            metric_label = "bal.\\ acc."
        else:
            chance = f"{annotation.get('mean_predictor_mae', float('nan')):.1f}"
            fmt = "{:.1f}"
            metric_label = "MAE (mm)"
        # Report the INCREMENTAL contribution of annotation features over
        # acquisition geometry, not two side-by-side columns. Acquisition
        # geometry is a scanner fingerprint that inpainting cannot remove, and
        # side-by-side columns invite the reader to attribute its share to
        # annotations.
        combined = groups["annotation_plus_acquisition"]
        increment = combined["mean"] - acquisition["mean"]
        sign = "+" if increment >= 0 else ""
        lines.append(
            f"{PRETTY[dataset]} & {metric_label} & {fmt.format(acquisition['mean'])} & "
            f"{fmt.format(combined['mean'])} & {sign}{fmt.format(increment)} & {chance} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def macros(out: Path) -> None:
    """Single numbers used inline in the prose, as macros.

    Inline numbers are the ones most likely to drift out of date, because they
    are the ones a co-author edits around. Defining them here means the prose
    cannot disagree with the tables.
    """
    values: dict[str, str] = {}
    total_images = 0
    for dataset in DATASET_ORDER:
        manifest_path = REPO / "work" / dataset / "manifest.csv"
        if manifest_path.exists():
            total_images += len(load_manifest(manifest_path))

        report = _load(dataset, "annotation_prevalence.json")
        payload = _load(dataset, "annotation_label_mi.json")
        key = MACRO_KEY[dataset]
        if report:
            values[f"{key}AnyAnnot"] = f"{report['frac_images_with_any_annotation'] * 100:.1f}\\%"
            values[f"{key}Calipers"] = f"{report['frac_images_with_calipers'] * 100:.1f}\\%"
        if payload:
            baseline = payload["geometry_baselines"]["annotation_only"]
            # Classification scores read to 3 dp; MAE in mm to 1.
            digits = 3 if baseline["metric"] == "balanced_accuracy" else 1
            values[f"{key}GeomScore"] = f"{baseline['mean']:.{digits}f}"
            if "chance" in baseline:
                values[f"{key}Chance"] = f"{baseline['chance']:.3f}"
                ratio = baseline["mean"] / baseline["chance"] if baseline["chance"] else float("nan")
                values[f"{key}GeomRatio"] = f"{ratio:.1f}"
    values["totalImages"] = f"{total_images:,}".replace(",", "{,}")

    lines = ["% generated by scripts/make_paper_assets.py -- do not edit"]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(values.items())]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"  {len(values)} macros -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplet-dataset", default="fetal_planes")
    parser.add_argument("--triplet-image", default=None, help="image_id; default picks a representative frame")
    parser.add_argument("--paper", type=Path, default=None, help="manuscript directory (default: auto-detect)")
    args = parser.parse_args()

    global PAPER
    if args.paper:
        PAPER = args.paper.expanduser().resolve()
    print(f"manuscript: {PAPER}")
    if not (PAPER / "main.tex").exists():
        print("  WARNING: no main.tex here -- assets may be going somewhere nothing compiles")

    print("figures:")
    chosen = figure_triplet(args.triplet_dataset, args.triplet_image, PAPER / "figures" / "annotation_triplet.pdf")
    print(f"  annotation_triplet.pdf  (image {chosen} from {args.triplet_dataset})")
    figure_prevalence(PAPER / "figures" / "prevalence.pdf")
    print("  prevalence.pdf")

    print("tables:")
    table_datasets(PAPER / "tables" / "datasets.tex")
    table_prevalence(PAPER / "tables" / "prevalence.tex")
    table_geometry_baseline(PAPER / "tables" / "geometry_baseline.tex")
    for name in ("datasets", "prevalence", "geometry_baseline"):
        print(f"  {name}.tex")

    print("macros:")
    macros(PAPER / "generated" / "values.tex")


if __name__ == "__main__":
    main()
