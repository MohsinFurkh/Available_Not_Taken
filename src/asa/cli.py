"""Command-line entry points.

    python -m asa manifest      --dataset fetal_planes
    python -m asa masks         --dataset fetal_planes
    python -m asa verify-export --dataset fetal_planes
    python -m asa verify-score  --dataset fetal_planes
    python -m asa mi            --dataset fetal_planes
    python -m asa package       --dataset fetal_planes --size 224
    python -m asa gate          --datasets fetal_planes bus_bra
    python -m asa matrix        --experiment full_matrix --dataset fetal_planes

Stages A and B (manifest / masks / package / mi) run locally on CPU; the gate
and matrix stages run wherever the GPU is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from .config import AnnotationConfig, DatasetConfig, load_experiment
from .utils import get_logger, write_json

LOGGER = get_logger("asa.cli")

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"
WORK_ROOT = REPO_ROOT / "work"


def _dataset_config(name: str) -> DatasetConfig:
    payload = yaml.safe_load((CONFIG_ROOT / "datasets" / f"{name}.yaml").read_text())
    payload.pop("notes", None)
    return DatasetConfig(**payload, notes="")


def _work(dataset: str) -> Path:
    path = WORK_ROOT / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------


def cmd_manifest(args: argparse.Namespace) -> None:
    from .data.manifest import build_manifest

    dataset = _dataset_config(args.dataset)
    manifest = build_manifest(args.dataset, dataset.root)
    out = _work(args.dataset) / "manifest.csv"
    manifest.to_csv(out, index=False)
    LOGGER.info("wrote %s (%d rows, %d groups)", out, len(manifest), manifest.patient_id.nunique())


def cmd_masks(args: argparse.Namespace) -> None:
    from .annot.pipeline import prevalence_report, run
    from .data.manifest import load_manifest

    work = _work(args.dataset)
    manifest = load_manifest(work / "manifest.csv")
    features = run(
        manifest,
        out_dir=work,
        cfg=AnnotationConfig(),
        use_ocr=not args.no_ocr,
        workers=args.workers,
        limit=args.limit,
    )
    report = prevalence_report(features)
    write_json(work / "annotation_prevalence.json", report)
    LOGGER.info("annotation prevalence: %s", report)


def cmd_verify_export(args: argparse.Namespace) -> None:
    from .annot.pipeline import export_verification_sample
    from .data.manifest import load_manifest

    work = _work(args.dataset)
    manifest = load_manifest(work / "manifest.csv")
    export_verification_sample(manifest, work / "masks", work / "verification", fraction=args.fraction)


def cmd_verify_score(args: argparse.Namespace) -> None:
    from .annot.pipeline import score_verification

    work = _work(args.dataset)
    result = score_verification(work / "verification" / "verification_sheet.csv")
    write_json(work / "verification" / "verification_scores.json", result)
    LOGGER.info("verification: %s", result)

    dataset = _dataset_config(args.dataset)
    sensitivities = [v["image_level_sensitivity"] for v in result.get("per_verifier", {}).values()]
    if sensitivities and min(sensitivities) < dataset.mask_sensitivity_floor:
        LOGGER.error(
            "GATE FAILED: mask sensitivity %.3f is below the floor %.3f for %s. Per Step 3 this "
            "dataset must be excluded -- every downstream conclusion depends on mask quality.",
            min(sensitivities), dataset.mask_sensitivity_floor, args.dataset,
        )
        sys.exit(2)


def cmd_mi(args: argparse.Namespace) -> None:
    from .data.manifest import load_manifest
    from .eval.mutual_info import annotation_label_mi, feature_group_report

    work = _work(args.dataset)
    dataset = _dataset_config(args.dataset)
    manifest = load_manifest(work / "manifest.csv")
    features = pd.read_csv(work / "annotation_features.csv", dtype={"image_id": str})
    table = manifest.merge(features, on="image_id", how="inner")
    if len(table) != len(manifest):
        LOGGER.warning("only %d of %d manifest rows have annotation features", len(table), len(manifest))

    label_column = dataset.label_column if dataset.task == "classification" else "target"
    baselines = feature_group_report(table, label_column, dataset.group_column, dataset.task)
    result = {
        "mutual_information": annotation_label_mi(table, label_column, dataset.task),
        "geometry_baselines": baselines,
    }
    write_json(work / "annotation_label_mi.json", result)
    for name, baseline in baselines.items():
        LOGGER.info(
            "%-28s %s = %.4f (chance %.4f, above chance %+.4f) top: %s",
            name, baseline["metric"], baseline["mean"],
            baseline.get("chance", float("nan")), baseline.get("above_chance", float("nan")),
            list(baseline["coefficient_importance"])[:3],
        )


def cmd_package(args: argparse.Namespace) -> None:
    from .data.manifest import load_manifest
    from .data.packaging import package

    work = _work(args.dataset)
    manifest = load_manifest(work / "manifest.csv")
    features_path = work / "annotation_features.csv"
    features = pd.read_csv(features_path, dtype={"image_id": str}) if features_path.exists() else None
    dataset = _dataset_config(args.dataset)
    grayscale = dataset.grayscale if args.channels == "auto" else args.channels == "gray"
    if not grayscale and dataset.grayscale:
        LOGGER.warning("packing %s as RGB although its config says grayscale; shards will be 3x larger", args.dataset)
    package(
        manifest,
        mask_dir=work / "masks",
        out_dir=work / f"pack_{args.size}",
        size=args.size,
        grayscale=grayscale,
        features=features,
        workers=args.workers,
    )


def _warn_if_stale(dataset: str, size: int) -> None:
    features = _work(dataset) / "annotation_features.csv"
    meta = _work(dataset) / f"pack_{size}" / "meta.json"
    if features.exists() and meta.exists() and meta.stat().st_mtime < features.stat().st_mtime:
        LOGGER.warning(
            "%s: pack_%d is OLDER than the annotation features -- the shards were built from "
            "masks that have since changed. Re-run: python -m asa package --dataset %s --size %d",
            dataset, size, dataset, size,
        )


def cmd_gate(args: argparse.Namespace) -> None:
    from .data.packaging import load_package
    from .train.gate import decide, evaluate_gate, write_gate_report
    from .train.matrix import run_matrix

    verdicts = []
    for name in args.datasets:
        _warn_if_stale(name, args.size)
        cfg = load_experiment(CONFIG_ROOT / "experiments" / "step4_gate.yaml", CONFIG_ROOT,
                             overrides={"dataset": name})
        index, arrays, _ = load_package(_work(name) / f"pack_{args.size}")
        results = run_matrix(
            cfg, index, arrays,
            out_dir=REPO_ROOT / cfg.output_dir / name,
            folds=list(range(args.folds)),
        )
        verdicts.append(evaluate_gate(results, name, cfg.dataset.task))

    decision = decide(verdicts, min_datasets_passing=args.min_datasets_passing)
    write_gate_report(decision, REPO_ROOT / "runs" / "step4_gate")
    LOGGER.info("GATE: %s", decision["recommendation"])
    if not decision["proceed"]:
        sys.exit(3)


def cmd_matrix(args: argparse.Namespace) -> None:
    from .data.packaging import load_package
    from .train.matrix import run_matrix, summarise_matrix

    overrides: dict = {"dataset": args.dataset}
    if args.backbone:
        overrides["model"] = {"backbone": args.backbone}
    if args.label_budget is not None:
        overrides["train"] = {"label_budget": args.label_budget}
    if args.synthetic_strength is not None:
        overrides["synthetic"] = {"enabled": True, "strength": args.synthetic_strength}
        if args.synthetic_cue:
            overrides["synthetic"]["cue"] = args.synthetic_cue
    cfg = load_experiment(CONFIG_ROOT / "experiments" / f"{args.experiment}.yaml", CONFIG_ROOT, overrides)

    _warn_if_stale(args.dataset, args.size)
    index, arrays, _ = load_package(_work(args.dataset) / f"pack_{args.size}")
    suffix = f"{cfg.model.backbone}_lb{cfg.train.label_budget}"
    if cfg.synthetic.enabled:
        suffix += f"_{cfg.synthetic.cue}{cfg.synthetic.strength}"
    out_dir = REPO_ROOT / cfg.output_dir / args.dataset / suffix
    results = run_matrix(cfg, index, arrays, out_dir=out_dir)
    summary = summarise_matrix(results)
    summary.to_csv(out_dir / "matrix_summary.csv", index=False)
    print(summary.to_string(index=False))


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("asa", description="Annotation Shortcut Audit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", help="build the tidy manifest for one dataset")
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("masks", help="detect annotations and parse content (Stage A)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="process only the first N images (smoke test)")
    p.add_argument("--no-ocr", action="store_true", help="skip Tesseract; masks are unaffected")
    p.set_defaults(func=cmd_masks)

    p = sub.add_parser("verify-export", help="write the human mask-verification sample")
    p.add_argument("--dataset", required=True)
    p.add_argument("--fraction", type=float, default=0.05)
    p.set_defaults(func=cmd_verify_export)

    p = sub.add_parser("verify-score", help="score the returned verification sheet; enforces the Step 3 gate")
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=cmd_verify_score)

    p = sub.add_parser("mi", help="annotation-label mutual information + geometry baseline (H4)")
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=cmd_mi)

    p = sub.add_parser("package", help="build variant shards for training (Stage B)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--size", type=int, default=224)
    p.add_argument(
        "--channels", choices=["auto", "gray", "rgb"], default="auto",
        help="auto follows the dataset config; MMOTU must stay rgb because its calipers are chromatic",
    )
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(func=cmd_package)

    p = sub.add_parser("gate", help="Step 4 minimal decisive experiment (go/no-go)")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--folds", type=int, default=3, help="how many folds to run at the gate")
    p.add_argument("--min-datasets-passing", type=int, default=2)
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("matrix", help="full train x test variant matrix (Step 6)")
    p.add_argument("--experiment", default="full_matrix")
    p.add_argument("--dataset", required=True)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--backbone", default=None)
    p.add_argument("--label-budget", type=float, default=None)
    p.add_argument("--synthetic-strength", type=float, default=None,
                   help="positive control: P(injected marker encodes the true label)")
    p.add_argument("--synthetic-cue", choices=["rectangle", "glyph"], default=None,
                   help="rectangle = easy block cue (upper bound); glyph = learnability-matched")
    p.set_defaults(func=cmd_matrix)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
