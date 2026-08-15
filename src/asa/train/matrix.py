"""The train x test variant matrix -- the study's central experiment.

The matrix is not a leaderboard: its diagonal is conventional performance and
its off-diagonals carry the scientific content.

    (train a_full,    test a_full)      conventional reported performance
    (train a_full,    test a_removed)   THE DEPLOYMENT PENALTY            -> H2
    (train a_full,    test a_only)      THE SHORTCUT, isolated            -> H1
    (train a_full,    test ctrl_random) the pixel-removal control
    (train a_removed, test a_full)      robustness to unseen annotations
    (train a_removed, test a_removed)   the field's implicit best practice

One model is trained per (regime, fold, seed) and then evaluated on every
variant, so every cell of a row comes from the *same weights* on the *same
patients*. Nothing but the input variant changes across a row, which is what
licenses reading the differences causally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import REGIME_TRAIN_VARIANT, ExperimentConfig
from ..data.datasets import VariantDataset, make_loader
from ..data.splits import apply_label_budget, make_folds, summarise_split
from ..eval.metrics import PRIMARY_METRIC, evaluate
from ..eval.stats import Interval, bootstrap_difference, bootstrap_metric
from ..utils import get_logger, provenance, write_json

LOGGER = get_logger("asa.matrix")


def _metric_fn(task: str, metric: str):
    from ..eval.metrics import classification_metrics, regression_metrics

    if task == "regression":
        return lambda y, o: regression_metrics(y.astype(float), o.reshape(-1))[metric]
    return lambda y, o: classification_metrics(y.astype(int), o)[metric]


def run_matrix(
    cfg: ExperimentConfig,
    index: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    out_dir: str | Path,
    aux_targets: dict[str, np.ndarray] | None = None,
    aux_categorical: dict[str, int] | None = None,
    aux_numeric: list[str] | None = None,
    folds: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Train every regime x fold x seed, evaluate on every variant, and write
    one long-format results table.

    Results are flushed after every trained model and completed cells are
    skipped on re-entry. This is not a nicety: Kaggle kills a session at 12
    hours, and the full matrix (3 regimes x 5 folds x 5 seeds per backbone) does
    not fit in one. Re-running the same command in a fresh session continues
    where the last one stopped.
    """
    from .engine import get_device, predict, train_one

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    task = cfg.dataset.task
    primary = PRIMARY_METRIC[task]

    label_column = cfg.dataset.label_column if task == "classification" else "target"
    labels_all = index[label_column].to_numpy()
    array_index = index["array_index"].to_numpy()
    groups_all = index[cfg.dataset.group_column].to_numpy()

    seed_list = list(seeds) if seeds is not None else [cfg.split.seed0 + i for i in range(cfg.split.n_seeds)]
    results_path = out_dir / "matrix_results.csv"
    records: list[dict[str, Any]] = []
    completed: set[tuple[str, int, int]] = set()

    if resume and results_path.exists():
        previous = pd.read_csv(results_path)
        records = previous.to_dict("records")
        completed = {
            (str(r["regime"]), int(r["fold"]), int(r["seed"]))
            for r in records
            if pd.notna(r.get("backbone")) and str(r["backbone"]) == cfg.model.backbone
        }
        LOGGER.info("resuming: %d cells already done in %s", len(completed), results_path)

    split_log: list[dict[str, Any]] = []

    for seed in seed_list:
        for split in make_folds(index, cfg.dataset, cfg.split, seed):
            if folds is not None and split.fold not in folds:
                continue
            split_log.append(summarise_split(index, split, cfg.dataset))

            for regime in cfg.regimes:
                if (regime, split.fold, seed) in completed:
                    LOGGER.info("skip (done): regime=%s fold=%d seed=%d", regime, split.fold, seed)
                    continue
                train_variant = REGIME_TRAIN_VARIANT[regime]
                train_idx = apply_label_budget(index, split.train_idx, cfg.dataset, cfg.train.label_budget, seed)

                LOGGER.info(
                    "seed=%d fold=%d regime=%s backbone=%s train_variant=%s n_train=%d n_test=%d",
                    seed, split.fold, regime, cfg.model.backbone, train_variant, len(train_idx), len(split.test_idx),
                )

                use_aux = cfg.aux.enabled and regime == "r_supervised"
                make = lambda idx, variant, train: VariantDataset(  # noqa: E731
                    arrays=arrays,
                    indices=array_index[idx],
                    labels=labels_all[idx],
                    variant=variant,
                    train=train,
                    image_size=cfg.model.image_size,
                    aux_targets=({k: v[idx] for k, v in (aux_targets or {}).items()} if use_aux else None),
                    annotation_randomising_aug=cfg.train.annotation_randomising_aug,
                    annotation_aug_prob=cfg.train.annotation_aug_prob,
                    synthetic=cfg.synthetic,
                    num_classes=cfg.dataset.num_classes,
                    seed=seed + split.fold,
                )

                train_loader = make_loader(make(train_idx, train_variant, True), cfg, shuffle=True, seed=seed)
                val_loader = make_loader(make(split.val_idx, train_variant, False), cfg, shuffle=False, seed=seed)

                result = train_one(
                    cfg, train_loader, val_loader, cfg.dataset.num_classes, seed,
                    aux_categorical if use_aux else None,
                    aux_numeric if use_aux else None,
                    device=device,
                )

                # One model, evaluated on every variant of the same held-out
                # patients. This is the row of the matrix.
                per_variant: dict[str, dict[str, Any]] = {}
                raw_outputs: dict[str, np.ndarray] = {}
                test_targets: np.ndarray | None = None
                test_groups: np.ndarray | None = None

                for test_variant in cfg.eval.test_variants:
                    from ..data.datasets import NOMARKER_SUFFIX

                    base = (
                        test_variant[: -len(NOMARKER_SUFFIX)]
                        if test_variant.endswith(NOMARKER_SUFFIX)
                        else test_variant
                    )
                    if base not in arrays:
                        continue
                    loader = make_loader(make(split.test_idx, test_variant, False), cfg, shuffle=False, seed=seed)
                    predicted = predict(result.model, loader, cfg, device)
                    metrics = evaluate(predicted["targets"], predicted["outputs"], task, cfg.eval.calibration_bins)
                    per_variant[test_variant] = metrics
                    raw_outputs[test_variant] = predicted["outputs"]
                    test_targets = predicted["targets"]
                    row_to_group = dict(zip(array_index, groups_all))
                    test_groups = np.array([row_to_group[r] for r in predicted["rows"]])

                # Per-image predictions, for the qualitative analysis the
                # qualitative analysis needs ("manual review of a sample of
                # cases correct under A-full and wrong under A-removed"). Not
                # recoverable from the aggregate table afterwards, and cheap:
                # a few hundred KB per run.
                if cfg.eval.save_predictions:
                    pred_dir = out_dir / "predictions"
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        pred_dir / f"{regime}_fold{split.fold}_seed{seed}.npz",
                        targets=test_targets,
                        rows=predicted["rows"],
                        groups=test_groups.astype(str),
                        **{f"probs__{v}": o for v, o in raw_outputs.items()},
                    )

                metric_fn = _metric_fn(task, primary)
                for test_variant, metrics in per_variant.items():
                    interval = bootstrap_metric(
                        test_targets, raw_outputs[test_variant], test_groups, metric_fn,
                        cfg.eval.bootstrap_replicates, seed=cfg.eval.bootstrap_seed,
                    )
                    record: dict[str, Any] = {
                        "dataset": cfg.dataset.name,
                        "experiment": cfg.name,
                        "regime": regime,
                        "train_variant": train_variant,
                        "test_variant": test_variant,
                        "backbone": cfg.model.backbone,
                        "pretrained": cfg.model.pretrained,
                        "label_budget": cfg.train.label_budget,
                        "annotation_aug": cfg.train.annotation_randomising_aug,
                        "synthetic_strength": cfg.synthetic.strength if cfg.synthetic.enabled else None,
                        "synthetic_cue": cfg.synthetic.cue if cfg.synthetic.enabled else None,
                        "synthetic_area_frac": cfg.synthetic.area_frac if cfg.synthetic.enabled else None,
                        "fold": split.fold,
                        "seed": seed,
                        "n_train": len(train_idx),
                        "n_test": len(split.test_idx),
                        "best_epoch": result.best_epoch,
                        "val_score": result.best_val_score,
                        "train_seconds": result.train_seconds,
                        "n_parameters": result.n_parameters,
                        f"{primary}_ci_low": interval.low,
                        f"{primary}_ci_high": interval.high,
                    }
                    record.update({k: v for k, v in metrics.items() if k != "reliability_curve"})
                    records.append(record)

                # Paired contrasts against the training variant: same patients,
                # same weights, so the interval is paired and the between-image
                # variance cancels.
                reference = raw_outputs.get(train_variant)
                if reference is not None:
                    for test_variant, outputs in raw_outputs.items():
                        if test_variant == train_variant:
                            continue
                        delta = bootstrap_difference(
                            test_targets, reference, outputs, test_groups, metric_fn,
                            cfg.eval.bootstrap_replicates, seed=cfg.eval.bootstrap_seed,
                        )
                        records.append(
                            {
                                "dataset": cfg.dataset.name,
                                "experiment": cfg.name,
                                "regime": regime,
                                "train_variant": train_variant,
                                "test_variant": f"delta:{train_variant}-{test_variant}",
                                "backbone": cfg.model.backbone,
                                "pretrained": cfg.model.pretrained,
                                "label_budget": cfg.train.label_budget,
                                "fold": split.fold,
                                "seed": seed,
                                primary: delta.estimate,
                                f"{primary}_ci_low": delta.low,
                                f"{primary}_ci_high": delta.high,
                                "is_contrast": True,
                            }
                        )

                del result

                # Flush after every trained model. A session killed at the
                # 12-hour cap must cost one cell, not the whole run.
                pd.DataFrame.from_records(records).to_csv(results_path, index=False)

    frame = pd.DataFrame.from_records(records)
    frame.to_csv(results_path, index=False)
    write_json(out_dir / "splits.json", split_log)
    write_json(out_dir / "config.json", cfg.to_dict())
    write_json(out_dir / "provenance.json", provenance({"stage": "C", "n_records": len(frame)}))
    LOGGER.info("wrote %s (%d rows)", out_dir / "matrix_results.csv", len(frame))
    return frame


def summarise_matrix(results: pd.DataFrame, metric: str = "balanced_accuracy") -> pd.DataFrame:
    """Mean +- SD over folds and seeds, one row per matrix cell."""
    cells = results[~results.get("is_contrast", pd.Series(False, index=results.index)).fillna(False).astype(bool)]
    grouped = cells.groupby(["dataset", "backbone", "regime", "train_variant", "test_variant"])[metric]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    return summary.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_sd", "count": "n_runs"})
