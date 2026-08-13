"""Patient-level splitting.

Proposal Section 9: "No image, augmented copy, or video frame from a patient may
appear in more than one partition." Every split function here operates on the
group column and never on the row index, and the assertions are load-bearing --
a grouping violation would invalidate every number in the study, and the
stage-1 review found exactly this failure mode in the folder corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

from ..config import DatasetConfig, SplitConfig


@dataclass
class Split:
    fold: int
    seed: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train_idx), "val": len(self.val_idx), "test": len(self.test_idx)}


def assert_disjoint_groups(manifest: pd.DataFrame, split: Split, group_column: str) -> None:
    groups = {
        name: set(manifest.iloc[idx][group_column])
        for name, idx in (("train", split.train_idx), ("val", split.val_idx), ("test", split.test_idx))
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[a] & groups[b]
        if overlap:
            raise AssertionError(
                f"patient-level leakage between {a} and {b}: {len(overlap)} shared groups, "
                f"e.g. {sorted(overlap)[:5]}"
            )


def _stratification_target(manifest: pd.DataFrame, dataset: DatasetConfig) -> np.ndarray | None:
    if dataset.task == "classification":
        return manifest[dataset.label_column].to_numpy()
    # Stratify regression on target quantiles, so that folds are comparable in
    # gestational range rather than merely in size.
    target = pd.to_numeric(manifest["target"], errors="coerce")
    return pd.qcut(target, q=5, labels=False, duplicates="drop").to_numpy()


def make_folds(
    manifest: pd.DataFrame,
    dataset: DatasetConfig,
    split_cfg: SplitConfig,
    seed: int,
) -> Iterator[Split]:
    """Yield one Split per fold.

    Published folds are used unchanged where the dataset provides them
    (BUS-BRA's K5P), so results stay comparable to the literature; otherwise
    stratified group k-fold on the patient column.
    """
    groups = manifest[dataset.group_column].to_numpy()
    n = len(manifest)
    rng = np.random.default_rng(seed)

    if (
        split_cfg.use_official_folds
        and dataset.official_fold_column
        and dataset.official_fold_column in manifest.columns
        and (manifest[dataset.official_fold_column] >= 0).all()
    ):
        fold_ids = manifest[dataset.official_fold_column].to_numpy()
        unique = sorted(set(fold_ids.tolist()))
        for fold, held in enumerate(unique):
            test_idx = np.flatnonzero(fold_ids == held)
            rest = np.flatnonzero(fold_ids != held)
            train_idx, val_idx = _carve_validation(manifest, rest, dataset, split_cfg, rng)
            split = Split(fold=fold, seed=seed, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
            assert_disjoint_groups(manifest, split, dataset.group_column)
            yield split
        return

    strat = _stratification_target(manifest, dataset)
    if split_cfg.stratify and strat is not None and pd.notna(strat).all():
        splitter = StratifiedGroupKFold(n_splits=split_cfg.n_folds, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(n), strat, groups)
    else:
        # GroupKFold has no random_state; shuffle the group->fold assignment
        # ourselves so that seeds actually change the partition.
        order = rng.permutation(n)
        splitter = GroupKFold(n_splits=split_cfg.n_folds)
        iterator = (
            (order[tr], order[te]) for tr, te in splitter.split(np.zeros(n), None, groups[order])
        )

    for fold, (rest, test_idx) in enumerate(iterator):
        train_idx, val_idx = _carve_validation(manifest, np.asarray(rest), dataset, split_cfg, rng)
        split = Split(fold=fold, seed=seed, train_idx=train_idx, val_idx=np.asarray(val_idx), test_idx=np.asarray(test_idx))
        assert_disjoint_groups(manifest, split, dataset.group_column)
        yield split


def _carve_validation(
    manifest: pd.DataFrame,
    rest: np.ndarray,
    dataset: DatasetConfig,
    split_cfg: SplitConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the non-test portion into train and validation, by group.

    The validation set is carved out of the *training* portion and never out of
    the test portion. P3 in the folder corpus drew its validation set from its
    test set; the proposal names that as a practice this study explicitly avoids
    (Section 9, "Model selection").
    """
    rest_groups = manifest.iloc[rest][dataset.group_column].to_numpy()
    unique_groups = np.unique(rest_groups)
    rng.shuffle(unique_groups)
    n_val = max(1, int(round(split_cfg.val_fraction * len(unique_groups))))
    val_groups = set(unique_groups[:n_val].tolist())
    is_val = np.array([g in val_groups for g in rest_groups])
    return rest[~is_val], rest[is_val]


def apply_label_budget(
    manifest: pd.DataFrame,
    train_idx: np.ndarray,
    dataset: DatasetConfig,
    budget: float,
    seed: int,
) -> np.ndarray:
    """Subsample the *manual labels* available for training (SO2).

    Sampling is by group, not by image, so that a 10% budget means 10% of
    patients rather than 10% of frames -- otherwise a budget curve on a
    multi-frame dataset measures frame redundancy rather than label efficiency.
    Class balance is preserved where the task is classification.
    """
    if budget >= 1.0:
        return train_idx
    rng = np.random.default_rng(seed)
    sub = manifest.iloc[train_idx]
    groups = sub[dataset.group_column].to_numpy()

    if dataset.task == "classification":
        # dominant label per group, so stratification is well defined
        group_label = sub.groupby(dataset.group_column)[dataset.label_column].agg(
            lambda s: s.value_counts().idxmax()
        )
        keep: list[str] = []
        for label, members in group_label.groupby(group_label):
            names = members.index.to_numpy()
            rng.shuffle(names)
            n_keep = max(1, int(round(budget * len(names))))
            keep.extend(names[:n_keep].tolist())
    else:
        names = np.unique(groups)
        rng.shuffle(names)
        keep = names[: max(1, int(round(budget * len(names))))].tolist()

    keep_set = set(keep)
    mask = np.array([g in keep_set for g in groups])
    return train_idx[mask]


def summarise_split(manifest: pd.DataFrame, split: Split, dataset: DatasetConfig) -> dict[str, object]:
    out: dict[str, object] = {"fold": split.fold, "seed": split.seed, **split.sizes()}
    for name, idx in (("train", split.train_idx), ("val", split.val_idx), ("test", split.test_idx)):
        part = manifest.iloc[idx]
        out[f"{name}_groups"] = int(part[dataset.group_column].nunique())
        if dataset.task == "classification":
            counts = part[dataset.label_column].value_counts().sort_index()
            out[f"{name}_class_counts"] = counts.to_dict()
    return out
