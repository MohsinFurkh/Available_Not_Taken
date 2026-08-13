"""Annotation-label mutual information (H4).

H4 is what turns a measurement into a rule. If shortcut magnitude is predictable
from annotation prevalence and annotation-label MI, the finding generalises even
where the magnitudes differ; if it is not, the paper's claim reduces to "measure
this yourself" plus a released protocol. Either way this quantity is computed
*before* any model is trained, so it is a prediction rather than a
post-hoc rationalisation, and Section 15 R3 uses it to answer "what is
modality-specific here?" -- operator-placed calipers should carry systematically
more label information than the incidental confounds of the radiography
shortcut literature.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#: Descriptors of the burnt-in annotations themselves. Nothing here can be
#: computed from anatomy, which is what makes the analysis interpretable.
ANNOTATION_FEATURES = [
    "annotation_area_frac",
    "text_area_frac",
    "text_n_components",
    "text_cx",
    "text_cy",
    "text_spread_x",
    "text_spread_y",
    "calipers_area_frac",
    "calipers_n_components",
    "calipers_cx",
    "calipers_cy",
    "caliper_max_sep",
    "caliper_mean_sep",
    "ticks_n_components",
    "overlay_area_frac",
    "overlay_cx",
    "overlay_cy",
    "redaction_area_frac",
    "chrome_area_frac",
    "chrome_mean_intensity",
]

#: Acquisition-geometry descriptors. These are NOT annotations -- image
#: dimensions and sector shape are a scanner/preset fingerprint, a separate
#: confound with its own literature. They are measured because they are real,
#: and kept in their own group because folding them into the annotation
#: baseline would overstate the annotation shortcut, which is exactly the kind
#: of inflation this study exists to measure rather than to commit.
ACQUISITION_FEATURES = [
    "image_h",
    "image_w",
    "aspect",
    "fan_area_frac",
]

DEFAULT_FEATURES = ANNOTATION_FEATURES


def available_features(table: pd.DataFrame, requested: Sequence[str] | None = None) -> list[str]:
    requested = list(requested or DEFAULT_FEATURES)
    return [f for f in requested if f in table.columns and table[f].notna().any()]


def annotation_label_mi(
    table: pd.DataFrame,
    label_column: str = "label",
    task: str = "classification",
    features: Sequence[str] | None = None,
    seed: int = 20260805,
) -> dict[str, Any]:
    """Per-feature and total mutual information between annotations and label."""
    columns = available_features(table, features)
    X = table[columns].fillna(0.0).to_numpy(dtype=float)
    y = table[label_column].to_numpy()

    if task == "classification":
        per_feature = mutual_info_classif(X, y.astype(int), random_state=seed)
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        label_entropy = float(-(probabilities * np.log(probabilities)).sum())
    else:
        per_feature = mutual_info_regression(X, y.astype(float), random_state=seed)
        label_entropy = float("nan")

    ranked = sorted(zip(columns, per_feature.tolist()), key=lambda kv: -kv[1])
    return {
        "features": columns,
        "mi_per_feature_nats": dict(ranked),
        "mi_max_nats": float(per_feature.max()) if len(per_feature) else 0.0,
        "mi_sum_nats": float(per_feature.sum()),
        "label_entropy_nats": label_entropy,
        # The interpretable headline: what fraction of the label's uncertainty is
        # explained by the single most informative annotation descriptor.
        "mi_max_over_entropy": float(per_feature.max() / label_entropy)
        if len(per_feature) and np.isfinite(label_entropy) and label_entropy > 0
        else float("nan"),
    }


def annotation_only_predictability(
    table: pd.DataFrame,
    label_column: str = "label",
    group_column: str = "patient_id",
    task: str = "classification",
    features: Sequence[str] | None = None,
    n_splits: int = 5,
    seed: int = 20260805,
) -> dict[str, Any]:
    """Section 10's hand-crafted annotation-geometry baseline.

    "If a five-feature logistic regression on annotation geometry matches the
    CNN's A-only performance, the shortcut is fully explained" -- and it is
    explained without any deep learning at all, which is the cheapest possible
    answer to reviewer attack R1 ("this is data cleaning dressed up as
    science").

    Grouped cross-validation, because these features are near-constant within a
    patient and an ungrouped split would report memorisation as predictability.
    """
    columns = available_features(table, features)
    X = table[columns].fillna(0.0).to_numpy(dtype=float)
    y = table[label_column].to_numpy()
    groups = table[group_column].to_numpy()

    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    scores: list[float] = []
    coefficients: list[np.ndarray] = []

    for train_idx, test_idx in splitter.split(X, y, groups):
        if task == "classification":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            )
            model.fit(X[train_idx], y[train_idx].astype(int))
            scores.append(float(balanced_accuracy_score(y[test_idx].astype(int), model.predict(X[test_idx]))))
            coefficients.append(np.abs(model[-1].coef_).mean(axis=0))
        else:
            model = make_pipeline(StandardScaler(), Ridge(random_state=seed))
            model.fit(X[train_idx], y[train_idx].astype(float))
            scores.append(float(mean_absolute_error(y[test_idx].astype(float), model.predict(X[test_idx]))))
            coefficients.append(np.abs(model[-1].coef_))

    scores_arr = np.array(scores)
    importance = dict(
        sorted(zip(columns, np.mean(coefficients, axis=0).tolist()), key=lambda kv: -kv[1])
    )
    out: dict[str, Any] = {
        "task": task,
        "features": columns,
        "n_features": len(columns),
        "fold_scores": scores,
        "mean": float(scores_arr.mean()),
        "sd": float(scores_arr.std(ddof=1)) if len(scores_arr) > 1 else float("nan"),
        "coefficient_importance": importance,
    }
    if task == "classification":
        out["chance"] = float(1.0 / len(np.unique(y)))
        out["above_chance"] = out["mean"] - out["chance"]
        out["metric"] = "balanced_accuracy"
    else:
        out["mean_predictor_mae"] = float(np.mean(np.abs(y.astype(float) - y.astype(float).mean())))
        out["metric"] = "mae"
    return out


def feature_group_report(
    table: pd.DataFrame,
    label_column: str = "label",
    group_column: str = "patient_id",
    task: str = "classification",
    seed: int = 20260805,
) -> dict[str, Any]:
    """The geometry baseline run on three feature groups, reported separately.

    Reporting only the combined number would let acquisition geometry (image
    size, sector shape) masquerade as an annotation effect. The three-way split
    is what lets the manuscript say how much of the annotation-free
    predictability is actually annotation-driven.
    """
    groups = {
        "annotation_only": ANNOTATION_FEATURES,
        "acquisition_only": ACQUISITION_FEATURES,
        "annotation_plus_acquisition": ANNOTATION_FEATURES + ACQUISITION_FEATURES,
    }
    return {
        name: annotation_only_predictability(
            table, label_column, group_column, task, features=features, seed=seed
        )
        for name, features in groups.items()
    }


def prevalence_vs_shortcut(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """H4's actual test: does dataset-level annotation statistics predict the
    per-dataset shortcut magnitude?

    Each record needs ``dataset``, ``annotation_prevalence``, ``mi_max_nats`` and
    ``shortcut_magnitude``. With four datasets this is a four-point regression --
    underpowered, and reported as descriptive rather than inferential, which is
    exactly the honesty the proposal's Section 15 R10 response commits to.
    """
    if len(records) < 3:
        return {"n_datasets": len(records), "note": "too few datasets for a correlation; report descriptively"}

    frame = pd.DataFrame(list(records))
    out: dict[str, Any] = {"n_datasets": len(frame)}
    from scipy import stats as sps

    for predictor in ("annotation_prevalence", "mi_max_nats"):
        if predictor not in frame:
            continue
        rho, p_value = sps.spearmanr(frame[predictor], frame["shortcut_magnitude"])
        out[predictor] = {
            "spearman_rho": float(rho),
            "p_value": float(p_value),
            "note": "descriptive only: n is the number of datasets, not the number of images",
        }
    return out
