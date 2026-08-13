"""Task metrics, calibration, and the chance rate.

The chance rate is not a formality here. H1's decisive claim is "above-chance
performance under A-only", so the reference against which A-only is judged has
to be computed from the *test fold's* class distribution, not assumed to be
1/K -- with FETAL_PLANES_DB's 34% 'Other' class, 1/K would understate chance by
a factor of two and manufacture a shortcut that is not there.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def chance_rate(y_true: np.ndarray, metric: str = "balanced_accuracy") -> float:
    """Empirical chance level for the metric in use.

    ``balanced_accuracy`` and ``macro_f1`` have distribution-free chance levels
    for a random-guessing classifier (1/K and roughly 1/K), while ``accuracy``
    does not -- its chance level is the majority-class prevalence.
    """
    classes, counts = np.unique(y_true, return_counts=True)
    k = len(classes)
    if metric == "accuracy":
        return float(counts.max() / counts.sum())
    if metric in ("balanced_accuracy", "macro_f1"):
        return float(1.0 / k)
    if metric == "auroc":
        return 0.5
    raise ValueError(f"no chance rate defined for metric {metric!r}")


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_pred = probabilities.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    n_classes = probabilities.shape[1]
    present = np.unique(y_true)
    try:
        if n_classes == 2:
            out["auroc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
        elif len(present) == n_classes:
            out["auroc"] = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
        else:
            # A fold missing a class cannot support a macro one-vs-rest AUROC;
            # report NaN rather than a silently restricted average.
            out["auroc"] = float("nan")
    except ValueError:
        out["auroc"] = float("nan")

    out["chance_balanced_accuracy"] = chance_rate(y_true, "balanced_accuracy")
    out["chance_accuracy"] = chance_rate(y_true, "accuracy")
    out["above_chance_balanced_accuracy"] = out["balanced_accuracy"] - out["chance_balanced_accuracy"]
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    baseline = np.full_like(y_true, float(np.mean(y_true)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "bias": float(residual.mean()),
        "r2": float(1.0 - np.sum(residual**2) / max(np.sum((y_true - y_true.mean()) ** 2), 1e-12)),
        # The "predict the training mean" reference. P2's USFM transfer scored
        # R2 = -10.5 on a fetal task, i.e. far worse than this; reporting it
        # explicitly keeps that failure mode visible rather than hidden in R2.
        "mae_mean_predictor": float(mean_absolute_error(y_true, baseline)),
    }


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15
) -> tuple[float, dict[str, list[float]]]:
    """ECE with equal-width confidence bins, plus the reliability curve.

    H2 is a calibration hypothesis: the clinically dangerous case is not that
    accuracy falls when annotations disappear, but that confidence does not fall
    with it. So confidence and accuracy are always reported as a pair.
    """
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    curve: dict[str, list[float]] = {"bin_confidence": [], "bin_accuracy": [], "bin_count": []}
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_confidence = float(confidence[in_bin].mean())
        bin_accuracy = float(correct[in_bin].mean())
        ece += (count / len(y_true)) * abs(bin_accuracy - bin_confidence)
        curve["bin_confidence"].append(bin_confidence)
        curve["bin_accuracy"].append(bin_accuracy)
        curve["bin_count"].append(count)
    return float(ece), curve


def calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> dict[str, Any]:
    ece, curve = expected_calibration_error(y_true, probabilities, n_bins)
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(float)
    eps = 1e-12
    return {
        "ece": ece,
        "mean_confidence": float(confidence.mean()),
        "mean_accuracy": float(correct.mean()),
        # Positive = overconfident. This single number is H2 in miniature.
        "confidence_minus_accuracy": float(confidence.mean() - correct.mean()),
        "nll": float(-np.log(probabilities[np.arange(len(y_true)), y_true] + eps).mean()),
        "brier": float(np.mean((probabilities - np.eye(probabilities.shape[1])[y_true]) ** 2).item()),
        "reliability_curve": curve,
    }


def evaluate(
    y_true: np.ndarray,
    outputs: np.ndarray,
    task: str = "classification",
    n_bins: int = 15,
) -> dict[str, Any]:
    if task == "regression":
        return regression_metrics(y_true.astype(float), outputs.reshape(-1).astype(float))
    probabilities = outputs if np.allclose(outputs.sum(axis=1), 1.0, atol=1e-3) else softmax(outputs)
    metrics = classification_metrics(y_true.astype(int), probabilities)
    metrics.update(calibration_metrics(y_true.astype(int), probabilities, n_bins))
    return metrics


PRIMARY_METRIC = {"classification": "balanced_accuracy", "regression": "mae"}
HIGHER_IS_BETTER = {"balanced_accuracy": True, "macro_f1": True, "auroc": True, "accuracy": True, "mae": False, "rmse": False}
