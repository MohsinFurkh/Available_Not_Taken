"""Statistical machinery.

Proposal Section 15, R9: "Statistical testing is inadequate / effects are within
noise" is the criticism stage 1 levelled at five of seven folder papers, so this
study cannot afford it. Everything reported carries a patient-level bootstrap
interval; every hypothesis test is paired across matched folds, corrected across
the hypothesis family, and accompanied by an effect size with its own interval;
and seed variance is reported separately from fold variance so a reader can tell
whether an effect exceeds training noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------
# patient-level bootstrap
# --------------------------------------------------------------------------


@dataclass
class Interval:
    estimate: float
    low: float
    high: float
    level: float = 0.95

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "ci_low": self.low, "ci_high": self.high, "ci_level": self.level}

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}]"


def bootstrap_metric(
    y_true: np.ndarray,
    outputs: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_replicates: int = 2000,
    level: float = 0.95,
    seed: int = 20260805,
) -> Interval:
    """Bootstrap by resampling *patients*, not images.

    Resampling images would treat 15 frames from one fetus as 15 independent
    observations and produce intervals several times too narrow. The resampling
    unit has to be the unit of dependence.
    """
    rng = np.random.default_rng(seed)
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(group_index == g) for g in range(len(unique_groups))]

    point = float(metric_fn(y_true, outputs))
    replicates = np.empty(n_replicates, dtype=float)
    n_groups = len(unique_groups)
    for r in range(n_replicates):
        picked = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([members[i] for i in picked])
        try:
            replicates[r] = metric_fn(y_true[rows], outputs[rows])
        except ValueError:
            replicates[r] = np.nan

    replicates = replicates[np.isfinite(replicates)]
    if replicates.size == 0:
        return Interval(point, float("nan"), float("nan"), level)
    alpha = (1.0 - level) / 2.0
    return Interval(point, float(np.quantile(replicates, alpha)), float(np.quantile(replicates, 1 - alpha)), level)


def bootstrap_difference(
    y_true: np.ndarray,
    outputs_a: np.ndarray,
    outputs_b: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_replicates: int = 2000,
    level: float = 0.95,
    seed: int = 20260805,
) -> Interval:
    """Paired bootstrap of metric(a) - metric(b) on the same patients.

    This is the interval for the headline quantity: the A-full minus A-removed
    gap for one model on one test fold. Pairing matters -- the two variants are
    the same images, so an unpaired interval would be dominated by between-image
    variance that cancels exactly.
    """
    rng = np.random.default_rng(seed)
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(group_index == g) for g in range(len(unique_groups))]

    point = float(metric_fn(y_true, outputs_a) - metric_fn(y_true, outputs_b))
    replicates = np.empty(n_replicates, dtype=float)
    n_groups = len(unique_groups)
    for r in range(n_replicates):
        picked = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([members[i] for i in picked])
        try:
            replicates[r] = metric_fn(y_true[rows], outputs_a[rows]) - metric_fn(y_true[rows], outputs_b[rows])
        except ValueError:
            replicates[r] = np.nan

    replicates = replicates[np.isfinite(replicates)]
    if replicates.size == 0:
        return Interval(point, float("nan"), float("nan"), level)
    alpha = (1.0 - level) / 2.0
    return Interval(point, float(np.quantile(replicates, alpha)), float(np.quantile(replicates, 1 - alpha)), level)


# --------------------------------------------------------------------------
# paired tests across matched folds
# --------------------------------------------------------------------------


def paired_test(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> dict[str, Any]:
    """Paired t or Wilcoxon signed-rank, chosen by a normality check on the
    differences (proposal Section 9), never by which one gives the answer."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    differences = a_arr - b_arr
    n = len(differences)
    result: dict[str, Any] = {"n_pairs": n, "mean_difference": float(differences.mean())}

    if n < 3 or np.allclose(differences, 0):
        result.update({"test": "none", "p_value": float("nan"), "normal": None})
        return result

    normal = True
    if n >= 8:
        try:
            normal = sps.shapiro(differences).pvalue > alpha
        except ValueError:
            normal = True
    result["normal"] = bool(normal)

    if normal:
        statistic, p_value = sps.ttest_rel(a_arr, b_arr)
        result.update({"test": "paired_t", "statistic": float(statistic), "p_value": float(p_value)})
    else:
        statistic, p_value = sps.wilcoxon(a_arr, b_arr)
        result.update({"test": "wilcoxon", "statistic": float(statistic), "p_value": float(p_value)})

    result.update(cohens_dz(differences))
    return result


def cohens_dz(differences: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Effect size for a paired design, with its confidence interval.

    Reported alongside every p-value, because the study's own critique of the
    folder corpus (P4 retained a module that worsened mean MAE on the strength
    of a larger SD) is exactly what an effect size without an interval permits.
    """
    n = len(differences)
    sd = float(differences.std(ddof=1))
    if sd == 0 or n < 2:
        return {"cohens_dz": float("nan"), "dz_ci_low": float("nan"), "dz_ci_high": float("nan")}
    dz = float(differences.mean() / sd)
    se = float(np.sqrt(1.0 / n + dz**2 / (2 * n)))
    z = float(sps.norm.ppf(1 - (1 - level) / 2))
    return {"cohens_dz": dz, "dz_ci_low": dz - z * se, "dz_ci_high": dz + z * se}


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni across the hypothesis family.

    The family is declared once, in the pre-registration, and corrected as a
    whole -- correcting only the tests that happened to be run is the failure
    this is meant to prevent.
    """
    items = [(name, p) for name, p in p_values.items() if np.isfinite(p)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, Any]] = {}
    previous = 0.0
    for rank, (name, p) in enumerate(items):
        adjusted = min(1.0, max(previous, (m - rank) * p))
        previous = adjusted
        out[name] = {"p_raw": float(p), "p_adjusted": float(adjusted), "reject": bool(adjusted < alpha)}
    for name, p in p_values.items():
        if name not in out:
            out[name] = {"p_raw": float(p), "p_adjusted": float("nan"), "reject": False}
    return out


# --------------------------------------------------------------------------
# variance decomposition
# --------------------------------------------------------------------------


def variance_components(records: Sequence[dict[str, Any]], metric: str) -> dict[str, float]:
    """Split observed variance into fold and seed components.

    Section 9 requires this so that a reader can tell whether an effect exceeds
    training noise. If seed variance dominates fold variance, no comparison of
    single runs means anything -- which is what the stage-1 review found in P4.
    """
    values = np.array([r[metric] for r in records], dtype=float)
    folds = np.array([r["fold"] for r in records])
    seeds = np.array([r["seed"] for r in records])

    def _within(grouping: np.ndarray) -> float:
        parts = [values[grouping == g] for g in np.unique(grouping)]
        parts = [p for p in parts if len(p) > 1]
        if not parts:
            return float("nan")
        return float(np.mean([p.var(ddof=1) for p in parts]))

    return {
        "total_var": float(values.var(ddof=1)) if len(values) > 1 else float("nan"),
        "within_fold_var_across_seeds": _within(folds),
        "within_seed_var_across_folds": _within(seeds),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        "n_runs": int(len(values)),
    }
