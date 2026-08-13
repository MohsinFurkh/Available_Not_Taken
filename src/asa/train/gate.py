"""Step 4: the minimal decisive experiment, and the project's kill switch.

Proposal, Recommended sequence, Step 4:

    "Before building anything else: train one ResNet-50 on one dataset
    (A-full), test on A-only. If performance is at chance across two datasets,
    the central hypothesis is refuted cheaply and the project should be
    redirected to D2. This is the go/no-go gate and it costs almost nothing."

The decision rule is written down here, in code, before any number exists. That
is the point: a gate whose threshold is chosen after seeing the result is not a
gate. The rule is deliberately conservative in the direction of killing the
project -- the bootstrap lower bound must clear chance, not the point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..eval.metrics import PRIMARY_METRIC
from ..utils import get_logger, write_json

LOGGER = get_logger("asa.gate")


@dataclass
class GateVerdict:
    dataset: str
    #: A-only performance: the model trained on annotated images, tested on
    #: images where the anatomy is destroyed and only the annotations remain.
    a_only_score: float
    a_only_ci_low: float
    a_only_ci_high: float
    chance: float
    #: A-full performance, for context on how much of it A-only recovers.
    a_full_score: float
    #: The deployment penalty: A-full-trained, A-removed-tested.
    a_removed_score: float
    #: The R5 control: equal area of anatomy inpainted instead.
    ctrl_score: float | None = None
    passed: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def shortcut_magnitude(self) -> float:
        """A-only performance above chance, as a fraction of the headroom the
        full model achieves. 0 = no shortcut, 1 = annotations alone reproduce
        the whole result."""
        headroom = self.a_full_score - self.chance
        if headroom <= 1e-6:
            return float("nan")
        return float((self.a_only_score - self.chance) / headroom)

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "shortcut_magnitude": self.shortcut_magnitude}


#: Minimum margin of the A-only bootstrap lower bound above chance, in the
#: primary metric. Pre-registered, not tuned.
MIN_MARGIN_ABOVE_CHANCE = 0.03


def evaluate_gate(
    results: pd.DataFrame,
    dataset: str,
    task: str = "classification",
    metric: str | None = None,
    min_margin: float = MIN_MARGIN_ABOVE_CHANCE,
) -> GateVerdict:
    metric = metric or PRIMARY_METRIC[task]
    rows = results[
        (results["dataset"] == dataset)
        & (results["train_variant"] == "a_full")
        & (~results.get("is_contrast", pd.Series(False, index=results.index)).fillna(False).astype(bool))
    ]
    if rows.empty:
        available = sorted(results.get("train_variant", pd.Series(dtype=str)).dropna().unique().tolist())
        raise ValueError(
            f"no a_full-trained rows for dataset {dataset!r}; train_variant values present: {available}.\n"
            "evaluate_gate implements the Step-4 gate rule, which is defined only for models trained "
            "on A-full. Experiments that train on another variant (step4b, and any run with a "
            "different regime) have no gate verdict by construction -- summarise them with "
            "asa.train.matrix.summarise_matrix instead."
        )

    def _mean(variant: str, column: str) -> float:
        subset = rows[rows["test_variant"] == variant]
        return float(subset[column].mean()) if len(subset) and column in subset else float("nan")

    chance_column = "chance_balanced_accuracy" if metric == "balanced_accuracy" else "chance_accuracy"
    verdict = GateVerdict(
        dataset=dataset,
        a_only_score=_mean("a_only", metric),
        a_only_ci_low=_mean("a_only", f"{metric}_ci_low"),
        a_only_ci_high=_mean("a_only", f"{metric}_ci_high"),
        chance=_mean("a_only", chance_column) if task == "classification" else float("nan"),
        a_full_score=_mean("a_full", metric),
        a_removed_score=_mean("a_removed", metric),
        ctrl_score=_mean("ctrl_random", metric),
    )

    reasons: list[str] = []
    if not np.isfinite(verdict.a_only_ci_low):
        reasons.append("A-only bootstrap interval unavailable")
    elif verdict.a_only_ci_low <= verdict.chance + min_margin:
        reasons.append(
            f"A-only lower bound {verdict.a_only_ci_low:.4f} does not clear chance "
            f"{verdict.chance:.4f} by the pre-registered margin {min_margin}"
        )

    # The R5 refutation: if inpainting an equal area of anatomy costs as much as
    # inpainting the annotations, the drop is about pixel removal and not about
    # annotations, and the hypothesis is refuted no matter what A-only says.
    if np.isfinite(verdict.ctrl_score) and np.isfinite(verdict.a_removed_score):
        annotation_drop = verdict.a_full_score - verdict.a_removed_score
        control_drop = verdict.a_full_score - verdict.ctrl_score
        if annotation_drop <= control_drop:
            reasons.append(
                f"equal-area control drop ({control_drop:.4f}) is at least as large as the "
                f"annotation-removal drop ({annotation_drop:.4f}): the effect is pixel removal, not annotations"
            )

    verdict.passed = not reasons
    verdict.reasons = reasons or ["A-only clears chance and the equal-area control does not explain the drop"]
    return verdict


def decide(verdicts: list[GateVerdict], min_datasets_passing: int = 2) -> dict[str, Any]:
    """The go/no-go decision.

    Two datasets, not one: the proposal's own novelty stress test names
    "dataset-specific improvement" as the main risk for D1, and a shortcut seen
    on exactly one annotation convention is a quirk rather than a finding.
    """
    # A verdict computed from nothing is worse than no verdict: 0 >= 2 is False,
    # so an empty list would silently produce a confident-looking STOP report
    # that no experiment supports. Observed in practice after a session restart
    # re-ran the summary cell without the training cell.
    if not verdicts:
        raise ValueError(
            "decide() received no verdicts. Nothing was evaluated, so there is no gate "
            "result -- run the training/evaluation cell before summarising. Refusing to "
            "emit a STOP verdict that no experiment supports."
        )

    passing = [v for v in verdicts if v.passed]
    proceed = len(passing) >= min_datasets_passing
    magnitudes = {v.dataset: v.shortcut_magnitude for v in verdicts}
    return {
        "proceed": proceed,
        "n_datasets_tested": len(verdicts),
        "n_datasets_passing": len(passing),
        "min_datasets_passing": min_datasets_passing,
        "shortcut_magnitude": magnitudes,
        "verdicts": [v.as_dict() for v in verdicts],
        "recommendation": (
            "PROCEED to Step 5 (pre-registration) and Step 6 (full matrix)."
            if proceed
            else "STOP. The central hypothesis is not supported at the gate. Per the proposal's "
                 "Recommended sequence, redirect to D2 (shift-valid conformal prediction for fetal "
                 "biometry) rather than continuing. Publish the null: it retires a stated but "
                 "untested concern in the literature."
        ),
    }


def write_gate_report(decision: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "gate_decision.json", decision)

    lines = [
        "# Step 4 gate: the minimal decisive experiment",
        "",
        f"**Decision: {'PROCEED' if decision['proceed'] else 'STOP'}** "
        f"({decision['n_datasets_passing']}/{decision['n_datasets_tested']} datasets passing, "
        f"{decision['min_datasets_passing']} required)",
        "",
        decision["recommendation"],
        "",
        "| Dataset | A-full | A-removed | A-only | chance | A-only 95% CI | shortcut magnitude | passed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for verdict in decision["verdicts"]:
        lines.append(
            f"| {verdict['dataset']} | {verdict['a_full_score']:.4f} | {verdict['a_removed_score']:.4f} | "
            f"{verdict['a_only_score']:.4f} | {verdict['chance']:.4f} | "
            f"[{verdict['a_only_ci_low']:.4f}, {verdict['a_only_ci_high']:.4f}] | "
            f"{verdict['shortcut_magnitude']:.3f} | {'yes' if verdict['passed'] else 'no'} |"
        )
    lines += ["", "## Reasons", ""]
    for verdict in decision["verdicts"]:
        lines.append(f"- **{verdict['dataset']}**: " + "; ".join(verdict["reasons"]))
    lines += [
        "",
        "## What this does and does not establish",
        "",
        "Above-chance A-only performance shows the model can classify from annotation pixels alone.",
        "It does *not* yet show how much of the headline A-full number is annotation-derived -- that is",
        "the full matrix (Step 6). Nor does it establish the calibration claim (H2), which needs the",
        "reliability analysis on every matrix cell.",
        "",
        "The equal-area random-region control is included here rather than later because it is the",
        "single result that can refute the hypothesis outright (proposal Section 15, R5).",
    ]
    report = out_dir / "gate_report.md"
    report.write_text("\n".join(lines) + "\n")
    LOGGER.info("wrote %s", report)
    return report
