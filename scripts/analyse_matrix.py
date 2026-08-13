#!/usr/bin/env python
"""Turn raw matrix results into the tables the manuscript needs.

    python scripts/analyse_matrix.py runs/full_matrix/fetal_planes/resnet50_lb1.0

Produces:
    matrix_cells.csv       mean +- SD per train x test cell, with n_runs
    hypothesis_tests.json  H1/H2 tests, Holm-Bonferroni corrected as one family
    variance.json          fold vs seed variance decomposition
    report.md              the above, readable

The hypothesis family is declared here, once, and corrected as a whole. Adding a
test later without re-running the correction is the failure this file exists to
prevent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asa.eval.stats import holm_bonferroni, paired_test, variance_components  # noqa: E402
from asa.utils import write_json  # noqa: E402


def cells(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    real = results[~results.get("is_contrast", pd.Series(False, index=results.index)).fillna(False).astype(bool)]
    grouped = real.groupby(["backbone", "regime", "train_variant", "test_variant"])
    summary = grouped.agg(
        **{
            f"{metric}_mean": (metric, "mean"),
            f"{metric}_sd": (metric, "std"),
            "ece_mean": ("ece", "mean") if "ece" in real else (metric, "size"),
            "confidence_minus_accuracy_mean": ("confidence_minus_accuracy", "mean")
            if "confidence_minus_accuracy" in real
            else (metric, "size"),
            "n_runs": (metric, "size"),
        }
    ).reset_index()
    return summary


def _paired(real: pd.DataFrame, train_variant: str, a: str, b: str, metric: str) -> dict:
    """Pair on (fold, seed): the same weights on the same patients."""
    subset = real[real.train_variant == train_variant]
    left = subset[subset.test_variant == a].set_index(["fold", "seed"])[metric]
    right = subset[subset.test_variant == b].set_index(["fold", "seed"])[metric]
    common = left.index.intersection(right.index)
    if len(common) < 3:
        return {"n_pairs": len(common), "p_value": float("nan"), "note": "too few matched runs"}
    return paired_test(left.loc[common].to_numpy(), right.loc[common].to_numpy())


def hypothesis_tests(results: pd.DataFrame, metric: str) -> dict:
    real = results[~results.get("is_contrast", pd.Series(False, index=results.index)).fillna(False).astype(bool)]
    tests: dict[str, dict] = {}

    # H1: annotations alone beat chance, and removing them costs performance.
    a_only = real[(real.train_variant == "a_full") & (real.test_variant == "a_only")]
    if len(a_only):
        chance_column = "chance_balanced_accuracy" if metric == "balanced_accuracy" else "chance_accuracy"
        chance = float(a_only[chance_column].mean()) if chance_column in a_only else np.nan
        observed = a_only[metric].to_numpy()
        tests["H1_a_only_above_chance"] = paired_test(observed, np.full_like(observed, chance))
        tests["H1_a_only_above_chance"]["chance"] = chance
        tests["H1_a_only_above_chance"]["mean_a_only"] = float(observed.mean())

    tests["H1_removal_costs_performance"] = _paired(real, "a_full", "a_full", "a_removed", metric)

    # R5: the effect must be annotations, not pixel removal.
    tests["R5_annotation_vs_equal_area_control"] = _paired(real, "a_full", "a_removed", "ctrl_random", metric)

    # H2: confidence must not track the accuracy loss.
    if "confidence_minus_accuracy" in real:
        tests["H2_overconfidence_under_removal"] = _paired(
            real, "a_full", "a_removed", "a_full", "confidence_minus_accuracy"
        )

    # A3: does clean training confer robustness?
    if (real.train_variant == "a_removed").any():
        clean = real[(real.train_variant == "a_removed") & (real.test_variant == "a_full")]
        naive = real[(real.train_variant == "a_full") & (real.test_variant == "a_removed")]
        if len(clean) >= 3 and len(naive) >= 3:
            tests["A3_clean_training_robustness"] = paired_test(
                clean[metric].to_numpy()[: min(len(clean), len(naive))],
                naive[metric].to_numpy()[: min(len(clean), len(naive))],
            )

    corrected = holm_bonferroni({k: v.get("p_value", float("nan")) for k, v in tests.items()})
    for name, payload in tests.items():
        payload.update(corrected.get(name, {}))
    return tests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--metric", default="balanced_accuracy")
    args = parser.parse_args()

    results = pd.read_csv(args.run_dir / "matrix_results.csv")
    metric = args.metric

    summary = cells(results, metric)
    summary.to_csv(args.run_dir / "matrix_cells.csv", index=False)

    tests = hypothesis_tests(results, metric)
    write_json(args.run_dir / "hypothesis_tests.json", tests)

    real = results[~results.get("is_contrast", pd.Series(False, index=results.index)).fillna(False).astype(bool)]
    variance = {
        f"{regime}:{variant}": variance_components(group.to_dict("records"), metric)
        for (regime, variant), group in real.groupby(["regime", "test_variant"])
        if len(group) > 1
    }
    write_json(args.run_dir / "variance.json", variance)

    lines = ["# Matrix results", "", f"Metric: `{metric}`", "", "## Cells", "", summary.to_markdown(index=False), ""]
    lines += ["## Hypothesis tests (Holm-Bonferroni across the family)", ""]
    lines += ["| test | n pairs | mean difference | effect size dz | p raw | p adjusted | reject |", "|---|---|---|---|---|---|---|"]
    for name, payload in tests.items():
        lines.append(
            f"| {name} | {payload.get('n_pairs', '')} | {payload.get('mean_difference', float('nan')):.4f} | "
            f"{payload.get('cohens_dz', float('nan')):.3f} | {payload.get('p_raw', float('nan')):.2e} | "
            f"{payload.get('p_adjusted', float('nan')):.2e} | {payload.get('reject', '')} |"
        )
    lines += [
        "",
        "## Fold vs seed variance",
        "",
        "If `within_fold_var_across_seeds` is comparable to `total_var`, single-run",
        "comparisons in this configuration are meaningless and only distributions",
        "should be reported.",
        "",
        "```json",
        __import__("json").dumps(variance, indent=2)[:4000],
        "```",
    ]
    (args.run_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
