"""Invariants that must hold or the study's numbers mean nothing.

Run with: python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asa.config import AnnotationConfig, DatasetConfig, SplitConfig
from asa.annot.detect import detect
from asa.annot.variants import build_variants
from asa.data.splits import apply_label_budget, assert_disjoint_groups, make_folds
from asa.eval.metrics import chance_rate, evaluate, expected_calibration_error
from asa.eval.stats import bootstrap_metric, holm_bonferroni, paired_test


# --------------------------------------------------------------------------
# synthetic ultrasound frame: speckle sector + rendered chrome
# --------------------------------------------------------------------------


def synthetic_frame(seed: int = 0, with_text: bool = True, with_calipers: bool = True) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    image = np.zeros((300, 400), dtype=np.uint8)
    # speckle inside a trapezoidal sector
    sector = np.zeros_like(image)
    cv2.fillConvexPoly(sector, np.array([[190, 10], [210, 10], [380, 290], [20, 290]]), 1)
    speckle = np.clip(rng.gamma(2.0, 30.0, image.shape), 0, 255).astype(np.uint8)
    image[sector > 0] = speckle[sector > 0]

    if with_text:
        cv2.putText(image, "Cardiac", (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
        cv2.putText(image, "Gn 5", (300, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
    if with_calipers:
        for cx, cy in ((150, 150), (250, 150)):
            cv2.line(image, (cx - 5, cy), (cx + 5, cy), 255, 1)
            cv2.line(image, (cx, cy - 5), (cx, cy + 5), 255, 1)
    return image


def test_detector_finds_rendered_chrome_and_not_speckle():
    det = detect(synthetic_frame(0))
    assert det.annotation.any(), "detector found no annotation on a frame that has text and calipers"
    # Precision guard: the chrome occupies well under 5% of the frame, so a mask
    # covering more than that is eating anatomy.
    assert det.features["annotation_area_frac"] < 0.05

    clean = detect(synthetic_frame(1, with_text=False, with_calipers=False))
    assert clean.features["annotation_area_frac"] < det.features["annotation_area_frac"], (
        "a frame with no rendered graphics produced as much annotation mask as one with them"
    )


def test_variants_differ_only_where_they_should():
    """A-removed must change pixels only inside the mask; A-only must preserve
    exactly those pixels. If either fails, every matrix cell is meaningless."""
    image = synthetic_frame(2)
    cfg = AnnotationConfig()
    det = detect(image, cfg)
    variants = build_variants(image, det, cfg, seed=0)

    mask = det.annotation
    full = variants["a_full"].astype(np.int16)
    removed = variants["a_removed"].astype(np.int16)
    only = variants["a_only"].astype(np.int16)
    if full.ndim == 3:
        mask = np.repeat(mask[..., None], full.shape[-1], axis=-1)

    assert np.abs(full - removed)[~mask].max() == 0, "A-removed altered pixels outside the annotation mask"
    assert np.abs(full - only)[mask].max() == 0, "A-only altered the annotation pixels it must preserve"
    assert np.abs(full - only)[~mask].mean() > 1.0, "A-only did not destroy the anatomy"


def test_control_removes_equal_area_of_anatomy():
    """Reviewer attack R5: the control must differ from A-removed in location
    only, so its area has to match."""
    image = synthetic_frame(3)
    cfg = AnnotationConfig()
    det = detect(image, cfg)
    variants = build_variants(image, det, cfg, seed=0)

    changed_removed = (variants["a_full"] != variants["a_removed"]).any(axis=-1) if variants["a_full"].ndim == 3 \
        else (variants["a_full"] != variants["a_removed"])
    changed_ctrl = (variants["a_full"] != variants["ctrl_random"]).any(axis=-1) if variants["a_full"].ndim == 3 \
        else (variants["a_full"] != variants["ctrl_random"])

    if changed_removed.sum() > 50:  # only meaningful when something was removed
        ratio = changed_ctrl.sum() / changed_removed.sum()
        assert 0.4 < ratio < 2.5, f"control area {ratio:.2f}x the annotation area; it must be comparable"


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------


def _toy_manifest(n_patients: int = 60, per_patient: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for p in range(n_patients):
        label = p % 3
        for k in range(per_patient):
            rows.append(
                {
                    "image_id": f"p{p}_{k}",
                    "patient_id": f"P{p:03d}",
                    "label": label,
                    "target": float(label) + rng.normal(0, 0.1),
                    "official_fold": -1,
                    "array_index": len(rows),
                }
            )
    return pd.DataFrame(rows)


def test_no_patient_straddles_a_split():
    manifest = _toy_manifest()
    dataset = DatasetConfig(name="toy", root=".", num_classes=3)
    for split in make_folds(manifest, dataset, SplitConfig(n_folds=5, val_fraction=0.2), seed=7):
        assert_disjoint_groups(manifest, split, "patient_id")
        assert len(split.test_idx) > 0 and len(split.val_idx) > 0


def test_label_budget_subsamples_patients_not_images():
    manifest = _toy_manifest()
    dataset = DatasetConfig(name="toy", root=".", num_classes=3)
    train_idx = np.arange(len(manifest))
    kept = apply_label_budget(manifest, train_idx, dataset, budget=0.25, seed=1)
    sub = manifest.iloc[kept]
    # every retained patient keeps all of their frames
    counts = sub.groupby("patient_id").size().unique()
    assert set(counts.tolist()) == {4}, "budget sampling split a patient's frames"
    assert 0.15 < sub.patient_id.nunique() / manifest.patient_id.nunique() < 0.4


# --------------------------------------------------------------------------
# metrics and statistics
# --------------------------------------------------------------------------


def test_chance_rate_uses_the_empirical_distribution():
    # 90/10 imbalance: accuracy chance is 0.9, balanced-accuracy chance is 0.5
    y = np.array([0] * 90 + [1] * 10)
    assert chance_rate(y, "accuracy") == pytest.approx(0.9)
    assert chance_rate(y, "balanced_accuracy") == pytest.approx(0.5)


def test_overconfidence_is_detected():
    """H2 in miniature: confident and wrong must show up as positive
    confidence-minus-accuracy and non-trivial ECE."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    wrong = (y + 1) % 3
    probabilities = np.full((400, 3), 0.02)
    probabilities[np.arange(400), wrong] = 0.96
    metrics = evaluate(y, probabilities, "classification")
    assert metrics["confidence_minus_accuracy"] > 0.8
    ece, _ = expected_calibration_error(y, probabilities)
    assert ece > 0.8


def test_bootstrap_resamples_patients():
    """Image-level resampling would give an interval several times too narrow;
    guard against a regression to it."""
    from asa.eval.metrics import classification_metrics

    rng = np.random.default_rng(0)
    n_patients, per_patient = 40, 10
    groups = np.repeat(np.arange(n_patients), per_patient)
    y = np.repeat(rng.integers(0, 2, n_patients), per_patient)
    # Errors are correlated *within* a patient -- which is the real situation
    # (frames from one fetus fail together) and the whole reason the resampling
    # unit has to be the patient.
    patient_wrong = rng.random(n_patients) < 0.2
    noisy = np.where(np.repeat(patient_wrong, per_patient), 1 - y, y)
    probabilities = np.zeros((len(y), 2))
    probabilities[np.arange(len(y)), noisy] = 1.0

    metric = lambda a, b: classification_metrics(a.astype(int), b)["balanced_accuracy"]  # noqa: E731
    grouped = bootstrap_metric(y, probabilities, groups, metric, n_replicates=200, seed=0)
    per_image = bootstrap_metric(y, probabilities, np.arange(len(y)), metric, n_replicates=200, seed=0)
    assert (grouped.high - grouped.low) > (per_image.high - per_image.low)


def test_holm_bonferroni_is_monotone():
    corrected = holm_bonferroni({"h1": 0.001, "h2": 0.02, "h3": 0.30, "h4": 0.60})
    adjusted = [corrected[k]["p_adjusted"] for k in ("h1", "h2", "h3", "h4")]
    assert adjusted == sorted(adjusted), "adjusted p-values must be non-decreasing in raw order"
    assert all(a >= corrected[k]["p_raw"] for k, a in zip(("h1", "h2", "h3", "h4"), adjusted))


def test_paired_test_reports_effect_size():
    a = [0.80, 0.82, 0.79, 0.81, 0.83, 0.80, 0.82, 0.81]
    b = [0.72, 0.74, 0.71, 0.73, 0.75, 0.72, 0.74, 0.73]
    result = paired_test(a, b)
    assert result["p_value"] < 0.01
    assert result["cohens_dz"] > 1.0
    assert np.isfinite(result["dz_ci_low"]) and np.isfinite(result["dz_ci_high"])


def test_gate_refutes_when_control_drops_as_much():
    """The R5 refutation must fire even when A-only looks impressive."""
    from asa.train.gate import evaluate_gate

    rows = []
    for variant, score in (("a_full", 0.90), ("a_removed", 0.70), ("a_only", 0.60), ("ctrl_random", 0.68)):
        rows.append(
            {
                "dataset": "toy", "train_variant": "a_full", "test_variant": variant,
                "balanced_accuracy": score, "balanced_accuracy_ci_low": score - 0.02,
                "balanced_accuracy_ci_high": score + 0.02, "chance_balanced_accuracy": 0.25,
                "is_contrast": False,
            }
        )
    verdict = evaluate_gate(pd.DataFrame(rows), "toy")
    assert not verdict.passed
    assert any("equal-area control" in reason for reason in verdict.reasons)


# --------------------------------------------------------------------------
# positive control
# --------------------------------------------------------------------------


def test_synthetic_shortcut_is_label_dependent_and_deterministic():
    """The injected marker must move with the label and be stable per image.

    If it were not deterministic, the model would face a different marker each
    epoch and the control would understate what is learnable -- reporting a
    false 'the design cannot detect shortcuts'.
    """
    from asa.config import SyntheticShortcutConfig
    from asa.data.shortcut import inject_synthetic_shortcut

    cfg = SyntheticShortcutConfig(enabled=True, area_frac=0.007, intensity=220, strength=1.0)
    blank = np.zeros((224, 224), dtype=np.uint8)

    a = inject_synthetic_shortcut(blank, label=0, num_classes=6, cfg=cfg, row=11)
    b = inject_synthetic_shortcut(blank, label=3, num_classes=6, cfg=cfg, row=11)
    again = inject_synthetic_shortcut(blank, label=0, num_classes=6, cfg=cfg, row=11)

    assert not np.array_equal(a, b), "marker does not move with the label"
    assert np.array_equal(a, again), "marker is not deterministic for a given image"

    # area matches the configured fraction, so the control is a fair power test
    covered = (a == cfg.intensity).mean()
    assert 0.5 * cfg.area_frac < covered < 2.0 * cfg.area_frac, covered


def test_synthetic_shortcut_strength_controls_correlation():
    """strength is a correlation dial: at 0 the marker must carry no label
    information, at 1 it must carry all of it."""
    from asa.config import SyntheticShortcutConfig
    from asa.data.shortcut import inject_synthetic_shortcut

    blank = np.zeros((224, 224), dtype=np.uint8)

    def agreement(strength: float) -> float:
        cfg = SyntheticShortcutConfig(enabled=True, strength=strength)
        matches = 0
        for row in range(400):
            label = row % 6
            marked = inject_synthetic_shortcut(blank, label, 6, cfg, row)
            truth = inject_synthetic_shortcut(blank, label, 6, SyntheticShortcutConfig(enabled=True, strength=1.0), row)
            matches += int(np.array_equal(marked, truth))
        return matches / 400

    assert agreement(1.0) == 1.0
    assert 0.5 < agreement(0.5) < 0.85          # 0.5 encode truth + chance agreement among the rest
    assert agreement(0.0) < 0.45


def test_nomarker_variant_suppresses_injection_only():
    """`__nomarker` must return the underlying variant untouched by the marker,
    and nothing else. Without this the positive control cannot measure what
    removing a used shortcut costs."""
    from asa.config import SyntheticShortcutConfig
    from asa.data.datasets import NOMARKER_SUFFIX, VariantDataset

    n = 8
    arrays = {"a_full": np.random.default_rng(0).integers(0, 255, (n, 32, 32), dtype=np.uint8)}
    labels = np.arange(n) % 4
    synthetic = SyntheticShortcutConfig(enabled=True, area_frac=0.05, intensity=250)

    common = dict(arrays=arrays, indices=np.arange(n), labels=labels, train=False,
                  augment=False, synthetic=synthetic, num_classes=4)
    marked = VariantDataset(variant="a_full", **common)
    clean = VariantDataset(variant="a_full" + NOMARKER_SUFFIX, **common)

    a = marked[3]["image"].numpy()
    b = clean[3]["image"].numpy()
    assert not np.allclose(a, b), "__nomarker returned the marked image"

    # and the clean one must equal the raw array
    raw = VariantDataset(variant="a_full", **{**common, "synthetic": None})[3]["image"].numpy()
    assert np.allclose(b, raw), "__nomarker altered something other than the marker"


def test_decide_refuses_an_empty_verdict_list():
    """0 >= 2 is False, so an empty list would produce a confident STOP report
    supported by no experiment at all."""
    from asa.train.gate import decide

    with pytest.raises(ValueError, match="no verdicts"):
        decide([], min_datasets_passing=2)
