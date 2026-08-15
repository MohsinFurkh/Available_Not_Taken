"""Configuration objects.

Everything the study does is driven by YAML so that a run is reproducible from
its config alone (see the released protocol).
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# The three image variants of the annotation triplet (see the released protocol),
# plus the equal-area random-region control demanded by the pixel-removal objection.
Variant = Literal["a_full", "a_removed", "a_only", "ctrl_random"]
VARIANTS: tuple[Variant, ...] = ("a_full", "a_removed", "a_only", "ctrl_random")

# The three supervision regimes (see the released protocol).
Regime = Literal["r_naive", "r_clean", "r_supervised", "r_annotation_only"]
REGIMES: tuple[Regime, ...] = ("r_naive", "r_clean", "r_supervised", "r_annotation_only")

#: Which variant each regime trains on. R-supervised trains on A-removed and
#: additionally receives parsed annotation content through an auxiliary head,
#: which is what separates "annotations help by being visible" (shortcut) from
#: "annotations help by carrying information" (supervision).
REGIME_TRAIN_VARIANT: dict[str, Variant] = {
    "r_naive": "a_full",
    "r_clean": "a_removed",
    "r_supervised": "a_removed",
    # Trains *on* A-only. A-only as a test condition asks whether an
    # archive-trained representation happens to route through the annotations;
    # training on it asks the different and prior question of whether the
    # annotations carry learnable label information at all. The two can come
    # apart -- a shortcut can be available without being taken -- and telling
    # them apart is what separates "no signal" from "signal not used".
    "r_annotation_only": "a_only",
}


@dataclass
class DatasetConfig:
    """One dataset, as declared in configs/datasets/*.yaml."""

    name: str
    root: str
    task: Literal["classification", "regression"] = "classification"
    #: Column of the manifest used as the supervised target.
    label_column: str = "label"
    #: Column grouping images that must never straddle a split. Patient-level
    #: separation is non-negotiable (see the released protocol).
    group_column: str = "patient_id"
    num_classes: int = 0
    class_names: list[str] = field(default_factory=list)
    #: Use the dataset's own published fold assignment where one exists, so
    #: results stay comparable to the published literature.
    official_fold_column: str | None = None
    #: Images whose annotation masks are known to be unreliable are excluded by
    #: the verification gate; this is the per-dataset mask sensitivity floor.
    mask_sensitivity_floor: float = 0.90
    #: Pack single-channel. B-mode is achromatic, so this is lossless and cuts
    #: the shard to a third -- which decides whether the package fits in a Drive
    #: upload. It must be False wherever colour carries annotation: MMOTU's
    #: calipers and measurement line ARE the chromatic overlay, and discarding
    #: colour there would delete the signal being measured.
    grayscale: bool = True
    notes: str = ""


@dataclass
class AnnotationConfig:
    """Annotation detection + variant construction (see the released protocol)."""

    # --- fan / imaging-sector detection -----------------------------------
    fan_threshold: int = 10
    fan_open_kernel: int = 15

    # --- rendered-graphic candidates --------------------------------------
    #: Candidates are top-hat responses, NOT absolute-intensity thresholds. The
    #: same burnt-in glyphs measure 170 / 209 / 49 across FETAL_PLANES_DB,
    #: BUS-BRA and HC18, so any fixed intensity cut-off drops a whole dataset.
    text_tophat_kernel: int = 7
    graphic_min_tophat: int = 18
    graphic_min_intensity: int = 30
    text_min_area: int = 5
    text_max_area: int = 2500
    #: Boundary gradient magnitude divided by the blob's own contrast. Rendered
    #: glyphs have a hard one-pixel transition; speckle is band-limited by the
    #: transducer point-spread function and cannot. Brightness-invariant by
    #: construction, which is what lets one threshold serve all four datasets.
    graphic_min_sharpness: float = 3.2

    # --- burnt-in text ----------------------------------------------------
    #: Contrast against the immediate surrounding ring. This is the single
    #: discriminator that separates rendered glyphs from bright speckle.
    text_min_contrast: float = 45.0
    #: Glyphs on one rendered line share a plateau value; speckle does not.
    text_max_peak_spread: float = 10.0
    #: A blob group only counts as text if it has more than this many members,
    #: OR passes the merged-word shape test below.
    text_min_line_members: int = 2
    #: Low-resolution renders merge whole words into one blob; accept those on
    #: shape (wide, thin, sparse) rather than on glyph count.
    text_min_word_aspect: float = 2.5
    text_max_word_fill: float = 0.7

    # --- calipers ---------------------------------------------------------
    caliper_min_area: int = 8
    caliper_max_area: int = 600
    caliper_max_aspect: float = 2.5
    caliper_min_contrast: float = 48.0
    #: Cross glyphs are sparse inside their bounding box; solid specular blobs
    #: are not. Components denser than this are rejected as anatomy.
    caliper_max_fill: float = 0.62

    # --- depth-ruler ticks ------------------------------------------------
    tick_edge_band_frac: float = 0.06
    tick_max_area: int = 400
    tick_min_contrast: float = 25.0
    tick_min_members: int = 4
    #: Coefficient of variation of the vertical gaps. The ruler is periodic.
    tick_max_spacing_cv: float = 0.55

    # --- curator redaction boxes ------------------------------------------
    redaction_max_intensity: int = 2
    redaction_min_area_frac: float = 0.0015
    redaction_min_rectangularity: float = 0.93
    redaction_max_area_frac: float = 0.15

    # --- colour overlays --------------------------------------------------
    #: B-mode ultrasound is achromatic, so any saturated pixel is an overlay.
    overlay_min_saturation: int = 40
    overlay_min_value: int = 45
    #: Colour Doppler is chromatic but it is imaging data, not annotation.
    #: Chromatic components above this area AND this density are rejected as
    #: flow mapping; thin structures (dotted measurement lines) are kept at any
    #: size. Without this, 9%% of MMOTU-2D exceeded a 5%% mask area.
    overlay_max_component_area_frac: float = 0.02
    overlay_max_component_density: float = 0.30
    #: Optional CLAHE + HSV K-means path, following a published removal pipeline.
    overlay_use_kmeans: bool = False
    overlay_kmeans_k: int = 5

    # --- mask post-processing --------------------------------------------
    dilate_px: int = 4
    #: Drop mask components smaller than this after the union (speckle).
    min_component_area: int = 4

    # --- variant construction --------------------------------------------
    inpaint_method: Literal["navier_stokes", "telea", "uniform", "noise"] = "navier_stokes"
    inpaint_radius: int = 3
    #: Fill used outside the annotation regions when building A-only.
    a_only_fill: Literal["uniform", "noise", "black"] = "noise"
    #: Equal-area control: inpaint random regions elsewhere in the fan.
    ctrl_max_placement_tries: int = 40


@dataclass
class SplitConfig:
    n_folds: int = 5
    n_seeds: int = 5
    seed0: int = 1000
    val_fraction: float = 0.15
    stratify: bool = True
    use_official_folds: bool = True


@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    pretrained: Literal["imagenet", "none", "usf_mae", "fetalclip"] = "imagenet"
    #: Path to third-party foundation-encoder weights, when pretrained is
    #: usf_mae or fetalclip.
    weights_path: str | None = None
    image_size: int = 224
    dropout: float = 0.0
    freeze_backbone: bool = False  # True = linear probe


@dataclass
class AuxConfig:
    """Auxiliary weak-supervision head fed by parsed annotation content."""

    enabled: bool = False
    #: Categorical fields parsed out of the burnt-in text (vendor preset,
    #: probe, laterality...). Each becomes one softmax head.
    categorical_fields: list[str] = field(default_factory=list)
    #: Continuous fields (depth in cm, gain, measurement values in mm).
    numeric_fields: list[str] = field(default_factory=list)
    #: L = L_task + lambda * L_aux
    lambda_aux: float = 1.0


@dataclass
class SyntheticShortcutConfig:
    """Positive control: inject an artificial shortcut of known strength.

    Why this is not optional. The Step-4 gate found that removing the real
    annotations costs nothing -- but the equal-area control showed that removing
    the same area of *anatomy* also costs nothing, both at ~0.07x the run-to-run
    noise. A null is only informative if the experiment could have detected an
    effect that was there, and nothing in the gate demonstrates that it could.

    So we plant a shortcut we control, at the same area and intensity as the
    real annotations, and sweep how strongly it correlates with the label. That
    yields a sensitivity curve: the weakest annotation-label association this
    design can detect. The real annotations can then be placed on that curve
    rather than merely declared undetectable.
    """

    enabled: bool = False
    #: Fraction of image pixels the marker covers. Default matches the measured
    #: median annotation coverage at 224 px (~0.7%), so the control is a fair
    #: power test rather than an obvious blob.
    area_frac: float = 0.007
    #: Marker intensity, matching burnt-in graphics rather than anatomy.
    intensity: int = 220
    #: P(marker encodes the true label). 1.0 is a perfect shortcut; 0.0 is a
    #: label-independent distractor. Sweeping this is the sensitivity curve.
    strength: float = 1.0
    #: Grid used to encode the class in the marker's position, as operator
    #: annotations encode meaning by where they are placed.
    grid: int = 4
    seed: int = 7
    #: "rectangle" -- a solid block at one of K canonical positions. Easy to
    #: read, and therefore an UPPER bound on the design's sensitivity.
    #: "glyph"     -- a rendered vendor-like string whose *content* carries the
    #: class, scattered in position, jittered in scale, sometimes occluded.
    #: Matched to real annotations on the property that governs learnability,
    #: and therefore the bound a calibration actually needs.
    cue: Literal["rectangle", "glyph"] = "rectangle"
    occlusion_prob: float = 0.25


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: Literal["adamw", "sgd"] = "adamw"
    scheduler: Literal["cosine", "none"] = "cosine"
    warmup_epochs: int = 1
    early_stop_patience: int = 8
    num_workers: int = 4
    amp: bool = True
    #: Fraction of available manual labels used, for the SO2 efficiency curve.
    label_budget: float = 1.0
    #: Mitigation arm: randomly strip / re-insert annotations at training time.
    annotation_randomising_aug: bool = False
    annotation_aug_prob: float = 0.5
    grad_clip: float | None = 1.0


@dataclass
class EvalConfig:
    #: Every trained model is evaluated on every variant, producing the
    #: train x test matrix whose off-diagonals are the scientific content.
    test_variants: list[str] = field(default_factory=lambda: list(VARIANTS))
    calibration_bins: int = 15
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 20260805
    #: Write per-image probabilities per test variant. Needed for qualitative
    #: figures and per-case review; costs a few hundred KB per run.
    save_predictions: bool = True
    saliency: bool = True
    saliency_max_images: int = 512


@dataclass
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    aux: AuxConfig = field(default_factory=AuxConfig)
    synthetic: SyntheticShortcutConfig = field(default_factory=SyntheticShortcutConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    regimes: list[str] = field(default_factory=lambda: ["r_naive", "r_clean"])
    output_dir: str = "runs"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls, payload: dict[str, Any] | None):
    payload = payload or {}
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - fields
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**payload)


def load_experiment(
    experiment_path: str | Path,
    config_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load an experiment YAML, resolving its ``dataset:`` reference.

    An experiment file names a dataset by stem (``dataset: fetal_planes``) and
    may override any dataset field under ``dataset_overrides:``.
    """
    experiment_path = Path(experiment_path)
    config_root = Path(config_root) if config_root else experiment_path.parent.parent
    raw = yaml.safe_load(experiment_path.read_text()) or {}
    if overrides:
        raw = _merge(raw, overrides)

    dataset_ref = raw.pop("dataset")
    if isinstance(dataset_ref, str):
        dataset_raw = yaml.safe_load((config_root / "datasets" / f"{dataset_ref}.yaml").read_text())
    else:
        dataset_raw = dataset_ref
    dataset_raw = _merge(dataset_raw, raw.pop("dataset_overrides", {}) or {})

    return ExperimentConfig(
        name=raw.pop("name", experiment_path.stem),
        dataset=_build(DatasetConfig, dataset_raw),
        annotation=_build(AnnotationConfig, raw.pop("annotation", None)),
        split=_build(SplitConfig, raw.pop("split", None)),
        model=_build(ModelConfig, raw.pop("model", None)),
        aux=_build(AuxConfig, raw.pop("aux", None)),
        synthetic=_build(SyntheticShortcutConfig, raw.pop("synthetic", None)),
        train=_build(TrainConfig, raw.pop("train", None)),
        eval=_build(EvalConfig, raw.pop("eval", None)),
        regimes=raw.pop("regimes", ["r_naive", "r_clean"]),
        output_dir=raw.pop("output_dir", "runs"),
    )
