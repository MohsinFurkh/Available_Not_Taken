"""Backbones and heads.

**No architecture is proposed by this study.** the protocol is explicit that the
contribution is the measurement and the reformulation, not the network, and
Ambsdorf et al. (2025) provide direct evidence that methodological innovation is
the wrong lever in this domain. The backbone exists to be varied so that the
finding can be shown to be architecture-independent (ablation A8), and for no
other reason.

Four options, matching the Section 10 baseline table:

    resnet50     ImageNet          the field's default
    vit_b_16     ImageNet          different inductive bias; transformers may be
                                   more or less prone to text-like shortcuts
    usf_mae      public weights    pretrained on an annotation-inpainted corpus,
                                   so it is the natural test of whether
                                   annotation-free pretraining confers
                                   shortcut resistance
    fetalclip    public weights    current fetal state of the art
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..utils import get_logger

LOGGER = get_logger("asa.models")


class Backbone(nn.Module):
    """Wraps a feature extractor and exposes ``out_features``."""

    def __init__(self, module: nn.Module, out_features: int, name: str) -> None:
        super().__init__()
        self.module = module
        self.out_features = out_features
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.module(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return features


def _torchvision_backbone(cfg: ModelConfig) -> Backbone:
    import torchvision.models as tvm

    imagenet = cfg.pretrained == "imagenet"
    if cfg.backbone == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if imagenet else None
        model = tvm.resnet50(weights=weights)
        out_features = model.fc.in_features
        model.fc = nn.Identity()
    elif cfg.backbone == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if imagenet else None
        model = tvm.resnet18(weights=weights)
        out_features = model.fc.in_features
        model.fc = nn.Identity()
    elif cfg.backbone == "vit_b_16":
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if imagenet else None
        model = tvm.vit_b_16(weights=weights)
        out_features = model.heads.head.in_features
        model.heads.head = nn.Identity()
    else:
        raise ValueError(f"unknown torchvision backbone {cfg.backbone!r}")
    return Backbone(model, out_features, cfg.backbone)


def _load_foundation_weights(model: nn.Module, path: str | Path, name: str) -> None:
    """Load third-party encoder weights, reporting exactly what matched.

    Silent key mismatches are the classic way a "foundation model baseline"
    quietly becomes a randomly initialised one, which would make ablation A8
    meaningless. Anything less than a near-complete match is logged loudly.
    """
    state = torch.load(str(path), map_location="cpu")
    for key in ("model", "state_dict", "module"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    matched = len(state) - len(result.unexpected_keys)
    LOGGER.info(
        "%s: loaded %d/%d tensors (missing=%d unexpected=%d)",
        name, matched, len(state), len(result.missing_keys), len(result.unexpected_keys),
    )
    if matched < 0.5 * len(state):
        raise RuntimeError(
            f"{name}: only {matched}/{len(state)} tensors matched -- refusing to report this as a "
            "pretrained baseline. Check the checkpoint layout."
        )


def build_backbone(cfg: ModelConfig) -> Backbone:
    if cfg.pretrained in ("imagenet", "none"):
        return _torchvision_backbone(cfg)

    if cfg.pretrained == "usf_mae":
        # USF-MAE is a ViT-B/16 MAE encoder; load its weights into the
        # torchvision ViT so the rest of the pipeline is unchanged.
        vit_cfg = ModelConfig(**{**cfg.__dict__, "backbone": "vit_b_16", "pretrained": "none"})
        backbone = _torchvision_backbone(vit_cfg)
        if not cfg.weights_path:
            raise ValueError("model.weights_path is required for pretrained: usf_mae")
        _load_foundation_weights(backbone.module, cfg.weights_path, "USF-MAE")
        backbone.name = "usf_mae"
        return backbone

    if cfg.pretrained == "fetalclip":
        if not cfg.weights_path:
            raise ValueError("model.weights_path is required for pretrained: fetalclip")
        backbone = _fetalclip_backbone(cfg)
        backbone.name = "fetalclip"
        return backbone

    raise ValueError(f"unknown pretrained option {cfg.pretrained!r}")


def _fetalclip_backbone(cfg: ModelConfig) -> Backbone:
    """FetalCLIP image encoder (Maani et al. 2025, npj Digital Medicine).

    Released as an open_clip checkpoint. open_clip is an optional dependency:
    the audit runs without it and only ablation A8 loses one row.
    """
    try:
        import open_clip
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "FetalCLIP needs open_clip: pip install open_clip_torch. "
            "Weights: https://github.com/biomedia-mbzuai/fetalclip"
        ) from exc

    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained=str(cfg.weights_path))
    visual = model.visual
    out_features = getattr(visual, "output_dim", None) or getattr(visual, "width", 768)
    return Backbone(visual, int(out_features), "fetalclip")


class AuditModel(nn.Module):
    """Backbone + task head + optional auxiliary weak-supervision heads.

    Fusion is deliberately trivial (the protocol): the auxiliary head is a linear
    layer on the same pooled features. A complex fusion mechanism would make it
    impossible to attribute an effect to the supervision signal rather than to
    the mechanism.
    """

    def __init__(
        self,
        backbone: Backbone,
        num_classes: int,
        task: str = "classification",
        dropout: float = 0.0,
        aux_categorical: dict[str, int] | None = None,
        aux_numeric: list[str] | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.task = task
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        out_dim = 1 if task == "regression" else num_classes
        self.head = nn.Linear(backbone.out_features, out_dim)

        self.aux_categorical = nn.ModuleDict(
            {name: nn.Linear(backbone.out_features, size) for name, size in (aux_categorical or {}).items()}
        )
        self.aux_numeric = nn.ModuleDict(
            {name: nn.Linear(backbone.out_features, 1) for name in (aux_numeric or [])}
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.dropout(self.backbone(x))
        out: dict[str, torch.Tensor] = {"features": features, "logits": self.head(features)}
        for name, layer in self.aux_categorical.items():
            out[f"aux_{name}"] = layer(features)
        for name, layer in self.aux_numeric.items():
            out[f"aux_{name}"] = layer(features).squeeze(-1)
        return out

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    model_cfg: ModelConfig,
    num_classes: int,
    task: str,
    aux_categorical: dict[str, int] | None = None,
    aux_numeric: list[str] | None = None,
) -> AuditModel:
    backbone = build_backbone(model_cfg)
    return AuditModel(
        backbone=backbone,
        num_classes=num_classes,
        task=task,
        dropout=model_cfg.dropout,
        aux_categorical=aux_categorical,
        aux_numeric=aux_numeric,
        freeze_backbone=model_cfg.freeze_backbone,
    )
