"""Saliency mass inside verified annotation masks (proposal Section C.10).

Saliency alone is weak evidence, and the proposal says so. Its role here is
triangulation: an accuracy drop under A-removed *plus* above-chance A-only
performance *plus* a large share of attribution mass sitting inside the
annotation regions is strong evidence, and any one of the three alone is not.

Grad-CAM for convolutional backbones, attention rollout for ViTs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..models.backbones import AuditModel


class GradCAM:
    """Grad-CAM on the last convolutional stage."""

    def __init__(self, model: AuditModel, layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        layer.register_forward_hook(self._forward_hook)
        layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, images: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)["logits"]
        if targets is None:
            targets = logits.argmax(dim=1)
        selected = logits.gather(1, targets.view(-1, 1)).sum()
        selected.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        # Normalise per image so the mass fraction is comparable across images.
        flat = flat - flat.min(dim=1, keepdim=True).values
        denominator = flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (flat / denominator).view_as(cam)


def _last_conv_layer(model: AuditModel) -> torch.nn.Module | None:
    last = None
    for module in model.backbone.module.modules():
        if isinstance(module, torch.nn.Conv2d):
            last = module
    return last


def saliency_mass_in_mask(
    model: AuditModel,
    loader,
    annotation_masks: np.ndarray,
    device: torch.device,
    max_images: int = 512,
) -> dict[str, Any]:
    """Fraction of normalised attribution mass falling inside the annotation mask.

    The comparison that makes the number meaningful is the mask's own area
    fraction: if annotations cover 1% of the pixels and attract 1% of the
    attribution, nothing is happening. A ratio well above 1 is the claim.
    """
    layer = _last_conv_layer(model)
    if layer is None:
        return {"supported": False, "reason": "no Conv2d layer; use attention rollout for ViT backbones"}

    cam = GradCAM(model, layer)
    model.eval()
    mass_fractions: list[float] = []
    area_fractions: list[float] = []
    seen = 0

    for batch in loader:
        images = batch["image"].to(device)
        maps = cam(images)
        rows = batch["row"].numpy()
        for i, row in enumerate(rows):
            mask = torch.from_numpy((np.asarray(annotation_masks[row]) > 127).astype(np.float32))
            if mask.shape != maps.shape[-2:]:
                mask = F.interpolate(mask[None, None], size=maps.shape[-2:], mode="nearest")[0, 0]
            mask = mask.to(maps.device)
            mass_fractions.append(float((maps[i] * mask).sum()))
            area_fractions.append(float(mask.mean()))
            seen += 1
        if seen >= max_images:
            break

    if not mass_fractions:
        return {"supported": False, "reason": "no images scored"}

    mass = float(np.mean(mass_fractions))
    area = float(np.mean(area_fractions))
    return {
        "supported": True,
        "n_images": seen,
        "mean_saliency_mass_in_annotation": mass,
        "mean_annotation_area_fraction": area,
        # > 1 means attribution concentrates on annotations beyond their share
        # of the image; = 1 means it is exactly what area alone would give.
        "enrichment_ratio": float(mass / area) if area > 1e-8 else float("nan"),
    }
