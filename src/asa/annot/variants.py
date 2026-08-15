"""Construction of the annotation triplet, plus the equal-area control.

    A-full       original image, annotations intact          (what archives contain)
    A-removed    annotations detected and inpainted          (what deployment looks like)
    A-only       annotation regions kept, anatomy destroyed  (the decisive condition)
    ctrl-random  equal-area random regions inpainted instead (the pixel-removal objection)

A-only is what makes the audit an audit. A drop under A-removed is ambiguous --
"you removed pixels and performance fell, so what?" -- but above-chance
performance under A-only is not ambiguous at all. And ctrl-random is what
separates "annotation removal hurt" from "inpainting anything hurts": if
performance falls just as far when an equal area of *anatomy* is inpainted, the
hypothesis is refuted, and that is pre-registered as the refutation criterion.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import AnnotationConfig
from .detect import Detection, to_bgr, to_gray

_CV_INPAINT = {"navier_stokes": cv2.INPAINT_NS, "telea": cv2.INPAINT_TELEA}


def _speckle_like(reference: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noise matched to the first two moments of the reference region.

    Used as a fill that is neither informative nor conspicuously smooth. A flat
    fill is itself a detectable cue -- the second shortcut that inpainting can
    create -- so the noise fill exists as the comparison
    arm for that ablation.
    """
    source = reference[mask] if mask.any() else reference.ravel()
    if source.size == 0:
        return np.zeros_like(reference)
    noise = rng.normal(float(source.mean()), max(float(source.std()), 1.0), reference.shape)
    return np.clip(noise, 0, 255).astype(np.uint8)


def inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: AnnotationConfig,
    method: str | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Fill ``mask`` in ``image``. Four methods, all reported side by side.

    The four exist because inpainting artefacts are themselves a candidate
    shortcut: Navier-Stokes leaves smooth scars that a
    network can read as "an annotation used to be here". Comparing NS against
    Telea, a uniform fill and a speckle-matched fill bounds how much of any
    measured drop is an artefact of the removal rather than of the annotation.
    """
    method = method or cfg.inpaint_method
    rng = rng or np.random.default_rng(0)
    if not mask.any():
        return image.copy()

    mask_u8 = mask.astype(np.uint8)
    if method in _CV_INPAINT:
        source = to_bgr(image) if image.ndim == 3 else image
        filled = cv2.inpaint(source, mask_u8, cfg.inpaint_radius, _CV_INPAINT[method])
        return filled

    out = image.copy()
    if method == "uniform":
        gray = to_gray(image)
        value = int(np.median(gray[~mask])) if (~mask).any() else 0
        out[mask] = value
    elif method == "noise":
        gray = to_gray(image)
        noise = _speckle_like(gray, ~mask, rng)
        if out.ndim == 3:
            out[mask] = noise[mask][:, None]
        else:
            out[mask] = noise[mask]
    else:
        raise ValueError(f"unknown inpaint method {method!r}")
    return out


def make_a_removed(image: np.ndarray, det: Detection, cfg: AnnotationConfig, rng=None) -> np.ndarray:
    return inpaint(image, det.annotation, cfg, rng=rng)


def make_a_only(image: np.ndarray, det: Detection, cfg: AnnotationConfig, rng=None) -> np.ndarray:
    """Keep annotation pixels; destroy everything else.

    The fill defaults to speckle-matched noise rather than black, because a
    black background would let a network locate the annotations from the *shape
    of the hole* even if it could not read them, which would inflate A-only
    performance and overstate the shortcut.
    """
    rng = rng or np.random.default_rng(0)
    keep = det.annotation
    gray = to_gray(image)

    if cfg.a_only_fill == "black":
        background = np.zeros_like(image)
    elif cfg.a_only_fill == "uniform":
        value = int(np.median(gray[det.fan])) if det.fan.any() else 0
        background = np.full_like(image, value)
    else:
        noise = _speckle_like(gray, det.fan & ~keep, rng)
        background = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR) if image.ndim == 3 else noise

    out = background.copy()
    out[keep] = image[keep]
    return out


def make_control(
    image: np.ndarray,
    det: Detection,
    cfg: AnnotationConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inpaint an equal-area random region of *anatomy* instead of the annotations.

    "You removed pixels and performance fell -- so what?" This is the control that
    answers that objection. Each annotation component is translated to a
    random position inside the imaging sector, preserving the component's exact
    shape and area, so the control differs from A-removed in *location only*.

    Returns the control image and the control mask.
    """
    rng = rng or np.random.default_rng(0)
    h, w = det.annotation.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(det.annotation.astype(np.uint8), connectivity=8)
    control_mask = np.zeros((h, w), dtype=bool)
    # Keep the displaced components clear of the real annotations, so that the
    # control removes anatomy and only anatomy.
    forbidden = cv2.dilate(det.annotation.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    interior = det.fan & ~forbidden

    for i in range(1, n):
        x, y, cw, ch = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                   cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        shape = labels[y : y + ch, x : x + cw] == i
        if cw >= w or ch >= h:
            continue
        for _ in range(cfg.ctrl_max_placement_tries):
            ny = int(rng.integers(0, h - ch)) if h > ch else 0
            nx = int(rng.integers(0, w - cw)) if w > cw else 0
            window = interior[ny : ny + ch, nx : nx + cw]
            if window.shape != shape.shape:
                continue
            # require the displaced component to land inside the sector
            if (shape & ~window).sum() > 0.1 * shape.sum():
                continue
            control_mask[ny : ny + ch, nx : nx + cw] |= shape
            break

    return inpaint(image, control_mask, cfg, rng=rng), control_mask


def build_variants(
    image: np.ndarray,
    det: Detection,
    cfg: AnnotationConfig,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """All four variants for one image, from one detection."""
    rng = np.random.default_rng(seed)
    control, _ = make_control(image, det, cfg, rng)
    return {
        "a_full": image.copy(),
        "a_removed": make_a_removed(image, det, cfg, rng),
        "a_only": make_a_only(image, det, cfg, rng),
        "ctrl_random": control,
    }


def randomise_annotations(
    image: np.ndarray,
    annotation_mask: np.ndarray,
    donor_image: np.ndarray | None,
    donor_mask: np.ndarray | None,
    cfg: AnnotationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mitigation: strip this image's annotations, optionally pasting
    a different image's annotations in their place.

    This is the cheapest plausible fix for the shortcut, and It must be tested *before* anything more elaborate is proposed: if it works, it is the paper's practical recommendation and worth
    more than a novel method.
    """
    out = inpaint(image, annotation_mask, cfg, rng=rng)
    if donor_image is None or donor_mask is None or not donor_mask.any():
        return out
    h, w = out.shape[:2]
    dh, dw = donor_mask.shape
    if (dh, dw) != (h, w):
        donor_image = cv2.resize(donor_image, (w, h), interpolation=cv2.INTER_LINEAR)
        donor_mask = cv2.resize(donor_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if out.ndim == 3 and donor_image.ndim == 2:
        donor_image = cv2.cvtColor(donor_image, cv2.COLOR_GRAY2BGR)
    elif out.ndim == 2 and donor_image.ndim == 3:
        donor_image = to_gray(donor_image)
    out[donor_mask] = donor_image[donor_mask]
    return out
