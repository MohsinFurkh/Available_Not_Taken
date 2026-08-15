"""Burnt-in annotation detection.

Design commitment: **the annotation mask does not depend on OCR.**

Published removal pipelines use Tesseract, and so do we -- but only in ``asa.annot.parse``, to read the
*content* of the text for the weak-supervision arm. The mask itself is built
from geometry and morphology, because mask quality is
"the single largest threat to validity": if the mask came from OCR then every
headline number would be bounded by OCR recall, and the audit would be
measuring Tesseract rather than measuring the shortcut.

Everything is built on one primitive. Burnt-in graphics are *rendered*: they sit
on a bright, sharp-edged plateau against their immediate surround, and glyphs on
one text line share that plateau value. Ultrasound anatomy never does -- speckle
is textured at every scale. So the pipeline is

    top-hat candidates  ->  per-component descriptors (peak, ring contrast)
                        ->  classified into text lines / calipers / edge ticks

with chromatic overlays and curator redaction boxes detected separately.

Absolute intensity thresholds were deliberately avoided: measured across the
four study datasets the same glyphs run at 170 (FETAL_PLANES Voluson S10), 209
(BUS-BRA) and 49 (HC18), so any fixed cut-off silently drops a whole dataset.
Ring contrast is stable across all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..config import AnnotationConfig

MaskDict = dict[str, np.ndarray]


@dataclass
class Detection:
    """All masks are bool arrays of shape (H, W)."""

    fan: np.ndarray
    text: np.ndarray
    calipers: np.ndarray
    ticks: np.ndarray
    overlay: np.ndarray
    redaction: np.ndarray
    annotation: np.ndarray
    #: Chromatic regions judged to be colour Doppler rather than annotation.
    #: Imaging data, never inpainted, but reported because it is a distinct
    #: confound that this study does not address.
    doppler: np.ndarray | None = None
    features: dict[str, float] = field(default_factory=dict)

    @property
    def components(self) -> MaskDict:
        return {
            "text": self.text,
            "calipers": self.calipers,
            "ticks": self.ticks,
            "overlay": self.overlay,
        }

    @property
    def chrome(self) -> np.ndarray:
        """Annotation lying outside the imaging sector (vendor banner, ruler)."""
        return self.annotation & ~self.fan


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _disk(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, int(size)),) * 2)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _drop_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1 or not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return np.isin(labels, keep) if keep else np.zeros_like(mask)


# --------------------------------------------------------------------------
# 1. imaging sector ("fan")
# --------------------------------------------------------------------------


def detect_fan(gray: np.ndarray, cfg: AnnotationConfig) -> np.ndarray:
    """Convex hull of the tissue-bearing sector.

    Morphological opening deletes text and calipers (thin, isolated) while
    leaving tissue (large, connected), so the largest surviving component is the
    sector. The convex hull then fills anechoic pockets -- amniotic fluid is
    inside the sector even though it is black.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bright = (blurred > cfg.fan_threshold).astype(np.uint8)
    opened = cv2.morphologyEx(bright, cv2.MORPH_OPEN, _disk(cfg.fan_open_kernel))
    if opened.sum() == 0:
        return np.ones_like(gray, dtype=bool)

    sector = _largest_component(opened)
    contours, _ = cv2.findContours(sector.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return sector
    hull = cv2.convexHull(np.vstack(contours))
    filled = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    return filled.astype(bool)


# --------------------------------------------------------------------------
# 2. rendered-graphic candidates
# --------------------------------------------------------------------------


@dataclass
class _Blob:
    index: int
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float
    peak: float
    contrast: float
    fill: float
    sharpness: float


def _graphic_blobs(gray: np.ndarray, cfg: AnnotationConfig) -> tuple[np.ndarray, np.ndarray, list[_Blob]]:
    """Top-hat candidates with per-component descriptors."""
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, _disk(cfg.text_tophat_kernel))
    tophat = cv2.subtract(gray, opened)
    candidate = ((tophat >= cfg.graphic_min_tophat) & (gray >= cfg.graphic_min_intensity)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    inner = cv2.dilate(candidate, np.ones((3, 3), np.uint8)).astype(bool)
    outer = cv2.dilate(candidate, np.ones((9, 9), np.uint8)).astype(bool)
    surround = outer & ~inner
    gray_f = gray.astype(np.float32)

    # Edge sharpness. A rendered glyph has a hard one-pixel transition; speckle
    # is band-limited by the transducer point-spread function and cannot. Scaled
    # by the blob's own contrast so the measure is brightness-invariant, which
    # matters because the same glyph runs at 49 in HC18 and 209 in BUS-BRA.
    # Measured separation on FETAL_PLANES: 4.9 for burnt-in text against 1.8 for
    # bright anatomy, with no overlap between the 10th and 90th percentiles.
    gradient = np.hypot(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    )
    eroded = cv2.erode(candidate, np.ones((3, 3), np.uint8)).astype(bool)
    boundary = candidate.astype(bool) & ~eroded

    blobs: list[_Blob] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < cfg.text_min_area or area > cfg.text_max_area:
            continue
        if h < 3 or h > gray.shape[0] * 0.12 or w > gray.shape[1] * 0.55:
            continue
        pad = 6
        y0, y1 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        window = labels[y0:y1, x0:x1] == i
        ring = surround[y0:y1, x0:x1] & ~window
        patch = gray_f[y0:y1, x0:x1]
        if ring.sum() < 4:
            continue
        values = patch[window]
        contrast = float(values.mean() - patch[ring].mean())
        edge = boundary[y0:y1, x0:x1] & window
        sharpness = float(gradient[y0:y1, x0:x1][edge].mean() / max(contrast, 1.0)) if edge.any() else 0.0
        blobs.append(
            _Blob(
                index=i,
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                cx=float(centroids[i][0]),
                cy=float(centroids[i][1]),
                peak=float(np.percentile(values, 90)),
                contrast=contrast,
                fill=area / float(w * h),
                sharpness=sharpness,
            )
        )
    return candidate.astype(bool), labels, blobs


# --------------------------------------------------------------------------
# 3. text lines
# --------------------------------------------------------------------------


def _group_lines(blobs: list[_Blob]) -> list[list[_Blob]]:
    """Single-link grouping on a shared baseline, comparable height and
    horizontal proximity. An isolated bright blob is anatomy until proven
    otherwise; a row of similar blobs at one baseline is text."""
    remaining = list(blobs)
    groups: list[list[_Blob]] = []
    while remaining:
        group = [remaining.pop()]
        changed = True
        while changed:
            changed = False
            for candidate in list(remaining):
                for member in group:
                    scale = max(member.h, candidate.h)
                    if abs(member.cy - candidate.cy) > 0.5 * scale:
                        continue
                    if not (0.55 <= member.h / max(candidate.h, 1) <= 1.8):
                        continue
                    if abs(member.cx - candidate.cx) > 4.0 * scale:
                        continue
                    group.append(candidate)
                    remaining.remove(candidate)
                    changed = True
                    break
        groups.append(group)
    return groups


def detect_text(labels: np.ndarray, blobs: list[_Blob], cfg: AnnotationConfig) -> np.ndarray:
    mask = np.zeros(labels.shape, dtype=bool)
    for group in _group_lines(blobs):
        contrasts = np.array([b.contrast for b in group])
        if np.median(contrasts) < cfg.text_min_contrast:
            continue
        # Median rather than mean: one genuinely sharp glyph should not carry a
        # group of speckle, and one blurred glyph should not sink a real line.
        if float(np.median([b.sharpness for b in group])) < cfg.graphic_min_sharpness:
            continue

        if len(group) >= cfg.text_min_line_members + 1:
            peaks = np.array([b.peak for b in group])
            if peaks.std() > cfg.text_max_peak_spread:
                continue
        else:
            # Small or low-resolution renders merge whole words into one blob
            # ("MAMA ESQUERDA" in BUS-BRA is two components, not eleven). Accept
            # those on shape instead: a word is a wide, thin, sparse box.
            x0 = min(b.x for b in group)
            x1 = max(b.x + b.w for b in group)
            y0 = min(b.y for b in group)
            y1 = max(b.y + b.h for b in group)
            width, height = x1 - x0, max(1, y1 - y0)
            if width / height < cfg.text_min_word_aspect:
                continue
            if max(b.fill for b in group) > cfg.text_max_word_fill:
                continue
        for b in group:
            mask |= labels == b.index
    return mask


# --------------------------------------------------------------------------
# 4. calipers
# --------------------------------------------------------------------------


def detect_calipers(labels: np.ndarray, blobs: list[_Blob], cfg: AnnotationConfig) -> np.ndarray:
    """Operator-placed cross / plus glyphs.

    Calipers are the causally interesting annotation: unlike a vendor banner
    they are placed *by the operator, on the finding, at the moment of
    diagnosis* (see the released protocol), so their position is downstream of the
    label rather than incidental to it.
    """
    mask = np.zeros(labels.shape, dtype=bool)
    for b in blobs:
        if not (cfg.caliper_min_area <= b.area <= cfg.caliper_max_area):
            continue
        if b.w < 3 or b.h < 3:
            continue
        if max(b.w / b.h, b.h / b.w) > cfg.caliper_max_aspect:
            continue
        if b.fill > cfg.caliper_max_fill:
            continue
        if b.contrast < cfg.caliper_min_contrast:
            continue
        if b.sharpness < cfg.graphic_min_sharpness:
            continue
        patch = labels[b.y : b.y + b.h, b.x : b.x + b.w] == b.index
        if _is_cross(patch):
            mask |= labels == b.index
    return mask


def _is_cross(patch: np.ndarray) -> bool:
    """A '+' or 'x' concentrates mass on the centre row *and* centre column (or
    on both diagonals) and leaves its corners empty."""
    h, w = patch.shape
    if h < 3 or w < 3:
        return False
    cy, cx = h // 2, w // 2
    band = max(1, min(h, w) // 6)
    row_hit = patch[max(0, cy - band) : cy + band + 1, :].any(axis=0).mean()
    col_hit = patch[:, max(0, cx - band) : cx + band + 1].any(axis=1).mean()
    plus_like = row_hit > 0.55 and col_hit > 0.55

    side = min(h, w)
    square = patch[:side, :side]
    diag = np.eye(side, dtype=bool)
    x_like = (square & diag).mean() > 0.3 and (square & diag[::-1]).mean() > 0.3

    corner = max(1, side // 4)
    corners = (
        patch[:corner, :corner].mean()
        + patch[:corner, -corner:].mean()
        + patch[-corner:, :corner].mean()
        + patch[-corner:, -corner:].mean()
    )
    return (plus_like or x_like) and corners < 1.2


# --------------------------------------------------------------------------
# 5. depth-ruler ticks along the frame edge
# --------------------------------------------------------------------------


def detect_ticks(labels: np.ndarray, blobs: list[_Blob], shape: tuple[int, int], cfg: AnnotationConfig) -> np.ndarray:
    """Regularly spaced marks in the outer edge band.

    The depth ruler is annotation that encodes the acquisition depth setting --
    i.e. a vendor/preset fingerprint -- but it is neither text-like nor
    cross-like, so it needs its own rule: small isolated blobs, hugging one
    vertical edge, at regular vertical spacing.
    """
    h, w = shape
    band = max(4, int(cfg.tick_edge_band_frac * w))
    mask = np.zeros(shape, dtype=bool)
    for side_blobs in (
        [b for b in blobs if b.x + b.w <= band],
        [b for b in blobs if b.x >= w - band],
    ):
        side_blobs = [b for b in side_blobs if b.area <= cfg.tick_max_area and b.contrast >= cfg.tick_min_contrast]
        if len(side_blobs) < cfg.tick_min_members:
            continue
        ys = np.sort(np.array([b.cy for b in side_blobs]))
        gaps = np.diff(ys)
        if len(gaps) < 2:
            continue
        # regular spacing: the ruler is periodic, speckle is not
        if gaps.std() > cfg.tick_max_spacing_cv * max(gaps.mean(), 1e-6):
            continue
        for b in side_blobs:
            mask |= labels == b.index
    return mask


# --------------------------------------------------------------------------
# 6. chromatic overlays
# --------------------------------------------------------------------------


def detect_overlay(image: np.ndarray, cfg: AnnotationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Any saturated pixel in a B-mode frame is an operator overlay.

    B-mode is achromatic by construction, so this detector has near-perfect
    precision. It is what catches MMOTU's yellow calipers and dotted measurement
    line, and the blue vendor text.

    Optionally routed through CLAHE + HSV K-means to follow P8's published
    pipeline exactly; the direct saturation threshold is the default because it
    is deterministic, and K-means would put a per-image random seed underneath
    every headline number.
    """
    if image.ndim == 2:
        empty = np.zeros(image.shape, dtype=bool)
        return empty, empty
    bgr = to_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if cfg.overlay_use_kmeans:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv = hsv.copy()
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        samples = hsv.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(samples, cfg.overlay_kmeans_k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        clusters = [k for k, c in enumerate(centers) if c[1] >= cfg.overlay_min_saturation]
        mask = np.isin(labels.reshape(image.shape[:2]), clusters)
        return _reject_doppler(mask & (hsv[:, :, 2] >= cfg.overlay_min_value), cfg)

    chromatic = (hsv[:, :, 1] >= cfg.overlay_min_saturation) & (hsv[:, :, 2] >= cfg.overlay_min_value)
    return _reject_doppler(chromatic, cfg)


def _reject_doppler(chromatic: np.ndarray, cfg: AnnotationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Split chromatic pixels into operator graphics and colour Doppler.

    Colour Doppler is chromatic but it is *imaging data*, not annotation:
    inpainting it would destroy the measurement the sonographer took, and
    counting it as annotation would report flow mapping as a shortcut. It is
    separated on size -- Doppler paints a large contiguous region, while a
    caliper glyph or a dotted measurement line is small or thin.

    Measured on MMOTU-2D, where 9% of images exceeded a 5% mask area before this
    filter existed; almost all of those were colour flow boxes.

    Returns (annotation overlay, rejected Doppler-like region).
    """
    if not chromatic.any():
        empty = np.zeros_like(chromatic)
        return chromatic, empty

    n, labels, stats, _ = cv2.connectedComponentsWithStats(chromatic.astype(np.uint8), connectivity=8)
    max_area = cfg.overlay_max_component_area_frac * chromatic.size
    keep, rejected = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        # A dotted measurement line is wide but nearly empty inside its bounding
        # box; a Doppler box is both large and dense. Keep thin structures at
        # any size, reject dense ones above the area bound.
        density = area / float(max(w * h, 1))
        if area > max_area and density > cfg.overlay_max_component_density:
            rejected.append(i)
        else:
            keep.append(i)
    return (
        np.isin(labels, keep) if keep else np.zeros_like(chromatic),
        np.isin(labels, rejected) if rejected else np.zeros_like(chromatic),
    )


# --------------------------------------------------------------------------
# 7. curator redaction rectangles
# --------------------------------------------------------------------------


def detect_redaction(gray: np.ndarray, fan: np.ndarray, cfg: AnnotationConfig) -> np.ndarray:
    """Solid uniform rectangles pasted over identifiers by dataset curators.

    HC18 ships with a black box over the bottom-right corner. That box is
    annotation removal *already performed by the curator*, undocumented and
    unmeasured -- precisely the "cleaning is undocumented and its effect
    unmeasured" problem named in the protocol, and a candidate second shortcut in
    its own right. It is detected separately so it can be reported
    rather than silently absorbed into either anatomy or annotation.

    ``fan`` is accepted but unused: the box is often pasted over the corner
    *outside* the sector, so requiring sector overlap would miss exactly the
    case this detector exists for. Rectangularity plus an upper area bound is
    what keeps ordinary black background out.
    """
    flat = (gray <= cfg.redaction_max_intensity).astype(np.uint8)
    flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN, _disk(5))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(flat, connectivity=8)
    out = np.zeros(gray.shape, dtype=bool)
    min_area = cfg.redaction_min_area_frac * gray.size
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area or w < 10 or h < 10:
            continue
        if area / float(w * h) < cfg.redaction_min_rectangularity:
            continue
        if area > cfg.redaction_max_area_frac * gray.size:
            continue  # the whole background, not a pasted box
        out |= labels == i
    return out


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def detect(image: np.ndarray, cfg: AnnotationConfig | None = None) -> Detection:
    """Run all detectors on a BGR / BGRA / grayscale uint8 image."""
    cfg = cfg or AnnotationConfig()
    gray = to_gray(image)

    fan = detect_fan(gray, cfg)
    _, labels, blobs = _graphic_blobs(gray, cfg)

    text = detect_text(labels, blobs, cfg)
    calipers = detect_calipers(labels, blobs, cfg)
    ticks = detect_ticks(labels, blobs, gray.shape, cfg)
    overlay, doppler = detect_overlay(image, cfg)
    redaction = detect_redaction(gray, fan, cfg)

    union = _drop_small(text | calipers | ticks | overlay, cfg.min_component_area)
    if cfg.dilate_px > 0:
        union = cv2.dilate(union.astype(np.uint8), _disk(cfg.dilate_px)).astype(bool)
    # Dilation must not eat the redaction box into the annotation mask; it is a
    # separate phenomenon with its own reporting line.
    annotation = union & ~redaction

    det = Detection(
        fan=fan,
        text=text,
        calipers=calipers,
        ticks=ticks,
        overlay=overlay,
        redaction=redaction,
        annotation=annotation,
        doppler=doppler,
    )
    det.features = geometry_features(det, gray)
    return det


# --------------------------------------------------------------------------
# geometry features
# --------------------------------------------------------------------------


def geometry_features(det: Detection, gray: np.ndarray) -> dict[str, float]:
    """Hand-crafted descriptors of the annotations, containing no anatomy.

    These feed two things: the Section 10 annotation-geometry baseline (if a
    logistic regression on these matches the CNN's A-only score, the shortcut is
    fully explained without any deep learning at all), and the H4
    annotation-label mutual-information analysis.
    """
    h, w = gray.shape
    area = float(h * w)
    feats: dict[str, float] = {
        "image_h": float(h),
        "image_w": float(w),
        "aspect": float(w) / h,
        "fan_area_frac": float(det.fan.sum()) / area,
    }

    for name, mask in det.components.items():
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        feats[f"{name}_area_frac"] = float(mask.sum()) / area
        feats[f"{name}_n_components"] = float(max(0, n - 1))
        if n > 1:
            cs = centroids[1:]
            weights = stats[1:, cv2.CC_STAT_AREA].astype(float)
            feats[f"{name}_cx"] = float((cs[:, 0] * weights).sum() / weights.sum() / w)
            feats[f"{name}_cy"] = float((cs[:, 1] * weights).sum() / weights.sum() / h)
            feats[f"{name}_spread_x"] = float(cs[:, 0].std() / w)
            feats[f"{name}_spread_y"] = float(cs[:, 1].std() / h)
            feats[f"{name}_mean_comp_area"] = float(weights.mean() / area)
        else:
            for suffix in ("cx", "cy", "spread_x", "spread_y", "mean_comp_area"):
                feats[f"{name}_{suffix}"] = 0.0

    # Caliper pair geometry: the distance between two calipers *is* the
    # measurement, so it is the most label-predictive annotation feature
    # obtainable without reading a single character of text.
    n, _, _, centroids = cv2.connectedComponentsWithStats(det.calipers.astype(np.uint8), connectivity=8)
    if n >= 3:
        cs = centroids[1:]
        dists = np.linalg.norm(cs[:, None, :] - cs[None, :, :], axis=-1)
        upper = dists[np.triu_indices_from(dists, k=1)]
        diagonal = float(np.hypot(h, w))
        feats["caliper_max_sep"] = float(upper.max()) / diagonal
        feats["caliper_min_sep"] = float(upper.min()) / diagonal
        feats["caliper_mean_sep"] = float(upper.mean()) / diagonal
    else:
        feats["caliper_max_sep"] = feats["caliper_min_sep"] = feats["caliper_mean_sep"] = 0.0

    annotation = det.annotation
    feats["annotation_area_frac"] = float(annotation.sum()) / area
    feats["annotation_in_fan_frac"] = float((annotation & det.fan).sum()) / max(1.0, float(annotation.sum()))
    feats["chrome_area_frac"] = float(det.chrome.sum()) / area
    feats["chrome_mean_intensity"] = float(gray[det.chrome].mean()) if det.chrome.any() else 0.0
    feats["redaction_area_frac"] = float(det.redaction.sum()) / area
    feats["doppler_area_frac"] = float(det.doppler.sum()) / area if det.doppler is not None else 0.0
    # Masks this large are almost always a detector failure rather than a
    # heavily annotated frame. Flagged rather than silently used: mask quality
    # names mask quality as the study's largest threat to validity, and the A7
    # ablation re-runs every headline number on high-confidence masks only.
    feats["mask_suspect"] = float(feats["annotation_area_frac"] > 0.15)
    return feats


def masks_to_uint8(det: Detection) -> dict[str, np.ndarray]:
    """Pack masks for on-disk storage as single-channel PNGs."""
    payload = {"fan": det.fan, "annotation": det.annotation, "redaction": det.redaction, **det.components}
    return {name: (mask.astype(np.uint8) * 255) for name, mask in payload.items()}


def summarise(det: Detection) -> dict[str, Any]:
    f = det.features
    return {
        "annotation_area_frac": f["annotation_area_frac"],
        "n_text_components": f["text_n_components"],
        "n_caliper_components": f["calipers_n_components"],
        "n_tick_components": f["ticks_n_components"],
        "overlay_area_frac": f["overlay_area_frac"],
        "redaction_area_frac": f["redaction_area_frac"],
    }


def overlay_visualisation(image: np.ndarray, det: Detection) -> np.ndarray:
    """Colour-coded mask overlay for the human verification sample."""
    vis = to_bgr(image).copy()
    for mask, colour in (
        (det.text, (0, 0, 255)),
        (det.calipers, (0, 255, 255)),
        (det.ticks, (255, 128, 0)),
        (det.overlay, (0, 255, 0)),
        (det.redaction, (255, 0, 255)),
        (det.doppler if det.doppler is not None else np.zeros_like(det.fan), (255, 255, 0)),
    ):
        if mask.any():
            vis[mask] = colour
    contours, _ = cv2.findContours(det.fan.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (128, 128, 128), 1)
    return vis
