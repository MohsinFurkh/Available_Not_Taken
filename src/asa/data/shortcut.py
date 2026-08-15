"""Synthetic shortcut injection for the positive control.

Deliberately free of any torch import. The injector is pure NumPy, and keeping
it that way means the invariants it has to satisfy -- label dependence,
determinism, area matching -- are testable on a machine with no GPU stack,
which is where they get run most often.
"""

from __future__ import annotations

import numpy as np


#: Vendor-like strings, one per class index. Chosen to resemble the burnt-in
#: preset and measurement text actually observed in these datasets, so that the
#: glyph control is matched to real annotations on the property that governs
#: learnability -- what a CNN must read to recover the label -- rather than only
#: on area and intensity.
GLYPH_VOCAB = [
    "Cardiac", "Har-mid", "OB Rout", "C7/M7", "Gn -6", "FF2/E2",
    "SRI II 3", "AO 97%", "22Hz", "14.0cm", "MI 0.9", "TIb 0.4",
]


def inject_synthetic_shortcut(
    image: np.ndarray,
    label: int,
    num_classes: int,
    cfg,
    row: int,
) -> np.ndarray:
    """Plant a marker whose *position* encodes the class.

    Position rather than shape, because that is how operator annotations carry
    meaning: a caliper says what it says by sitting on the finding. The marker
    matches the real annotations in area and intensity so the control measures
    sensitivity at the scale that actually occurs, not at an easier one.

    Deterministic in ``row``, so an image gets the same marker every epoch and
    in every variant -- otherwise the model would face a moving target and the
    control would understate what is learnable.
    """
    if num_classes <= 1:
        return image
    rng = np.random.default_rng(cfg.seed * 1_000_003 + int(row))

    # With probability (1 - strength) the marker encodes a random class instead
    # of the true one, which is what makes strength a correlation dial rather
    # than an on/off switch.
    encoded = int(label) if rng.random() < cfg.strength else int(rng.integers(num_classes))

    height, width = image.shape[:2]
    side = max(3, int(round(np.sqrt(cfg.area_frac * height * width))))
    cell = int(encoded % (cfg.grid * cfg.grid))
    row_cell, col_cell = divmod(cell, cfg.grid)
    cell_h, cell_w = height // cfg.grid, width // cfg.grid
    y0 = row_cell * cell_h + max(0, (cell_h - side) // 2)
    x0 = col_cell * cell_w + max(0, (cell_w - side) // 2)
    y1, x1 = min(height, y0 + side), min(width, x0 + side)

    out = image.copy()
    out[y0:y1, x0:x1] = cfg.intensity
    return out


def inject_glyph_shortcut(
    image: np.ndarray,
    label: int,
    num_classes: int,
    cfg,
    row: int,
) -> np.ndarray:
    """A learnability-matched control: a rendered glyph string whose *content*
    encodes the class.

    The rectangle control in ``inject_synthetic_shortcut`` matches real
    annotations on area and intensity, which are not the properties that decide
    whether a CNN can learn a cue. It is a solid block at one of K canonical
    positions, identical in every image -- far easier to read than a caliper or
    a burnt-in word. That makes its sweep an *upper* bound on the design's
    sensitivity, not the lower bound a calibration needs.

    This control closes that gap. The cue is a short vendor-like string drawn in
    a small font; the class is carried by *which string* appears, not by where
    it sits. Position is scattered per image, the font scale jitters, and a
    fraction of the glyphs are occluded -- the three properties the discussion
    previously offered as an untested explanation of the null.

    Reported alongside the rectangle curve rather than replacing it: together
    they bracket the range of cue difficulty, and the real annotations can be
    placed against the harder of the two.
    """
    import cv2

    if num_classes <= 1:
        return image
    rng = np.random.default_rng(cfg.seed * 7_654_321 + int(row))
    encoded = int(label) if rng.random() < cfg.strength else int(rng.integers(num_classes))
    text = GLYPH_VOCAB[encoded % len(GLYPH_VOCAB)]

    out = image.copy()
    height, width = out.shape[:2]
    # Scale the font so the rendered string covers roughly the configured area,
    # keeping the two controls comparable in ink even though they differ in
    # difficulty.
    scale = float(np.clip(np.sqrt(cfg.area_frac * height * width) / 42.0, 0.28, 1.1))
    scale *= float(rng.uniform(0.85, 1.15))
    thickness = 1 if scale < 0.7 else 2
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

    # Operator-scattered placement, anywhere the string fits.
    x = int(rng.integers(2, max(3, width - text_w - 2)))
    y = int(rng.integers(text_h + 2, max(text_h + 3, height - 2)))

    colour = int(cfg.intensity) if out.ndim == 2 else (int(cfg.intensity),) * out.shape[2]
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)

    # Partial occlusion on a fraction of frames, as happens when anatomy or a
    # later overlay is drawn across burnt-in text.
    if rng.random() < getattr(cfg, "occlusion_prob", 0.25):
        cut = int(text_w * rng.uniform(0.2, 0.45))
        x0 = x if rng.random() < 0.5 else max(x, x + text_w - cut)
        patch = out[max(0, y - text_h) : y + 2, x0 : min(width, x0 + cut)]
        if patch.size:
            patch[...] = image[max(0, y - text_h) : y + 2, x0 : min(width, x0 + cut)]
    return out
