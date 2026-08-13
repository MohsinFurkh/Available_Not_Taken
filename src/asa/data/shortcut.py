"""Synthetic shortcut injection for the positive control.

Deliberately free of any torch import. The injector is pure NumPy, and keeping
it that way means the invariants it has to satisfy -- label dependence,
determinism, area matching -- are testable on a machine with no GPU stack,
which is where they get run most often.
"""

from __future__ import annotations

import numpy as np


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
