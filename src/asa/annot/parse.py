"""Reading the *content* of burnt-in annotations, for the weak-supervision arm.

This is the only place OCR is used, and it is deliberately isolated: if
Tesseract is unavailable the audit (H1, H2) still runs in full, and only the
supervision arm (H3) degrades. The proposal is explicit that the paper's weight
sits on the audit, so the dependency structure of the code should match.

Nothing here claims novelty. Extraction of structured content from sonographer
annotations is BUSClean's contribution (Bunnell et al. 2024, PLOS ONE); this
module reproduces it for the fetal/ovarian datasets BUSClean does not cover, and
the proposal commits to reporting extraction performance as a reproduction
rather than as a result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from ..config import AnnotationConfig
from .detect import Detection, to_gray


@lru_cache(maxsize=1)
def _tesseract():
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return pytesseract
    except Exception:
        return None


def ocr_available() -> bool:
    return _tesseract() is not None


@dataclass
class ParsedAnnotation:
    """Structured content recovered from one image's burnt-in text."""

    text: str = ""
    tokens: list[str] = field(default_factory=list)
    numeric: dict[str, float] = field(default_factory=dict)
    categorical: dict[str, str] = field(default_factory=dict)
    n_calipers: int = 0
    #: Caliper separation in image diagonals; multiplied by a known pixel size
    #: this is the measurement itself.
    caliper_separation: float = 0.0
    ocr_confidence: float = 0.0

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "ocr_text": self.text,
            "ocr_confidence": self.ocr_confidence,
            "n_calipers": self.n_calipers,
            "caliper_separation": self.caliper_separation,
        }
        row.update({f"num_{k}": v for k, v in self.numeric.items()})
        row.update({f"cat_{k}": v for k, v in self.categorical.items()})
        return row


# --------------------------------------------------------------------------
# regexes over vendor overlay conventions
# --------------------------------------------------------------------------

# Patterns observed directly in the four study datasets, e.g.
#   "22Hz/14.0cm"  "Gn  5"  "AO 97%"  "C7/M7"  "MI 0.9"  "SRI II 3/CRI 3"
#   "Routine 2 Trim./OB"  "Cardiac"  "Har-mid"  "MAMA ESQUERDA"  "QSE"
NUMERIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "depth_cm": re.compile(r"(\d{1,2}(?:\.\d)?)\s*cm\b", re.I),
    "frame_rate_hz": re.compile(r"(\d{1,3})\s*Hz\b", re.I),
    "gain": re.compile(r"\bGn\s*(-?\d{1,3})\b", re.I),
    "acoustic_output_pct": re.compile(r"\bAO\s*(\d{1,3})\s*%", re.I),
    "mechanical_index": re.compile(r"\bMI\s*[:=]?\s*(\d\.\d{1,2})\b", re.I),
    "thermal_index": re.compile(r"\bTI[bsc]?\s*[:=]?\s*(\d\.\d{1,2})\b", re.I),
    "measurement_mm": re.compile(r"(\d{1,3}(?:\.\d{1,2})?)\s*mm\b", re.I),
}

CATEGORICAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "probe": re.compile(r"\b([CLIE]\d{1,2}\s*/\s*M\d{1,2})\b", re.I),
    "vendor": re.compile(r"\b(Voluson|Aloka|Logiq|Aplio|Mindray|Philips|Siemens|Samsung)\b", re.I),
    "preset": re.compile(
        r"\b(Cardiac|OB|Obstetric|Gyn|Breast|Abdomen|Vascular|Har-\w+|Routine\s*\d\s*Trim\.?(?:/OB)?)\b", re.I
    ),
    "laterality": re.compile(r"\b(MAMA\s+(?:ESQUERDA|DIREITA)|LEFT|RIGHT|ESQUERDA|DIREITA|LT|RT)\b", re.I),
    "quadrant": re.compile(r"\b(Q[SI][EI]|UOQ|UIQ|LOQ|LIQ)\b"),
}


def parse_text(text: str) -> tuple[dict[str, float], dict[str, str]]:
    numeric: dict[str, float] = {}
    for key, pattern in NUMERIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            try:
                numeric[key] = float(match.group(1))
            except ValueError:
                pass
    categorical: dict[str, str] = {}
    for key, pattern in CATEGORICAL_PATTERNS.items():
        match = pattern.search(text)
        if match:
            categorical[key] = re.sub(r"\s+", " ", match.group(1)).strip().lower()
    return numeric, categorical


# --------------------------------------------------------------------------
# OCR over the detected text regions only
# --------------------------------------------------------------------------


def _crop_to_text(image: np.ndarray, det: Detection) -> np.ndarray:
    """Render the detected glyphs as clean black-on-white for Tesseract.

    Feeding Tesseract the raw frame wastes most of its effort on speckle and
    produces hallucinated tokens. Feeding it only the regions our own detector
    already accepted as glyphs is both faster and far more precise, and it keeps
    the OCR strictly downstream of the mask.
    """
    gray = to_gray(image)
    canvas = np.full_like(gray, 255)
    if det.text.any():
        values = gray[det.text]
        # invert so glyphs are dark on a light field
        canvas[det.text] = np.clip(255 - values.astype(np.int32), 0, 255).astype(np.uint8)
    return canvas


def run_ocr(image: np.ndarray, det: Detection, psm: int = 6) -> tuple[str, float]:
    tess = _tesseract()
    if tess is None or not det.text.any():
        return "", 0.0
    canvas = _crop_to_text(image, det)
    try:
        data = tess.image_to_data(
            canvas, config=f"--psm {psm}", output_type=tess.Output.DICT
        )
    except Exception:
        return "", 0.0
    words, confidences = [], []
    for token, conf in zip(data["text"], data["conf"]):
        token = token.strip()
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        if token and conf > 0:
            words.append(token)
            confidences.append(conf)
    return " ".join(words), float(np.mean(confidences)) if confidences else 0.0


def parse_annotation(
    image: np.ndarray,
    det: Detection,
    cfg: AnnotationConfig | None = None,
    use_ocr: bool = True,
) -> ParsedAnnotation:
    """Full content extraction for one image.

    Caliper count and separation are recovered geometrically and therefore work
    with no OCR at all -- which matters, because for a measurement task the
    separation between two calipers *is* the label.
    """
    del cfg  # kept for signature symmetry with the rest of asa.annot
    parsed = ParsedAnnotation()
    parsed.n_calipers = int(det.features.get("calipers_n_components", 0))
    parsed.caliper_separation = float(det.features.get("caliper_max_sep", 0.0))

    if use_ocr:
        parsed.text, parsed.ocr_confidence = run_ocr(image, det)
        parsed.tokens = [t for t in re.split(r"\W+", parsed.text) if t]
        parsed.numeric, parsed.categorical = parse_text(parsed.text)
    return parsed


def build_aux_vocabularies(rows: list[dict[str, Any]], fields: list[str], min_count: int = 20) -> dict[str, list[str]]:
    """Vocabulary per categorical auxiliary field.

    Rare parses are folded into ``<other>`` rather than becoming near-empty
    classes, since an auxiliary head with singleton classes contributes noise to
    the very label-efficiency comparison it is meant to inform.
    """
    vocabularies: dict[str, list[str]] = {}
    for name in fields:
        counts: dict[str, int] = {}
        for row in rows:
            value = row.get(f"cat_{name}")
            if isinstance(value, str) and value:
                counts[value] = counts.get(value, 0) + 1
        kept = sorted(v for v, c in counts.items() if c >= min_count)
        vocabularies[name] = ["<other>", *kept]
    return vocabularies
