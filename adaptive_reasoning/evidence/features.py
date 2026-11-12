"""
features.py — the raw numbers legibility is made of.

Given one plate observation (crop + OCR lines + arrows), compute the feature
vector φ(p).  This file computes RAW quantities only.  It contains no weights,
no thresholds, and no notion of "readable": those live in calib/w.json (fit in
T6 by calibrate.py) and are applied by legibility.py.  Standardization
(mean/std per feature) also happens at fit time and is stored alongside w.

The six features (order fixed by FEATURES):
  text_height_px   median OCR-line height in pixels at native resolution.
                   For a fixed-size sign and fixed focal length this scales as
                   f·H/d — a depth-free proxy for distance, and directly the
                   pixels-per-character the VLM will see.
  sharpness        variance of the Laplacian of the grayscale crop.  Low when
                   the plate is motion-blurred or defocused.
  ocr_conf         mean OCR confidence over the plate's lines (from docTR).
  completeness     1.0 if the plate has BOTH destination-like text (a line with
                   >=2 alphanumeric characters) AND a directional cue (>=1
                   arrow), else 0.0.  A door plate has text but no arrow; an
                   arrow-only wall marker has the converse.
  foreshortening   min/max of the left and right edge heights of the plate
                   quad, in (0,1]; 1 = viewed frontally, small = oblique.
                   Falls back to 1.0 when only an axis-aligned box is known.
  agreement_k      fraction of the last k frames in which this plate's
                   normalized text was recognized identically (from
                   StringAgreement).  Flicker → low; stable lock → 1.0.

Everything is a pure function of the observation; StringAgreement is the one
stateful helper (a rolling window) and is owned by the caller.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .relevance import normalize

FEATURES = ["text_height_px", "sharpness", "ocr_conf",
            "completeness", "foreshortening", "agreement_k"]


# ----------------------------------------------------------------------------
# Observation structure (detect.py will produce this at runtime)
# ----------------------------------------------------------------------------

@dataclass
class OcrLine:
    text: str
    conf: float                          # [0,1] from the OCR model
    height_px: float                     # line box height at native resolution
    arrows: list[str] = field(default_factory=list)


@dataclass
class PlateObservation:
    lines: list[OcrLine]
    crop: Optional[np.ndarray] = None    # HxW or HxWx3, native resolution
    quad: Optional[np.ndarray] = None    # (4,2) corners TL,TR,BR,BL; None if box-only
    box: Optional[tuple[float, float, float, float]] = None   # x0,y0,x1,y1 native px

    @property
    def text(self) -> str:
        return " | ".join(l.text for l in self.lines)


# ----------------------------------------------------------------------------
# Individual features (pure)
# ----------------------------------------------------------------------------

def text_height_px(p: PlateObservation) -> float:
    hs = [l.height_px for l in p.lines if l.text.strip()]
    return float(np.median(hs)) if hs else 0.0


_LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def sharpness(p: PlateObservation) -> float:
    """Variance of the Laplacian of the grayscale crop (0 if no crop)."""
    if p.crop is None or p.crop.size == 0 or min(p.crop.shape[:2]) < 3:
        return 0.0
    g = p.crop.astype(np.float64)
    if g.ndim == 3:
        g = g @ np.array([0.299, 0.587, 0.114])
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def ocr_conf(p: PlateObservation) -> float:
    cs = [l.conf for l in p.lines if l.text.strip()]
    return float(np.mean(cs)) if cs else 0.0


def completeness(p: PlateObservation) -> float:
    has_text = any(sum(c.isalnum() for c in l.text) >= 2 for l in p.lines)
    has_arrow = any(l.arrows for l in p.lines)
    return 1.0 if (has_text and has_arrow) else 0.0


def foreshortening(p: PlateObservation) -> float:
    if p.quad is None:
        return 1.0
    q = np.asarray(p.quad, dtype=np.float64)      # TL,TR,BR,BL
    h_left = np.linalg.norm(q[3] - q[0])
    h_right = np.linalg.norm(q[2] - q[1])
    hi, lo = max(h_left, h_right), min(h_left, h_right)
    return float(lo / hi) if hi > 0 else 1.0


class StringAgreement:
    """Rolling window of the plate's normalized text over the last k frames."""

    def __init__(self, k: int):
        self.k = k
        self._win: deque[str] = deque(maxlen=k)

    def update(self, text: str) -> float:
        """Push this frame's text; return fraction of window matching it."""
        key = normalize(text.replace("|", " "))
        self._win.append(key)
        if not key:
            return 0.0
        return sum(1 for s in self._win if s == key) / self.k


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------

def phi(p: PlateObservation, agreement: float) -> dict[str, float]:
    """The full feature dict for one plate at one frame.  `agreement` comes
    from the caller's StringAgreement for this plate track."""
    return {
        "text_height_px": text_height_px(p),
        "sharpness": sharpness(p),
        "ocr_conf": ocr_conf(p),
        "completeness": completeness(p),
        "foreshortening": foreshortening(p),
        "agreement_k": float(agreement),
    }


def phi_vector(d: dict[str, float]) -> np.ndarray:
    return np.array([d[f] for f in FEATURES], dtype=np.float64)
