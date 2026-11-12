"""
legibility.py — ℓ(p) = P(the reasoner can extract the goal's direction from p).

Ten lines of math: standardize φ with the fit-time mean/std, apply the fitted
logistic, done.  The entire content of this file's honesty is WHERE the numbers
come from:

  calib/w.json   —  written by replay/calibrate.py (T6/E2), which fits
                    {mean, std, w, b} on VLM success/failure labels from
                    replayed approaches.  Format:
                    {"features": [...], "mean": [...], "std": [...],
                     "w": [...], "b": ..., "meta": {...}}

If the file is absent, Legibility falls back to a PLACEHOLDER that squashes
each feature to [0,1] with fixed scales and averages them — good enough to
exercise the gate in tests and demos, and it warns UNCALIBRATED on first use
so a placeholder can never silently reach a paper number.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .features import FEATURES, phi_vector


@dataclass
class Legibility:
    mean: np.ndarray
    std: np.ndarray
    w: np.ndarray
    b: float
    calibrated: bool = True
    _warned: bool = False

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = "adaptive_reasoning/calib/w.json") -> "Legibility":
        p = Path(path)
        if not p.exists():
            return cls.placeholder()
        d = json.loads(p.read_text())
        if d.get("features") != FEATURES:
            raise ValueError(
                f"{p} was fit for features {d.get('features')} but the code computes "
                f"{FEATURES}; refit (T6) before running."
            )
        return cls(np.array(d["mean"]), np.array(d["std"]), np.array(d["w"]),
                   float(d["b"]), calibrated=True)

    @classmethod
    def placeholder(cls) -> "Legibility":
        # Fixed squash scales so each feature lands roughly in [0,1]; equal
        # weights; slight negative bias.  Sharpness (Laplacian variance) spans
        # orders of magnitude, so the placeholder works on log1p(sharpness).
        # DEMO/TEST ONLY — calibrate.py fits the real thing on raw features.
        mean = np.zeros(len(FEATURES))
        std = np.array([40.0, 10.0, 1.0, 1.0, 1.0, 1.0])   # px, log1p(lap-var), rest [0,1]
        w = np.full(len(FEATURES), 1.2)
        return cls(mean, std, w, b=-3.5, calibrated=False)

    # -- the number -----------------------------------------------------------
    def __call__(self, phi: dict[str, float] | np.ndarray) -> float:
        if not self.calibrated and not self._warned:
            warnings.warn("Legibility is UNCALIBRATED (placeholder weights). "
                          "Fit calib/w.json in T6 before producing any result.")
            self._warned = True
        x = phi_vector(phi) if isinstance(phi, dict) else np.asarray(phi, dtype=float)
        if not self.calibrated:
            x = x.copy()
            x[FEATURES.index("sharpness")] = math.log1p(max(x[FEATURES.index("sharpness")], 0.0))
        z = (x - self.mean) / np.where(self.std == 0, 1.0, self.std)
        z = np.clip(z, -8.0, 8.0)         # numerical guard; harmless post-fit
        return 1.0 / (1.0 + math.exp(-(float(self.w @ z) + self.b)))

    # -- persistence (used by calibrate.py) -----------------------------------
    def save(self, path: str | Path, meta: Optional[dict] = None) -> None:
        Path(path).write_text(json.dumps({
            "features": FEATURES, "mean": self.mean.tolist(), "std": self.std.tolist(),
            "w": self.w.tolist(), "b": self.b, "meta": meta or {},
        }, indent=2))
