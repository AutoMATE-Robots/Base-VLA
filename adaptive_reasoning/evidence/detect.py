"""
detect.py — frame → plates.  The only file that touches a vision model.

Pipeline:  image → OCR lines (engine) → cluster lines into plates (pure
geometry) → PlateObservation per plate (crop + lines + arrows), ready for
features.py / relevance.py / resolve.py.

Engine-agnostic on purpose: `OcrEngine` is any callable image → [DetectedLine].
  * DocTREngine — the real one (docTR, `ar` env, lazy import so this module
    imports anywhere).
  * tests use a scripted fake, so the clustering and the whole downstream
    pipeline are tested without a GPU or model weights.

Clustering convention (stated, not tuned): two lines belong to the same plate
when their vertical gap is below `vgap_scale` line-heights AND they overlap
horizontally or sit within `hgap_scale` line-heights sideways.  These are
document-layout conventions in units of text height, not fitted quantities;
they are config fields so the E1 replay logs them, and their end-to-end effect
is measured by the same metrics as everything else.

Arrows: OCR engines read arrow glyphs (←→↑↓ and sometimes < > ^) as text; we
parse those out of each line via the same table resolve.py uses.  A dedicated
visual arrow detector (the old scorer's contour logic) can be plugged in as
`arrow_detector` once ported — without one, plates simply resolve less often
and the VLM handles direction, which the fast-path-rate metric reports
honestly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from .features import OcrLine, PlateObservation
from .resolve import ARROW_CHARS, _ASCII_ARROW_RE, _GLYPH_ARROW_RE


# ----------------------------------------------------------------------------
# OCR interface
# ----------------------------------------------------------------------------

@dataclass
class DetectedLine:
    text: str
    conf: float                      # [0,1]
    box: tuple[float, float, float, float]   # x0,y0,x1,y1 in native pixels


class OcrEngine(Protocol):
    def __call__(self, image: np.ndarray) -> list[DetectedLine]: ...


class DocTREngine:
    """docTR ocr_predictor wrapped to the DetectedLine interface (lazy import)."""

    def __init__(self, det_arch: str = "db_resnet50", reco_arch: str = "crnn_vgg16_bn"):
        from doctr.models import ocr_predictor  # type: ignore
        self._pred = ocr_predictor(det_arch, reco_arch, pretrained=True)
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                self._pred = self._pred.cuda()
                print("DocTREngine: using CUDA")
            else:
                print("DocTREngine: CUDA not available, running on CPU (slow at 1080p)")
        except Exception as e:  # pragma: no cover
            print(f"DocTREngine: staying on CPU ({e})")

    def __call__(self, image: np.ndarray) -> list[DetectedLine]:
        H, W = image.shape[:2]
        out = self._pred([image]).export()
        lines: list[DetectedLine] = []
        for block in out["pages"][0]["blocks"]:
            for ln in block["lines"]:
                words = ln["words"]
                if not words:
                    continue
                text = " ".join(w["value"] for w in words)
                conf = float(np.mean([w["confidence"] for w in words]))
                (x0, y0), (x1, y1) = ln["geometry"]
                lines.append(DetectedLine(text, conf, (x0 * W, y0 * H, x1 * W, y1 * H)))
        return lines


# ----------------------------------------------------------------------------
# Arrows out of OCR text
# ----------------------------------------------------------------------------

def split_arrows(text: str) -> tuple[str, list[str]]:
    """Pull arrow glyphs/tokens out of a line's text; return (clean_text, arrows)."""
    arrows = [ARROW_CHARS[m.group(0)] for m in _GLYPH_ARROW_RE.finditer(text)]
    arrows += [ARROW_CHARS[m.group(0)] for m in _ASCII_ARROW_RE.finditer(text)]
    clean = _GLYPH_ARROW_RE.sub(" ", text)
    clean = _ASCII_ARROW_RE.sub(" ", clean)
    return " ".join(clean.split()), arrows


# ----------------------------------------------------------------------------
# Clustering (pure)
# ----------------------------------------------------------------------------

@dataclass
class DetectConfig:
    vgap_scale: float = 1.5      # max vertical gap between plate lines, in line-heights
    hgap_scale: float = 2.0      # max horizontal gap when lines don't overlap sideways
    # Crop margins are asymmetric ON PURPOSE: the box is a union of OCR *text*
    # lines, but arrows are graphics OCR never reports — they sit BESIDE the
    # text, so a tight crop slices them off (seen on t3: "Main Elevators" text
    # with its arrow clipped → underdetermined VLM payload).  Wide horizontal
    # margin captures the arrow column; modest vertical margin captures
    # over/under arrows.
    crop_margin_x: float = 0.40  # fraction of box WIDTH added on each side
    crop_margin_y: float = 0.25  # fraction of box HEIGHT added top and bottom
    # At distance docTR splits a panel into small fragments; a fired fragment's
    # crop can be a few dozen pixels of bare text (observed on Tate 17_13_22:
    # the payload was the word "B50" alone).  The payload crop therefore grows
    # to swallow NEARBY text boxes (reconstructing the panel) and is never
    # smaller than min_crop_px on its long side.
    payload_near: float = 1.5    # neighbour within this × box size joins the crop
    min_crop_px: int = 256       # minimum long-side of the payload crop
    context_factor: float = 2.5  # second image: the grown box scaled by this, same centre
    min_line_conf: float = 0.0   # keep everything by default; filtering is the gate's job


def _linked(a: DetectedLine, b: DetectedLine, cfg: DetectConfig) -> bool:
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box
    h = min(ay1 - ay0, by1 - by0)
    if h <= 0:
        return False
    vgap = max(by0 - ay1, ay0 - by1, 0.0)
    if vgap > cfg.vgap_scale * h:
        return False
    hgap = max(bx0 - ax1, ax0 - bx1, 0.0)   # 0 when they overlap horizontally
    return hgap <= cfg.hgap_scale * h


def cluster_lines(lines: Sequence[DetectedLine], cfg: DetectConfig) -> list[list[int]]:
    """Single-linkage grouping of line indices into plates (union-find)."""
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if _linked(lines[i], lines[j], cfg):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(lines)):
        groups.setdefault(find(i), []).append(i)
    # reading order: top-to-bottom by group top edge, lines within group by y
    out = [sorted(g, key=lambda i: lines[i].box[1]) for g in groups.values()]
    return sorted(out, key=lambda g: lines[g[0]].box[1])


def _union_box(boxes: Sequence[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    return x0, y0, x1, y1


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def detect(image: np.ndarray, engine: OcrEngine,
           cfg: Optional[DetectConfig] = None,
           arrow_detector: Optional[Callable[[np.ndarray, tuple], list[str]]] = None,
           ) -> list[PlateObservation]:
    """Run OCR, cluster lines into plates, build PlateObservations with crops."""
    cfg = cfg or DetectConfig()
    raw = [l for l in engine(image) if l.conf >= cfg.min_line_conf and l.text.strip()]
    plates: list[PlateObservation] = []
    H, W = image.shape[:2]
    for group in cluster_lines(raw, cfg):
        lines: list[OcrLine] = []
        for i in group:
            clean, arrows = split_arrows(raw[i].text)
            x0, y0, x1, y1 = raw[i].box
            lines.append(OcrLine(clean, raw[i].conf, y1 - y0, arrows))
        box = _union_box([raw[i].box for i in group])
        mx = cfg.crop_margin_x * (box[2] - box[0]); my = cfg.crop_margin_y * (box[3] - box[1])
        cx0, cy0 = max(int(box[0] - mx), 0), max(int(box[1] - my), 0)
        cx1, cy1 = min(int(box[2] + mx), W), min(int(box[3] + my), H)
        crop = image[cy0:cy1, cx0:cx1] if cy1 > cy0 and cx1 > cx0 else None
        if arrow_detector is not None and crop is not None:
            extra = arrow_detector(crop, box)
            if extra and lines:
                lines[-1].arrows.extend(a for a in extra if a not in lines[-1].arrows)
        plates.append(PlateObservation(lines, crop=crop, quad=None, box=box))
    return plates


# ----------------------------------------------------------------------------
# Payload crop (SHARED by the live gate and by stored calibration crops —
# they must be identical, or ℓ is calibrated for a payload we never send)
# ----------------------------------------------------------------------------

def _grow(a: tuple, b: tuple) -> tuple:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _is_near(a: tuple, b: tuple, factor: float) -> bool:
    """b sits within `factor` × a's size of a (gap measured per axis)."""
    aw, ah = max(a[2] - a[0], 1.0), max(a[3] - a[1], 1.0)
    gap_x = max(b[0] - a[2], a[0] - b[2], 0.0)
    gap_y = max(b[1] - a[3], a[1] - b[3], 0.0)
    return gap_x <= factor * aw and gap_y <= factor * ah


def payload_box(box: tuple, all_boxes, cfg: Optional[DetectConfig] = None) -> tuple:
    """Grow `box` to include nearby text boxes (transitively), then apply margins."""
    cfg = cfg or DetectConfig()
    cur = tuple(float(v) for v in box)
    used = set()
    changed = True
    while changed:
        changed = False
        for i, b in enumerate(all_boxes):
            if i in used or b is None:
                continue
            b = tuple(float(v) for v in b)
            if _is_near(cur, b, cfg.payload_near):
                cur = _grow(cur, b)
                used.add(i)
                changed = True
    mx = cfg.crop_margin_x * (cur[2] - cur[0])
    my = cfg.crop_margin_y * (cur[3] - cur[1])
    return (cur[0] - mx, cur[1] - my, cur[2] + mx, cur[3] + my)


def payload_crop(image: np.ndarray, box: tuple, all_boxes=(),
                 cfg: Optional[DetectConfig] = None) -> np.ndarray:
    """The image actually sent to the reasoner for a plate."""
    cfg = cfg or DetectConfig()
    H, W = image.shape[:2]
    x0, y0, x1, y1 = payload_box(box, list(all_boxes), cfg)
    # enforce a minimum long side, expanding about the centre
    long_side = max(x1 - x0, y1 - y0)
    if long_side < cfg.min_crop_px:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        scale = cfg.min_crop_px / max(long_side, 1.0)
        hw, hh = (x1 - x0) * scale / 2, (y1 - y0) * scale / 2
        hw = max(hw, cfg.min_crop_px / 2 * (x1 - x0) / max(long_side, 1.0))
        x0, x1 = cx - hw, cx + hw
        y0, y1 = cy - hh, cy + hh
    ix0, iy0 = max(int(x0), 0), max(int(y0), 0)
    ix1, iy1 = min(int(x1), W), min(int(y1), H)
    if ix1 <= ix0 or iy1 <= iy0:
        return image
    return image[iy0:iy1, ix0:ix1]


def payload_crops(image: np.ndarray, box: tuple, all_boxes=(),
                  cfg: Optional[DetectConfig] = None) -> list[np.ndarray]:
    """[tight payload crop, context crop].  The tight crop carries the digits;
    the context crop (context_factor × the grown box, same centre, clipped to
    the frame) carries the arrow column and neighbouring plates, so a clipped
    arrow or a mis-selected plate is recoverable without sending the whole
    frame (which invites corridor-geometry hallucinations)."""
    cfg = cfg or DetectConfig()
    tight = payload_crop(image, box, all_boxes, cfg)
    H, W = image.shape[:2]
    x0, y0, x1, y1 = payload_box(box, list(all_boxes), cfg)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) * cfg.context_factor / 2, (y1 - y0) * cfg.context_factor / 2
    ix0, iy0 = max(int(cx - hw), 0), max(int(cy - hh), 0)
    ix1, iy1 = min(int(cx + hw), W), min(int(cy + hh), H)
    ctx = image[iy0:iy1, ix0:ix1] if (ix1 > ix0 and iy1 > iy0) else image
    return [tight, ctx]
