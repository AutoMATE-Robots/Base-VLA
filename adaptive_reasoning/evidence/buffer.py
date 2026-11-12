"""
buffer.py — E_t: the same physical sign across frames.

detect.py sees each frame in isolation; the gate needs continuity — "this is
the same directory I saw last frame, now bigger and sharper" — so that
(a) agreement can be computed, (b) memory keys are stable, and (c) OCR flicker
doesn't look like a brand-new sign every frame.

What real Keller footage taught us (t1 replay, Aug 2026):
  * OCR letters mutate constantly ("to"→"105"→"t05") but the DIGIT content of
    a directory is rock-stable: 5-231/5-250/5-201/5-217/5-117/5-196 appear in
    every read.  So association keys on the **digit signature** — the set of
    numeric tokens on the plate — compared by Jaccard overlap.  This also
    bridges the case where the detector splits one sign into top/bottom
    halves, because a half's codes are a subset of the whole's.
  * Association must be TWO-PHASE: match every detection against the tracks'
    pre-frame state, THEN apply updates.  Matching-and-updating in one pass
    let the first half of a split sign overwrite the track before the second
    half was compared (the identity-churn bug seen at f146/f184).
  * Detections that map to the same track in the same frame are MERGED back
    into one observation (lines in top-to-bottom order, union box), undoing
    the detector's split.

Association rule per detection, against pre-frame track state, in order:
  1. exact normalized-text match (last or best text), else
  2. digit-signature Jaccard ≥ sig_jaccard, else
  3. fuzzy text match (difflib ≥ text_sim, vs last AND best text), else
  4. box IoU ≥ min_iou, else new track.
Tracks age out after miss_tolerance consecutive unseen frames.

Plate identity (`plate_id`) is the signature of the best read
("SIG:5-117 5-196 ..."), so memory keys survive OCR noise and half/whole
splits.  Known limit, stated: two physically distinct plates listing the same
room codes would share an identity; acceptable indoors, noted in the paper.

`sig_jaccard`, `text_sim`, `min_iou`, `miss_tolerance` are association
conventions (config fields, logged by E1), not fitted quantities.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Optional

from .features import OcrLine, PlateObservation, StringAgreement
from .relevance import normalize
from ..memory import plate_key

def digit_signature(text: str) -> frozenset[str]:
    """All 3-digit windows of every digit run.  Windowing (not whole tokens)
    is the point: the OCR mutation "to"→"105" GLUES codes ("5-201105-217"),
    and windows let the true codes (201, 217) survive inside glued runs."""
    runs = re.findall(r"\d+", normalize(text.replace("|", " ")))
    sig = {run[i:i + 3] for run in runs if len(run) >= 3 for i in range(len(run) - 2)}
    # 2-digit room numbers (Rapson: "Rooms 43-58", "1-37, 63-71") would otherwise leave
    # the plate with NO signature and a brittle raw-text identity; tag them so they
    # cannot collide with 3-digit windows
    sig |= {"d2:" + run for run in runs if len(run) == 2}
    return frozenset(sig)


def sig_overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """Overlap coefficient |a∩b|/min(|a|,|b|): a half-sign is a subset of its
    whole, so ∩/min scores it ~1 where Jaccard would dilute.  Plates with a
    single code (door plates) never sig-match (return 0) so they cannot be
    absorbed into a directory that lists their room."""
    if min(len(a), len(b)) < 2:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def stable_plate_id(text: str) -> str:
    """Signature-based identity when the plate has numbers; normalized text otherwise."""
    sig = digit_signature(text)
    return "SIG:" + " ".join(sorted(sig)) if sig else plate_key(text)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def merge_observations(group: list[PlateObservation]) -> PlateObservation:
    """Re-join detections the detector split (one physical sign): lines in
    top-to-bottom order, union box, crop from the largest piece."""
    if len(group) == 1:
        return group[0]
    group = sorted(group, key=lambda o: o.box[1] if o.box else 0.0)
    lines: list[OcrLine] = [l for o in group for l in o.lines]
    boxes = [o.box for o in group if o.box is not None]
    box = (min(b[0] for b in boxes), min(b[1] for b in boxes),
           max(b[2] for b in boxes), max(b[3] for b in boxes)) if boxes else None
    biggest = max(group, key=lambda o: ((o.box[2] - o.box[0]) * (o.box[3] - o.box[1]))
                  if o.box else 0.0)
    return PlateObservation(lines, crop=biggest.crop, quad=None, box=box)


@dataclass
class PlateTrack:
    track_id: int
    obs: PlateObservation
    agreement: StringAgreement
    best_text: str
    best_conf: float
    last_seen: int
    agreement_frac: float = 0.0
    _sticky_id: Optional[str] = None

    @property
    def plate_id(self) -> str:
        """Sticky: locked at the first signature-bearing read so identity never
        drifts under OCR churn (memory keys depend on this)."""
        if self._sticky_id is None or not self._sticky_id.startswith("SIG:"):
            cand = stable_plate_id(self.best_text)
            if self._sticky_id is None or cand.startswith("SIG:"):
                self._sticky_id = cand
        return self._sticky_id


@dataclass
class BufferConfig:
    k: int = 6                 # agreement window (frames)
    sig_overlap: float = 0.5   # digit-signature overlap coefficient for association
    text_sim: float = 0.8      # difflib ratio for fuzzy association
    min_iou: float = 0.3
    miss_tolerance: int = 30   # frames a track survives unseen (3 s at 10 Hz;
                               # t1 showed OCR dropouts >1 s that reincarnated tracks)


class EvidenceBuffer:
    def __init__(self, cfg: Optional[BufferConfig] = None):
        self.cfg = cfg or BufferConfig()
        self._tracks: list[PlateTrack] = []
        self._next_id = 0

    # -- matching against a PRE-FRAME snapshot --------------------------------
    def _match(self, obs: PlateObservation,
               snapshot: list[tuple["PlateTrack", str, str, frozenset, Optional[tuple]]]
               ) -> Optional["PlateTrack"]:
        key = plate_key(obs.text)
        sig = digit_signature(obs.text)
        # 1. exact
        for tr, last_key, best_key, _, _ in snapshot:
            if key and key in (last_key, best_key):
                return tr
        # 2. digit signature
        best, best_j = None, 0.0
        for tr, _, _, tsig, _ in snapshot:
            j = sig_overlap(sig, tsig)
            if j > best_j:
                best, best_j = tr, j
        if best is not None and best_j >= self.cfg.sig_overlap:
            return best
        # 3. fuzzy text (vs last AND best read)
        best, best_r = None, 0.0
        for tr, last_key, best_key, _, _ in snapshot:
            r = max(difflib.SequenceMatcher(None, key, last_key).ratio(),
                    difflib.SequenceMatcher(None, key, best_key).ratio())
            if r > best_r:
                best, best_r = tr, r
        if best is not None and best_r >= self.cfg.text_sim:
            return best
        # 4. IoU vs pre-frame box
        if obs.box is not None:
            for tr, _, _, _, tbox in snapshot:
                if tbox is not None and iou(obs.box, tbox) >= self.cfg.min_iou:
                    return tr
        return None

    def update(self, plates: list[PlateObservation], t: int) -> list["PlateTrack"]:
        """Two-phase: (1) associate every detection against the pre-frame track
        state; (2) merge same-track groups and apply updates.  Returns the
        tracks seen THIS frame."""
        snapshot = [(tr, plate_key(tr.obs.text), plate_key(tr.best_text),
                     digit_signature(tr.best_text) | digit_signature(tr.obs.text),
                     tr.obs.box) for tr in self._tracks]
        groups: dict[int, list[PlateObservation]] = {}
        track_of: dict[int, PlateTrack] = {}
        new_groups: list[list[PlateObservation]] = []
        for obs in plates:
            tr = self._match(obs, snapshot)
            if tr is None:
                new_groups.append([obs])          # each unmatched detection = its own track
            else:
                groups.setdefault(tr.track_id, []).append(obs)
                track_of[tr.track_id] = tr

        seen: list[PlateTrack] = []
        for tid, group in groups.items():
            tr = track_of[tid]
            self._apply(tr, merge_observations(group), t)
            seen.append(tr)
        for group in new_groups:
            obs = merge_observations(group)
            tr = PlateTrack(self._next_id, obs, StringAgreement(self.cfg.k),
                            obs.text, max((l.conf for l in obs.lines), default=0.0), t)
            self._next_id += 1
            self._tracks.append(tr)
            tr.agreement_frac = tr.agreement.update(obs.text)
            seen.append(tr)

        self._tracks = [tr for tr in self._tracks
                        if t - tr.last_seen <= self.cfg.miss_tolerance]
        return seen

    def _apply(self, tr: "PlateTrack", obs: PlateObservation, t: int) -> None:
        tr.agreement_frac = tr.agreement.update(obs.text)
        tr.plate_id  # touch: lock sticky identity on first signature-bearing read
        conf = max((l.conf for l in obs.lines), default=0.0)
        runs = lambda txt: len(re.findall(r"\d{3,}", txt))
        # canonical text: higher confidence wins; at equal confidence, fuller read wins
        if conf > tr.best_conf or (conf == tr.best_conf and runs(obs.text) > runs(tr.best_text)):
            tr.best_conf, tr.best_text = conf, obs.text
        tr.obs, tr.last_seen = obs, t

    @property
    def live_tracks(self) -> list["PlateTrack"]:
        return list(self._tracks)
