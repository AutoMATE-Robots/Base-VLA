"""
memory.py — M_g: which sign plates have already been *consumed* for the current goal.

"Consumed" means one of:
  * the VLM was asked about it and answered (decision or not-applicable),
  * the fast path read a direction off it,
  * a human/operator told us to ignore it.
A consumed plate can never trigger another call for the same goal — this is
the novelty term E_t \\ M_g in the gate equation, and it replaces the old
"ignore for 5 seconds" cooldown with a goal-scoped fact.

Scope: ONE goal.  When the goal changes, start a fresh Memory (the richer
cross-goal memory — the reasoning-event graph for Section C — will wrap this,
not replace it).

Purity: the gate never mutates memory.  It returns MemoryUpdate objects; the
caller (replay loop or server) applies them with memory.apply(update).  That
keeps gate.tick a pure function and makes replays reproducible.

Keys: plate_id is whatever detect.py / buffer.py decide is the stable identity
of a plate (normalized text for now).  Merging OCR variants of the same sign
is the buffer's job, not memory's — memory does exact lookups only.

WHERE (added for multi-junction trips, a light version of paper Eq. 4): text alone
cannot tell two physical signs apart — the "Elevators ->" at junction 1 and the
"Elevators ->" at junction 2 have the same id, so text-only memory suppressed the
second sign and the robot drove past its second turn.  When the caller supplies
the robot pose (set_pose, from /odom), every consumed entry is stamped with the
pose it was read from, and a later read only counts as the SAME sign if it is also
within `radius_m` of that pose AND seen with a heading within `yaw_tol_rad`.
  * same sign, still approaching it      -> near, same heading  -> consumed
  * next junction after a turn           -> heading differs     -> novel
  * repeated sign further down a corridor -> far                -> novel
Without a pose (replays, tests, no odometry) the old text-only rule applies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

from .evidence.relevance import normalize


def plate_key(text: str) -> str:
    """Default plate identity: normalized text with lines joined.  Exact, no fuzz."""
    return normalize(text.replace("|", " "))


@dataclass(frozen=True)
class MemoryUpdate:
    plate_id: str
    outcome: str                  # 'fast' | 'vlm' | 'not_applicable' | 'ignored'
    decision: Optional[str] = None
    t: Optional[int] = None
    pose: Optional[tuple] = None  # (x, y, yaw) in the odom frame when the sign was consumed


@dataclass
class Memory:
    goal: str
    consumed: dict[str, MemoryUpdate] = field(default_factory=dict)

    # a consumed plate is recognised again if its digit signature overlaps a consumed
    # one by >= this (the evidence buffer's association rule); exact-id equality
    # broke on a single OCR dropout ("43-58" vs "43 58") and re-asked the same sign.
    # fuzzy=False -> exact-id consumption only.  Baselines keep exact identity:
    # their papers dedup per tracked instance, and OUR overlap rule must never
    # decide when a BASELINE's sign counts as already-seen (a false match would
    # suppress a call they needed — tuning against them).
    match_overlap: float = 0.5
    fuzzy: bool = True
    # location rule (only used when poses are known): see module docstring
    radius_m: float = 8.0
    yaw_tol_rad: float = math.radians(45.0)
    pose: Optional[tuple] = None                 # current robot pose (x, y, yaw), set per frame
    events: list = field(default_factory=list)   # every consumed entry, incl. same text at other places

    def set_pose(self, pose: Optional[tuple]) -> None:
        self.pose = tuple(pose) if pose is not None else None

    def _same_place(self, u: MemoryUpdate) -> bool:
        if self.pose is None or u.pose is None:
            return True                          # no location information: text decides
        dx, dy = self.pose[0] - u.pose[0], self.pose[1] - u.pose[1]
        dyaw = abs((self.pose[2] - u.pose[2] + math.pi) % (2 * math.pi) - math.pi)
        return math.hypot(dx, dy) <= self.radius_m and dyaw <= self.yaw_tol_rad

    @staticmethod
    def _sig(plate_id: str) -> frozenset:
        if plate_id.startswith("SIG:"):
            return frozenset(plate_id[4:].split())
        return frozenset(t for t in plate_id.split() if len(t) >= 3)   # text key: word tokens

    def _match_event(self, plate_id: str) -> Optional[MemoryUpdate]:
        """The consumed entry this read refers to (text rule + location rule), or None."""
        here = [u for u in self.events if self._same_place(u)]
        for u in here:
            if u.plate_id == plate_id:
                return u
        if not self.fuzzy:
            return None
        a = self._sig(plate_id)
        if len(a) < 2:
            return None
        best, best_j = None, 0.0
        for u in here:
            k = u.plate_id
            b = self._sig(k)
            if len(b) < 2 or (k.startswith("SIG:") != plate_id.startswith("SIG:")):
                continue
            j = len(a & b) / min(len(a), len(b))
            if j > best_j:
                best, best_j = u, j
        return best if best_j >= self.match_overlap else None

    def matches(self, plate_id: str):
        """The consumed plate id this read refers to, or None."""
        u = self._match_event(plate_id)
        return u.plate_id if u else None

    def is_novel(self, plate_id: str) -> bool:
        return self.matches(plate_id) is None

    def apply(self, update: MemoryUpdate) -> None:
        # the gate is pure and does not know the pose; stamp it here
        if update.pose is None and self.pose is not None:
            update = replace(update, pose=self.pose)
        self.consumed[update.plate_id] = update
        self.events.append(update)

    def outcome(self, plate_id: str) -> Optional[str]:
        u = self._match_event(plate_id)
        return u.outcome if u else None

    def __len__(self) -> int:
        return len(self.events)