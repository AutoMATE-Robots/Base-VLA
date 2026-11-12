"""
resolve.py — "Can I read the goal's direction off this plate myself, without the VLM?"

This is the *fast path*.  Relevance (relevance.py) says a plate is about the
goal.  Resolution asks the stronger question: is there exactly one arrow on
the goal's line, so the direction is already known?  If yes, the gate sets
necessity N_t = 0, hands the direction to the VLA as its standing prompt, and
no reasoning call is made.  This is the same mechanism as IROS's System One
(OCR + condition→action); we keep it and put the evidence gate on top.

Input : a Plate — lines of text, each with the arrows attached to it.  Offline
        this is parsed from the annotated `sign_text` ("6-201 to 6-250 > | Main
        Elevators <"); at runtime detect.py builds the same structure from OCR
        boxes and detected arrow glyphs.
Output: a Resolution with
        status    'resolved' | 'ambiguous' | 'unresolved'
        direction sign-frame arrow direction ('left','right','up','down',...)
        vla_prompt the robot action, or None
        source    'line' (arrow on the goal's own line) | 'plate' (one arrow serves the whole plate)
        channel   'struct' | 'sem' — how the goal's line was matched
        reason    a sentence you can read

What resolve.py deliberately does NOT decide:
  * whether the sign is frontal enough for "left on the sign" to mean "left
    for the robot".  That is a property of the crop (foreshortening, in
    features.py) and the reliability of the fast path as a function of it is
    *measured* in E5 and calibrated like legibility — not assumed here.
  * what a DOWN arrow or a diagonal means.  Conventions differ between
    buildings; those cases return 'unresolved' and go to the VLM.
  * anything about memory: but resolve_all() returns every line's direction
    so memory.py can store the whole plate and answer later goals for free.

CLI:  python -m adaptive_reasoning.evidence.resolve "6-201 to 6-250 > | Main Elevators <" 6-217
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .relevance import (
    Embedder, Goal, RelevanceResult, SemanticCalibration,
    _DASH_MAP, relevance_lines, split_lines,
)

# ----------------------------------------------------------------------------
# 1. Arrows
# ----------------------------------------------------------------------------

# Annotation tokens and Unicode glyphs → a direction word.  This is a format
# table (what the characters denote), not a policy.
ARROW_CHARS: dict[str, str] = {
    "<": "left", ">": "right", "^": "up", "v": "down", "V": "down",
    "\u2190": "left", "\u2192": "right", "\u2191": "up", "\u2193": "down",
    "\u21d0": "left", "\u21d2": "right", "\u21d1": "up", "\u21d3": "down",
    "\u2196": "up_left", "\u2197": "up_right", "\u2198": "down_right", "\u2199": "down_left",
}
# Standalone ASCII tokens (so the 'v' in "Elevators" is not an arrow) or any glyph anywhere.
_ASCII_ARROW_RE = re.compile(r"(?<!\S)[<>^vV](?!\S)")
_GLYPH_ARROW_RE = re.compile("[" + "".join(c for c in ARROW_CHARS if len(c) == 1 and ord(c) > 127) + "]")

# Sign-frame direction → VLA prompt.  Only the three unambiguous cases.
# 'down' and diagonals are building conventions → left to the VLM.
DIRECTION_TO_VLA: dict[str, str] = {"left": "turn_left", "right": "turn_right", "up": "straight"}


# ----------------------------------------------------------------------------
# 2. Plate structure (shared with detect.py later)
# ----------------------------------------------------------------------------

@dataclass
class PlateLine:
    text: str                 # line text with arrows removed
    arrows: list[str] = field(default_factory=list)   # directions, reading order


@dataclass
class Plate:
    lines: list[PlateLine]

    @classmethod
    def from_text(cls, sign_text: str) -> "Plate":
        """Parse the annotation format: lines joined by ' | ', arrows as < > ^ v or glyphs."""
        out = []
        for raw in split_lines(sign_text):
            raw = raw.translate(_DASH_MAP)
            arrows = [ARROW_CHARS[m.group(0)] for m in _GLYPH_ARROW_RE.finditer(raw)]
            arrows += [ARROW_CHARS[m.group(0)] for m in _ASCII_ARROW_RE.finditer(raw)]
            text = _GLYPH_ARROW_RE.sub(" ", raw)
            text = _ASCII_ARROW_RE.sub(" ", text)
            out.append(PlateLine(re.sub(r"\s+", " ", text).strip(), arrows))
        return cls(out)

    @property
    def texts(self) -> list[str]:
        return [ln.text for ln in self.lines]

    @property
    def all_directions(self) -> set[str]:
        return {d for ln in self.lines for d in ln.arrows}


# ----------------------------------------------------------------------------
# 3. Resolution
# ----------------------------------------------------------------------------

@dataclass
class Resolution:
    status: str                      # resolved | ambiguous | unresolved
    direction: Optional[str] = None  # sign-frame
    vla_prompt: Optional[str] = None
    line_index: int = -1
    source: str = "none"             # line | plate | none
    channel: str = "none"            # struct | sem | none
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"


def resolve(
    plate: Plate,
    goal: Goal,
    rel: Optional[RelevanceResult] = None,
    embedder: Optional[Embedder] = None,
    sem_calib: Optional[SemanticCalibration] = None,
    require_struct: bool = True,
) -> Resolution:
    """Try to read the goal's direction from the plate's arrows.

    require_struct=True  → only a line that *literally* lists the goal (R_struct = 1)
                           may resolve.  Semantic matches ("Amundson Hall <" for a
                           room in Amundson) still count as relevant but go to the VLM.
    require_struct=False → the best relevant line may resolve regardless of channel.
    Both variants are measured in E5; the default is the conservative one.
    """
    rel = rel or relevance_lines(plate.texts, goal, embedder, sem_calib)
    if rel.score <= 0.0 or rel.best_line < 0:
        return Resolution("unresolved", reason="no line relevant to the goal")

    if require_struct:
        cands = [i for i, lr in enumerate(rel.lines) if lr.struct >= 1.0]
        if not cands:
            return Resolution("unresolved", line_index=rel.best_line, channel="sem",
                              reason="goal line matched only semantically; require_struct=True")
        channel = "struct"
    else:
        cands = [i for i, lr in enumerate(rel.lines) if lr.score >= rel.score]
        channel = "struct" if rel.lines[rel.best_line].struct >= 1.0 else "sem"

    dirs = {d for i in cands for d in plate.lines[i].arrows}
    source = "line"
    if not dirs:
        # No arrow on the goal line.  If the whole plate carries exactly one
        # direction (an arrow column or an arrow-only line), it serves every line.
        dirs = plate.all_directions
        source = "plate"
        if not dirs:
            return Resolution("unresolved", line_index=cands[0], channel=channel,
                              reason="no arrow on the goal line and none on the plate")
    if len(dirs) > 1:
        return Resolution("ambiguous", line_index=cands[0], source=source, channel=channel,
                          reason=f"more than one direction applies: {sorted(dirs)}")

    d = next(iter(dirs))
    vla = DIRECTION_TO_VLA.get(d)
    if vla is None:
        return Resolution("unresolved", direction=d, line_index=cands[0], source=source, channel=channel,
                          reason=f"arrow '{d}' has no unambiguous robot action (building convention)")
    return Resolution("resolved", d, vla, cands[0], source, channel,
                      reason=f"line {cands[0]} '{plate.lines[cands[0]].text}' has one arrow: {d} ({source})")


def resolve_all(plate: Plate) -> list[Optional[str]]:
    """Direction for every line, or None where a line has zero or several arrows.
    A plate with one shared direction gives that direction to all lines.
    memory.py stores this so a later goal on the same plate needs no call."""
    shared = plate.all_directions
    out: list[Optional[str]] = []
    for ln in plate.lines:
        ds = set(ln.arrows) or (shared if len(shared) == 1 else set())
        out.append(next(iter(ds)) if len(ds) == 1 else None)
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, warnings
    warnings.simplefilter("ignore")
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    plate = Plate.from_text(sys.argv[1])
    goal = Goal.parse(sys.argv[2], sys.argv[3:])
    r = resolve(plate, goal)
    print(f"{r.status.upper():10s} direction={r.direction} vla={r.vla_prompt} source={r.source} channel={r.channel}")
    print("  " + r.reason)
    print("  per-line:", resolve_all(plate))
