"""
relevance.py — "Is this sign about my goal?"

Input : the text of ONE sign plate (OCR output at runtime, or the annotated
        `sign_text` column offline) and a Goal.
Output: R in [0, 1] for the plate, plus the same number per line and a
        human-readable reason, so we can always see *why* a plate scored.

A sign can be about the goal in two ways, and we take the larger of the two:

  R_struct  (exact, pure parsing, no model)
      The plate literally lists the goal's room number, or a range that
      contains it ("6-201 to 6-250" contains "6-217").  Returns 1.0 or 0.0.
      Typed goals (elevator, exit, ...) have no structural channel.

  R_sem     (semantic, text embeddings)
      The plate names something on the way to the goal — building, wing,
      floor, department, landmark — that the goal's descriptor set V_g
      mentions ("Amundson Hall" for a room inside Amundson; "Elevators" for a
      goal "elevator").  V_g is produced
      ONCE per goal (see goals/ and, later, expand_goal.py), so this costs a
      cosine similarity per line at runtime.  The cosine → probability map is
      calibrated offline (T6); until then it is identity and flagged.

  R(plate, goal) = max over lines of max(R_struct, R_sem)

What this file deliberately does NOT do:
  * It never looks at arrows.  "Which arrow belongs to the goal's line" is
    *resolution*, not relevance, and lives in resolve.py.  Keeping the two
    apart is what lets a door plate be relevant-but-insufficient.
  * It never loads a vision model.  The default embedder is a tiny
    bag-of-words so tests run anywhere; a sentence-transformer can be
    plugged in with one line.

Run from the command line to check a sign against a goal (handy for annotators):
    python -m adaptive_reasoning.evidence.relevance "6-201 to 6-250 > | Main Elevators <" 6-217
"""
from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

# ----------------------------------------------------------------------------
# 1. Text normalisation
# ----------------------------------------------------------------------------

# Every Unicode dash we've seen on signs and in OCR output → plain "-".
_DASH_MAP = {ord(c): "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"}
# Arrow glyphs and the annotation tokens (< > ^ v) used in sign_text.
_ARROW_GLYPHS = "\u2190\u2192\u2191\u2193\u21d0\u21d2\u21d1\u21d3"
_ARROW_TOKEN_RE = re.compile(r"(?<!\S)[<>^V](?!\S)")  # standalone < > ^ v after upper-casing
_LINE_SPLIT_RE = re.compile(r"\s*(?:\||\n|;)\s*")


def normalize(text: str) -> str:
    """Upper-case, unify dashes, drop arrows, collapse whitespace."""
    t = (text or "").translate(_DASH_MAP).upper()
    t = re.sub(f"[{_ARROW_GLYPHS}]", " ", t)
    t = _ARROW_TOKEN_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def split_lines(text: str) -> list[str]:
    """A plate is several lines.  Annotations join them with ' | '; OCR may use newlines."""
    return [ln for ln in (s.strip() for s in _LINE_SPLIT_RE.split(text or "")) if ln]


# ----------------------------------------------------------------------------
# 2. Room codes and ranges (the structural part)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Room:
    """A room code.  prefix = floor/wing part ("6" in "6-210", "B" in "B12", "" if none)."""
    prefix: str
    number: int
    suffix: str = ""

    def __str__(self) -> str:
        p = f"{self.prefix}-" if self.prefix else ""
        return f"{p}{self.number}{self.suffix}"


@dataclass(frozen=True)
class RoomRange:
    lo: Room
    hi: Room

    @property
    def prefix(self) -> str:
        return self.lo.prefix

    def contains(self, room: Room) -> tuple[bool, str]:
        """Return (inside?, kind). kind is 'exact' | 'number_only' | 'no'.
        'number_only' = the sign (or goal) omitted the floor prefix, so we matched on number alone."""
        if not (self.lo.number <= room.number <= self.hi.number):
            return False, "no"
        if self.prefix and room.prefix:
            return (self.prefix == room.prefix, "exact" if self.prefix == room.prefix else "no")
        return True, "number_only"

    def __str__(self) -> str:
        return f"{self.lo}..{self.hi}"


def _code_pattern(tag: str) -> str:
    # A room code is: optional prefix + 2-4 digit number + optional letter suffix.
    #   digit prefix MUST be followed by a dash  ("6-201"), so "6201" is not "62"+"01";
    #   letter prefix MAY omit the dash           ("B12", "B-12").
    return (
        rf"(?:(?P<dp{tag}>\d{{1,2}})-|(?P<lp{tag}>[A-Z]{{1,2}})-?)?"
        rf"(?P<num{tag}>\d{{2,4}})(?P<suf{tag}>[A-Z])?"
    )


_B = r"(?<![A-Z0-9])"   # left boundary: not preceded by alnum
_E = r"(?![A-Z0-9])"    # right boundary
_CODE_RE = re.compile(_B + _code_pattern("") + _E)
_RANGE_RE = re.compile(
    _B + _code_pattern("a") + r"\s*(?P<sep>TO|THRU|THROUGH|-|\.\.\.|…)\s*" + _code_pattern("b") + _E
)
# Small bare ranges with a 1-digit start ("1-37", "5-9"): the code pattern needs a
# 2-digit number, so these never match _RANGE_RE.  Seen on Rapson signs
# ("Rooms 1-37, 63-71").
_SMALL_RANGE_RE = re.compile(_B + r"(?P<a>\d{1,2})\s*-\s*(?P<b>\d{1,2})" + _E)


def _room_from_match(m: re.Match, tag: str, inherit_prefix: str = "") -> Room:
    prefix = m.group(f"dp{tag}") or m.group(f"lp{tag}") or inherit_prefix
    return Room(prefix, int(m.group(f"num{tag}")), m.group(f"suf{tag}") or "")


def parse_room(text: str) -> Optional[Room]:
    """Parse a single room code from a short string (used for goals).  None if not a room."""
    t = normalize(text)
    m = _CODE_RE.fullmatch(t)
    return _room_from_match(m, "") if m else None


def parse_rooms_and_ranges(text: str) -> tuple[list[RoomRange], list[Room]]:
    """Find all ranges ("6-201 to 6-250", "201-250") and standalone rooms in a line."""
    t = normalize(text)
    ranges: list[RoomRange] = []
    spans: list[tuple[int, int]] = []
    for m in _RANGE_RE.finditer(t):
        a = _room_from_match(m, "a")
        # Ambiguity guard: "12-201" is floor 12, room 201 — not the range 12..201.
        # A bare dash with a 1–2 digit left part and a 3+ digit right part is a
        # floor-room code.  "43-58" (both sides small) IS a range: buildings with
        # two-digit room numbers (Rapson) print ranges exactly like that, and a
        # 43rd floor is not a plausible reading.
        if (m.group("sep") == "-" and not a.prefix and len(m.group("numa")) <= 2
                and len(m.group("numb")) >= 3):
            continue
        b = _room_from_match(m, "b", inherit_prefix=a.prefix)
        if b.number < a.number:
            continue
        ranges.append(RoomRange(a, b))
        spans.append(m.span())
    # Blank out the ranges, then small 1-digit-start ranges, then standalone codes.
    masked = list(t)
    for s, e in spans:
        for i in range(s, e):
            masked[i] = " "
    for m in _SMALL_RANGE_RE.finditer("".join(masked)):
        a_n, b_n = int(m.group("a")), int(m.group("b"))
        if b_n >= a_n:
            ranges.append(RoomRange(Room("", a_n, ""), Room("", b_n, "")))
            for i in range(*m.span()):
                masked[i] = " "
    rooms = [_room_from_match(m, "") for m in _CODE_RE.finditer("".join(masked))]
    return ranges, rooms


# ----------------------------------------------------------------------------
# 3. Goal
# ----------------------------------------------------------------------------

@dataclass
class Goal:
    """What the robot was asked to reach.

    text        : exactly as given ("6-217", "elevator")
    room        : parsed Room if text is a room code, else None (a "typed" goal)
    descriptors : V_g — strings that name things on the way to the goal
                  (building, wing, floor, department, landmarks) and, for typed
                  goals, the words signs use for it ("Elevators", "Lift").
                  Filled once per goal by expand_goal (later file) or by hand and
                  stored in goals/<text>.json.  There is no synonym table in code:
                  typed goals are matched through V_g like everything else.
    """
    text: str
    room: Optional[Room] = None
    descriptors: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str, descriptors: Iterable[str] = ()) -> "Goal":
        text = text.strip()
        room = parse_room(text)
        desc = list(descriptors)
        if room is None and not desc:
            desc = [text]          # typed goal with no V_g yet: at least match its own word
        return cls(text=text, room=room, descriptors=desc)

    @classmethod
    def load(cls, path: str | Path) -> "Goal":
        d = json.loads(Path(path).read_text())
        return cls.parse(d["goal"], d.get("descriptors", []))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"goal": self.text, "descriptors": self.descriptors}, indent=2))


# ----------------------------------------------------------------------------
# 4. Structural relevance
# ----------------------------------------------------------------------------

def structural_relevance(line: str, goal: Goal) -> tuple[float, str]:
    """1.0 if the line literally lists the goal's room (or a range containing it), else 0.0."""
    if goal.room is None:
        return 0.0, "typed goal: no structural channel (see descriptors)"
    ranges, rooms = parse_rooms_and_ranges(line)
    for rng in ranges:
        inside, kind = rng.contains(goal.room)
        if inside:
            return 1.0, f"range {rng} ({kind})"
    for r in rooms:
        if r.number == goal.room.number and (not r.prefix or not goal.room.prefix or r.prefix == goal.room.prefix):
            return 1.0, f"room {r}"
    return 0.0, "no room/range match"


# ----------------------------------------------------------------------------
# 5. Semantic relevance
# ----------------------------------------------------------------------------

# An embedder turns a list of strings into a list of equal-length float vectors.
Embedder = Callable[[Sequence[str]], list[list[float]]]


def bow_embedder(texts: Sequence[str], dim: int = 2048) -> list[list[float]]:
    """Tiny hashed bag-of-words embedder.  Only here so the code runs with no
    model installed; swap in sentence_transformer_embedder for real use."""
    out = []
    for t in texts:
        v = [0.0] * dim
        for tok in re.findall(r"[A-Z0-9]{2,}", normalize(t)):
            v[hash(tok) % dim] += 1.0
        out.append(v)
    return out


def sentence_transformer_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Build a real embedder (lazy import so this file stays dependency-free)."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True).tolist()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else sum(x * y for x, y in zip(a, b)) / (na * nb)


@dataclass
class SemanticCalibration:
    """Maps cosine → P(line is relevant).  Fit offline in T6 (isotonic or logistic on
    the annotated sign_text / goal pairs).  Until then: identity, clipped, and flagged."""
    kind: str = "identity"
    a: float = 1.0   # logistic slope   (if kind == 'logistic')
    b: float = 0.0   # logistic offset
    _warned: bool = field(default=False, repr=False)

    def __call__(self, cos: float) -> float:
        if self.kind == "identity":
            if not self._warned:
                warnings.warn("SemanticCalibration is UNCALIBRATED (identity). Fit it in T6 → calib/sem.json")
                self._warned = True
            return max(0.0, min(1.0, cos))
        return 1.0 / (1.0 + math.exp(-(self.a * cos + self.b)))

    @classmethod
    def load(cls, path: str | Path) -> "SemanticCalibration":
        d = json.loads(Path(path).read_text())
        return cls(kind=d.get("kind", "logistic"), a=d["a"], b=d["b"])


def semantic_relevance(line: str, goal: Goal, embedder: Embedder, calib: SemanticCalibration) -> tuple[float, str]:
    if not goal.descriptors:
        return 0.0, "no descriptors"
    vecs = embedder([line, *goal.descriptors])
    best_i, best_cos = -1, -1.0
    for i, dv in enumerate(vecs[1:]):
        c = cosine(vecs[0], dv)
        if c > best_cos:
            best_i, best_cos = i, c
    return calib(best_cos), f"semantic '{goal.descriptors[best_i]}' cos={best_cos:.2f}"


# ----------------------------------------------------------------------------
# 6. Putting it together
# ----------------------------------------------------------------------------

@dataclass
class LineRelevance:
    line: str
    struct: float
    sem: float
    reason: str

    @property
    def score(self) -> float:
        return max(self.struct, self.sem)


@dataclass
class RelevanceResult:
    score: float                 # R(plate, goal) = max over lines
    best_line: int               # index into lines, -1 if no lines
    lines: list[LineRelevance]

    def explain(self) -> str:
        rows = [f"  [{i}] {lr.score:.2f}  struct={lr.struct:.0f} sem={lr.sem:.2f}  '{lr.line}'  ← {lr.reason}"
                for i, lr in enumerate(self.lines)]
        return f"R = {self.score:.2f} (best line {self.best_line})\n" + "\n".join(rows)


def relevance_lines(
    plate_lines: Sequence[str],
    goal: Goal,
    embedder: Optional[Embedder] = None,
    sem_calib: Optional[SemanticCalibration] = None,
) -> RelevanceResult:
    """R for a plate given as a list of line strings.  Index i of the result
    always corresponds to plate_lines[i] (empty lines score 0) — resolve.py
    relies on this to line arrows up with text."""
    embedder = embedder or bow_embedder
    sem_calib = sem_calib or SemanticCalibration()
    lines: list[LineRelevance] = []
    for ln in plate_lines:
        if not normalize(ln):
            lines.append(LineRelevance(ln, 0.0, 0.0, "empty line"))
            continue
        s, why = structural_relevance(ln, goal)
        if s >= 1.0 or not goal.descriptors:
            lines.append(LineRelevance(ln, s, 0.0, why))
            continue
        m, why_sem = semantic_relevance(ln, goal, embedder, sem_calib)
        lines.append(LineRelevance(ln, s, m, why_sem))
    if not lines:
        return RelevanceResult(0.0, -1, [])
    best = max(range(len(lines)), key=lambda i: lines[i].score)
    return RelevanceResult(lines[best].score, best, lines)


def relevance(
    sign_text: str,
    goal: Goal,
    embedder: Optional[Embedder] = None,
    sem_calib: Optional[SemanticCalibration] = None,
) -> RelevanceResult:
    """R(plate, goal) from a plate string ('line | line | ...' or newline-separated).
    Structural first; semantic only if structural missed and V_g exists."""
    return relevance_lines(split_lines(sign_text), goal, embedder, sem_calib)


# ----------------------------------------------------------------------------
# CLI for annotators:  python -m adaptive_reasoning.evidence.relevance "<sign_text>" <goal> [descriptor ...]
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    g = Goal.parse(sys.argv[2], sys.argv[3:])
    print(relevance(sys.argv[1], g).explain())
