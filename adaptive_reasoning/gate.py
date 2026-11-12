"""
gate.py — WHEN to reason.  The equation, as a pure function.

    fire at t   iff   N_t · q_t ≥ τ ,      q_t = max over novel plates p of R(p,g)·ℓ(p)

  N_t  necessity  — 1 while no decision is held for the upcoming decision point,
                    0 once one is held (or a call is in flight).
  R    relevance  — computed upstream (evidence/relevance.py)
  ℓ    legibility — computed upstream (evidence/legibility.py)
  novel           — not in memory M_g (memory.py)
  τ               — the operating point.  Comes from calib/tau.json (T8).  It is
                    the ONLY number in this file and it is never defaulted.

When the bar is met the gate ACTS.  There are two ways to act:
  * fast path — the plate already resolves (evidence/resolve.py found one
    arrow on the goal's line): hand the direction to the VLA, no call.
  * call      — otherwise fire the VLM (non-blocking).
The same evidence bar applies to both; the only difference is whether the
cheap parser could already read the direction.

State machine (what "FSM" means here):
  NO_SIGN  — no novel relevant evidence in view.                      N = 1
  ARMED    — novel relevant evidence in view, but q_t < τ.  Waiting
             for it to get bigger/sharper.  Nothing is sent.           N = 1
  PENDING  — a call has been fired and has not returned yet.  No
             second call.  The VLA keeps driving on the old prompt.    N = 0
  DECIDED  — a direction is held as the VLA's standing prompt (from
             fast path or VLM).  Held until the turn is executed.      N = 0
  Transitions:
    NO_SIGN ⇄ ARMED         as evidence appears / disappears (every tick)
    ARMED   → PENDING       fire (q ≥ τ, not resolvable)
    ARMED   → DECIDED       fast path (q ≥ τ, resolvable)
    PENDING → DECIDED       VLM returned a direction
    PENDING → NO_SIGN       VLM said not-applicable (plate is consumed)
    DECIDED → NO_SIGN       on_turn_done() from the controller (decision executed)
    any     → NO_SIGN       new goal (fresh state + fresh memory)

Purity: tick() takes the previous GateState and returns a new one plus any
MemoryUpdate it wants applied.  It never mutates anything.  Replay and the
live server call the same bytes.

Known simplification (documented, not hidden): in DECIDED the gate ignores
new evidence, even if a later sign contradicts the held decision.  Adding a
"reconsider" clause to necessity is a one-line change we will evaluate, not
assume.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional, Protocol, Sequence

from .evidence.resolve import Resolution
from .memory import Memory, MemoryUpdate


# ----------------------------------------------------------------------------
# 1. Inputs
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PlateEvidence:
    """One plate as the gate sees it: identity + the two numbers + (optional) a resolution."""
    plate_id: str
    relevance: float                     # R(p, g) in [0, 1]
    legibility: float                    # ℓ(p)    in [0, 1]
    resolution: Optional[Resolution] = None
    text: str = ""

    @property
    def q(self) -> float:
        return self.relevance * self.legibility

    @property
    def resolvable(self) -> bool:
        return self.resolution is not None and self.resolution.resolved


class State(str, Enum):
    NO_SIGN = "NO_SIGN"
    ARMED = "ARMED"
    PENDING = "PENDING"
    DECIDED = "DECIDED"


@dataclass(frozen=True)
class GateState:
    state: State = State.NO_SIGN
    decision: Optional[str] = None           # standing VLA prompt, e.g. 'turn_left'
    decision_source: Optional[str] = None    # 'fast' | 'vlm'
    decision_plate: Optional[str] = None
    pending_plate: Optional[str] = None
    fired_at: Optional[int] = None
    decided_at: Optional[int] = None

    @property
    def necessity(self) -> int:
        return 0 if self.state in (State.PENDING, State.DECIDED) else 1


@dataclass(frozen=True)
class GateConfig:
    tau: float                    # operating point — REQUIRED, read from calib/tau.json
    fast_path: bool = True        # allow resolve.py to act without a call (ablation switch)


@dataclass(frozen=True)
class VLMResponse:
    plate_id: str
    applicable: bool
    vla_prompt: Optional[str] = None   # 'turn_left' | 'turn_right' | 'straight' | 'stop'


# ----------------------------------------------------------------------------
# 2. Output
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class TickResult:
    state: GateState
    fire: bool                              # send a VLM call now?
    necessity: int
    q: float                                # q_t (0 if nothing novel & relevant)
    best: Optional[PlateEvidence]
    memory_updates: tuple[MemoryUpdate, ...] = ()
    reason: str = ""


# ----------------------------------------------------------------------------
# 3. Policy interface (baselines implement the same three methods)
# ----------------------------------------------------------------------------

class GatePolicy(Protocol):
    def tick(self, prev: GateState, plates: Sequence[PlateEvidence], memory: Memory, t: int) -> TickResult: ...
    def on_response(self, prev: GateState, resp: VLMResponse, t: int) -> tuple[GateState, tuple[MemoryUpdate, ...]]: ...
    def on_turn_done(self, prev: GateState, t: int) -> GateState: ...


# ----------------------------------------------------------------------------
# 4. The evidence gate
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceGate:
    cfg: GateConfig

    # -- hooks (baselines override these; the FSM below is shared) ------------
    def _candidates(self, plates: Sequence[PlateEvidence], memory: Memory) -> list[PlateEvidence]:
        return [p for p in plates if p.relevance > 0.0 and memory.is_novel(p.plate_id)]

    def _score(self, p: PlateEvidence) -> float:
        return p.q

    # -- per frame ------------------------------------------------------------
    def tick(self, prev: GateState, plates: Sequence[PlateEvidence], memory: Memory, t: int) -> TickResult:
        novel = self._candidates(plates, memory)
        best = max(novel, key=self._score) if novel else None
        q = self._score(best) if best else 0.0

        # N_t = 0: a decision is held or a call is in flight.  Log q, do nothing.
        if prev.necessity == 0:
            return TickResult(prev, False, 0, q, best, reason=f"{prev.state.value}: necessity 0")

        if best is None:
            return TickResult(replace(prev, state=State.NO_SIGN), False, 1, 0.0, None,
                              reason="no novel relevant evidence")

        if q < self.cfg.tau:
            return TickResult(replace(prev, state=State.ARMED), False, 1, q, best,
                              reason=f"armed: q={q:.3f} < tau={self.cfg.tau:.3f}")

        # Bar met → act.
        if self.cfg.fast_path and best.resolvable:
            d = best.resolution.vla_prompt
            new = replace(prev, state=State.DECIDED, decision=d, decision_source="fast",
                          decision_plate=best.plate_id, decided_at=t, pending_plate=None)
            upd = MemoryUpdate(best.plate_id, "fast", d, t)
            return TickResult(new, False, 1, q, best, (upd,),
                              reason=f"fast path: {best.resolution.reason} → {d}")

        new = replace(prev, state=State.PENDING, pending_plate=best.plate_id, fired_at=t)
        return TickResult(new, True, 1, q, best, reason=f"FIRE: q={q:.3f} ≥ tau={self.cfg.tau:.3f}")

    # -- events ---------------------------------------------------------------
    def on_response(self, prev: GateState, resp: VLMResponse, t: int) -> tuple[GateState, tuple[MemoryUpdate, ...]]:
        if prev.state != State.PENDING:
            return prev, ()                        # stale/duplicate response: ignore
        if resp.applicable and resp.vla_prompt:
            new = replace(prev, state=State.DECIDED, decision=resp.vla_prompt, decision_source="vlm",
                          decision_plate=resp.plate_id, decided_at=t, pending_plate=None)
            return new, (MemoryUpdate(resp.plate_id, "vlm", resp.vla_prompt, t),)
        new = replace(prev, state=State.NO_SIGN, pending_plate=None)
        return new, (MemoryUpdate(resp.plate_id, "not_applicable", None, t),)

    def on_turn_done(self, prev: GateState, t: int) -> GateState:
        """Controller reports the held decision was executed → necessity returns to 1."""
        return GateState()


def new_goal() -> GateState:
    return GateState()
