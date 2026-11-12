"""Tests for gate.py + memory.py.

Each test is one claim from the paper text or one FSM transition.  If a test
name reads like a sentence from Section B, that's on purpose.
"""
import pytest

from adaptive_reasoning.evidence.resolve import Resolution
from adaptive_reasoning.gate import (
    EvidenceGate, GateConfig, GateState, PlateEvidence, State, VLMResponse, new_goal,
)
from adaptive_reasoning.memory import Memory, MemoryUpdate, plate_key

TAU = 0.5   # a test value only; the real τ comes from calib/tau.json


def gate(tau=TAU, fast_path=True):
    return EvidenceGate(GateConfig(tau=tau, fast_path=fast_path))


def plate(pid="A", R=1.0, L=1.0, resolved=None, text=""):
    res = None
    if resolved is not None:
        res = Resolution("resolved", resolved, {"left": "turn_left", "right": "turn_right", "up": "straight"}[resolved],
                         0, "line", "struct", "test")
    return PlateEvidence(pid, R, L, res, text)


def run(g, mem, frames, s=None, t0=0):
    """Tick through a list of plate-lists; apply memory updates; return list of results."""
    s = s or new_goal()
    out = []
    for i, plates in enumerate(frames):
        r = g.tick(s, plates, mem, t0 + i)
        for u in r.memory_updates:
            mem.apply(u)
        s = r.state
        out.append(r)
    return out


# ---------------- memory ----------------

def test_memory_novel_then_consumed():
    m = Memory("6-217")
    assert m.is_novel("A")
    m.apply(MemoryUpdate("A", "vlm", "turn_left", 3))
    assert not m.is_novel("A") and m.outcome("A") == "vlm" and len(m) == 1


def test_plate_key_is_exact_normalized_text():
    assert plate_key("6-201 to 6-250 > | Main Elevators <") == plate_key("6\u2013201 TO 6\u2013250 \u2192 | main elevators \u2190")


# ---------------- necessity & product form ----------------

def test_no_plates_is_no_sign_and_necessity_one():
    r = run(gate(), Memory("g"), [[]])[0]
    assert r.state.state == State.NO_SIGN and not r.fire and r.necessity == 1 and r.q == 0.0


def test_sharp_but_irrelevant_plate_never_fires():
    # The product: R = 0 kills a perfectly legible notice.  (This is the bug the old sum had.)
    frames = [[plate("notice", R=0.0, L=1.0)]] * 5
    rs = run(gate(), Memory("g"), frames)
    assert all(r.state.state == State.NO_SIGN and not r.fire for r in rs)


def test_relevant_but_blurry_plate_arms_and_waits():
    r = run(gate(), Memory("g"), [[plate("dir", R=1.0, L=0.1)]])[0]
    assert r.state.state == State.ARMED and not r.fire and r.necessity == 1 and r.q == pytest.approx(0.1)


def test_fires_exactly_once_at_first_crossing_then_pending():
    # ℓ rising along the approach: 0.1, 0.3, 0.5, 0.7, 0.9
    frames = [[plate("dir", R=1.0, L=l)] for l in (0.1, 0.3, 0.5, 0.7, 0.9)]
    rs = run(gate(tau=0.5), Memory("g"), frames)
    fires = [i for i, r in enumerate(rs) if r.fire]
    assert fires == [2]                                             # first frame with q ≥ τ
    assert rs[2].state.state == State.PENDING and rs[2].state.fired_at == 2
    assert all(r.state.state == State.PENDING and not r.fire and r.necessity == 0 for r in rs[3:])


def test_boundary_q_equal_tau_fires():
    r = run(gate(tau=0.5), Memory("g"), [[plate("dir", R=1.0, L=0.5)]])[0]
    assert r.fire


def test_best_plate_is_argmax_of_product_not_relevance():
    a = plate("a", R=1.0, L=0.2)     # q 0.2
    b = plate("b", R=0.8, L=0.9)     # q 0.72
    r = run(gate(tau=0.5), Memory("g"), [[a, b]])[0]
    assert r.fire and r.best.plate_id == "b" and r.state.pending_plate == "b"


def test_tau_to_zero_recovers_reactive_gate():
    # The paper's claim: IROS-style reactive necessity-only gating is τ → 0.
    r = run(gate(tau=0.0), Memory("g"), [[plate("dir", R=0.2, L=0.05)]])[0]
    assert r.fire


# ---------------- responses ----------------

def test_applicable_response_decides_and_blocks_further_fires():
    g = gate(tau=0.5); m = Memory("g")
    s = run(g, m, [[plate("dir", R=1.0, L=0.9)]])[0].state
    s, upd = g.on_response(s, VLMResponse("dir", True, "turn_left"), t=7)
    for u in upd: m.apply(u)
    assert s.state == State.DECIDED and s.decision == "turn_left" and s.decision_source == "vlm" and s.decided_at == 7
    assert not m.is_novel("dir")
    # Even a brand-new, perfect plate does not fire while DECIDED (necessity 0).
    r = g.tick(s, [plate("other", R=1.0, L=1.0)], m, 8)
    assert not r.fire and r.necessity == 0 and r.state.state == State.DECIDED


def test_not_applicable_consumes_plate_and_returns_to_no_sign():
    g = gate(tau=0.5); m = Memory("g")
    s = run(g, m, [[plate("dir", R=1.0, L=0.9)]])[0].state
    s, upd = g.on_response(s, VLMResponse("dir", False), t=7)
    for u in upd: m.apply(u)
    assert s.state == State.NO_SIGN and m.outcome("dir") == "not_applicable"
    # Same plate again: ignored.  A different plate: fires.
    assert not g.tick(s, [plate("dir", R=1.0, L=1.0)], m, 8).fire
    assert g.tick(s, [plate("dir2", R=1.0, L=1.0)], m, 9).fire


def test_stale_response_when_not_pending_is_ignored():
    g = gate(); s = new_goal()
    s2, upd = g.on_response(s, VLMResponse("x", True, "turn_left"), t=1)
    assert s2 == s and upd == ()


# ---------------- turn done / new goal ----------------

def test_turn_done_restores_necessity_but_memory_persists():
    g = gate(tau=0.5); m = Memory("g")
    s = run(g, m, [[plate("dir", R=1.0, L=0.9)]])[0].state
    s, upd = g.on_response(s, VLMResponse("dir", True, "turn_left"), t=5)
    for u in upd: m.apply(u)
    s = g.on_turn_done(s, t=20)
    assert s.state == State.NO_SIGN and s.decision is None and s.necessity == 1
    assert not g.tick(s, [plate("dir", R=1.0, L=1.0)], m, 21).fire      # consumed stays consumed
    assert g.tick(s, [plate("next", R=1.0, L=1.0)], m, 22).fire          # new sign fires


# ---------------- fast path ----------------

def test_fast_path_decides_without_a_call_when_bar_is_met():
    g = gate(tau=0.5); m = Memory("g")
    r = run(g, m, [[plate("dir", R=1.0, L=0.9, resolved="right")]])[0]
    assert not r.fire and r.state.state == State.DECIDED
    assert r.state.decision == "turn_right" and r.state.decision_source == "fast"
    assert m.outcome("dir") == "fast"


def test_fast_path_still_waits_below_the_bar():
    # Resolvable but blurry: the same evidence bar applies to acting as to calling.
    r = run(gate(tau=0.5), Memory("g"), [[plate("dir", R=1.0, L=0.2, resolved="right")]])[0]
    assert r.state.state == State.ARMED and not r.fire


def test_fast_path_disabled_fires_instead():
    r = run(gate(tau=0.5, fast_path=False), Memory("g"), [[plate("dir", R=1.0, L=0.9, resolved="right")]])[0]
    assert r.fire and r.state.state == State.PENDING


def test_unresolved_resolution_object_does_not_take_fast_path():
    p = PlateEvidence("dir", 1.0, 0.9, Resolution("ambiguous", reason="two arrows"))
    r = run(gate(tau=0.5), Memory("g"), [[p]])[0]
    assert r.fire and r.state.state == State.PENDING


# ---------------- purity ----------------

def test_tick_does_not_mutate_inputs():
    g = gate(tau=0.5); m = Memory("g"); s = new_goal()
    p = plate("dir", R=1.0, L=0.9, resolved="left")
    r = g.tick(s, [p], m, 0)
    assert s == new_goal() and len(m) == 0 and r.memory_updates    # update returned, not applied
