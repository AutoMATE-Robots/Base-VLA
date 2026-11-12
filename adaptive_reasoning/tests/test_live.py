"""test_live.py — the inference loop end to end with a scripted OCR engine and a
fake VLM: acquire → fire → (call in flight) → decide → turn done → memory."""
import json
import threading

import numpy as np
import pytest

from adaptive_reasoning.evidence.detect import DetectedLine
from adaptive_reasoning.evidence.legibility import Legibility
from adaptive_reasoning.live import LiveConfig, LiveReasoner, parse_goal
from adaptive_reasoning.reasoner import FakeVLM, Reasoner

IMG = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)

FAR = [DetectedLine("6-201 to 6-250", 0.9, (600, 300, 640, 310))]      # 10 px tall
NEAR = [DetectedLine("6-201 to 6-250", 0.95, (500, 200, 760, 280)),    # 80 px tall
        DetectedLine("6-101 to 6-150", 0.95, (500, 290, 760, 370))]
OTHER = [DetectedLine("Restrooms", 0.95, (100, 100, 400, 180))]


class ScriptedOcr:
    """Returns the next scripted list of lines on every call (last one repeats)."""

    def __init__(self, frames):
        self.frames, self.i = frames, 0

    def __call__(self, image):
        out = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return list(out)


class GatedVLM(FakeVLM):
    """FakeVLM that blocks until released, to observe the PENDING state."""

    def __init__(self, script):
        super().__init__(script)
        self.release = threading.Event()

    def complete(self, prompt, images):
        self.release.wait(5)
        return super().complete(prompt, images)


class BrokenVLM:
    calls = 0

    def complete(self, prompt, images):
        BrokenVLM.calls += 1
        raise ConnectionError("network down")


ANSWER = {"6-217": {"applicable": True, "direction": "turn_right", "confidence": 0.9,
                    "summary": "6-201 to 6-250 has a right arrow"}}


def make(frames, client, tmp_path, goal="room 6-217", tau=0.3, min_plate_h=60.0, clock=None):
    reasoner = Reasoner(client, model_tag="fake", cache_dir=tmp_path / "cache", save_payload=False)
    kw = {"clock": clock} if clock else {}
    return LiveReasoner(goal, LiveConfig(tau=tau, min_plate_h=min_plate_h), ScriptedOcr(frames),
                        reasoner, Legibility.placeholder(), **kw)


def test_parse_goal_strips_room_word():
    assert str(parse_goal("room 6-217").room) == "6-217"
    assert str(parse_goal("Rm 49").room) == "49"
    g = parse_goal("Main Elevators")
    assert g.room is None and g.descriptors == ["Main Elevators"]


def test_acquire_fire_decide_turn_done_and_memory(tmp_path):
    vlm = GatedVLM(ANSWER)
    live = make([FAR, NEAR, NEAR, NEAR, NEAR, NEAR], vlm, tmp_path)
    try:
        r0 = live.step(IMG, 0)                       # far: too few pixels → ACQUIRE, no call
        assert r0.acquiring and not r0.fired and r0.state == "ARMED"
        assert r0.prompt == "straight"

        r1 = live.step(IMG, 1)                       # near: fire, call runs in background
        assert r1.fired and r1.state == "PENDING" and live.calls == 1
        assert r1.prompt == "straight"               # robot keeps driving on the old prompt

        r2 = live.step(IMG, 2)                       # still in flight: no second call
        assert r2.state == "PENDING" and not r2.fired and live.calls == 1

        vlm.release.set()
        live.wait_for_call()
        r3 = live.step(IMG, 3)                       # answer lands
        assert r3.answered["direction"] == "turn_right"
        assert r3.state == "DECIDED" and r3.prompt == "turn_right"

        r4 = live.step(IMG, 4)                       # decision is held, nothing new fires
        assert r4.prompt == "turn_right" and not r4.fired

        live.turn_done(5)                            # controller: turn executed
        assert live.prompt == "straight"
        r5 = live.step(IMG, 6)                       # same sign still visible: already consumed
        assert not r5.fired and live.calls == 1 and r5.state == "NO_SIGN"
    finally:
        live.close()


def test_payload_carries_parser_hint(tmp_path):
    seen = {}

    class Spy(FakeVLM):
        def complete(self, prompt, images):
            seen["prompt"], seen["n"] = prompt, len(images)
            return super().complete(prompt, images)

    live = make([NEAR, NEAR], Spy(ANSWER), tmp_path)
    try:
        live.step(IMG, 0)
        live.wait_for_call()
        live.step(IMG, 1)
        assert seen["n"] == 1                        # exactly the cropped sign
        assert 'Goal: "room 6-217"' in seen["prompt"]
        assert "6-201 to 6-250" in seen["prompt"] and "Parser check" in seen["prompt"]
    finally:
        live.close()


def test_fast_path_decides_without_a_call(tmp_path):
    lines = [DetectedLine("6-201 to 6-250 ←", 0.95, (500, 200, 760, 280))]
    vlm = FakeVLM(ANSWER)
    live = make([lines], vlm, tmp_path)
    try:
        r = live.step(IMG, 0)
        assert r.fast_path and r.state == "DECIDED" and r.prompt == "turn_left"
        assert vlm.calls == 0 and live.calls == 0
    finally:
        live.close()


def test_irrelevant_sign_never_fires(tmp_path):
    vlm = FakeVLM(ANSWER)
    live = make([OTHER] * 5, vlm, tmp_path)
    try:
        for t in range(5):
            r = live.step(IMG, t)
            assert r.state == "NO_SIGN" and not r.fired
        assert vlm.calls == 0
    finally:
        live.close()


def test_not_applicable_consumes_plate(tmp_path):
    vlm = FakeVLM({})                                # default answer: not applicable
    live = make([NEAR] * 4, vlm, tmp_path)
    try:
        live.step(IMG, 0)
        live.wait_for_call()
        r = live.step(IMG, 1)
        assert r.answered and r.answered["applicable"] is False
        assert r.state == "NO_SIGN" and r.prompt == "straight"
        live.step(IMG, 2)
        assert live.calls == 1                       # never asked twice
    finally:
        live.close()


def test_failed_call_retries_after_backoff(tmp_path):
    now = [100.0]
    BrokenVLM.calls = 0
    live = make([NEAR] * 6, BrokenVLM(), tmp_path, clock=lambda: now[0])
    try:
        live.step(IMG, 0)
        live.wait_for_call()
        r1 = live.step(IMG, 1)                       # error lands; plate NOT consumed
        assert r1.answered and "error" in r1.answered
        assert r1.state == "ARMED" and r1.acquiring and live.calls == 1   # backing off
        now[0] += 5.0
        r2 = live.step(IMG, 2)                       # back-off over → fires again
        assert r2.fired and live.calls == 2
    finally:
        live.close()


def test_new_goal_ignores_stale_answer(tmp_path):
    vlm = GatedVLM(ANSWER)
    live = make([NEAR] * 4, vlm, tmp_path)
    try:
        live.step(IMG, 0)
        assert live.state.state.value == "PENDING"
        live.set_goal("6-120")
        vlm.release.set()
        r = live.step(IMG, 1)
        assert r.answered is None                    # the old goal's answer is dropped
        assert live.prompt != "turn_right"
    finally:
        live.close()


class SequenceVLM:
    """Answers scripted calls in order (one dict per call)."""

    def __init__(self, answers):
        self.answers, self.calls = list(answers), 0

    def complete(self, prompt, images):
        self.calls += 1
        self.last_prompt = prompt
        return json.dumps(self.answers.pop(0)), 0.5


def test_multi_turn_trip_to_the_elevators_then_arrival(tmp_path):
    """'go to the elevators': two identical 'Elevators' signs at two junctions (right,
    right), then the label at the elevator doors.  Text-only memory used to drop sign 2."""
    import math
    sign = [DetectedLine("Elevators", 0.95, (500, 200, 760, 280))]
    vlm = SequenceVLM([
        {"applicable": True, "direction": "turn_right", "confidence": .9, "summary": "Elevators line, right arrow"},
        {"applicable": True, "direction": "turn_right", "confidence": .9, "summary": "Elevators line, right arrow"},
        {"applicable": True, "direction": "stop", "confidence": .9, "summary": "ARRIVED: label at the elevator doors"},
    ])
    # OCR per step: junction 1 (2 frames), just after the turn (sign 1 is out of view),
    # junction 2 (2 frames), elevator doors (2 frames), then the doors again
    live = make([sign, sign, [], sign, sign, sign, sign, sign], vlm, tmp_path, goal="Elevators")
    try:
        def read_sign(t, pose):
            r = live.step(IMG, t, pose)
            assert r.fired, f"sign at {pose} should be read"
            assert r.text_h_px == 80.0
            live.wait_for_call()
            return live.step(IMG, t + 1, pose)

        r = read_sign(0, (0.0, 0.0, 0.0))                             # junction 1
        assert r.prompt == "turn_right"
        live.turn_done(2)
        assert not live.step(IMG, 3, (4.0, -2.0, -math.pi / 2)).fired  # new corridor, no sign yet

        r = read_sign(4, (4.0, -12.0, -math.pi / 2))                  # junction 2: SAME text
        assert r.prompt == "turn_right" and live.calls == 2
        live.turn_done(6)

        r = read_sign(7, (-8.0, -14.0, math.pi))                      # elevator doors
        assert r.prompt == "stop" and r.arrived and live.arrived
        assert live.calls == 3
        assert "ARRIVAL" in vlm.last_prompt                           # arrival rule was in the payload
        assert live.step(IMG, 9, (-8.5, -14.0, math.pi)).prompt == "stop"   # trip over: held
    finally:
        live.close()


def test_latency_estimate_tracks_measured_calls(tmp_path):
    vlm = SequenceVLM([{"applicable": False, "direction": None, "confidence": .1, "summary": "no"}])
    live = make([NEAR] * 3, vlm, tmp_path)
    try:
        assert live.latency_estimate == 4.0                           # default before any call
        live.step(IMG, 0); live.wait_for_call(); live.step(IMG, 1)
        assert live.latency_estimate == 0.5
    finally:
        live.close()
