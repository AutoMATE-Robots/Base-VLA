"""
live.py — the adaptive-reasoning loop for INFERENCE: one camera frame in,
standing VLA prompt out.  No ROS, no bag, no pre-dumped jsonl.

This is the live counterpart of replay/dump_features.py + replay/render_submission.py
in the SignWay research repo.  Those ran docTR over a recorded bag first, wrote a
jsonl, and then replayed the gate over it with a simulated call latency.  Here
every step happens on the current frame, and the VLM call runs in a background
thread so the robot keeps driving on its current prompt while the call is in flight
(the PENDING state of the gate).

Per frame (step):
  1. collect a finished VLM call, if any        → gate.on_response → memory
  2. detect plates (docTR)                      → evidence/detect.py
  3. track plates across frames                 → evidence/buffer.py
  4. per track: R (relevance), resolve (fast path), φ → ℓ (legibility)
  5. gate.tick: NO_SIGN / ARMED / fire / fast path         → gate.py
  6. on fire: payload crop + parser hint → Reasoner.ask in the background

What the caller does with it:
  * read `.prompt` after each step and send it to the VLA (it is always one of
    straight / turn_left / turn_right / stop);
  * call `turn_done()` when the held maneuver is finished (the ROS node decides
    this from odometry), which re-arms the gate for the next sign;
  * call `set_goal()` to start a new trip (fresh gate + memory + buffer).

Nothing here is tuned: τ, the minimum plate height and the legibility weights all
come from the caller's config (ROS params file / CLI flags / calib/w.json).

CLI (laptop or Jetson, no ROS): feed a sequence of frames as if the robot drove past
them, with real docTR and a real VLM call:
    export GEMINI_API_KEY=...        # or OPENAI_API_KEY with --base-url
    python -m adaptive_reasoning.live --goal 49 --tau 0.45 frame_000.png frame_001.png ...
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .evidence.buffer import BufferConfig, EvidenceBuffer
from .evidence.detect import DetectConfig, OcrEngine, detect, payload_crop
from .evidence.features import phi, text_height_px
from .evidence.legibility import Legibility
from .evidence.relevance import Goal, relevance_lines
from .evidence.resolve import Plate, PlateLine, resolve
from .gate import EvidenceGate, GateConfig, GateState, PlateEvidence, State, VLMResponse, new_goal
from .memory import Memory
from .reasoner import Reasoner, VLMAnswer, structural_hint

DEFAULT_PROMPT = "straight"

# Appended to the parser hint on every call.  Without it the deployed prompt only knows
# "follow the arrow" or "not applicable", so the trip could never END: the robot would
# follow signs forever.  "stop" is already a valid direction in reasoner.py; ARRIVED in
# the summary separates "you are here" from a closure notice that also answers "stop".
ARRIVAL_CLAUSE = (
    "\nARRIVAL: if this sign does not POINT to the goal but MARKS the goal itself "
    "(e.g. the goal room's own door plate showing exactly the goal number, or an "
    "\"Elevators\" label mounted at the elevator doors with no arrow), the robot has "
    "arrived: answer applicable=true, direction \"stop\", and begin the summary with "
    "\"ARRIVED\". A sign that lists the goal next to an arrow is NOT arrival.")
_ROOM_WORD_RE = re.compile(r"^\s*(?:room|rm\.?|office)\s+", re.IGNORECASE)


def parse_goal(goal_text: str, goals_dir: Optional[Path] = None) -> Goal:
    """'room 6-217' → Goal(room=6-217); 'Main Elevators' → typed goal.

    Goal.parse only recognises a bare room code, so a leading 'room'/'rm' word is
    stripped for parsing (the full text still goes to the VLM).  A cached descriptor
    file goals/<goal>.json (written by evidence/expand_goal.py) supplies V_g."""
    key = _ROOM_WORD_RE.sub("", goal_text).strip()
    if goals_dir is not None:
        for name in (goal_text.strip(), key):
            f = Path(goals_dir) / f"{name}.json"
            if f.exists():
                g = Goal.load(f)
                return Goal.parse(key, g.descriptors)
    return Goal.parse(key)


@dataclass
class LiveConfig:
    tau: float                       # gate operating point (required, no default on purpose)
    fast_path: bool = True           # decide from OCR text + arrow glyph without a call
    # Acquire, don't fire, while the plate is shorter than this many pixels in the source
    # frame.  Stand-in for calibrated legibility (same knob as render_submission
    # --min-plate-h).  0 disables.
    min_plate_h: float = 60.0
    # After a failed call (network/API error) wait this long before the gate may fire again.
    retry_after_s: float = 2.0
    # VLM latency assumed until real calls have been measured (paper median: 4.32 s)
    default_latency_s: float = 4.0


@dataclass
class StepResult:
    t: int
    state: str                       # NO_SIGN | ARMED | PENDING | DECIDED
    prompt: str                      # what the VLA should be driving on now
    fired: bool = False              # a VLM call was started this frame
    answered: Optional[dict] = None  # a VLM call finished this frame
    fast_path: bool = False          # decided from the sign text this frame (no call)
    acquiring: bool = False          # fire vetoed: sign too small, keep approaching
    q: float = 0.0
    best_text: str = ""
    n_plates: int = 0
    text_h_px: float = 0.0           # median text-line height of the best relevant sign (speed cap)
    arrived: bool = False            # the trip's goal was reached this frame
    reason: str = ""
    ms: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class LiveReasoner:
    """Owns gate state, memory and the evidence buffer for one goal at a time."""

    def __init__(self, goal: str, cfg: LiveConfig, engine: OcrEngine, reasoner: Reasoner,
                 legibility: Optional[Legibility] = None, goals_dir: Optional[Path] = None,
                 det_cfg: Optional[DetectConfig] = None, buf_cfg: Optional[BufferConfig] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.engine = engine
        self.reasoner = reasoner
        self.legibility = legibility or Legibility.load(Path(__file__).parent / "calib" / "w.json")
        self.goals_dir = goals_dir
        self.det_cfg = det_cfg or DetectConfig()
        self.buf_cfg = buf_cfg
        self.clock = clock
        self.gate = EvidenceGate(GateConfig(tau=cfg.tau, fast_path=cfg.fast_path))
        self._pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")
        self._future: Optional[cf.Future] = None
        self._pending_plate: Optional[str] = None
        self._no_fire_until = 0.0
        self.calls = 0
        self.last_answer: Optional[VLMAnswer] = None
        self.latencies: list[float] = []
        self.arrived = False
        self._fire_pose: Optional[tuple] = None
        self.set_goal(goal)

    # ------------------------------------------------------------------ goal
    def set_goal(self, goal: str) -> None:
        """Start a new trip: fresh gate, memory and buffer.  An in-flight call for the
        old goal is abandoned (its answer is ignored when it lands)."""
        self.goal_text = goal.strip()
        self.goal = parse_goal(self.goal_text, self.goals_dir)
        self.state: GateState = new_goal()
        self.memory = Memory(self.goal_text)
        self.buffer = EvidenceBuffer(self.buf_cfg)
        self._future, self._pending_plate = None, None
        self.arrived = False

    @property
    def prompt(self) -> str:
        return self.state.decision or DEFAULT_PROMPT

    def turn_done(self, t: int = 0) -> None:
        """The held maneuver is complete → necessity back to 1 (memory is kept, so a
        sign already used for this goal is never asked about again)."""
        if self.state.state == State.DECIDED:
            self.state = self.gate.on_turn_done(self.state, t)

    @property
    def latency_estimate(self) -> float:
        """Median of the last 5 measured (non-cached) call latencies."""
        recent = sorted(self.latencies[-5:])
        return recent[len(recent) // 2] if recent else self.cfg.default_latency_s

    @property
    def call_in_flight(self) -> bool:
        return self._future is not None

    def wait_for_call(self, timeout: Optional[float] = None) -> None:
        if self._future is not None:
            cf.wait([self._future], timeout=timeout)

    def collect(self, t: int) -> Optional[dict]:
        """Land a finished VLM call now (step() also does this at the start of every frame)."""
        return self._collect(t)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ step
    def _collect(self, t: int) -> Optional[dict]:
        """Apply a finished VLM call to the gate.  Returns a summary dict or None."""
        fut = self._future
        if fut is None or not fut.done():
            return None
        plate_id = self._pending_plate
        self._future, self._pending_plate = None, None
        if self.state.state != State.PENDING or self.state.pending_plate != plate_id:
            return None                                   # stale (goal changed meanwhile)
        try:
            ans: VLMAnswer = fut.result()
        except Exception as e:                            # noqa: BLE001  network / API error
            # Do NOT consume the plate: the sign is still unread.  Back to NO_SIGN so the
            # gate can fire again on it after a short back-off.
            self.state = replace(self.state, state=State.NO_SIGN, pending_plate=None)
            self._no_fire_until = self.clock() + self.cfg.retry_after_s
            return {"error": f"{type(e).__name__}: {e}"[:300], "plate_id": plate_id}
        self.last_answer = ans
        if not ans.cached:
            self.latencies.append(ans.latency_s)
        self.state, upds = self.gate.on_response(
            self.state, VLMResponse(plate_id, ans.applicable, ans.direction), t)
        for u in upds:
            # stamp with the pose the sign was READ from, not where the answer landed
            self.memory.apply(replace(u, pose=self._fire_pose) if self._fire_pose else u)
        arrived = bool(ans.applicable and ans.direction == "stop"
                       and "ARRIVED" in (ans.summary or "").upper())
        self.arrived = self.arrived or arrived
        return {"plate_id": plate_id, "applicable": ans.applicable, "direction": ans.direction,
                "confidence": ans.confidence, "summary": ans.summary, "arrived": arrived,
                "latency_s": round(ans.latency_s, 3), "cached": ans.cached}

    def step(self, img: np.ndarray, t: int, pose: Optional[tuple] = None) -> StepResult:
        """One frame.  `pose` = (x, y, yaw) from odometry, or None if unavailable."""
        t0 = time.perf_counter()
        self.memory.set_pose(pose)
        answered = self._collect(t)

        plates = detect(img, self.engine, self.det_cfg)
        frame_boxes = [p.box for p in plates if p.box is not None]
        evidence: list[PlateEvidence] = []
        obs_of: dict[str, object] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")               # UNCALIBRATED warns once at startup instead
            for tr in self.buffer.update(plates, t):
                obs = tr.obs
                texts = [l.text for l in obs.lines]
                rel = relevance_lines(texts, self.goal)
                res = resolve(Plate([PlateLine(l.text, list(l.arrows)) for l in obs.lines]),
                              self.goal, rel=rel)
                ell = self.legibility(phi(obs, tr.agreement_frac))
                evidence.append(PlateEvidence(tr.plate_id, rel.score, ell, res, obs.text))
                obs_of[tr.plate_id] = obs

        prev = self.state
        tick = self.gate.tick(prev, evidence, self.memory, t)
        for u in tick.memory_updates:
            self.memory.apply(u)

        out = StepResult(t=t, state=tick.state.state.value, prompt=self.prompt, q=round(tick.q, 3),
                         best_text=tick.best.text if tick.best else "", n_plates=len(plates),
                         reason=tick.reason, answered=answered,
                         arrived=bool(answered and answered.get("arrived")))
        if tick.best is not None and tick.best.plate_id in obs_of:
            out.text_h_px = round(text_height_px(obs_of[tick.best.plate_id]), 1)

        if tick.fire:
            obs = obs_of.get(tick.best.plate_id)
            box = getattr(obs, "box", None)
            plate_h = float(box[3] - box[1]) if box is not None else 0.0
            if self.clock() < self._no_fire_until:
                out.acquiring, out.reason = True, "fire held: backing off after a failed call"
            elif plate_h < self.cfg.min_plate_h:
                # Readable enough by q, but still too few pixels for the VLM: keep the
                # pre-tick state (ARMED) and keep approaching.  This is ACQUIRE.
                out.acquiring = True
                out.reason = f"acquire: plate {plate_h:.0f}px < min_plate_h {self.cfg.min_plate_h:.0f}px"
            else:
                crop = payload_crop(img, box, frame_boxes, self.det_cfg) if box is not None else img
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hint = structural_hint(obs.text, self.goal.text) + ARRIVAL_CLAUSE
                self._fire_pose = pose
                summary = f"{len(self.memory)} signs consumed"
                self.state = tick.state                    # PENDING
                self._pending_plate = tick.best.plate_id
                self._future = self._pool.submit(
                    self.reasoner.ask, self.goal_text, [obs.text], [crop], None, summary,
                    {"t": t, "plate_id": tick.best.plate_id}, hint)
                self.calls += 1
                out.fired = True
        else:
            self.state = tick.state
            out.fast_path = any(u.outcome == "fast" for u in tick.memory_updates)

        if out.acquiring:
            self.state = replace(prev, state=State.ARMED)
        out.state = self.state.state.value
        out.prompt = self.prompt
        out.ms = round((time.perf_counter() - t0) * 1e3, 1)
        return out


# ----------------------------------------------------------------------------
# factory shared by the ROS node and the CLI
# ----------------------------------------------------------------------------

def build_live(goal: str, tau: float, *, model: str, base_url: str,
               reasoning: Optional[str] = "low", timeout_s: float = 15.0, max_retries: int = 1,
               min_plate_h: float = 60.0, fast_path: bool = True,
               calib: Optional[str] = None, goals_dir: Optional[str] = None,
               log_dir: Optional[str] = None, engine: Optional[OcrEngine] = None,
               client=None, expand_typed_goals: bool = True) -> LiveReasoner:
    """Wire docTR + the OpenAI-compatible VLM client + the gate into a LiveReasoner.
    The API key is read from the environment (GEMINI_API_KEY / OPENAI_API_KEY), never
    from a parameter, so it cannot end up in a params file or a log."""
    from .reasoner import OpenAICompatVLM
    if client is None:
        # rpm=0: no client-side pacing (the gate already makes ~1 call per sign);
        # few retries: a call that takes 60 s is useless to a moving robot.
        client = OpenAICompatVLM(model=model, base_url=base_url, reasoning_effort=reasoning,
                                 timeout=timeout_s, rpm=0.0, max_retries=max_retries)
    if engine is None:
        from .evidence.detect import DocTREngine
        engine = DocTREngine()
    log = Path(log_dir).expanduser() if log_dir else None
    reasoner = Reasoner(client, model_tag=f"{model}:{reasoning}",
                        cache_dir=(log / "vlm_cache") if log else Path("vlm_cache"),
                        payload_dir=(log / "payloads") if log else None,
                        save_payload=log is not None,
                        ledger_path=(log / "ledger.jsonl") if log else None,
                        eval_tag="live")
    gdir = Path(goals_dir).expanduser() if goals_dir else Path(__file__).parent / "goals"
    leg = Legibility.load(calib) if calib else Legibility.load(Path(__file__).parent / "calib" / "w.json")
    if not leg.calibrated:
        warnings.warn("legibility is UNCALIBRATED (no calib/w.json): using placeholder weights; "
                      "min_plate_h and tau carry the acquire decision.")
    if expand_typed_goals:
        ensure_goal_descriptors(goal, gdir, client, model)
    return LiveReasoner(goal, LiveConfig(tau=tau, fast_path=fast_path, min_plate_h=min_plate_h),
                        engine, reasoner, leg, goals_dir=gdir)


def ensure_goal_descriptors(goal: str, goals_dir: Path, client, model: str) -> Optional[Path]:
    """Typed goals ('Main Elevators') need V_g for semantic relevance.  One text-only
    call per goal, cached in goals/<goal>.json (expand_goal.py).  Room-code goals skip
    this: the structural channel handles them."""
    g = parse_goal(goal, goals_dir)
    f = Path(goals_dir) / f"{goal.strip()}.json"
    if g.room is not None or f.exists():
        return None
    from .evidence.expand_goal import expand_goal, write_goal_file
    try:
        desc = expand_goal(goal.strip(), client)
    except Exception as e:                                # noqa: BLE001
        warnings.warn(f"goal expansion failed ({e}); matching on the goal text alone")
        return None
    return write_goal_file(goal.strip(), desc, goals_dir, model, "university campus building")


# ----------------------------------------------------------------------------
# CLI: frames on disk → the live loop (real docTR, real VLM)
# ----------------------------------------------------------------------------

def main() -> None:
    import argparse
    from PIL import Image
    ap = argparse.ArgumentParser(description="run the live adaptive-reasoning loop over frames on disk")
    ap.add_argument("frames", nargs="+", help="images in driving order")
    ap.add_argument("--goal", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--min-plate-h", type=float, default=60.0)
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta/openai/")
    ap.add_argument("--reasoning", default="low", help="'none' to omit reasoning_effort")
    ap.add_argument("--log-dir", default="signway_logs/cli")
    ap.add_argument("--wait", action="store_true",
                    help="block on each VLM call (frames on disk do not move while the call runs)")
    a = ap.parse_args()
    reasoning = None if a.reasoning.lower() in ("", "none", "off") else a.reasoning
    live = build_live(a.goal, a.tau, model=a.model, base_url=a.base_url, reasoning=reasoning,
                      min_plate_h=a.min_plate_h, log_dir=a.log_dir)
    try:
        for t, path in enumerate(a.frames):
            img = np.asarray(Image.open(path).convert("RGB"))
            r = live.step(img, t)
            if a.wait and live.call_in_flight:
                live.wait_for_call()
                r.answered = live.collect(t)              # land the answer on the same frame
                r.state, r.prompt = live.state.state.value, live.prompt
            print(json.dumps({"frame": Path(path).name, **r.to_dict()}))
        if live.call_in_flight:
            live.wait_for_call()
            print(json.dumps({"final": live.collect(len(a.frames)), "prompt": live.prompt}))
    finally:
        live.close()
    print(f"VLM calls: {live.calls}  final prompt: {live.prompt}  state: {live.state.state.value}")


if __name__ == "__main__":
    main()
