#!/usr/bin/env python3
"""
signway_node_viz.py -- run the REAL signway_reasoner_node unchanged, plus read-only hooks that
publish what the gate saw on every step as JSON on /signway/debug (for signway_viewer.py).

Nothing in the decision path is changed: each hook calls the original function with the same
arguments and returns its result untouched; it only copies values out. Same CLI as the node:

  source ~/v8_env.sh && source ~/signway_handoff/cudnn_tegra_env.sh && source ~/.signway_secrets
  unset ROS_NAMESPACE
  python3 -u ~/signway_viz/signway_node_viz.py --ros-args \
      --params-file ~/weights/corridor_nav/current/signway_params.yaml \
      -p goal:="Elevators" -p publish_prompts:=false

Not part of the repo (test tooling for the Jetson handoff).
"""
import importlib.util
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

NODE_PATH = Path(os.environ.get(
    "SIGNWAY_NODE", "~/weights/corridor_nav/current/signway_reasoner_node.py")).expanduser()
spec = importlib.util.spec_from_file_location("signway_reasoner_node", NODE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)          # adds the repo root to sys.path and imports adaptive_reasoning

import rclpy                                               # noqa: E402
from rclpy.executors import MultiThreadedExecutor          # noqa: E402
from std_msgs.msg import String                            # noqa: E402
import adaptive_reasoning.live as live_mod                 # noqa: E402
from adaptive_reasoning.evidence.buffer import EvidenceBuffer   # noqa: E402
from adaptive_reasoning.gate import EvidenceGate           # noqa: E402
from adaptive_reasoning.reasoner import Reasoner           # noqa: E402

if os.environ.get("SIGNWAY_VLM", "").lower() == "jev":
    # TypeSafe Jev instead of the OpenAI-compatible VLM: build_live() looks the class up in
    # adaptive_reasoning.reasoner at call time, so swapping it here is enough.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import adaptive_reasoning.reasoner as _reasoner        # noqa: E402
    from jev_vlm import JevVLM                             # noqa: E402
    _reasoner.OpenAICompatVLM = JevVLM
    print("signway_node_viz: VLM = TypeSafe Jev (SIGNWAY_VLM=jev)", flush=True)

VIZ_DIR = Path("~/signway_logs/viz_payloads").expanduser()
VIZ_DIR.mkdir(parents=True, exist_ok=True)

_step = {}                # filled during one LiveReasoner.step (steps never overlap: _live_lock)
_calls = []               # one entry per VLM ask (written from the VLM thread)
_calls_lock = threading.Lock()


def _box(b):
    return None if b is None else [round(float(v), 1) for v in b]


# ---------------------------------------------------------------- read-only hooks
_orig_detect = live_mod.detect


def detect_hook(img, engine, cfg=None):
    plates = _orig_detect(img, engine, cfg)
    _step["ocr"] = [{"box": _box(getattr(p, "box", None)), "text": getattr(p, "text", "")[:120],
                     "lines": [{"text": l.text[:60], "box": _box(getattr(l, "box", None)),
                                "arrows": list(getattr(l, "arrows", []) or [])}
                               for l in getattr(p, "lines", [])]}
                    for p in plates]
    _step["img_hw"] = list(img.shape[:2])
    return plates


live_mod.detect = detect_hook

_orig_update = EvidenceBuffer.update


def update_hook(self, plates, t):
    tracks = _orig_update(self, plates, t)
    _step["tracks"] = {tr.plate_id: {"box": _box(tr.obs.box), "text": tr.obs.text[:120],
                                     "agree": round(float(tr.agreement_frac), 2),
                                     "h": round(float(tr.obs.box[3] - tr.obs.box[1]), 1)
                                     if tr.obs.box is not None else 0.0}
                       for tr in tracks}
    return tracks


EvidenceBuffer.update = update_hook

_orig_tick = EvidenceGate.tick


def tick_hook(self, prev, plates, memory, t):
    res = _orig_tick(self, prev, plates, memory, t)
    ev = []
    for p in plates:
        r = p.resolution
        ev.append({"id": p.plate_id, "R": round(p.relevance, 3), "L": round(p.legibility, 3),
                   "q": round(p.q, 3), "novel": bool(memory.is_novel(p.plate_id)),
                   "res": None if r is None else {"status": r.status, "prompt": r.vla_prompt,
                                                  "reason": r.reason[:80]}})
    _step["evidence"] = ev
    _step["tick"] = {"best": res.best.plate_id if res.best else None, "fire": bool(res.fire),
                     "reason": res.reason, "q": round(res.q, 3), "tau": self.cfg.tau}
    return res


EvidenceGate.tick = tick_hook

_orig_ask = Reasoner.ask


def ask_hook(self, goal, plate_texts, crops, *a, **kw):
    n = None
    with _calls_lock:
        n = len(_calls) + 1
        path = VIZ_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_call{n:02d}.jpg"
        if crops:
            try:
                cv2.imwrite(str(path), cv2.cvtColor(np.asarray(crops[0]), cv2.COLOR_RGB2BGR))
            except Exception:                                   # noqa: BLE001
                path = None
        _calls.append({"n": n, "t_start": time.time(), "crop": str(path) if path else None,
                       "texts": [s[:120] for s in plate_texts], "done": False})
    try:
        ans = _orig_ask(self, goal, plate_texts, crops, *a, **kw)
    except Exception as e:                                      # noqa: BLE001
        with _calls_lock:
            _calls[n - 1].update(done=True, error=f"{type(e).__name__}: {e}"[:200],
                                 latency=round(time.time() - _calls[n - 1]["t_start"], 2))
        raise
    with _calls_lock:
        _calls[n - 1].update(done=True, latency=round(time.time() - _calls[n - 1]["t_start"], 2),
                             answer={"applicable": ans.applicable, "direction": ans.direction,
                                     "confidence": ans.confidence, "arrived": getattr(ans, "arrived", False),
                                     "summary": (ans.summary or "")[:200]})
    return ans


Reasoner.ask = ask_hook

_orig_step = live_mod.LiveReasoner.step


def step_hook(self, img, t, pose=None):
    r = _orig_step(self, img, t, pose)
    _step["result"] = r.to_dict()
    return r


live_mod.LiveReasoner.step = step_hook


def instrument(node):
    pub = node.create_publisher(String, "/signway/debug", 10)
    orig_reason_step = node._reason_step

    def reason_step():
        _step.clear()
        msg = node._img
        stamp = None if msg is None else msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        orig_reason_step()
        if "result" not in _step:                     # inactive / decode failure / step error
            return
        mem = node.live.memory
        with _calls_lock:
            calls = [dict(c) for c in _calls[-5:]]
        d = {
            "wall": time.time(), "stamp": stamp, **_step,
            "goal": node.live.goal_text, "phase": node._phase, "held": node._held,
            "published": node._published, "publish_prompts": node.publish_prompts,
            "cap": round(node._cap, 3), "calls": node.live.calls,
            "latency_est": round(node.live.latency_estimate, 2),
            "min_plate_h": node.live.cfg.min_plate_h,
            "fy": node.cap_kw.get("fy_px"), "letter_h": node.cap_kw.get("letter_h_m"),
            "open_streak": node._open_streak, "opening_scans": node.opening_scans,
            "wait_for_opening": node.wait_for_opening,
            "yaw_accum_deg": round(math.degrees(node._yaw_accum), 1),
            "dist_accum": round(node._dist_accum, 2), "max_wait_m": node.max_wait_m,
            "turn_done_deg": round(math.degrees(node.turn_done_rad), 1),
            "pose": None if node._pose is None else [round(v, 3) for v in node._pose],
            "odom_ok": node._odom_ok(), "lidar_ok": node._lidar_ok(),
            "memory": [{"id": k, "outcome": u.outcome, "decision": u.decision,
                        "pose": None if u.pose is None else [round(v, 2) for v in u.pose]}
                       for k, u in list(mem.consumed.items())[-8:]],
            "vlm_calls": calls,
        }
        m = String()
        m.data = json.dumps(d, default=str)
        pub.publish(m)

    node._reason_step = reason_step
    # the node's timer captured the bound method at construction; re-point it
    for tmr in node.timers:
        if getattr(tmr.callback, "__name__", "") == "_reason_step":
            tmr.callback = reason_step
    node.get_logger().info("viz hooks on: publishing /signway/debug")


def main():
    rclpy.init(args=sys.argv)
    node = mod.SignWayReasonerNode()
    instrument(node)
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
