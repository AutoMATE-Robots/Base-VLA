#!/usr/bin/env python3
"""
signway_reasoner_node.py -- goal-conditioned sign reasoning for the v8 drive stack.

    /image_raw --> docTR --> plate tracks --> adaptive gate --(fire)--> VLM API (cropped sign + goal)
                                                  |                          |
                                                  |  fast path (no call)     v  decision
                                                  +----------------> /vla/prompt --> v8_drive_node
    /odom ------> turn / pass-through completion --> re-arm the gate for the next sign

This node ONLY publishes prompts (straight / turn_left / turn_right / stop) on the same
topic the operator used to publish by hand. v8_drive_node turns the prompt into 8x2
waypoints and the LiDAR still has the final say on motion. The worst this node can do
is hold the wrong direction; it never touches /cmd_vel.

Gate (adaptive_reasoning/gate.py, paper Sec. III-B):
  NO_SIGN  no novel goal-relevant sign in view               -> keep current prompt
  ARMED    relevant sign seen, not readable yet (ACQUIRE)     -> keep driving closer
  PENDING  VLM call in flight (background thread)             -> keep current prompt
  DECIDED  direction held as the VLA prompt until the maneuver is done

A TRIP IS A CHAIN OF SIGNS, not one sign. After every executed decision the gate re-arms
and the robot looks for the next relevant sign, as many junctions as it takes, until a
sign says the robot has ARRIVED (the VLM answers "stop" with summary "ARRIVED ...", e.g.
the goal room's door plate or the label at the elevator doors).

Per decision (maneuver phases):
  turn_left / turn_right
    waiting   : decision held, prompt stays "straight" until /lidar/openings_array shows a
                passage on that side (wait_for_opening), so an early answer does not turn
                the robot into the first opening before the junction. Fallback: release
                the turn after max_wait_m of driving without seeing the opening.
    executing : prompt = the turn, until |net yaw| from /odom >= turn_done_deg
  straight    : executing until straight_done_m driven
  stop        : held; the trip is over (arrived, or a closure with no way round).
                Publish a new goal on /signway/goal to continue.
  then        : prompt back to "straight", gate re-armed. Memory keeps the signs already
                used, stamped with where they were read (/odom pose), so the SAME sign is
                not asked twice but an identical-looking sign at the next junction is.
  no odometry : turns are released at once and re-armed after decision_timeout_s.

Speed while a relevant sign is being read (ARMED / PENDING): /signway/speed_cap tells the
drive node to slow down according to the size of the sign's text, so the VLM answer lands
before the robot reaches the sign (adaptive_reasoning/maneuver.py). 0 = no cap.

The VLM API key is read from the environment ONLY (GEMINI_API_KEY or OPENAI_API_KEY);
it is never a ROS parameter, so it cannot leak into a params file or a log.

Run (after the drive stack from ROBOT_COMMANDS.md is up):
    source ~/v8_env.sh && source ~/.signway_secrets && unset ROS_NAMESPACE
    python3 -u ~/weights/corridor_nav/current/signway_reasoner_node.py --ros-args \
        --params-file ~/weights/corridor_nav/current/signway_params.yaml -p goal:="room 6-217"

Change the goal mid-run (fresh gate + memory):
    ros2 topic pub --once /signway/goal std_msgs/msg/String "{data: 'Main Elevators'}"
"""
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, String

# adaptive_reasoning/ lives at the repo root (two levels up from this file). If the node
# was copied elsewhere (e.g. ~/), fall back to $SIGNWAY_ROOT or ~/weights.
for _root in (Path(__file__).resolve().parents[2],
              Path(os.environ.get("SIGNWAY_ROOT", "~/weights")).expanduser()):
    if (_root / "adaptive_reasoning").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
        break
from adaptive_reasoning.live import build_live, ensure_goal_descriptors  # noqa: E402
from adaptive_reasoning.maneuver import opening_on_side, speed_cap_from_text  # noqa: E402

VALID_PROMPTS = ("straight", "turn_left", "turn_right", "stop")
TURNS = ("turn_left", "turn_right")


def decode_image(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> (H,W,3) uint8 RGB. Same decoder as v8_drive_node (honours step)."""
    h, w, step = msg.height, msg.width, msg.step
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if buf.size < h * step:
        raise ValueError(f"short image buffer: {buf.size} < {h * step}")
    rows = buf[: h * step].reshape(h, step)
    if enc in ("rgb8", "bgr8"):
        img = rows[:, : w * 3].reshape(h, w, 3)
        return np.ascontiguousarray(img[:, :, ::-1] if enc == "bgr8" else img)
    if enc == "mono8":
        return np.ascontiguousarray(np.repeat(rows[:, :w, None], 3, axis=2))
    if enc in ("uyvy", "uyvy422", "yuv422", "yuyv", "yuv422_yuy2"):
        packed = rows[:, : w * 2].reshape(h, w // 2, 4).astype(np.float32)
        if enc in ("yuyv", "yuv422_yuy2"):
            y0, u, y1, v = packed[..., 0], packed[..., 1], packed[..., 2], packed[..., 3]
        else:
            u, y0, v, y1 = packed[..., 0], packed[..., 1], packed[..., 2], packed[..., 3]
        cu, cv = u - 128.0, v - 128.0
        out = np.empty((h, w, 3), dtype=np.float32)
        for k, y in ((0, y0), (1, y1)):
            out[:, k::2, 0] = y + 1.402 * cv
            out[:, k::2, 1] = y - 0.344136 * cu - 0.714136 * cv
            out[:, k::2, 2] = y + 1.772 * cu
        return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))
    raise ValueError(f"unsupported encoding {msg.encoding!r}")


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class SignWayReasonerNode(Node):
    def __init__(self):
        super().__init__("signway_reasoner")
        p = self.declare_parameter
        p("goal", "")
        p("image_topic", "/image_raw")
        p("odom_topic", "/odom")
        p("prompt_topic", "/vla/prompt")
        p("reason_hz", 2.0)
        p("tau", -1.0)                       # REQUIRED: comes from signway_params.yaml
        p("min_plate_h", 60.0)
        p("fast_path", True)
        p("vlm_model", "gemini-3.6-flash")
        p("vlm_base_url", "https://generativelanguage.googleapis.com/v1beta/openai/")
        p("vlm_reasoning", "low")            # 'none' to omit reasoning_effort
        p("vlm_timeout_s", 15.0)
        p("vlm_max_retries", 1)
        p("calib_path", "")                  # '' = adaptive_reasoning/calib/w.json if present
        p("goals_dir", "")                   # '' = adaptive_reasoning/goals
        p("log_dir", "~/signway_logs")
        p("publish_prompts", True)           # false = log what would be published, publish nothing
        # maneuver completion
        p("turn_done_deg", 80.0)
        p("straight_done_m", 3.0)
        p("decision_timeout_s", 30.0)
        # hold a turn until the LiDAR sees the opening on that side
        p("wait_for_opening", True)
        p("opening_scans", 3)                # consecutive /lidar/openings_array scans
        p("max_wait_m", 20.0)                # release the turn anyway after this far
        # memory location rule
        p("memory_radius_m", 8.0)
        p("memory_yaw_tol_deg", 45.0)
        # speed cap from sign text size (adaptive_reasoning/maneuver.py)
        p("speed_cap", True)
        p("camera_fy_px", 2060.0)
        p("sign_letter_h_m", 0.05)
        p("cap_margin_s", 1.0)
        p("cap_buffer_m", 1.0)
        p("cap_min_speed", 0.15)
        g = lambda n: self.get_parameter(n).value

        tau = float(g("tau"))
        if tau < 0:
            raise ValueError("tau is required: pass --params-file signway_params.yaml or -p tau:=...")
        self.publish_prompts = bool(g("publish_prompts"))
        self.turn_done_rad = math.radians(float(g("turn_done_deg")))
        self.straight_done_m = float(g("straight_done_m"))
        self.decision_timeout = float(g("decision_timeout_s"))
        self.wait_for_opening = bool(g("wait_for_opening"))
        self.opening_scans = int(g("opening_scans"))
        self.max_wait_m = float(g("max_wait_m"))
        self.use_cap = bool(g("speed_cap"))
        self.cap_kw = dict(fy_px=float(g("camera_fy_px")), letter_h_m=float(g("sign_letter_h_m")),
                           margin_s=float(g("cap_margin_s")), d_buffer_m=float(g("cap_buffer_m")),
                           v_min=float(g("cap_min_speed")))
        self.period = 1.0 / max(float(g("reason_hz")), 0.1)
        reasoning = g("vlm_reasoning")
        reasoning = None if str(reasoning).lower() in ("", "none", "off") else reasoning

        self.session = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(g("log_dir")).expanduser() / self.session
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._events = open(self.log_dir / "events.jsonl", "a")

        goal = str(g("goal")).strip()
        self.get_logger().info("loading docTR + VLM client ...")
        self.live = build_live(
            goal or "none", tau, model=g("vlm_model"), base_url=g("vlm_base_url"),
            reasoning=reasoning, timeout_s=float(g("vlm_timeout_s")),
            max_retries=int(g("vlm_max_retries")), min_plate_h=float(g("min_plate_h")),
            fast_path=bool(g("fast_path")), calib=g("calib_path") or None,
            goals_dir=g("goals_dir") or None, log_dir=str(self.log_dir),
            expand_typed_goals=bool(goal))
        self.mem_radius = float(g("memory_radius_m"))
        self.mem_yaw = math.radians(float(g("memory_yaw_tol_deg")))
        self._configure_memory()
        self.active = bool(goal)

        self._live_lock = threading.Lock()   # self.live (held for a whole OCR step)
        self._man_lock = threading.Lock()    # maneuver phase + publishing (held briefly)
        self._img = None
        self._t = 0
        self._published = None               # last prompt we sent
        self._published_at = 0.0
        self._drive_prompt = None            # what v8_drive_node reports it is using
        # maneuver: phase in idle | waiting | executing | done | arrived
        self._phase = "idle"
        self._held = None
        self._held_since = 0.0
        self._yaw_accum = 0.0
        self._dist_accum = 0.0                # since the phase started
        self._pose = None                     # latest (x, y, yaw)
        self._odom_seen = 0.0
        self._open_streak = 0
        self._openings_seen = 0.0
        self._last_openings = []
        self._cap = 0.0

        cg_reason = MutuallyExclusiveCallbackGroup()
        cg_io = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, g("image_topic"), self._on_image,
                                 QoSPresetProfiles.SENSOR_DATA.value, callback_group=cg_io)
        self.create_subscription(Odometry, g("odom_topic"), self._on_odom, 50, callback_group=cg_io)
        self.create_subscription(Float32MultiArray, "/lidar/openings_array", self._on_openings, 10,
                                 callback_group=cg_io)
        self.create_subscription(String, "/signway/goal", self._on_goal, 10, callback_group=cg_io)
        self.create_subscription(String, "/vla/status", self._on_drive_status, 10, callback_group=cg_io)
        self.pub_prompt = self.create_publisher(String, g("prompt_topic"), 10)
        self.pub_status = self.create_publisher(String, "/signway/status", 10)
        self.pub_cap = self.create_publisher(Float32, "/signway/speed_cap", 10)
        self.create_timer(self.period, self._reason_step, callback_group=cg_reason)
        self._odom_check = self.create_timer(5.0, self._check_odom, callback_group=cg_io)

        self.get_logger().info(
            f"signway_reasoner up | goal='{goal or '(none: publish /signway/goal)'}' | tau={tau} "
            f"min_plate_h={g('min_plate_h')} | vlm={g('vlm_model')} | {g('reason_hz')} Hz | "
            f"wait_for_opening={self.wait_for_opening} speed_cap={self.use_cap} | "
            f"publish_prompts={self.publish_prompts} | logs={self.log_dir}")
        if self.active:
            self._publish("straight", why="trip start")

    def _configure_memory(self):
        self.live.memory.radius_m = self.mem_radius
        self.live.memory.yaw_tol_rad = self.mem_yaw

    # ------------------------------------------------------------------ helpers
    def _odom_ok(self) -> bool:
        return (time.monotonic() - self._odom_seen) < 2.0

    def _lidar_ok(self) -> bool:
        return (time.monotonic() - self._openings_seen) < 2.0

    def _check_odom(self):
        self._odom_check.cancel()
        if not self._odom_ok():
            self.get_logger().error(
                "NO /odom after 5 s. Turns cannot be confirmed (timeout fallback), signs cannot be "
                "told apart by location, and turns are not held for the opening. "
                "Check: ros2 topic hz /odom ; ros2 topic list | grep odom")
        else:
            self.get_logger().info("/odom OK")

    def _desired_prompt(self) -> str:
        """What the VLA should drive on, given the maneuver phase (call under _man_lock)."""
        if self._phase in ("executing", "arrived"):
            return self._held
        return "straight"                      # idle, waiting (turn held), done

    def _start_phase(self, phase: str):
        self._phase = phase
        self._yaw_accum = self._dist_accum = 0.0
        self._held_since = time.monotonic()

    # ------------------------------------------------------------------ inputs
    def _on_image(self, msg: Image):
        self._img = msg                      # keep only the newest frame

    def _on_goal(self, msg: String):
        goal = msg.data.strip()
        if not goal:
            return
        with self._live_lock:
            ensure_goal_descriptors(goal, self.live.goals_dir, self.live.reasoner.client,
                                    self.live.reasoner.model_tag)
            self.live.set_goal(goal)
            self._configure_memory()
            self.active = True
            with self._man_lock:
                self._held, self._phase = None, "idle"
        self.get_logger().info(f"NEW GOAL '{goal}' -> fresh gate + memory")
        self._log({"event": "goal", "goal": goal})
        self._publish("straight", why="new goal")

    def _on_drive_status(self, msg: String):
        # v8_drive_node publishes "prompt=X latency_ms=..."; if it restarted and lost our
        # prompt, re-send it.
        for tok in msg.data.split():
            if tok.startswith("prompt="):
                self._drive_prompt = tok[len("prompt="):]
        want = self._published
        settled = time.monotonic() - self._published_at > 3.0
        if self.active and want and settled and self._drive_prompt and self._drive_prompt != want:
            self._publish(want, why="drive node out of sync", force=True)

    def _on_openings(self, msg: Float32MultiArray):
        self._openings_seen = time.monotonic()
        self._last_openings = list(msg.data)
        with self._man_lock:
            if self._phase != "waiting":
                self._open_streak = 0
                return
            self._open_streak = self._open_streak + 1 if opening_on_side(msg.data, self._held) else 0
            if self._open_streak >= self.opening_scans:
                self._release_turn(f"LiDAR sees the {self._held.split('_')[1]} opening")

    def _release_turn(self, why: str):
        """waiting -> executing (call under _man_lock)."""
        self._start_phase("executing")
        self.get_logger().info(f"TURN RELEASED '{self._held}' ({why})")
        self._log({"event": "turn_released", "prompt": self._held, "why": why})
        self._publish_locked(self._held, why)

    def _on_odom(self, msg: Odometry):
        pos = msg.pose.pose.position
        pose = (pos.x, pos.y, yaw_from_quat(msg.pose.pose.orientation))
        self._odom_seen = time.monotonic()
        last, self._pose = self._pose, pose
        if last is None:
            return
        with self._man_lock:
            if self._phase not in ("waiting", "executing"):
                return
            # signed accumulation (left/right wobble cancels on a straight approach)
            self._yaw_accum += wrap(pose[2] - last[2])
            self._dist_accum += math.hypot(pose[0] - last[0], pose[1] - last[1])
            held = self._held
            if self._phase == "waiting":
                if self._dist_accum >= self.max_wait_m:
                    self.get_logger().warn(f"no {held.split('_')[1]} opening seen in "
                                           f"{self.max_wait_m:.0f} m -> releasing the turn anyway")
                    self._release_turn("max_wait_m reached")
                return
            done = ((held in TURNS and abs(self._yaw_accum) >= self.turn_done_rad) or
                    (held == "straight" and self._dist_accum >= self.straight_done_m))
            if done:
                self._phase = "done"
                self.get_logger().info(
                    f"MANEUVER DONE '{held}' (yaw {math.degrees(self._yaw_accum):+.0f} deg, "
                    f"{self._dist_accum:.1f} m) -> straight, looking for the next sign")
                # flip the prompt NOW; the reasoning thread re-arms the gate on its next step
                self._publish_locked("straight", f"{held} complete")

    # ------------------------------------------------------------------ the loop
    def _reason_step(self):
        msg = self._img
        if not self.active or msg is None:
            return
        try:
            img = decode_image(msg)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"decode: {e}", throttle_duration_sec=5.0)
            return
        pose = self._pose if self._odom_ok() else None
        with self._live_lock:
            self._t += 1
            self._rearm_if_done()
            try:
                r = self.live.step(img, self._t, pose)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"reasoning step failed: {type(e).__name__}: {e}",
                                        throttle_duration_sec=5.0)
                return
            self._rearm_if_done()             # odom may have finished the maneuver meanwhile
            self._start_decision(r)
            lat = self.live.latency_estimate

        self._report(r)
        self._update_cap(r, lat)
        with self._man_lock:
            want = self._desired_prompt()
            phase = self._phase
        if want != self._published:
            self._publish(want, why=f"gate {r.state} / {phase}")
        st = String()
        st.data = json.dumps({"goal": self.live.goal_text, "calls": self.live.calls, "phase": phase,
                              "held": self._held, "speed_cap": round(self._cap, 3), **r.to_dict()})
        self.pub_status.publish(st)
        if r.fired or r.answered or r.fast_path or r.acquiring or r.state != "NO_SIGN":
            self._log({**r.to_dict(), "phase": phase, "pose": pose, "speed_cap": self._cap})

    def _start_decision(self, r):
        """A new DECIDED state -> begin its maneuver (call under _live_lock)."""
        if self.live.state.state.value != "DECIDED":
            return
        with self._man_lock:
            if self._phase != "idle":
                return
            self._held = self.live.prompt
            src = self.live.state.decision_source
            if self._held == "stop":
                self._start_phase("arrived")
                what = "GOAL REACHED" if self.live.arrived else "STOP (route closed?)"
                self.get_logger().info(f"{what}: holding 'stop' | VLM calls this trip: "
                                       f"{self.live.calls} | publish /signway/goal to continue")
            elif self._held in TURNS and self.wait_for_opening and self._odom_ok() and self._lidar_ok():
                self._start_phase("waiting")
                self._open_streak = 0
                self.get_logger().info(f"DECISION '{self._held}' ({src}): holding 'straight' until "
                                       f"the LiDAR sees the opening")
                if opening_on_side(self._last_openings, self._held):
                    self._open_streak = 1          # already at the junction: confirm on next scans
            else:
                if self._held in TURNS and self.wait_for_opening:
                    self.get_logger().warn("no /odom or /lidar/openings_array: turning without waiting")
                self._start_phase("executing")
                self.get_logger().info(f"DECISION '{self._held}' ({src}) -> executing")
            self._log({"event": "decision", "prompt": self._held, "source": src,
                       "phase": self._phase, "arrived": self.live.arrived})

    def _rearm_if_done(self):
        """Maneuver finished (odom) or timed out (no odom) -> re-arm the gate (under _live_lock)."""
        with self._man_lock:
            if self._phase not in ("executing", "done"):
                return
            timed_out = (self._phase == "executing" and not self._odom_ok()
                         and time.monotonic() - self._held_since > self.decision_timeout)
            if self._phase != "done" and not timed_out:
                return
            if timed_out:
                self.get_logger().warn(f"no /odom: re-arming after {self.decision_timeout:.0f}s")
            held = self._held
            self._held, self._phase = None, "idle"
        self.live.turn_done(self._t)
        self._log({"event": "maneuver_done", "held": held, "timeout": timed_out})

    def _update_cap(self, r, latency_s: float):
        cap = 0.0
        if self.use_cap and (r.state in ("ARMED", "PENDING") or r.acquiring):
            v = speed_cap_from_text(r.text_h_px, latency_s=latency_s, **self.cap_kw)
            # sign out of view while the call is in flight: keep the last cap
            cap = v if v is not None else (self._cap if r.state == "PENDING" else 0.0)
        if cap != self._cap and (cap == 0.0 or self._cap == 0.0):
            self.get_logger().info(f"SPEED CAP {'off' if cap == 0 else f'{cap:.2f} m/s'} "
                                   f"(text {r.text_h_px:.0f}px, VLM latency ~{latency_s:.1f}s)")
        self._cap = cap
        m = Float32()
        m.data = float(cap)
        self.pub_cap.publish(m)

    def _report(self, r):
        if r.fired:
            self.get_logger().info(f"FIRE q={r.q} sign='{r.best_text[:60]}' -> VLM call "
                                   f"#{self.live.calls} (robot keeps driving)")
        if r.answered:
            a = r.answered
            if "error" in a:
                self.get_logger().warn(f"VLM call failed: {a['error']}  (will retry)")
            else:
                self.get_logger().info(
                    f"VLM ANSWER applicable={a['applicable']} direction={a['direction']} "
                    f"arrived={a['arrived']} conf={a['confidence']:.2f} {a['latency_s']}s | "
                    f"{a['summary'][:100]}")
        if r.fast_path:
            self.get_logger().info(f"FAST PATH '{r.best_text[:60]}' -> {r.prompt} (no VLM call)")
        if r.acquiring:
            self.get_logger().info(f"ACQUIRE {r.reason}", throttle_duration_sec=2.0)
        self.get_logger().info(
            f"[{r.state}/{self._phase}] prompt={self._published} q={r.q} plates={r.n_plates} "
            f"read='{r.best_text[:40]}' {r.ms:.0f}ms", throttle_duration_sec=2.0)

    # ------------------------------------------------------------------ outputs
    def _publish(self, prompt: str, why: str = "", force: bool = False):
        with self._man_lock:
            self._publish_locked(prompt, why, force)

    def _publish_locked(self, prompt: str, why: str = "", force: bool = False):
        if prompt not in VALID_PROMPTS:
            self.get_logger().error(f"refusing to publish invalid prompt {prompt!r}")
            return
        if prompt == self._published and not force:
            return
        self._published = prompt
        self._published_at = time.monotonic()
        if self.publish_prompts:
            m = String()
            m.data = prompt
            self.pub_prompt.publish(m)
            self.get_logger().info(f"PROMPT -> {prompt}  ({why})")
        else:
            self.get_logger().warn(f"[publish_prompts=false] would send PROMPT -> {prompt} ({why})")
        self._log({"event": "prompt", "prompt": prompt, "why": why})

    def _log(self, row: dict):
        try:
            self._events.write(json.dumps({"ts": time.time(), **row}) + "\n")
            self._events.flush()
        except Exception:  # noqa: BLE001
            pass

    def shutdown(self):
        self.live.close()
        try:
            m = Float32()
            m.data = 0.0
            self.pub_cap.publish(m)          # never leave a cap behind
        except Exception:  # noqa: BLE001
            pass
        self._events.close()
        self.get_logger().info(f"VLM calls this run: {self.live.calls} | logs: {self.log_dir}")


def main():
    rclpy.init(args=sys.argv)
    node = SignWayReasonerNode()
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
