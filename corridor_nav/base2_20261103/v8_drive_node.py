#!/usr/bin/env python3
"""
v8_drive_node.py -- drive the v8 trajectory VLA on the Jetson.

    /image_raw --> [inference timer] --> 8 waypoints --> [20 Hz pure pursuit] --> /cmd_vel
    /vla/prompt (std_msgs/String) --> which prompt the policy is asked
    /vla/path   (nav_msgs/Path)   --> predicted trajectory, for rviz

DRY RUN IS THE DEFAULT. Nothing is published to /cmd_vel and the publisher is not even
created unless you pass --live. Run:

    source /opt/ros/humble/setup.bash && source ~/v8env/bin/activate
    python3 ~/v8_drive_node.py                       # dry run, prints what it WOULD send
    python3 ~/v8_drive_node.py --live                # actually drives. Robot on blocks.

VERIFIED ON THIS MACHINE (2026-10-21)
  * rclpy imports fine inside ~/v8env with no venv changes: ROS's setup.bash puts
    /opt/ros/humble/local/lib/python3.10/dist-packages on PYTHONPATH, and PYTHONPATH is
    searched even by a venv built without --system-site-packages (venv isolation blocks
    the system site-packages dir, not PYTHONPATH). numpy still resolves to the venv's
    1.26.4 because ROS ships no numpy of its own. Source ROS first, then activate.
  * /image_raw publishes encoding "rgb8", NOT uyvy -- 1920x1280, step 5760 (= 1920*3).
    `pixel_format:=uyvy` selects the CAPTURE format; v4l2_camera converts to RGB before
    publishing. The uyvy branch below is kept for other drivers but does not run here.
  * Measured ~25-28 Hz on /image_raw (the framerate:=20.0 parameter is not honored; the
    sensor runs at 30 fps and the node passes frames through).
  * The base listens on /cmd_vel -- NOT namespaced. One subscriber, node name "hunter",
    namespace "/", type geometry_msgs/msg/Twist, RELIABLE.
  * ~/.bashrc exports ROS_NAMESPACE=robot1. That would push this node's topics to
    /robot1/cmd_vel, where NOTHING is listening. Every topic default below is therefore
    ABSOLUTE (leading "/"), which ROS does not namespace-remap. Do not make them relative.

CAVEAT worth knowing before the first live run: waypoints are cumulative positions in the
frame of the image that produced them. Between inferences the robot moves, so by the end
of an inference period the lookahead point is expressed in a slightly stale frame. At low
speed with a short stale_after this is small, and it is what the open-loop eval assumed,
but if you push the speed up the correct fix is to transform the stored waypoints by the
odometry delta since traj_stamp. This node deliberately does not subscribe to /odom.
"""
import math
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, String

VALID_PROMPTS = ("straight", "turn_left", "turn_right", "stop")


# ---------------------------------------------------------------------------------------
# image decoding -- numpy only, no cv_bridge
# ---------------------------------------------------------------------------------------
def decode_image(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> contiguous (H,W,3) uint8 RGB.

    Honours msg.step rather than assuming width*channels: ROS rows may be padded, and
    slicing a padded buffer as if it were tight shears the image diagonally.
    """
    h, w, step = msg.height, msg.width, msg.step
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if buf.size < h * step:
        raise ValueError(f"short image buffer: {buf.size} < {h * step}")
    rows = buf[: h * step].reshape(h, step)

    if enc in ("rgb8", "bgr8"):
        img = rows[:, : w * 3].reshape(h, w, 3)
        if enc == "bgr8":
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)

    if enc == "mono8":
        g = rows[:, :w]
        return np.ascontiguousarray(np.repeat(g[:, :, None], 3, axis=2))

    if enc in ("uyvy", "uyvy422", "yuv422", "yuyv", "yuv422_yuy2"):
        # 4 bytes carry 2 pixels; the two orderings differ only in which byte holds Y.
        packed = rows[:, : w * 2].reshape(h, w // 2, 4).astype(np.float32)
        if enc in ("yuyv", "yuv422_yuy2"):
            y0, u, y1, v = packed[..., 0], packed[..., 1], packed[..., 2], packed[..., 3]
        else:
            u, y0, v, y1 = packed[..., 0], packed[..., 1], packed[..., 2], packed[..., 3]
        cu, cv = u - 128.0, v - 128.0
        out = np.empty((h, w, 3), dtype=np.float32)
        for k, y in ((0, y0), (1, y1)):                       # BT.601, full range
            out[:, k::2, 0] = y + 1.402 * cv
            out[:, k::2, 1] = y - 0.344136 * cu - 0.714136 * cv
            out[:, k::2, 2] = y + 1.772 * cu
        return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))

    raise ValueError(f"unsupported encoding {msg.encoding!r} "
                     "(handled: rgb8, bgr8, mono8, uyvy/yuyv)")


# ---------------------------------------------------------------------------------------
# pure pursuit
# ---------------------------------------------------------------------------------------
def waypoints_to_twist(wp, *, speed, lookahead_dist, max_speed, max_omega,
                       max_curvature=0.0, min_forward=0.05):
    """(8,2) cumulative waypoints -> (v, omega) target, before any ramping.

    Returns (0.0, 0.0) whenever the geometry is not safely drivable, so the caller can
    treat "stop" and "cannot drive" identically.

    Pure pursuit proper: pick the first waypoint at least `lookahead_dist` along the
    path, then the arc through the origin, tangent to the robot's heading, that reaches
    (x,y) has curvature

        kappa = 2*y / L**2,      L**2 = x**2 + y**2

    and omega = v * kappa. Selecting the lookahead by DISTANCE rather than by a fixed
    index matters here: the 8 waypoints span ~0.8 s, so at low speed a fixed far index
    is an aggressive target and at high speed a fixed near index is a twitchy one.
    """
    wp = np.asarray(wp, dtype=np.float64)
    if wp.ndim != 2 or wp.shape[1] != 2 or wp.shape[0] < 1:
        return 0.0, 0.0
    if not np.all(np.isfinite(wp)):                      # NaN/inf from a bad forward pass
        return 0.0, 0.0

    d = np.hypot(wp[:, 0], wp[:, 1])
    idx = int(np.argmax(d >= lookahead_dist)) if np.any(d >= lookahead_dist) else len(wp) - 1
    x, y = float(wp[idx, 0]), float(wp[idx, 1])

    # Target at or behind the axle: pure pursuit is undefined, and the base is Ackermann
    # (AgileX Hunter) so it cannot rotate in place to recover. Stop instead of spinning.
    if x <= min_forward:
        return 0.0, 0.0

    L2 = x * x + y * y
    if L2 < 1e-9:
        return 0.0, 0.0
    kappa = 2.0 * y / L2
    if max_curvature > 0.0:                              # steering-angle limit, if set
        kappa = float(np.clip(kappa, -max_curvature, max_curvature))

    v = float(np.clip(speed, 0.0, max_speed))
    omega = float(np.clip(v * kappa, -max_omega, max_omega))
    return v, omega


def arc_free_lengths(kappas, ox, oy, half_width, horizon, side_half_width=None,
                     front_reach=0.0, margin_ramp=0.0):
    """Free distance along each constant-curvature arc before the robot's swept band
    (|lateral offset from the arc| <= half_width) touches an obstacle point.

    kappas (K,) in 1/m, + = left. ox, oy (P,) obstacle points in base_link. Returns (K,)
    arc lengths in metres, capped at horizon. Only the half-circle ahead is considered,
    so returns behind the robot never block a forward arc.

    Points within front_reach of arc length are BESIDE the body, not ahead of it, and are
    tested against the narrower side_half_width. Using the full ahead-margin there made
    an object 0.5-0.6 m to the side (a 12-22 cm gap) read as blocked 0.07 m ahead, and the
    robot stopped next to things it was not going to touch. Past the bumper the band
    widens linearly to half_width over margin_ramp metres: the next half metre of path is
    near-certain, so a full margin there only stops the robot beside things it would clear.
    """
    if side_half_width is None:
        side_half_width = half_width
    kappas = np.asarray(kappas, dtype=np.float64)
    out = np.full(kappas.shape, float(horizon))
    if ox.size == 0:
        return out
    for i, k in enumerate(kappas):
        if abs(k) < 1e-3:
            off, s_all, fwd = np.abs(oy), ox, ox > 0.0
        else:
            R = 1.0 / k                        # signed; centre of the turn is (0, R)
            dy = oy - R
            r = np.hypot(ox, dy)
            # Angle swept from the start point (0, 0), which sits at -90 deg about a left
            # centre / +90 deg about a right one; left turns run CCW, right turns CW.
            th = np.arctan2(dy, ox)
            th0 = math.atan2(-R, 0.0)
            phi = np.mod((th - th0) if k > 0 else (th0 - th), 2.0 * math.pi)
            off, s_all, fwd = np.abs(r - abs(R)), abs(R) * phi, phi <= math.pi
        grow = np.clip((s_all - front_reach) / margin_ramp, 0.0, 1.0) if margin_ramp > 0 else \
            (s_all >= front_reach).astype(np.float64)
        hw = side_half_width + (half_width - side_half_width) * grow
        hit = fwd & (off <= hw)
        s = s_all[hit]
        if s.size:
            out[i] = min(float(horizon), float(np.min(s)))
    return out


class V8DriveNode(Node):
    def __init__(self, live: bool):
        super().__init__("v8_drive")
        self.live = live

        p = self.declare_parameter
        # Absolute topic names on purpose -- see ROS_NAMESPACE note in the docstring.
        p("image_topic", "/image_raw")
        p("cmd_topic", "/cmd_vel")
        p("prompt_topic", "/vla/prompt")
        p("checkpoint", "/home/hunter/weights/v8_vla")
        p("prompt", "straight")
        p("infer_hz", 2.0)
        p("control_hz", 20.0)
        p("speed", 0.4)             # fixed commanded speed, m/s
        p("lookahead_dist", 0.6)    # m along the predicted path
        p("max_speed", 0.6)
        p("max_omega", 1.0)
        p("max_curvature", 0.0)     # 1/m; 0 disables. Ackermann limit = tan(delta_max)/wheelbase
        p("accel", 0.5)             # m/s^2 ramp up
        p("decel", 1.5)             # m/s^2 ramp down (faster than accel, on purpose)
        p("angular_accel", 2.0)     # rad/s^2
        p("stale_after", 1.0)       # s; older waypoints -> hard zero
        p("jpeg_roundtrip", True)   # matches training preprocessing; costs latency
        # --- safety (obstacle avoidance). Deliberately strict: these OVERRIDE the policy. ---
        p("use_safety", True)
        p("stop_distance", 1.00)    # m of clear path ahead below which v is forced to 0
        p("slow_distance", 4.00)    # m below which v is ramped down linearly toward stop.
        # Was 0.70 / 1.80: by the time the robot reacted it was too close to steer round
        # anything, given the ~1.6 m minimum turning radius.
        p("emergency_distance", 0.40)  # m to ANY forward return -> hard zero
        p("safety_stale", 0.7)      # s without a clearance message -> treat as blocked
        # --- creep recovery: a stop must be escapable. Ackermann cannot rotate in place,
        # so the only way out of a blocked lane is to roll slowly while steering. ---
        p("creep_speed", 0.10)      # m/s while recovering. NOTE min turning radius ~1.6 m,
        # so at 0.10 m/s the best turn rate is 0.062 rad/s and 45 deg takes 12.6 s, past
        # the 8 s timeout. Use ~0.25 in practice, or raise recovery_timeout.
        p("creep_lookahead", 1.0)   # m; pure-pursuit target distance toward the opening
        p("recovery_timeout", 8.0)  # s in creep before giving up and hard-stopping
        p("recovery_exit_margin", 0.25)  # m of clearance ABOVE stop_distance before exiting
        p("recovery_retarget_deg", 50.0) # re-pick the target only if it moves beyond this
        # --- arc avoidance: when the policy's path is blocked, steer onto the clearest arc
        # instead of driving at the obstacle. Needs /lidar/bins from lidar_openings_node. ---
        p("use_avoidance", True)
        p("avoid_distance", 3.0)    # m; engage once the policy's arc is blocked closer than this
        p("avoid_max_curvature", 0.6)   # 1/m; ~1.6 m min turning radius of the Hunter
        p("avoid_n_arcs", 25)       # candidate arcs spread over +-avoid_max_curvature
        p("avoid_margin", 0.25)     # m added to half the robot width for the swept band
        p("avoid_side_margin", 0.05)    # m beyond half the robot width, for points BESIDE the body
        p("avoid_front_reach", 0.50)    # m from base_link to the front bumper
        p("avoid_margin_ramp", 1.50)    # m past the bumper over which the margin grows to full
        p("avoid_need", 3.5)        # m; an arc clear this far is "enough": take the gentlest such
        p("avoid_horizon", 6.0)     # m; how far ahead obstacles are looked for along each arc
        p("avoid_dev_weight", 1.0)  # m of clearance traded per 1/m of deviation from the policy
        p("avoid_min_gain", 0.3)    # m; only override if the new arc is at least this much clearer
        p("avoid_switch_penalty", 1.0)  # m; cost of flipping sides once committed
        p("avoid_settle", 2.0)      # s after avoiding during which turning is limited
        p("avoid_settle_omega", 0.15)   # rad/s cap during that settle window
        # --- corridor centring: nudges omega to hold the middle of a corridor ---
        p("use_centering", True)
        p("center_gain", 0.45)      # rad/s per metre of lateral error (P)
        p("center_damp", 0.35)      # rad/s per (m/s) of error RATE (D) -- kills the weave
        p("center_deadband", 0.05)  # m; ignore errors this small so it stops twitching
        p("center_max", 0.35)       # rad/s cap on the centring term
        p("center_only_corridor", True)   # do not fight the policy mid-turn
        g = lambda n: self.get_parameter(n).value

        self.speed = float(g("speed"))
        self.lookahead_dist = float(g("lookahead_dist"))
        self.max_speed = float(g("max_speed"))
        self.max_omega = float(g("max_omega"))
        self.max_curvature = float(g("max_curvature"))
        self.accel = float(g("accel"))
        self.decel = float(g("decel"))
        self.angular_accel = float(g("angular_accel"))
        self.stale_after = float(g("stale_after"))
        self.use_safety = bool(g("use_safety"))
        self.stop_distance = float(g("stop_distance"))
        self.slow_distance = float(g("slow_distance"))
        self.emergency_distance = float(g("emergency_distance"))
        self.safety_stale = float(g("safety_stale"))
        self.creep_speed = float(g("creep_speed"))
        self.creep_lookahead = float(g("creep_lookahead"))
        self.recovery_timeout = float(g("recovery_timeout"))
        self.recovery_exit_margin = float(g("recovery_exit_margin"))
        self.recovery_retarget_deg = float(g("recovery_retarget_deg"))
        self.use_avoidance = bool(g("use_avoidance"))
        self.avoid_distance = float(g("avoid_distance"))
        self.avoid_max_curvature = float(g("avoid_max_curvature"))
        self.avoid_n_arcs = max(3, int(g("avoid_n_arcs")))
        self.avoid_half_width = 0.5 * 0.75 + float(g("avoid_margin"))   # robot_width 0.75
        self.avoid_side_half_width = 0.5 * 0.75 + float(g("avoid_side_margin"))
        self.avoid_front_reach = float(g("avoid_front_reach"))
        self.avoid_margin_ramp = float(g("avoid_margin_ramp"))
        self.avoid_need = float(g("avoid_need"))
        self.avoid_settle = float(g("avoid_settle"))
        self.avoid_settle_omega = float(g("avoid_settle_omega"))
        self.avoid_horizon = float(g("avoid_horizon"))
        self.avoid_dev_weight = float(g("avoid_dev_weight"))
        self.avoid_min_gain = float(g("avoid_min_gain"))
        self.avoid_switch_penalty = float(g("avoid_switch_penalty"))
        self.use_centering = bool(g("use_centering"))
        self.center_gain = float(g("center_gain"))
        self.center_damp = float(g("center_damp"))
        self.center_deadband = float(g("center_deadband"))
        self.center_max = float(g("center_max"))
        self.center_only_corridor = bool(g("center_only_corridor"))

        self.prompt = g("prompt")
        if self.prompt not in VALID_PROMPTS:
            raise ValueError(f"prompt must be one of {VALID_PROMPTS}")

        # shared state between the two timers
        self._lock = threading.Lock()
        self._msg = None            # latest raw Image, decoded on the inference timer
        self._traj = None
        self._traj_stamp = 0.0
        self._v = 0.0               # last commanded, for the ramps
        self._w = 0.0
        self._last_ctrl = time.monotonic()
        self._lat = []
        self._clear = None          # /lidar/clearance  (robot-wide lane ahead)
        self._near = None           # /lidar/min_dist   (anything in the forward half)
        self._safety_stamp = 0.0
        self._cerr = None           # /lidar/center_error
        self._cerr_stamp = 0.0
        self._cerr_rate = 0.0       # filtered d(e)/dt, the damping term
        self._scene = None          # /lidar/junction_type
        self._gate = "init"         # why the last twist was limited, for the console
        self._ops = []              # [(bearing_deg, width_deg, depth_m), ...] from LiDAR
        self._recover_t0 = None     # monotonic time creep recovery began, None = not in it
        self._recover_bearing = None  # LATCHED escape bearing; see _creep_recovery
        self._obs = None            # (ox, oy) obstacle points from /lidar/bins
        self._obs_stamp = 0.0
        self._avoid_side = 0        # +1 left / -1 right once committed to going round, 0 = not
        self._avoid_clear_t = -1e9  # monotonic time avoidance last released

        self.get_logger().info("loading policy (takes ~15 s)...")
        sys.path.insert(0, "/home/hunter")
        from v8_policy import load_policy
        self.policy = load_policy(g("checkpoint"), jpeg_roundtrip=bool(g("jpeg_roundtrip")))
        self.get_logger().info("policy ready")

        # Two callback groups so a ~hundreds-of-ms inference cannot stall the 20 Hz
        # control loop. With a single-threaded executor the control timer would simply
        # not run while predict() was busy, and the twist would go stale and jerky.
        self.cg_infer = MutuallyExclusiveCallbackGroup()
        self.cg_ctrl = MutuallyExclusiveCallbackGroup()

        self.create_subscription(Image, g("image_topic"), self._on_image,
                                 QoSPresetProfiles.SENSOR_DATA.value,
                                 callback_group=self.cg_infer)
        self.create_subscription(String, g("prompt_topic"), self._on_prompt, 10,
                                 callback_group=self.cg_infer)
        # Safety and centring run on the CONTROL group: they must keep flowing at 20 Hz
        # even while a ~500 ms inference is occupying the other thread.
        self.create_subscription(Float32, "/lidar/clearance", self._on_clear, 10,
                                 callback_group=self.cg_ctrl)
        self.create_subscription(Float32, "/lidar/min_dist", self._on_near, 10,
                                 callback_group=self.cg_ctrl)
        self.create_subscription(Float32, "/lidar/center_error", self._on_cerr, 10,
                                 callback_group=self.cg_ctrl)
        self.create_subscription(String, "/lidar/junction_type", self._on_scene, 10,
                                 callback_group=self.cg_ctrl)
        self.create_subscription(Float32MultiArray, "/lidar/openings_array", self._on_ops, 10,
                                 callback_group=self.cg_ctrl)
        self.create_subscription(Float32MultiArray, "/lidar/bins", self._on_bins, 10,
                                 callback_group=self.cg_ctrl)

        # In dry run the cmd_vel publisher is never created: belt and braces, so even a
        # coding error cannot put a Twist on the wire.
        self.pub_cmd = self.create_publisher(Twist, g("cmd_topic"), 10) if live else None
        self.pub_path = self.create_publisher(Path, "/vla/path", 10)
        self.pub_status = self.create_publisher(String, "/vla/status", 10)

        self.create_timer(1.0 / max(0.2, float(g("infer_hz"))), self._infer_step,
                          callback_group=self.cg_infer)
        self.create_timer(1.0 / max(1.0, float(g("control_hz"))), self._control_step,
                          callback_group=self.cg_ctrl)

        self.get_logger().info(
            f"{'LIVE -- PUBLISHING' if live else 'DRY RUN -- publishing nothing'} | "
            f"image={g('image_topic')} cmd={g('cmd_topic')} | prompt={self.prompt} | "
            f"speed={self.speed} lookahead={self.lookahead_dist}m "
            f"caps v<={self.max_speed} w<={self.max_omega}")

    # ---------------- callbacks ----------------
    def _on_prompt(self, msg: String):
        d = msg.data.strip()
        if d not in VALID_PROMPTS:
            self.get_logger().warn(f"ignoring unknown prompt {d!r}; valid: {VALID_PROMPTS}")
            return
        with self._lock:
            self.prompt = d
        self.get_logger().info(f"prompt -> {d}")

    def _on_clear(self, msg: Float32):
        with self._lock:
            self._clear = float(msg.data)
            self._safety_stamp = time.monotonic()

    def _on_near(self, msg: Float32):
        with self._lock:
            self._near = float(msg.data)

    def _on_cerr(self, msg: Float32):
        # Differentiate here, not in the 20 Hz control loop: e arrives at the 10 Hz scan
        # rate, so the control loop sees each value twice and would read a rate of zero on
        # every other tick.
        with self._lock:
            t = time.monotonic()
            e = float(msg.data)
            prev, prev_t = self._cerr, self._cerr_stamp
            if (prev is not None and np.isfinite(prev) and np.isfinite(e)
                    and 1e-3 < (t - prev_t) < 0.5):
                raw = (e - prev) / (t - prev_t)
                # The sector medians jitter a few cm scan to scan; an unfiltered derivative
                # is mostly that noise amplified by 10 Hz.
                self._cerr_rate = 0.7 * self._cerr_rate + 0.3 * raw
            self._cerr = e
            self._cerr_stamp = t

    def _on_scene(self, msg: String):
        with self._lock:
            self._scene = msg.data

    def _on_ops(self, msg: Float32MultiArray):
        d = list(msg.data)
        ops = [(d[i], d[i + 1], d[i + 2]) for i in range(0, len(d) - 2, 3)]
        with self._lock:
            self._ops = ops

    def _on_bins(self, msg: Float32MultiArray):
        r = np.asarray(msg.data, dtype=np.float64)
        if r.size == 0:
            return
        step = 360.0 / r.size
        th = np.radians(np.arange(r.size) * step - 180.0 + step / 2.0)
        ok = np.isfinite(r)
        with self._lock:
            self._obs = (r[ok] * np.cos(th[ok]), r[ok] * np.sin(th[ok]))
            self._obs_stamp = time.monotonic()

    def _on_image(self, msg: Image):
        # Keep only the newest frame; decoding and inference happen on the timer so a
        # slow forward pass cannot build a backlog of stale frames.
        with self._lock:
            self._msg = msg

    def _infer_step(self):
        with self._lock:
            msg, prompt = self._msg, self.prompt
        if msg is None:
            self.get_logger().warn("no frame on image topic yet", throttle_duration_sec=5.0)
            return
        try:
            rgb = decode_image(msg)
            from PIL import Image as PILImage
            t0 = time.monotonic()
            wp = self.policy.predict(PILImage.fromarray(rgb), prompt)
            dt = time.monotonic() - t0
        except Exception as e:
            # Leave the old trajectory to go stale rather than publishing garbage; the
            # control loop will hard-zero once stale_after passes.
            self.get_logger().error(f"inference failed: {type(e).__name__}: {e}")
            return

        with self._lock:
            self._traj = wp
            self._traj_stamp = time.monotonic()
            self._lat.append(dt)

        self._publish_path(wp, msg.header)
        # Mirror what the control loop will actually do, including the stop prompt --
        # otherwise this line reports the raw geometry and reads as if a "stop" were
        # still commanding speed.
        if prompt == "stop":
            v, w = 0.0, 0.0
        else:
            v, w = waypoints_to_twist(
                wp, speed=self.speed, lookahead_dist=self.lookahead_dist,
                max_speed=self.max_speed, max_omega=self.max_omega,
                max_curvature=self.max_curvature)
        self.get_logger().info(
            f"[{prompt}] {dt * 1e3:6.1f}ms  wp0=({wp[0][0]:+.3f},{wp[0][1]:+.3f}) "
            f"wp7=({wp[-1][0]:+.3f},{wp[-1][1]:+.3f})  -> v={v:+.3f} w={w:+.3f}")
        st = String()
        st.data = f"prompt={prompt} latency_ms={dt * 1e3:.1f} v={v:.3f} w={w:.3f}"
        self.pub_status.publish(st)

    def _control_step(self):
        now = time.monotonic()
        dt = min(max(now - self._last_ctrl, 1e-3), 0.25)
        self._last_ctrl = now

        with self._lock:
            traj, stamp, prompt = self._traj, self._traj_stamp, self.prompt

        # Faults -> hard zero, bypassing the ramp. A fault should stop the robot now, not
        # over the next few hundred ms.
        if traj is None or (now - stamp) > self.stale_after:
            if traj is not None:
                self.get_logger().warn(
                    f"waypoints stale ({now - stamp:.2f}s > {self.stale_after}s) -> zero",
                    throttle_duration_sec=2.0)
            self._v = self._w = 0.0
            self._send(0.0, 0.0)
            return

        if prompt == "stop":
            v_t, w_t = 0.0, 0.0        # commanded stop: smooth, via the ramp below
            self._gate = "stop"
        else:
            v_t, w_t = waypoints_to_twist(
                traj, speed=self.speed, lookahead_dist=self.lookahead_dist,
                max_speed=self.max_speed, max_omega=self.max_omega,
                max_curvature=self.max_curvature)
            self._gate = "policy"

            # ---- corridor centring -------------------------------------------------
            # e > 0 means the robot sits LEFT of centre, so it must steer RIGHT, which is
            # negative omega: hence the minus. Held off during a turn by default, since at
            # a junction e measures the branch mouth rather than a wall and would fight
            # the policy exactly when the policy is doing the interesting part.
            with self._lock:
                cerr, cstamp, scene = self._cerr, self._cerr_stamp, self._scene
                crate = self._cerr_rate
            settling = self._avoid_side != 0 or (now - self._avoid_clear_t) < self.avoid_settle
            if settling:
                # Just went round something. The centring term reads the obstacle as a wall
                # and the policy wants to swing back; together they over-turned. Hold
                # steering gentle until the robot has settled.
                w_t = float(np.clip(w_t, -self.avoid_settle_omega, self.avoid_settle_omega)) \
                    if self._avoid_side == 0 else w_t
            if (not settling and self.use_centering and cerr is not None and np.isfinite(cerr)
                    and (now - cstamp) <= self.stale_after
                    and (not self.center_only_corridor or scene == "corridor")):
                # PD, not P. Proportional alone on a car-like base overshoots the centre
                # and weaves -- seen live as "it overdoes the left or right after seeing a
                # wall". The D term opposes the rate at which the error is closing, so the
                # correction eases off before the robot reaches the middle. The deadband
                # stops it hunting over a few centimetres of LiDAR jitter.
                e_db = cerr - float(np.clip(cerr, -self.center_deadband, self.center_deadband))
                w_c = -(self.center_gain * e_db + self.center_damp * crate)
                w_t += float(np.clip(w_c, -self.center_max, self.center_max))
                w_t = float(np.clip(w_t, -self.max_omega, self.max_omega))

            # ---- obstacle avoidance ------------------------------------------------
            # Overrides the policy unconditionally. The VLA has no obstacle notion of its
            # own, and a stale clearance channel is treated as blocked rather than clear.
            if self.use_safety:
                v_t, w_t = self._apply_safety(v_t, w_t, now)

        # Ramps: accel limit going up, the larger decel limit coming down.
        dv_up, dv_dn = self.accel * dt, self.decel * dt
        self._v = float(np.clip(v_t, self._v - dv_dn, self._v + dv_up))
        dw = self.angular_accel * dt
        self._w = float(np.clip(w_t, self._w - dw, self._w + dw))
        self._send(self._v, self._w)

    def _apply_safety(self, v, w, now):
        """Clamp (v, w) by LiDAR clearance. Fails closed: unknown or stale -> stopped."""
        with self._lock:
            clear, near, stamp = self._clear, self._near, self._safety_stamp
        if clear is None or (now - stamp) > self.safety_stale:
            # Never heard from the LiDAR, or it went quiet. A missing safety channel is
            # the one case where guessing "probably clear" is unacceptable.
            self._gate = "no-lidar"
            self._recover_t0 = None
            self._recover_bearing = None
            self._avoid_side = 0
            self.get_logger().warn(
                "no clearance data (is lidar_openings_node running?) -> holding still",
                throttle_duration_sec=3.0)
            return 0.0, 0.0
        if near is not None and near < self.emergency_distance:
            self._gate = f"EMERG {near:.2f}m"
            if self._recover_t0 is not None:
                self.get_logger().warn("recovery ABORT: emergency lane breached")
                self._recover_t0 = None
                self._recover_bearing = None
            return 0.0, 0.0
        plan = self._plan_avoid(v, w, now)
        if plan is not None:
            return self._apply_arc_safety(v, w, plan, now)
        # Hysteresis: enter at stop_distance, leave only once clearance is margin ABOVE
        # it. Measured at a real T-junction, d_front jittered 0.74 / 1.16 / 0.74 m between
        # scans, which with a single threshold made recovery enter and exit every ~0.1 s.
        exit_at = self.stop_distance + self.recovery_exit_margin
        if clear < (exit_at if self._recover_t0 is not None else self.stop_distance):
            return self._creep_recovery(clear, now)
        if self._recover_t0 is not None:
            self.get_logger().info(
                f"recovery EXIT after {now - self._recover_t0:.1f}s (clear={clear:.2f}m)")
            self._recover_t0 = None
            self._recover_bearing = None
        if clear < self.slow_distance:
            scale = (clear - self.stop_distance) / max(1e-6, self.slow_distance - self.stop_distance)
            self._gate = f"slow {clear:.2f}m"
            return v * float(np.clip(scale, 0.0, 1.0)), w
        return v, w

    def _plan_avoid(self, v, w, now):
        """Pick the arc to drive. -> (kappa, free_m, tag) or None if avoidance is off or
        /lidar/bins is missing/stale (the caller then falls back to the straight lane).

        Never chooses an arc with less free distance than the policy's own: avoidance can
        only steer AWAY from what is ahead, not towards it.
        """
        if not self.use_avoidance:
            return None
        with self._lock:
            obs, stamp = self._obs, self._obs_stamp
        if obs is None or (now - stamp) > self.safety_stale:
            return None
        ox, oy = obs
        k_pol = (w / v) if v > 1e-3 else 0.0
        kmax = self.avoid_max_curvature
        ks = np.append(np.linspace(-kmax, kmax, self.avoid_n_arcs), k_pol)
        free = arc_free_lengths(ks, ox, oy, self.avoid_half_width, self.avoid_horizon,
                                side_half_width=self.avoid_side_half_width,
                                front_reach=self.avoid_front_reach,
                                margin_ramp=self.avoid_margin_ramp)
        free_pol = float(free[-1])

        release = self.avoid_distance + self.recovery_exit_margin
        if free_pol >= (release if self._avoid_side else self.avoid_distance):
            if self._avoid_side:
                self.get_logger().info(f"avoid CLEAR: policy path free {free_pol:.2f}m")
                self._avoid_clear_t = now
            self._avoid_side = 0
            return k_pol, free_pol, None

        dev = ks - k_pol
        # Commit to one side. Re-deciding from scratch every tick is what made it dither
        # left/right in front of an obstacle and end up hitting it.
        other = ((np.sign(dev) == -self._avoid_side) & (np.abs(dev) > 0.05)
                 if self._avoid_side else np.zeros(ks.shape, dtype=bool))
        # Take the GENTLEST arc that is clear far enough, not the clearest one: picking
        # the clearest swung the robot into hard turns it did not need, then hard back.
        enough = free >= min(self.avoid_need, self.avoid_horizon)
        if np.any(enough & ~other):
            enough &= ~other
        if np.any(enough):
            i = int(np.argmin(np.where(enough, np.abs(dev), np.inf)))
        else:
            score = free - self.avoid_dev_weight * np.abs(dev) - self.avoid_switch_penalty * other
            i = int(np.argmax(score))
        if free[i] < free_pol + self.avoid_min_gain:
            return k_pol, free_pol, f"blocked {free_pol:.2f}m"
        side = int(np.sign(dev[i])) if abs(dev[i]) > 0.05 else self._avoid_side
        if side and side != self._avoid_side:
            self.get_logger().info(
                f"avoid {'LEFT' if side > 0 else 'RIGHT'}: policy path blocked at "
                f"{free_pol:.2f}m, arc k={ks[i]:+.2f} free {free[i]:.2f}m")
            self._avoid_side = side
        return float(ks[i]), float(free[i]), f"AVOID {'L' if self._avoid_side > 0 else 'R'}"

    def _apply_arc_safety(self, v, w, plan, now):
        """Speed gate on the free length of the chosen arc, steering onto it if it differs
        from the policy's. Mirrors the straight-lane logic in _apply_safety."""
        kappa, free, tag = plan
        exit_at = self.stop_distance + self.recovery_exit_margin
        if free < (exit_at if self._recover_t0 is not None else self.stop_distance):
            if tag is None or not tag.startswith("AVOID"):
                # No arc is clearer than straight ahead: creeping would only close on the
                # obstacle, so hold still rather than roll towards it.
                self._gate = f"blocked {free:.2f}m no-way-round"
                if self._recover_t0 is not None:
                    self.get_logger().warn("recovery ABORT: no arc leads away -> hard stop")
                    self._recover_t0 = None
                self.get_logger().warn(
                    f"path blocked at {free:.2f} m and no arc leads away -> holding still",
                    throttle_duration_sec=5.0)
                return 0.0, 0.0
            return self._creep_arc(kappa, free, now)
        if self._recover_t0 is not None:
            self.get_logger().info(
                f"recovery EXIT after {now - self._recover_t0:.1f}s (free={free:.2f}m)")
            self._recover_t0 = None
        if tag is not None and tag.startswith("AVOID"):
            w = float(np.clip(v * kappa, -self.max_omega, self.max_omega))
        scale = 1.0
        if free < self.slow_distance:
            scale = (free - self.stop_distance) / max(1e-6, self.slow_distance - self.stop_distance)
        self._gate = (f"{tag} {free:.2f}m" if tag else
                      (f"slow {free:.2f}m" if scale < 1.0 else "policy"))
        return v * float(np.clip(scale, 0.0, 1.0)), w

    def _creep_arc(self, kappa, free, now):
        """Every arc is blocked inside stop_distance: roll slowly along the clearest one so
        the Ackermann base can still steer out, until the emergency lane or timeout."""
        if free < self.emergency_distance + 0.10:
            self._gate = f"blocked {free:.2f}m no-free-arc"
            if self._recover_t0 is not None:
                self.get_logger().warn("recovery ABORT: no free arc -> hard stop")
                self._recover_t0 = None
            else:
                self.get_logger().warn(
                    f"every arc blocked within {free:.2f} m -> holding still",
                    throttle_duration_sec=5.0)
            return 0.0, 0.0
        if self._recover_t0 is None:
            self._recover_t0 = now
            self.get_logger().info(
                f"recovery ENTER: best arc k={kappa:+.2f} free {free:.2f}m, "
                f"creeping at {self.creep_speed:.2f} m/s")
        elapsed = now - self._recover_t0
        if elapsed > self.recovery_timeout:
            self._gate = f"creep exhausted {elapsed:.1f}s"
            self.get_logger().warn(
                f"creep recovery exhausted after {elapsed:.1f}s -> hard stop",
                throttle_duration_sec=5.0)
            return 0.0, 0.0
        v = self.creep_speed
        w = float(np.clip(v * kappa, -self.max_omega, self.max_omega))
        self._gate = f"CREEP arc k={kappa:+.2f} free {free:.2f}m {elapsed:.1f}s"
        return v, w

    def _creep_recovery(self, clear, now):
        """Lane ahead blocked but the emergency lane still clear -> roll slowly and steer
        out, instead of freezing. Returns the (v, w) to command.

        A hard zero here is a trap: the Hunter is Ackermann, so omega with v=0 does
        nothing, and a stationary robot never changes its own clearance. Every stop was
        therefore terminal until something pushed it.
        """
        with self._lock:
            ops = list(self._ops)
        if not ops:
            self._gate = f"blocked {clear:.2f}m no-opening"
            if self._recover_t0 is not None:
                self.get_logger().warn("recovery ABORT: no opening detected -> hard stop")
                self._recover_t0 = None
                self._recover_bearing = None
            else:
                self.get_logger().warn(
                    f"blocked at {clear:.2f} m with no opening detected -> holding still",
                    throttle_duration_sec=5.0)
            return 0.0, 0.0

        if self._recover_t0 is None:
            self._recover_t0 = now
            self.get_logger().info(
                f"recovery ENTER: clear={clear:.2f}m < stop={self.stop_distance:.2f}m, "
                f"{len(ops)} opening(s), creeping at {self.creep_speed:.2f} m/s")
        elapsed = now - self._recover_t0
        if elapsed > self.recovery_timeout:
            self._gate = f"creep exhausted {elapsed:.1f}s"
            self.get_logger().warn(
                f"creep recovery exhausted after {elapsed:.1f}s -> hard stop",
                throttle_duration_sec=5.0)
            return 0.0, 0.0

        # LATCH the escape direction. Picking the widest opening afresh each tick looks
        # reasonable and is wrong: at a real T-junction the two arms measured 29 deg and
        # 28 deg wide, so the argmax flipped between +87 and -110 on consecutive scans and
        # the robot would have steered left, right, left instead of escaping.
        if self._recover_bearing is None:
            bearing, width, depth = max(ops, key=lambda o: o[1])
            self._recover_bearing = bearing
        else:
            near = min(ops, key=lambda o: abs(o[0] - self._recover_bearing))
            if abs(near[0] - self._recover_bearing) <= self.recovery_retarget_deg:
                bearing, width, depth = near
                self._recover_bearing = bearing      # track it as the robot rotates
            else:
                bearing, width, depth = max(ops, key=lambda o: o[1])
                self.get_logger().info(
                    f"recovery RETARGET {self._recover_bearing:+.0f} -> {bearing:+.0f} deg "
                    f"(latched opening lost)")
                self._recover_bearing = bearing

        th = math.radians(bearing)
        L = max(0.3, min(self.creep_lookahead, depth))
        y = L * math.sin(th)
        kappa = 2.0 * y / (L * L)
        if self.max_curvature > 0.0:
            kappa = float(np.clip(kappa, -self.max_curvature, self.max_curvature))
        v = self.creep_speed
        w = float(np.clip(v * kappa, -self.max_omega, self.max_omega))
        self._gate = f"CREEP {clear:.2f}m -> {bearing:+.0f}deg {elapsed:.1f}s"
        return v, w

    # ---------------- output ----------------
    def _send(self, v: float, w: float):
        """The single chokepoint through which anything reaches /cmd_vel."""
        if not np.isfinite(v) or not np.isfinite(w):
            v, w = 0.0, 0.0
        if not self.live or self.pub_cmd is None:
            with self._lock:
                clear, cerr = self._clear, self._cerr
            self.get_logger().info(
                f"[dry_run] v={v:+.3f} w={w:+.3f} | {self._gate}"
                f" | clr={'n/a' if clear is None else format(clear, '.2f')}"
                f" e={'n/a' if cerr is None else format(cerr, '+.2f')}",
                throttle_duration_sec=1.0)
            return
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.pub_cmd.publish(t)

    def _publish_path(self, wp, header):
        path = Path()
        path.header.stamp = header.stamp
        path.header.frame_id = "base_link"
        for x, y in wp:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)

    def report(self):
        if self._lat:
            a = np.array(self._lat)
            self.get_logger().info(
                f"latency over {len(a)} frames: mean {a.mean() * 1e3:.1f}ms "
                f"p50 {np.percentile(a, 50) * 1e3:.1f}ms p95 {np.percentile(a, 95) * 1e3:.1f}ms "
                f"-> {1.0 / a.mean():.2f} Hz sustained")


def main():
    live = "--live" in sys.argv
    rclpy.init(args=[a for a in sys.argv if a != "--live"])
    node = V8DriveNode(live=live)
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        try:
            if rclpy.ok() and live:
                node._send(0.0, 0.0)     # leave the base stopped
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
