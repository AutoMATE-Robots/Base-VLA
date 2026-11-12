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


# Hunter 2.0 footprint ~0.98 x 0.75 m, base_link at its centre. Three circles of radius
# ROBOT_COVER at -BODY_DS, 0, +BODY_DS along the body cover it, corners included:
# (0.49 - 0.3)^2 + 0.375^2 <= 0.425^2.
ROBOT_COVER = 0.425
BODY_DS = 0.30
# Ackermann: the REAR AXLE follows the path and the body pivots about it, so the front
# swings wide on a turn. Hunter 2.0 wheelbase 0.65 m -> rear axle ~0.33 m behind centre.
# Treating base_link as the point that follows the path let the front corner clip what
# the robot was passing (simulation: 7 cm, then wedged beside the obstacle).
REAR_AXLE = 0.33


def offset_paths(k_base, offsets, swerve_len, s):
    """Candidate paths: a base arc of curvature k_base[i], shifted sideways by offsets[i].

    The shift follows a smoothstep over swerve_len[i] metres and then holds, so a path is
    "move over by dy while bending by k". Plain arcs cannot express "swerve then carry
    on": every arc that cleared an obstacle kept turning into the wall beside it. Plain
    shifts along the current heading cannot straighten back out once the robot has turned.

    -> x, y of shape (C, S) in base_link, for C candidates and S arc-length samples.
    """
    k = np.asarray(k_base, dtype=np.float64)[:, None]
    straight = np.abs(k) < 1e-6
    kk = np.where(straight, 1.0, k)
    bx = np.where(straight, s[None, :], np.sin(kk * s) / kk)
    by = np.where(straight, 0.0, (1.0 - np.cos(kk * s)) / kk)
    nx = np.where(straight, 0.0, -np.sin(kk * s))           # left normal of the base arc
    ny = np.where(straight, 1.0, np.cos(kk * s))
    t = np.clip(s[None, :] / np.asarray(swerve_len)[:, None], 0.0, 1.0)
    lat = np.asarray(offsets)[:, None] * (t * t * (3.0 - 2.0 * t))
    return bx + lat * nx, by + lat * ny


def balance_plan(ox, oy, k_pol, prev_side, *, bases, max_offset, offset_step, swerve_kmax,
                 side_hw, full_hw, front_reach, margin_ramp, horizon, stations, clear_cap,
                 w_clear, w_tight, w_free, w_off, w_head, w_side, w_curv, need, lookahead, kmax,
                 min_clear, comfort, near_dist=1.5, side_at=2.0):
    """Choose the path that keeps the robot in the middle of the free space.

    Candidates are every sideways shift (0..+-max_offset) on every base arc (the policy's
    own curvature, straight, and `bases`); see offset_paths. Each is scored on
      * clearance -- mean distance from the body to the nearest obstacle at the scoring
        stations, capped at clear_cap. Maximising the nearest distance on BOTH sides is
        what puts the robot midway between a wall and an object, instead of swerving away
        from one and into the other. Capped, so once everything is further than
        clear_cap away the policy drives undisturbed;
      * free length -- how far it runs before the body would touch anything. Paths are
        traced by the rear axle (REAR_AXLE) and the body -- three circles at -BODY_DS, 0,
        +BODY_DS -- is placed rigidly on them at the path heading, radius growing from
        side_hw beside the robot to full_hw ahead. One circle at the centre missed the
        front corners, which stick out ~0.2 m beyond it, and letting the centre follow
        the path missed the front swinging wide on Ackermann steering;
      * how far it shifts off, and bends away from, the policy's path; how sharply it has
        to swerve; and whether it ends up on the other side (measured side_at metres
        ahead) from the side it committed to -- no dithering in front of things.
    Paths blocked closer than `need`, or that squeeze the body closer than min_clear,
    are excluded while any path is not.

    -> dict(k, free, cmin, cnear, dev, k_pol, free_pol, fits, leads_away). dev is how far
    left (+) of the policy's path the chosen one is at side_at; cmin the tightest body
    clearance on it, cnear the tightest within near_dist (what the robot is squeezing
    past now, for the speed limit); fits whether any path keeps cmin >= min_clear.
    """
    n = int(round(max_offset / offset_step))
    shifts = np.arange(-n, n + 1) * offset_step
    # Shortest swerve the steering can do: smoothstep peak curvature is 6*dy/L^2. Offer
    # that and a gentler 1.6x version; the curvature cost picks between them.
    l_min = np.maximum(1.0, np.sqrt(6.0 * np.abs(shifts) / swerve_kmax))
    side = shifts != 0
    shifts, lens = np.concatenate([shifts, shifts[side]]), np.concatenate([l_min, 1.6 * l_min[side]])
    kbs = np.clip(np.concatenate([[k_pol, 0.0], np.asarray(bases, dtype=np.float64)]), -kmax, kmax)
    kbs = kbs[np.concatenate([[True], np.abs(kbs[1:] - k_pol) > 1e-3])]   # policy arc first
    kb = np.repeat(kbs, len(shifts))
    offs, lens = np.tile(shifts, len(kbs)), np.tile(lens, len(kbs))
    i_pol = n                                   # base k_pol, shift 0

    ds = 0.15
    sb = np.arange(0, int(round(horizon / ds)) + 1) * ds    # rear-axle arc length
    rx, ry = offset_paths(kb, offs, lens, sb)               # rear-axle paths ...
    rx = rx - REAR_AXLE                                     # ... starting where it is now
    hx, hy = np.gradient(rx, ds, axis=1), np.gradient(ry, ds, axis=1)
    hn = np.maximum(np.hypot(hx, hy), 1e-9)
    hx, hy = hx / hn, hy / hn                               # heading along each path
    px, py = rx + REAR_AXLE * hx, ry + REAR_AXLE * hy       # base_link along each path

    reach_y = float(np.max(np.abs(py))) + 2.0 * ROBOT_COVER
    # Nothing behind the rear axle can be hit going forward: with Ackermann steering the
    # rear axle follows the path and the body ahead of it does the sweeping. Including
    # those returns stopped the robot for clutter just behind its rear corner.
    near = (ox > -REAR_AXLE) & (ox < horizon + 1.0) & (np.abs(oy) < reach_y)
    ox, oy = ox[near].astype(np.float32), oy[near].astype(np.float32)
    # Body clearance at each sample: nearest of the three body circles, placed rigidly on
    # the robot at that sample's heading.
    db = np.full(px.shape, 10.0)
    if ox.size:
        px32, py32, hx32, hy32 = (a.astype(np.float32) for a in (px, py, hx, hy))
        d2 = None
        for a in (-BODY_DS, 0.0, BODY_DS):
            cx = (px32 + np.float32(a) * hx32)[..., None]
            cy = (py32 + np.float32(a) * hy32)[..., None]
            m = np.min((cx - ox) ** 2 + (cy - oy) ** 2, axis=-1)
            d2 = m if d2 is None else np.minimum(d2, m)
        db = np.sqrt(d2).astype(np.float64)
    hw = side_hw + (full_hw - side_hw) * np.clip((sb - front_reach) / max(margin_ramp, 1e-6), 0, 1)
    hit = db < hw[None, :]
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), len(sb))
    free = np.where(first < len(sb), sb[np.maximum(first - 1, 0)], horizon)

    st = np.asarray(stations, dtype=np.float64)
    si = np.clip(np.round(st / ds).astype(int), 0, len(sb) - 1)
    reach = st[None, :] <= free[:, None]
    clear = np.minimum(np.where(reach, db[:, si], 0.0), clear_cap).mean(axis=1)

    # Which side of the policy's path each candidate goes, judged side_at metres ahead.
    j2 = min(int(np.searchsorted(sb, side_at)), len(sb) - 1)
    dev = py[:, j2] - py[i_pol, j2]
    flip = ((np.sign(dev) == -prev_side) & (np.abs(dev) > 0.1)) if prev_side else 0.0
    curv = 6.0 * np.abs(offs) / (lens * lens)

    # Tightest squeeze on each path, on the dense samples (a 0.5 m deep object fits between
    # scoring stations). The first few samples are skipped: beside the robot every path is
    # the same, and counting that would make it stop next to things it is already passing.
    ahead = (sb[None, :] >= 2 * BODY_DS) & (sb[None, :] <= free[:, None])
    cmin_all = np.where(ahead, db, np.inf).min(axis=1)
    cnear_all = np.where(ahead & (sb[None, :] <= near_dist), db, np.inf).min(axis=1)
    # The mean clearance barely notices the one stretch where the robot passes an object,
    # so every sample where the body is closer than `comfort` to something is penalised
    # by how much closer. This is what centres it in a gap. Rewarding only the single
    # tightest point did not: once alongside an object that point is where the robot
    # already is, the same for every path, and it held 11 cm off a box with 38 cm free
    # on the other side. Counted from the first sample, so a path that turns back right
    # after passing, swinging the body into the object, pays for it too.
    short = np.where(sb[None, :] <= free[:, None], np.maximum(0.0, ROBOT_COVER + comfort - db),
                     ROBOT_COVER + comfort)[:, 1:].mean(axis=1)
    score = (w_clear * clear - w_tight * short + w_free * free / horizon
             - w_off * np.abs(offs) - w_head * np.abs(kb - k_pol) - w_curv * curv
             - w_side * flip)
    ok = free >= need
    fits = ok & (cmin_all >= min_clear)
    if np.any(fits):
        ok = fits                   # only paths the body actually fits through
    if np.any(ok):
        score = np.where(ok, score, -np.inf)
    i = int(np.argmax(score))

    if i == i_pol:
        k = float(k_pol)            # nothing to get round: the policy steers, unchanged
    else:                           # pure pursuit for the rear axle on its chosen path
        j = min(int(np.searchsorted(sb, lookahead)), len(sb) - 1)
        tx, ty = rx[i, j] + REAR_AXLE, ry[i, j]
        k = float(np.clip(2.0 * ty / max(tx * tx + ty * ty, 1e-6), -kmax, kmax))
    free_pol = float(free[i_pol])
    return dict(k=k, free=float(free[i]), cmin=float(min(cmin_all[i], clear_cap)),
                cnear=float(min(cnear_all[i], clear_cap)), dev=float(dev[i]),
                k_pol=float(k_pol), free_pol=free_pol, fits=bool(np.any(fits)),
                leads_away=bool(free[i] >= free_pol + 0.3))


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
        # --- obstacle avoidance by balancing: every tick, steer along the candidate path
        # (policy path shifted sideways) that keeps the robot in the middle of the free
        # space between whatever is on each side. Needs /lidar/bins from lidar_openings_node.
        p("use_avoidance", True)
        p("self_filter_margin", 0.10)   # m; LiDAR returns this close to the body are the robot itself
        p("avoid_max_offset", 1.5)  # m; furthest sideways shift considered
        p("avoid_offset_step", 0.15)    # m between candidate shifts
        p("avoid_swerve_kmax", 0.45)    # 1/m; peak curvature allowed in a swerve (Hunter max ~0.6)
        p("avoid_max_curvature", 0.6)   # 1/m; hard cap on the commanded curvature
        p("avoid_lookahead", 1.0)   # m; pure-pursuit distance along the chosen path
        p("avoid_margin", 0.10)     # m added to the body circles ahead of the robot. A hard
        # limit only: keeping a comfortable distance is the clearance score's job. At 0.25 a
        # 1.25 m gap beside a box read as blocked and the robot stopped instead of going through.
        p("avoid_side_margin", 0.02)    # m added to the body circles BESIDE the robot (the circles
        # already reach ~5 cm past the body's sides, so the real side gap is ~7 cm)
        p("avoid_front_reach", 0.50)    # m from base_link to the front bumper
        p("avoid_margin_ramp", 1.50)    # m past the bumper over which the margin grows to full
        p("avoid_horizon", 6.0)     # m; how far ahead obstacles are looked for along each path
        p("avoid_stations", [0.75, 1.25, 1.75, 2.25, 2.75, 3.25])  # m along each path where clearance is scored
        p("avoid_clear_cap", 0.90)  # m; obstacles further than this from the robot centre are ignored
        p("avoid_w_clear", 3.0)     # weight on mean clearance (the balancing term)
        p("avoid_comfort", 0.25)    # m; body gap below which a path is penalised (centres in gaps)
        p("avoid_w_tight", 10.0)    # weight on that penalty (mean shortfall in metres)
        p("avoid_w_free", 0.5)      # weight on free length / horizon
        p("avoid_w_off", 0.5)       # cost per metre of sideways shift off the policy's line
        p("avoid_w_curv", 0.5)      # cost per 1/m of swerve curvature (prefer gentle)
        p("avoid_w_side", 2.0)      # cost of switching sides from last tick
        p("avoid_bases", [-0.3, -0.15, 0.15, 0.3])  # 1/m; base arcs besides the policy's and straight
        p("avoid_w_head", 1.0)      # cost per 1/m of bending away from the policy's curvature
        p("avoid_release", 1.0)     # s of "no shift needed" before the chosen side is let go
        p("avoid_body_gap", 0.05)   # m; side gap below this = no way through
        p("avoid_gap_slow", 0.30)   # m of side gap below which speed is reduced
        p("avoid_gap_min_scale", 0.35)  # slowest fraction of speed when squeezing through a gap
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
        self.self_filter_margin = float(g("self_filter_margin"))
        self.avoid_max_offset = float(g("avoid_max_offset"))
        self.avoid_offset_step = float(g("avoid_offset_step"))
        self.avoid_swerve_kmax = float(g("avoid_swerve_kmax"))
        self.avoid_max_curvature = float(g("avoid_max_curvature"))
        self.avoid_lookahead = float(g("avoid_lookahead"))
        self.avoid_half_width = ROBOT_COVER + float(g("avoid_margin"))
        self.avoid_side_half_width = ROBOT_COVER + float(g("avoid_side_margin"))
        self.avoid_front_reach = float(g("avoid_front_reach"))
        self.avoid_margin_ramp = float(g("avoid_margin_ramp"))
        self.avoid_horizon = float(g("avoid_horizon"))
        self.avoid_stations = [float(x) for x in g("avoid_stations")]
        self.avoid_clear_cap = float(g("avoid_clear_cap"))
        self.avoid_w_clear = float(g("avoid_w_clear"))
        self.avoid_w_tight = float(g("avoid_w_tight"))
        self.avoid_comfort = float(g("avoid_comfort"))
        self.avoid_w_free = float(g("avoid_w_free"))
        self.avoid_w_off = float(g("avoid_w_off"))
        self.avoid_w_curv = float(g("avoid_w_curv"))
        self.avoid_w_side = float(g("avoid_w_side"))
        self.avoid_bases = [float(x) for x in g("avoid_bases")]
        self.avoid_w_head = float(g("avoid_w_head"))
        self.avoid_release = float(g("avoid_release"))
        self.avoid_body_gap = float(g("avoid_body_gap"))
        self.avoid_gap_slow = float(g("avoid_gap_slow"))
        self.avoid_gap_min_scale = float(g("avoid_gap_min_scale"))
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
        self._plan_cache = None     # (key, k_pol, plan); see _plan_avoid
        self._side_prev = 0         # side (+1 left / -1 right) committed to, 0 = none
        self._side_zero_t = None    # when the planner first stopped asking for a shift
        self._steering = False      # planner currently overriding the policy (for logging)

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
        x, y = r[ok] * np.cos(th[ok]), r[ok] * np.sin(th[ok])
        # Drop returns inside the robot's own footprint (+ self_filter_margin). Measured
        # 2026-11-03: the scan always holds points at the rear-left, inside the body --
        # e.g. (-0.47, +0.10), (-0.30, +0.31) -- i.e. the robot itself or something
        # mounted on it. Forward-only checks never saw them; the planner checks beside
        # and behind too, read them as touching, and refused to move in a clear corridor.
        m = self.self_filter_margin
        own = (np.abs(x) <= 0.49 + m) & (np.abs(y) <= 0.375 + m)
        with self._lock:
            self._obs = (x[~own], y[~own])
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
            # The balance planner already centres the robot in the free space, obstacles
            # included. Adding the wall-only centring on top made the two fight beside an
            # object, so centring only runs when the planner is not available.
            planner_live = self.use_safety and self.use_avoidance and self._bins_fresh(now)
            if (not planner_live and self.use_centering and cerr is not None and np.isfinite(cerr)
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
            self._side_prev = 0
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

    def _bins_fresh(self, now):
        with self._lock:
            return self._obs is not None and (now - self._obs_stamp) <= self.safety_stale

    def _plan_avoid(self, v, w, now):
        """Run the balance planner. -> balance_plan() dict, or None if avoidance is off or
        /lidar/bins is missing/stale (the caller then falls back to the straight lane).

        Runs EVERY tick rather than switching on when the path is blocked. The switched
        version flapped on/off every 50 ms in live runs (its own override changed the
        policy arc it was judging) and so effectively never steered -- it hit two objects.
        """
        if not self.use_avoidance or not self._bins_fresh(now):
            return None
        with self._lock:
            ox, oy = self._obs
            stamp = self._obs_stamp
        k_pol = (w / v) if v > 1e-3 else 0.0
        # Scans arrive at 10 Hz and control runs at 20 Hz: re-plan only on a new scan, a
        # change of side, or a policy that has noticeably changed its mind. A plan costs
        # ~25 ms here, half the control period, on a CPU the VLA is also using.
        key = (stamp, self._side_prev)
        c = self._plan_cache
        if c is not None and c[0] == key and abs(c[1] - k_pol) < 0.03:
            return c[2]
        plan = balance_plan(
            ox, oy, k_pol, self._side_prev, bases=self.avoid_bases,
            w_head=self.avoid_w_head, max_offset=self.avoid_max_offset,
            offset_step=self.avoid_offset_step, swerve_kmax=self.avoid_swerve_kmax,
            side_hw=self.avoid_side_half_width, full_hw=self.avoid_half_width,
            front_reach=self.avoid_front_reach, margin_ramp=self.avoid_margin_ramp,
            horizon=self.avoid_horizon, stations=self.avoid_stations,
            clear_cap=self.avoid_clear_cap, w_clear=self.avoid_w_clear, w_tight=self.avoid_w_tight,
            w_free=self.avoid_w_free, w_off=self.avoid_w_off, w_curv=self.avoid_w_curv,
            w_side=self.avoid_w_side, need=self.stop_distance + 0.5,
            lookahead=self.avoid_lookahead, kmax=self.avoid_max_curvature,
            min_clear=ROBOT_COVER + self.avoid_body_gap, comfort=self.avoid_comfort)
        self._plan_cache = (key, k_pol, plan)
        return plan

    def _apply_arc_safety(self, v, w, plan, now):
        """Steer onto the planner's arc and gate speed on its free length and side gap.
        Mirrors the straight-lane logic in _apply_safety."""
        kappa, free, cmin = plan["k"], plan["free"], plan["cmin"]
        exit_at = self.stop_distance + self.recovery_exit_margin
        blocked = free < (exit_at if self._recover_t0 is not None else self.stop_distance)
        if blocked or not plan["fits"]:
            if not plan["leads_away"] or not plan["fits"]:
                # Nothing is clearer than straight ahead, or no arc fits through: creeping
                # would only close on the obstacle, so hold still rather than roll at it.
                self._gate = (f"blocked {free:.2f}m no-way-round" if plan["fits"] else
                              f"gap too narrow (body clearance {cmin - ROBOT_COVER:.2f}m)")
                if self._recover_t0 is not None:
                    self.get_logger().warn("recovery ABORT: no arc leads away -> hard stop")
                    self._recover_t0 = None
                self.get_logger().warn(f"{self._gate} -> holding still", throttle_duration_sec=5.0)
                self._side_prev = 0
                return 0.0, 0.0
            return self._creep_arc(kappa, free, now)
        if self._recover_t0 is not None:
            self.get_logger().info(
                f"recovery EXIT after {now - self._recover_t0:.1f}s (free={free:.2f}m)")
            self._recover_t0 = None

        off = plan["dev"]
        if abs(off) > 0.1:
            self._side_prev, self._side_zero_t = int(np.sign(off)), None
        elif self._side_prev:
            # Let go of the side only after a sustained run of "no shift". A single tick
            # of offset 0 used to wipe the commitment, and the next pick went the other way.
            if self._side_zero_t is None:
                self._side_zero_t = now
            elif now - self._side_zero_t >= self.avoid_release:
                self._side_prev, self._side_zero_t = 0, None
        steering = abs(off) > 0.1
        if steering != self._steering:
            self._steering = steering
            self.get_logger().info(
                f"balance {'SHIFT ' + format(off, '+.2f') + 'm' if steering else 'RELEASE'}: "
                f"k={kappa:+.2f} (policy {plan['k_pol']:+.2f}) free {free:.2f}m "
                f"policy-path free {plan['free_pol']:.2f}m centre-clear {cmin:.2f}m")
        w = float(np.clip(v * kappa, -self.max_omega, self.max_omega))

        scale = 1.0
        if free < self.slow_distance:
            scale = (free - self.stop_distance) / max(1e-6, self.slow_distance - self.stop_distance)
        gap = plan["cnear"] - ROBOT_COVER          # metres between body and what it is passing now
        if gap < self.avoid_gap_slow:
            scale = min(scale, max(self.avoid_gap_min_scale, gap / self.avoid_gap_slow))
        self._gate = (f"{'BAL ' + format(off, '+.2f') if steering else 'policy'} k={kappa:+.2f} free {free:.2f}m "
                      f"gap {gap:.2f}m" + (f" x{scale:.2f}" if scale < 1.0 else ""))
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
