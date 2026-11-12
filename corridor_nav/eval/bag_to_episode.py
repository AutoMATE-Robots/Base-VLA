#!/usr/bin/env python3
"""
bag_to_episode.py -- turn ONE ROS2 bag into training steps for the trajectory VLA.

FINAL DESIGN (settled with the team):

    input   :  image  +  prompt        prompt in {straight, turn_left, turn_right, stop}
    output  :  trajectory              next N future poses (fwd,left) in the robot's current
                                        frame, taken directly from odometry

Two things the model must learn, and where each comes from:
  * WHICH WAY to turn  -> the prompt. It is the decision the VLM makes by reading the sign.
  * WHEN to turn       -> the image. Junction-nearness is visible in the pixels; the model
                          learns to keep going straight while the junction is far and to turn
                          only when it looks close -- even though the prompt already says
                          "left". So "left" is a STANDING CONDITION, not a trigger.

The prompt is NOT derived from geometry. It flips from "straight" to the decision at the
frame where the sign first becomes readable / the VLM could reason -- which we take from the
adaptive-reasoning labels (the legible frame). This keeps training and inference consistent:
in both, "left" means "turn left at the junction you are approaching", never "turn now".

There is NO separate input goal-pose. The trajectory (from odometry) is the output; the prompt
is the only conditioning input. This removes the redundancy between an input-goal and the
prompt that we ran into.

The output is a plain list of step dicts -- framework-free, so it can be inspected, plotted and
unit-tested WITHOUT TensorFlow. tfds_builder.py adds the RLDS/TFDS packaging on top.

    python bag_to_episode.py --bag /path/to/rosbag2_dir \
        --flip-frame 47 --decision turn_left --horizon 8 --plot

Robot frame: x forward (+), y left (+), theta CCW.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

PROMPTS = ("straight", "turn_left", "turn_right", "stop")


# ---- pose helpers --------------------------------------------------------------------------
def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def to_robot_frame(px, py, ox, oy, otheta):
    """World point (px,py) into the frame of a robot at (ox,oy,otheta) -> (forward, left)."""
    dx, dy = px - ox, py - oy
    c, s = math.cos(-otheta), math.sin(-otheta)
    return c * dx - s * dy, s * dx + c * dy


# ---- image decoders (self-contained; no dependency on bag_to_mp4) --------------------------
def _yuv422_to_rgb(data, height, width, step, order):
    """UYVY or YUYV 4:2:2 -> RGB. Vectorised, row-stride aware."""
    rows = []
    for r in range(height):
        row = data[r * step:r * step + width * 2]
        if row.size < width * 2:
            break
        row = row.reshape(-1, 4).astype(np.int32)   # two pixels per 4 bytes
        if order == "uyvy":
            u, y0, v, y1 = row[:, 0], row[:, 1], row[:, 2], row[:, 3]
        else:  # yuyv
            y0, u, y1, v = row[:, 0], row[:, 1], row[:, 2], row[:, 3]
        rows.append((y0, u, v, y1))
    if not rows:
        return None
    y0 = np.concatenate([r[0] for r in rows]).reshape(len(rows), -1)
    u = np.concatenate([r[1] for r in rows]).reshape(len(rows), -1)
    v = np.concatenate([r[2] for r in rows]).reshape(len(rows), -1)
    y1 = np.concatenate([r[3] for r in rows]).reshape(len(rows), -1)
    h_ = y0.shape[0]
    out = np.zeros((h_, y0.shape[1] * 2, 3), np.uint8)

    def yuv(Y, U, V):
        c = Y - 16
        d = U - 128
        e = V - 128
        R = np.clip((298 * c + 409 * e + 128) >> 8, 0, 255)
        G = np.clip((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255)
        B = np.clip((298 * c + 516 * d + 128) >> 8, 0, 255)
        return np.stack([R, G, B], -1).astype(np.uint8)

    out[:, 0::2] = yuv(y0, u, v)
    out[:, 1::2] = yuv(y1, u, v)
    return out


def _raw_to_rgb(data, height, width, step, encoding):
    """rgb8/bgr8/mono8/rgba8/bgra8 -> RGB, honouring row stride."""
    enc = encoding.lower()
    ch = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4}.get(enc)
    if ch is None:
        return None
    rows = []
    for r in range(height):
        row = data[r * step:r * step + width * ch]
        if row.size < width * ch:
            break
        rows.append(row.reshape(width, ch))
    if not rows:
        return None
    img = np.stack(rows, 0)
    if enc == "mono8":
        img = np.repeat(img, 3, axis=2)
    elif enc == "bgr8":
        img = img[:, :, ::-1]
    elif enc == "rgba8":
        img = img[:, :, :3]
    elif enc == "bgra8":
        img = img[:, :, :3][:, :, ::-1]
    return np.ascontiguousarray(img)


def _decode_image(msg):
    enc = getattr(msg, "encoding", "").lower()
    h, w, step = msg.height, msg.width, msg.step
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("yuv422", "uyvy"):
        return _yuv422_to_rgb(data, h, w, step, "uyvy")
    if enc in ("yuv422_yuy2", "yuyv"):
        return _yuv422_to_rgb(data, h, w, step, "yuyv")
    if enc:
        return _raw_to_rgb(data, h, w, step, enc)
    return None


def _make_typestore(distro: str):
    """A rosbags typestore for the given ROS distro, if this rosbags version supports it."""
    try:
        from rosbags.typesys import Stores, get_typestore
        store = {"humble": Stores.ROS2_HUMBLE, "iron": Stores.ROS2_IRON,
                 "jazzy": Stores.ROS2_JAZZY, "foxy": Stores.ROS2_FOXY}.get(
                     distro, Stores.ROS2_HUMBLE)
        return get_typestore(store)
    except Exception:
        return None


def _open_reader(bag_path, typestore):
    """AnyReader over a bag directory, passing a typestore if this rosbags version accepts one."""
    from rosbags.highlevel import AnyReader
    if typestore is not None:
        try:
            return AnyReader([bag_path], default_typestore=typestore)
        except TypeError:
            pass
    return AnyReader([bag_path])


# ---- bag reading (self-contained via rosbags) ----------------------------------------------
def read_bag(bag_path: Path, image_topic: str, odom_topic: str, ros_distro: str = "humble"):
    """Return (frames, odom):
         frames = [(t_sec, HxWx3 uint8 rgb), ...]
         odom   = [(t_sec, x, y, yaw), ...]
    Reads directly with rosbags — no dependency on bag_to_mp4. Point bag_path at the bag
    DIRECTORY (the folder with metadata.yaml and the .db3/.mcap inside)."""
    try:
        import rosbags  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            f"the 'rosbags' package is required to read bags but isn't importable here "
            f"({type(e).__name__}: {e}).\nActivate the env that has it, e.g.:\n"
            f"    conda activate /scratch.global/$USER/conda_envs/dino\n"
            f"or: pip install rosbags")

    typestore = _make_typestore(ros_distro)
    frames, odom = [], []
    with _open_reader(bag_path, typestore) as reader:
        conns = {c.topic: c for c in reader.connections}
        for name, topic in (("image", image_topic), ("odom", odom_topic)):
            if topic not in conns:
                raise SystemExit(
                    f"{name} topic '{topic}' not in bag; available: {sorted(conns)}")
        for conn, ts, raw in reader.messages(
                connections=[conns[image_topic], conns[odom_topic]]):
            t = ts * 1e-9
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == image_topic:
                rgb = _decode_image(msg)
                if rgb is not None:
                    frames.append((t, rgb))
            else:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                odom.append((t, float(p.x), float(p.y), yaw_from_quat(q.x, q.y, q.z, q.w)))
    frames.sort(key=lambda r: r[0])
    odom.sort(key=lambda r: r[0])
    return frames, odom




# ---- time alignment ------------------------------------------------------------------------
def interp_odom(odom, t: float):
    times = [o[0] for o in odom]
    if t <= times[0]:
        return odom[0][1:]
    if t >= times[-1]:
        return odom[-1][1:]
    i = np.searchsorted(times, t)
    t0, x0, y0, th0 = odom[i - 1]
    t1, x1, y1, th1 = odom[i]
    a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    dth = math.atan2(math.sin(th1 - th0), math.cos(th1 - th0))
    return x0 + a * (x1 - x0), y0 + a * (y1 - y0), th0 + a * dth


# ---- the extraction ------------------------------------------------------------------------
def build_steps(frames, odom, flip_frame: int, decision: str,
                turn_done_frame: int = None, horizon: int = 8, stride: int = 1,
                crop_start: int = None, crop_end: int = None):
    """Per-frame training steps.

    The prompt is a STANDING DECISION that is consumed at the junction:

        frame  <  flip_frame                    -> "straight"   (sign not yet readable)
        flip_frame <= frame < turn_done_frame   -> `decision`   (turn pending / executing)
        frame >= turn_done_frame                -> "straight"   (turn done, new corridor)

    flip_frame      : where the prompt switches from "straight" to `decision`
                      (= the legible frame from the adaptive-reasoning labels). Use 0 if the
                      sign is readable from the very first frame.
    decision        : "turn_left" | "turn_right" | "stop" -- the pending instruction.
    turn_done_frame : where the turn completes and the prompt clears back to "straight"
                      (robot has rotated into the new corridor). None = decision holds to the end
                      (e.g. if the bag ends at/at the turn). For "stop", leave None.
    horizon         : number of future waypoints in the trajectory target (action chunk N).
    stride          : subsample frames.
    crop_start,     : RAW frame indices bounding which frames to KEEP. Used to trim the long
    crop_end          straight approach/exit so turn frames are not drowned out (the turn is a
                      small, rare signal; without cropping the model regresses to "always
                      straight"). None = no crop on that side. Prompt schedule is unchanged --
                      flip/turn_done are still absolute raw indices; cropping only selects which
                      frames become steps. Keep a margin of straight frames around the turn so
                      the model still learns the straight->turn->straight transition.

    Each step:
      image        HxWx3 uint8   -- current forward frame
      prompt       str           -- per the schedule above
      action       (16,)         -- ViNT/NoMaD/OmniVLA-style CUMULATIVE waypoints: positions of
                                    the next `horizon`(=8) strided frames, EACH expressed in
                                    THIS frame's robot coordinates (fwd,left), flattened
                                    [x1,y1,...,x8,y8] metres. During a turn the later waypoints
                                    carry a LARGE lateral offset (~0.3-0.5m) -- unlike the
                                    per-step displacement (~0.002m) whose turn signal collapsed
                                    to zero in training. Paired with NUM_ACTIONS_CHUNK=1,
                                    ACTION_DIM=16: the head predicts the whole trajectory in
                                    one shot, so OFT's temporal window is 1 and there is no
                                    double-chunking.
      frame_index  int
    """
    if decision not in PROMPTS:
        raise ValueError(f"decision must be one of {PROMPTS}, got {decision}")
    posed = [(t, rgb, interp_odom(odom, t)) for (t, rgb) in frames]
    n = len(posed)
    lo = 0 if crop_start is None else max(0, crop_start)
    hi = n if crop_end is None else min(n, crop_end)
    steps = []
    for i in range(lo, hi, stride):
        if i + stride * horizon >= n:
            break
        _, rgb_i, (ox, oy, oth) = posed[i]
        # cumulative waypoints: pose of frames i+stride, i+2*stride, ..., i+horizon*stride,
        # all in frame i's coordinates -- the turn accumulates across waypoints instead of
        # hiding in a tiny per-step delta.
        wps = []
        for k in range(1, horizon + 1):
            nx, ny, _ = posed[i + stride * k][2]
            wps.extend(to_robot_frame(nx, ny, ox, oy, oth))
        action = np.asarray(wps, dtype=np.float32)          # (2*horizon,) = (16,)
        if i < flip_frame:
            prompt = "straight"
        elif turn_done_frame is not None and i >= turn_done_frame:
            prompt = "straight"                     # decision consumed; back to driving
        else:
            prompt = decision
        steps.append({"image": rgb_i, "prompt": prompt,
                      "action": action, "frame_index": i})
    return steps


def summarize(steps, flip_frame, decision, turn_done_frame=None):
    from collections import Counter
    c = Counter(s["prompt"] for s in steps)
    td = f", turn done at {turn_done_frame}" if turn_done_frame is not None else ""
    print(f"  {len(steps)} steps  |  flip at frame {flip_frame} -> {decision}{td}")
    print(f"  prompt counts: " + ", ".join(f"{k}={v}" for k, v in c.items()))
    for s in steps:
        fi = s["frame_index"]
        if fi < flip_frame:
            want = "straight"
        elif turn_done_frame is not None and fi >= turn_done_frame:
            want = "straight"
        else:
            want = decision
        assert s["prompt"] == want, f"frame {fi}: prompt {s['prompt']} != expected {want}"
    print(f"  OK: prompt schedule verified")


def plot_episode(steps, out_path, flip_frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"straight": "#5FB37A", "turn_left": "#4C9BE8",
              "turn_right": "#E8994C", "stop": "#E86A6A"}
    # Each step now stores a single-step (2,) displacement. Chain them into a global path,
    # integrating heading, and color each segment by that step's prompt -- so you SEE the
    # prompt flip land on the path (straight-colored approach, then turn-colored at the bend).
    x = y = th = 0.0
    xs, ys, prompts = [0.0], [0.0], []
    for s in steps:
        # action is now (16,) cumulative waypoints; waypoint 0 = next-frame position in this
        # frame = the single-step displacement, so chain wp[0] to rebuild the driven path.
        wp = np.asarray(s["action"], dtype=np.float32).reshape(-1, 2)
        fwd, left = float(wp[0][0]), float(wp[0][1])
        x += fwd * np.cos(th) - left * np.sin(th)
        y += fwd * np.sin(th) + left * np.cos(th)
        th += np.arctan2(left, fwd) if (abs(fwd) + abs(left)) > 1e-6 else 0.0
        xs.append(x); ys.append(y); prompts.append(s["prompt"])
    xs, ys = np.array(xs), np.array(ys)
    for i in range(len(prompts)):
        ax.plot(ys[i:i+2], xs[i:i+2], color=colors.get(prompts[i], "#888"),
                lw=3, solid_capstyle="round")
    ax.scatter([0], [0], marker="^", s=140, color="k", zorder=5)
    handles = [plt.Line2D([], [], color=c, lw=3, label=k) for k, c in colors.items()]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel("left  <--  0  -->  right   [m]")
    ax.set_ylabel("forward  [m]")
    ax.invert_xaxis()   # +left mirrored so physical RIGHT shows on the screen's right
    ax.set_title(f"reconstructed path colored by prompt (flip at frame {flip_frame})")
    ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--flip-frame", type=int, required=True,
                    help="frame where prompt flips straight->decision (the legible frame); 0 if "
                         "the sign is readable from the first frame")
    ap.add_argument("--decision", required=True, choices=["turn_left", "turn_right", "stop"])
    ap.add_argument("--turn-done-frame", type=int, default=None,
                    help="frame where the turn completes and prompt clears back to straight; "
                         "omit if the decision holds to the end of the bag")
    ap.add_argument("--image-topic", default="/c1/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="episode_check")
    args = ap.parse_args()

    frames, odom = read_bag(Path(args.bag), args.image_topic, args.odom_topic, args.ros_distro)
    print(f"read {len(frames)} images, {len(odom)} odom msgs")
    steps = build_steps(frames, odom, args.flip_frame, args.decision,
                        args.turn_done_frame, args.horizon, args.stride)
    summarize(steps, args.flip_frame, args.decision, args.turn_done_frame)
    if args.plot:
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, os.path.basename(args.bag.rstrip("/")) + "_episode.png")
        plot_episode(steps, p, args.flip_frame)
        print(f"plot -> {p}")


if __name__ == "__main__":
    main()