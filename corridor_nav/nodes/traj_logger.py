#!/usr/bin/env python3
"""
traj_logger.py -- record every trajectory the VLA emits, for later plotting.

Drop this next to pepper_vla_node.py on the Jetson and add three lines to the
node (see INTEGRATION below). It writes one JSON object per prediction to a
.jsonl file.

WHY EACH FIELD IS THERE (they are all needed to draw the figure):

  wp        the prediction itself, in the CURRENT BODY FRAME, metres.
  odom      the robot pose when that prediction was made. Body-frame waypoints
            are meaningless on a third-person photo until they are lifted to the
            world frame:  p_world_k = p_t + R(psi_t) @ w_k.  Without the pose at
            prediction time the whole figure is impossible, so this is the field
            not to drop.
  t_image   ROS stamp of the frame the policy actually consumed. This is what
            syncs a logged prediction to the recorded bag and to the tripod
            video -- wall-clock alone drifts.
  cmd       what pure pursuit did with the prediction, so the figure can show
            intent (waypoints) against execution (odom) if useful.

    from traj_logger import TrajectoryLogger
    log = TrajectoryLogger("~/traj_logs", meta={...})
    log.write(...)   # once per prediction
    log.close()

INTEGRATION (pepper_vla_node.py) -- three edits:

  1. near the top:
        from traj_logger import TrajectoryLogger

  2. in __init__, after the parameters are read:
        self.tlog = TrajectoryLogger(
            "~/traj_logs",
            meta=dict(checkpoint=self.checkpoint if hasattr(self, "checkpoint") else "",
                      max_speed=max_speed, max_omega=max_omega,
                      server=self.server if hasattr(self, "server") else ""))

  3. wherever the node already prints its per-frame line
     ("[straight] turned=... wp1=... wp8=..."), right after the waypoints come
     back from the policy server:
        self.tlog.write(prompt=self.decision, wp=wps,
                        odom=(self.x, self.y, self.yaw),
                        t_image=img_stamp_sec, latency_ms=lat_ms,
                        cmd=(v, w), turned_deg=self.turned_deg,
                        done=self.turn_done)

     (Rename the locals to match the node -- the point is that `wp` is the
     8x2 array the server returned and `odom` is the pose at that instant.)

  4. optional, in the shutdown path:
        self.tlog.close()

Nothing here can block the control loop: writes are line-buffered appends and
failures are swallowed with a one-time warning.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


class TrajectoryLogger:
    """Append-only JSONL log of VLA predictions."""

    def __init__(self, out_dir="~/traj_logs", prefix="traj", meta=None,
                 logger=None):
        self.dir = Path(os.path.expanduser(out_dir))
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%m%d_%H%M%S")
        self.path = self.dir / f"{prefix}_{stamp}.jsonl"
        self.seq = 0
        self._warned = False
        self._log = logger
        self._fh = None
        try:
            self._fh = open(self.path, "w", buffering=1)   # line buffered
            header = dict(_meta=dict(
                started=time.time(),
                started_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                **(meta or {})))
            self._fh.write(json.dumps(header) + "\n")
            self._say(f"trajectory log -> {self.path}")
        except Exception as e:                              # noqa: BLE001
            self._say(f"could not open trajectory log: {e}", warn=True)

    # ------------------------------------------------------------------ #
    def write(self, prompt, wp, odom=None, t_image=None, latency_ms=None,
              cmd=None, turned_deg=None, done=None, **extra):
        """One prediction.

        prompt      str, the standing instruction in force
        wp          iterable of 8 (fwd, left) pairs in the CURRENT body frame [m]
        odom        (x, y, yaw) at prediction time [m, m, rad]
        t_image     ROS stamp (float seconds) of the frame the policy consumed
        latency_ms  round trip to the policy server
        cmd         (v, omega) issued by pure pursuit
        """
        if self._fh is None:
            return
        try:
            rec = dict(seq=self.seq, t_wall=time.time(), prompt=str(prompt))
            rec["wp"] = [[round(float(a), 5), round(float(b), 5)]
                         for a, b in _pairs(wp)]
            if odom is not None:
                x, y, yaw = odom
                rec["odom"] = dict(x=round(float(x), 5), y=round(float(y), 5),
                                   yaw=round(float(yaw), 6))
            if t_image is not None:
                rec["t_image"] = float(t_image)
            if latency_ms is not None:
                rec["latency_ms"] = round(float(latency_ms), 1)
            if cmd is not None:
                v, w = cmd
                rec["cmd"] = dict(v=round(float(v), 4), w=round(float(w), 4))
            if turned_deg is not None:
                rec["turned_deg"] = round(float(turned_deg), 2)
            if done is not None:
                rec["done"] = bool(done)
            if extra:
                rec.update(extra)
            self._fh.write(json.dumps(rec) + "\n")
            self.seq += 1
        except Exception as e:                              # noqa: BLE001
            if not self._warned:
                self._say(f"trajectory log write failed: {e}", warn=True)
                self._warned = True

    def event(self, kind, **fields):
        """Mark something worth finding later: a decision arriving, the turn
        completing, the operator pressing start. Useful for cropping the figure
        to the junction."""
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps(
                dict(event=str(kind), t_wall=time.time(), **fields)) + "\n")
        except Exception:                                   # noqa: BLE001
            pass

    def close(self):
        if self._fh is not None:
            try:
                self._fh.write(json.dumps(
                    dict(_meta=dict(ended=time.time(), n=self.seq))) + "\n")
                self._fh.close()
                self._say(f"trajectory log closed: {self.seq} predictions -> "
                          f"{self.path}")
            except Exception:                               # noqa: BLE001
                pass
            self._fh = None

    # ------------------------------------------------------------------ #
    def _say(self, msg, warn=False):
        if self._log is not None:
            (self._log.warn if warn else self._log.info)(msg)
        else:
            print(("[traj_logger][WARN] " if warn else "[traj_logger] ") + msg)


def _pairs(wp):
    """Accept (8,2) arrays, flat 16-vectors, or lists of pairs."""
    seq = list(wp)
    if seq and hasattr(seq[0], "__len__") and len(seq[0]) == 2:
        return [(p[0], p[1]) for p in seq]
    flat = [float(v) for v in seq]
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]


# ===================================================================== #
# reader -- used by the plotting tool
# ===================================================================== #
def read_log(path):
    """-> (meta, records, events). Records keep insertion order."""
    meta, recs, evs = {}, [], []
    for line in open(os.path.expanduser(path)):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "_meta" in d:
            meta.update(d["_meta"])
        elif "event" in d:
            evs.append(d)
        else:
            recs.append(d)
    return meta, recs, evs


def to_world(rec):
    """Body-frame waypoints -> world frame, using the pose logged with them.

        p_world_k = p_t + R(psi_t) @ w_k

    Returns [(X, Y), ...] or None when the record carries no pose.
    """
    if "odom" not in rec:
        return None
    o = rec["odom"]
    c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    return [(o["x"] + c * fwd - s * left,
             o["y"] + s * fwd + c * left) for fwd, left in rec["wp"]]


def executed_path(recs):
    """The robot's own path, straight from the logged poses."""
    return [(r["odom"]["x"], r["odom"]["y"]) for r in recs if "odom" in r]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: traj_logger.py <log.jsonl>   # summarise a log")
    meta, recs, evs = read_log(sys.argv[1])
    print("meta:", json.dumps(meta, indent=1)[:400])
    print(f"{len(recs)} predictions, {len(evs)} events")
    if recs:
        pr = {}
        for r in recs:
            pr[r["prompt"]] = pr.get(r["prompt"], 0) + 1
        print("prompts:", pr)
        w = to_world(recs[len(recs) // 2])
        if w:
            print("mid-log world waypoints:",
                  [(round(a, 2), round(b, 2)) for a, b in w[:3]], "...")
        p = executed_path(recs)
        if len(p) > 1:
            dx = p[-1][0] - p[0][0]
            dy = p[-1][1] - p[0][1]
            print(f"path extent: {math.hypot(dx, dy):.2f} m over {len(p)} poses")
    for e in evs[:6]:
        print("event:", e)
