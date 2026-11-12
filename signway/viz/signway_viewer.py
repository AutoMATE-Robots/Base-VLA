#!/usr/bin/env python3
"""
signway_viewer.py -- live window for the SignWay reasoner (pair with signway_node_viz.py).

  Left : camera video + every docTR plate, colour-coded by what the gate thinks of it
  Top  : gate state banner (NO_SIGN / ARMED / ACQUIRE / PENDING / DECIDED / ARRIVED) + maneuver phase
  Right: q vs tau, plate height vs min_plate_h, distance estimate, speed cap, VLM calls + last
         answer + the exact crop sent, LiDAR openings (left/right sectors), odom, event timeline

Box colours:  grey = OCR text, not goal-relevant (R = 0)      orange = relevant, q < tau (ARMED)
              yellow = q >= tau but plate too small (ACQUIRE)  blue = sent to the VLM (PENDING)
              green = decided from this sign                    cyan = already used (memory)

Keys: q / Esc quit, s = screenshot, l = toggle live video vs the exact frame OCR ran on,
      r = toggle recording. Records the window to ~/signway_logs/viz_<time>.mp4 by default.

  source ~/v8_env.sh && unset ROS_NAMESPACE && python3 ~/signway_viz/signway_viewer.py [--no-record]
"""
import argparse
import collections
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, String

VW, VH = 1200, 800          # video area (1920x1280 * 0.625)
PW = 470                    # right panel width
BH = 64                     # banner height
W, H = VW + PW, VH + BH

C = {  # BGR
    "bg": (24, 24, 28), "panel": (36, 36, 42), "fg": (235, 235, 235), "dim": (150, 150, 155),
    "grey": (150, 150, 150), "orange": (0, 140, 255), "yellow": (0, 220, 255),
    "blue": (255, 140, 40), "green": (80, 220, 80), "cyan": (230, 210, 40), "red": (60, 60, 230),
    "magenta": (220, 80, 220),
}
STATE_COL = {"NO_SIGN": C["grey"], "ARMED": C["orange"], "ACQUIRE": C["yellow"],
             "PENDING": C["blue"], "DECIDED": C["green"], "ARRIVED": C["magenta"]}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, txt, org, scale=0.5, col=C["fg"], th=1):
    cv2.putText(img, txt, org, FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, FONT, scale, col, th, cv2.LINE_AA)


def wrap_text(txt, width_chars):
    out, line = [], ""
    for w in str(txt).split():
        if len(line) + len(w) + 1 > width_chars:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


def decode(msg: Image):
    buf = np.frombuffer(msg.data, np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
        return img[:, :, ::-1] if msg.encoding == "rgb8" else img
    if msg.encoding in ("yuv422", "uyvy", "yuv422_yuy2", "yuyv"):
        img = buf.reshape(msg.height, msg.width, 2)
        code = cv2.COLOR_YUV2BGR_UYVY if msg.encoding in ("yuv422", "uyvy") else cv2.COLOR_YUV2BGR_YUY2
        return cv2.cvtColor(img, code)
    if msg.encoding == "mono8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported encoding {msg.encoding}")


class Viewer(Node):
    def __init__(self):
        super().__init__("signway_viewer")
        self.lock = threading.Lock()
        self.frames = collections.deque(maxlen=40)       # (stamp, small BGR)
        self.full_hw = (1280, 1920)
        self.dbg = None
        self.dbg_rx = 0.0
        self.status = None
        self.cap = 0.0
        self.vla_prompt = None
        self.vla_status = None
        self.odom = None
        self.speed = 0.0
        self.openings = []
        self.events = collections.deque(maxlen=14)
        self._prev = {}
        self.crop_cache = {}
        self.step_ms = collections.deque(maxlen=20)
        self.step_times = collections.deque(maxlen=20)
        be = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Image, "/image_raw", self.on_img, be)
        self.create_subscription(String, "/signway/debug", self.on_dbg, 10)
        self.create_subscription(Float32, "/signway/speed_cap", lambda m: setattr(self, "cap", m.data), 10)
        self.create_subscription(String, "/vla/prompt", lambda m: setattr(self, "vla_prompt", m.data), 10)
        self.create_subscription(String, "/vla/status", lambda m: setattr(self, "vla_status", m.data), 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(Float32MultiArray, "/lidar/openings_array",
                                 lambda m: setattr(self, "openings", list(m.data)), 10)

    # ------------------------------------------------------------ callbacks
    def on_img(self, msg):
        try:
            img = decode(msg)
        except Exception as e:                       # noqa: BLE001
            self.get_logger().warn(f"decode: {e}", throttle_duration_sec=5.0)
            return
        small = cv2.resize(img, (VW, VH), interpolation=cv2.INTER_AREA)
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.full_hw = img.shape[:2]
            self.frames.append((st, small))

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.odom = (p.x, p.y, yaw)
        self.speed = m.twist.twist.linear.x

    def ev(self, txt, col=C["fg"]):
        self.events.appendleft((time.strftime("%H:%M:%S"), txt, col))

    def on_dbg(self, msg):
        d = json.loads(msg.data)
        r = d.get("result", {})
        now = time.time()
        self.step_ms.append(r.get("ms", 0))
        self.step_times.append(now)
        st = self.view_state(d)
        p = self._prev
        if st != p.get("st"):
            col = STATE_COL.get(st, C["fg"])
            extra = f" '{r.get('best_text', '')[:28]}'" if st in ("ARMED", "ACQUIRE") else ""
            self.ev(f"{p.get('st', '-')} -> {st}{extra}", col)
        if r.get("acquiring") and not p.get("acq"):
            self.ev(f"ACQUIRE: {r.get('reason', '')[:44]}", C["yellow"])
        if r.get("fired"):
            self.ev(f"FIRE q={r.get('q')} -> VLM call #{d.get('calls')}", C["blue"])
        a = r.get("answered")
        if a:
            if "error" in a:
                self.ev(f"VLM ERROR {a['error'][:40]}", C["red"])
            else:
                self.ev(f"VLM: {a.get('direction')} conf={a.get('confidence', 0):.2f} "
                        f"{a.get('latency_s')}s appl={a.get('applicable')}", C["green"])
        if r.get("fast_path"):
            self.ev(f"FAST PATH -> {r.get('prompt')} (no VLM call)", C["green"])
        if d.get("phase") != p.get("phase"):
            self.ev(f"phase: {p.get('phase', '-')} -> {d.get('phase')}"
                    + (f" ({d.get('held')})" if d.get("held") else ""), C["cyan"])
        if d.get("published") != p.get("pub"):
            self.ev(f"PROMPT -> {d.get('published')}"
                    + ("" if d.get("publish_prompts") else "  [not sent]"), C["magenta"])
        if (d.get("cap", 0) > 0) != (p.get("cap", 0) > 0):
            self.ev(f"SPEED CAP {'%.2f m/s' % d['cap'] if d.get('cap', 0) > 0 else 'off'}", C["orange"])
        self._prev = {"st": st, "acq": r.get("acquiring"), "phase": d.get("phase"),
                      "pub": d.get("published"), "cap": d.get("cap", 0)}
        with self.lock:
            self.dbg, self.dbg_rx = d, now

    @staticmethod
    def view_state(d):
        r = d.get("result", {})
        if d.get("phase") == "arrived" or r.get("arrived"):
            return "ARRIVED"
        if r.get("acquiring"):
            return "ACQUIRE"
        return r.get("state", "NO_SIGN")

    # ------------------------------------------------------------ drawing
    def box_style(self, d, pid):
        ev = {e["id"]: e for e in d.get("evidence", [])}.get(pid)
        tick = d.get("tick", {})
        r = d.get("result", {})
        if ev is None:
            return C["grey"], 1, None
        lab = f"R{ev['R']:.2f} L{ev['L']:.2f} q{ev['q']:.2f}"
        if not ev["novel"]:
            return C["cyan"], 2, lab + " used"
        if ev["R"] <= 0:
            return C["grey"], 1, lab
        if pid == tick.get("best"):
            if r.get("state") == "PENDING" or r.get("fired"):
                return C["blue"], 3, lab + " ASKING"
            if r.get("state") == "DECIDED":
                return C["green"], 3, lab + " DECIDED"
            if r.get("acquiring"):
                return C["yellow"], 3, lab + " ACQUIRE"
        if ev["q"] >= tick.get("tau", 1):
            return C["yellow"], 2, lab
        return C["orange"], 2, lab

    def draw_video(self, canvas, d, live_mode):
        with self.lock:
            frames = list(self.frames)
            fh, fw = self.full_hw
        if not frames:
            put(canvas, "waiting for /image_raw ...", (40, BH + 60), 0.9, C["red"], 2)
            return
        frame = frames[-1][1]
        if d and not live_mode and d.get("stamp"):
            frame = min(frames, key=lambda f: abs(f[0] - d["stamp"]))[1]
        v = frame.copy()
        if d:
            ih, iw = d.get("img_hw", [fh, fw])
            sx, sy = VW / iw, VH / ih
            for o in d.get("ocr", []):                      # raw OCR lines (thin)
                for ln in o.get("lines", []):
                    b = ln.get("box")
                    if b:
                        cv2.rectangle(v, (int(b[0] * sx), int(b[1] * sy)), (int(b[2] * sx), int(b[3] * sy)),
                                      (110, 110, 110), 1)
            for pid, tr in d.get("tracks", {}).items():   # tracked plates (what the gate scores)
                b = tr.get("box")
                if not b:
                    continue
                col, th, lab = self.box_style(d, pid)
                p0, p1 = (int(b[0] * sx), int(b[1] * sy)), (int(b[2] * sx), int(b[3] * sy))
                cv2.rectangle(v, p0, p1, col, th)
                txt = tr.get("text", "")[:42]
                put(v, f"{txt}", (p0[0], max(14, p0[1] - 22)), 0.48, col, 1)
                if lab:
                    put(v, f"{lab}  h={tr.get('h', 0):.0f}px", (p0[0], max(28, p0[1] - 6)), 0.45, col, 1)
            age = time.time() - self.dbg_rx
            if age > 2.0:
                put(v, f"NO /signway/debug for {age:.0f}s - is signway_node_viz.py running?",
                    (20, VH - 20), 0.7, C["red"], 2)
        put(v, "LIVE video (boxes from last OCR step)" if live_mode else "OCR frame (exact)",
            (VW - 380, 24), 0.5, C["dim"])
        canvas[BH:BH + VH, 0:VW] = v

    def draw_banner(self, canvas, d):
        st = self.view_state(d) if d else "NO_SIGN"
        col = STATE_COL.get(st, C["fg"])
        cv2.rectangle(canvas, (0, 0), (VW, BH), C["bg"], -1)
        cv2.rectangle(canvas, (8, 8), (270, BH - 8), col, -1)
        cv2.putText(canvas, st, (20, 46), FONT, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
        if not d:
            put(canvas, "waiting for the reasoner ...", (290, 42), 0.75)
            return
        r = d.get("result", {})
        phase = d.get("phase", "idle")
        ph = {"idle": "looking for signs", "waiting": f"HOLDING '{d.get('held')}' until LiDAR opening "
                                                     f"({d.get('open_streak')}/{d.get('opening_scans')})",
              "executing": f"EXECUTING '{d.get('held')}' yaw {d.get('yaw_accum_deg'):+.0f}/"
                           f"{d.get('turn_done_deg'):.0f} deg",
              "done": "maneuver done", "arrived": "GOAL REACHED - holding stop"}.get(phase, phase)
        put(canvas, f"goal: {d.get('goal')}", (290, 26), 0.6)
        put(canvas, ph, (290, 52), 0.6, C["cyan"] if phase != "idle" else C["dim"])
        pp = d.get("published") or "-"
        put(canvas, f"PROMPT: {pp}", (VW - 330, 30), 0.75, C["magenta"], 2)
        put(canvas, "sent to VLA" if d.get("publish_prompts") else "NOT SENT (publish_prompts=false)",
            (VW - 330, 54), 0.45, C["green"] if d.get("publish_prompts") else C["yellow"])
        if st == "PENDING":
            calls = d.get("vlm_calls", [])
            if calls and not calls[-1].get("done"):
                el = time.time() - calls[-1]["t_start"]
                put(canvas, f"asking VLM ... {el:.1f}s", (VW - 600, 42), 0.7, C["blue"], 2)

    def bar(self, img, x, y, w, val, lo, hi, thr, label, unit=""):
        cv2.rectangle(img, (x, y), (x + w, y + 14), (70, 70, 75), -1)
        f = 0 if hi <= lo else max(0.0, min(1.0, (val - lo) / (hi - lo)))
        col = C["green"] if val >= thr else C["orange"]
        cv2.rectangle(img, (x, y), (x + int(w * f), y + 14), col, -1)
        tx = x + int(w * max(0, min(1, (thr - lo) / (hi - lo))))
        cv2.line(img, (tx, y - 4), (tx, y + 18), C["fg"], 2)
        put(img, f"{label} {val:.2f}{unit}  (thr {thr:g}{unit})", (x, y - 6), 0.47)

    def draw_panel(self, canvas, d):
        x0 = VW
        cv2.rectangle(canvas, (x0, 0), (W, H), C["panel"], -1)
        x, y = x0 + 14, 26
        put(canvas, "SignWay reasoner", (x, y), 0.7, C["fg"], 2)
        if self.step_times and len(self.step_times) > 1:
            hz = (len(self.step_times) - 1) / max(1e-3, self.step_times[-1] - self.step_times[0])
            put(canvas, f"OCR step {np.median(self.step_ms):.0f} ms  @ {hz:.1f} Hz", (x + 230, y), 0.45, C["dim"])
        y += 14
        if not d:
            put(canvas, "no /signway/debug yet", (x, y + 30), 0.6, C["red"])
            return
        r, tick = d.get("result", {}), d.get("tick", {})
        # --- gate meters
        y += 30
        self.bar(canvas, x, y, PW - 40, r.get("q", 0.0), 0, 1, tick.get("tau", 0.45), "q = R x L")
        best = tick.get("best")
        tr = d.get("tracks", {}).get(best, {}) if best else {}
        y += 40
        self.bar(canvas, x, y, PW - 40, tr.get("h", 0.0), 0, max(150, d.get("min_plate_h", 60) * 2.5),
                 d.get("min_plate_h", 60), "plate h", "px")
        y += 26
        th = r.get("text_h_px", 0) or 0
        dist = (d.get("fy") or 0) * (d.get("letter_h") or 0) / th if th > 0 else None
        put(canvas, f"text h {th:.0f}px -> est. dist {dist:.1f} m (letters {d.get('letter_h', 0)*100:.0f} cm)"
            if dist else "text h -   est. dist -", (x, y), 0.47)
        y += 20
        for ln in wrap_text(f"gate: {tick.get('reason') or r.get('reason', '')}", 58)[:2]:
            put(canvas, ln, (x, y), 0.45, C["dim"])
            y += 18
        # --- speed
        y += 8
        cap = d.get("cap", 0)
        put(canvas, f"speed cap: {'%.2f m/s' % cap if cap > 0 else 'none'}   odom v {self.speed:+.2f} m/s",
            (x, y), 0.55, C["orange"] if cap > 0 else C["fg"])
        y += 20
        put(canvas, f"VLM latency est {d.get('latency_est')} s   calls this trip: {d.get('calls')}", (x, y), 0.47)
        # --- last VLM call
        y += 26
        calls = d.get("vlm_calls", [])
        if calls:
            c = calls[-1]
            put(canvas, f"VLM call #{c['n']}:", (x, y), 0.55, C["blue"], 1)
            if not c.get("done"):
                put(canvas, f"in flight {time.time() - c['t_start']:.1f}s", (x + 140, y), 0.55, C["blue"])
            elif c.get("error"):
                put(canvas, f"ERROR {c['error'][:40]}", (x + 140, y), 0.45, C["red"])
            else:
                a = c.get("answer", {})
                put(canvas, f"{a.get('direction')}  conf {a.get('confidence', 0):.2f}  {c.get('latency')}s",
                    (x + 140, y), 0.55, C["green"] if a.get("applicable") else C["yellow"], 1)
                for ln in wrap_text(a.get("summary", ""), 58)[:2]:
                    y += 18
                    put(canvas, ln, (x, y), 0.43, C["dim"])
            crop = self.load_crop(c.get("crop"))
            if crop is not None:
                y += 8
                ch, cw = crop.shape[:2]
                s = min((PW - 28) / cw, 110 / ch)
                th_ = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))))
                canvas[y:y + th_.shape[0], x:x + th_.shape[1]] = th_
                put(canvas, "crop sent", (x + 4, y + 14), 0.4, C["fg"])
                y += th_.shape[0]
        else:
            put(canvas, "VLM: no calls yet", (x, y), 0.5, C["dim"])
        # --- LiDAR + odom
        y += 14
        self.draw_lidar(canvas, x, y, d)
        ox = x + 150
        pose = d.get("pose")
        put(canvas, "odom " + ("OK" if d.get("odom_ok") else "MISSING"), (ox, y + 14), 0.5,
            C["green"] if d.get("odom_ok") else C["red"])
        if pose:
            put(canvas, f"x {pose[0]:+.2f}  y {pose[1]:+.2f}", (ox, y + 34), 0.47)
            put(canvas, f"yaw {math.degrees(pose[2]):+.0f} deg", (ox, y + 52), 0.47)
        put(canvas, f"phase dist {d.get('dist_accum', 0):.1f} m / max wait {d.get('max_wait_m', 0):.0f} m",
            (ox, y + 72), 0.45, C["dim"])
        put(canvas, "lidar " + ("OK" if d.get("lidar_ok") else "MISSING"), (ox, y + 92), 0.5,
            C["green"] if d.get("lidar_ok") else C["red"])
        mem = d.get("memory", [])
        put(canvas, f"memory: {len(mem)} sign(s) used", (ox, y + 112), 0.47, C["cyan"])
        y += 172
        # --- timeline
        put(canvas, "events", (x, y), 0.55, C["fg"], 1)
        for ts, txt, col in list(self.events):
            y += 19
            if y > H - 8:
                break
            put(canvas, f"{ts} {txt}"[:62], (x, y), 0.43, col)

    def draw_lidar(self, canvas, x, y, d):
        R = 64
        cx, cy = x + R + 4, y + R + 4
        cv2.circle(canvas, (cx, cy), R, (70, 70, 75), 1)
        held = d.get("held")
        waiting = d.get("phase") == "waiting"
        for side, a0, a1 in (("turn_left", 25, 155), ("turn_right", -155, -25)):
            col = C["yellow"] if waiting and held == side else (55, 55, 60)
            # robot frame: +x forward = up on screen, +y left = left on screen
            cv2.ellipse(canvas, (cx, cy), (R, R), -90, -a1, -a0, col, 2)
        o = self.openings
        for i in range(0, len(o) - 2, 3):
            b, wdt, dep = o[i], o[i + 1], o[i + 2]
            rr = int(R * min(1.0, dep / 10.0))
            cv2.ellipse(canvas, (cx, cy), (rr, rr), -90, -(b + wdt / 2), -(b - wdt / 2), C["green"], -1)
        cv2.rectangle(canvas, (cx - 4, cy - 6), (cx + 4, cy + 6), C["fg"], -1)
        put(canvas, "openings", (x, y + 2 * R + 20), 0.42, C["dim"])

    def load_crop(self, path):
        if not path:
            return None
        if path not in self.crop_cache:
            self.crop_cache[path] = cv2.imread(path)
        return self.crop_cache[path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args()
    rclpy.init()
    node = Viewer()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    win = "SignWay live"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    live_mode, rec = True, not a.no_record
    writer, rec_path = None, None
    shots = Path("~/signway_logs").expanduser()
    shots.mkdir(parents=True, exist_ok=True)
    period = 1.0 / a.fps
    try:
        while rclpy.ok():
            t0 = time.time()
            with node.lock:
                d = node.dbg
            canvas = np.full((H, W, 3), C["bg"], np.uint8)
            node.draw_video(canvas, d, live_mode)
            node.draw_banner(canvas, d)
            node.draw_panel(canvas, d)
            if rec:
                if writer is None:
                    rec_path = shots / f"viz_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    writer = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
                    node.get_logger().info(f"recording -> {rec_path}")
                writer.write(canvas)
                cv2.circle(canvas, (W - 18, H - 18), 7, C["red"], -1)
            cv2.imshow(win, canvas)
            k = cv2.waitKey(max(1, int((period - (time.time() - t0)) * 1000))) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("l"):
                live_mode = not live_mode
            if k == ord("s"):
                p = shots / f"viz_shot_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(str(p), canvas)
                node.get_logger().info(f"screenshot -> {p}")
            if k == ord("r"):
                rec = not rec
                if not rec and writer is not None:
                    writer.release()
                    writer = None
                    node.get_logger().info(f"saved {rec_path}")
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            print(f"saved {rec_path}")
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
