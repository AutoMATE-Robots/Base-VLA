#!/usr/bin/env python3
r"""
pepper_vla_node.py -- RAW PASSTHROUGH version. Runs on the JETSON.
  /image_raw --(HTTP)--> policy_server (MSI A40) --(8 waypoints)--> pure pursuit --> /cmd_vel
                                                  \--> /pepper_vla/path  (nav_msgs/Path, base_link)
  /odom  --> heading integrator --> turn-completion state machine --> which prompt to send
WHAT "RAW" MEANS HERE
  * Speed comes from the MODEL: |waypoint_1| / 0.1s. No fixed speed, no scaling.
  * Steering comes from pure pursuit on the MODEL's waypoints, untouched.
  * NO omega_gain, NO dead_band / turn gate, NO trim, NO smoothing. Removed.
  * max_speed / max_omega remain, but default HIGH so they do not bind in normal
    operation -- they exist only to bound a garbage prediction (e.g. a near-zero
    forward waypoint makes curvature = 2y/L^2 explode). Set them lower while testing.
  A controller is unavoidable: the model outputs a path, the base takes (v, omega).
  Pure pursuit is the standard conversion and what ViNT / NoMaD / OmniVLA use. It
  applies no scaling of its own -- it just follows the predicted trajectory.
TURN COMPLETION (not an output modification -- it sets the PROMPT)
  During data collection `turn_done_frame` was a human deciding the turn was over.
  Deployment has no such human, so without this the prompt stays "turn_right" forever
  and the robot keeps turning. The node integrates NET yaw change from /odom and flips
  the prompt back to "straight" past turn_done_deg. Disable with -p auto_turn_done:=false.

  NET, not accumulated-absolute (fixed Aug 10 2026): the previous version summed
  abs(dyaw) every odom sample from the moment the turn prompt armed. Absolute value
  means noise NEVER cancels -- straight-approach wobble at 20 Hz burned through the
  whole turn budget before the junction, flipping the prompt to "straight" early.
  Signed accumulation makes left/right wobble cancel to ~0 over a straight run, so
  `turned` measures actual rotation. Signed sums (not wrap-of-difference) so turns
  beyond 180 deg still work.
SAFETY
  * Wheels off the ground first. Confirm the sign of angular.z matches the decision.
  * Server error / timeout / stale waypoints -> ZERO velocity.
  * No obstacle avoidance. Human on the e-stop.
Run:
    export POLICY_URL=http://127.0.0.1:8000/predict
    python3 pepper_vla_node.py --ros-args -p decision:=turn_right
"""
import base64
import math
import os
import time
import cv2
import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from std_msgs.msg import String

from jetson_refine_client import RefineClient          # top of file

          # in __init__
                                                       # policy server's 16 numbers
                                                       # currently go to the executor
                                                       
TURNS = ("turn_left", "turn_right")
DT = 0.1          # seconds between waypoints (10 Hz dataset)
def yaw_from_quat(q):
    s = 2.0 * (q.w * q.z + q.x * q.y)
    c = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(s, c)
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi
class PepperVLANode(Node):
    def __init__(self):
        super().__init__("pepper_vla")
        self.declare_parameter("image_topic", "/image_raw")
        self.refine = RefineClient(self, shadow=False)
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("decision", "straight")
        self.declare_parameter("auto_turn_done", True)
        self.declare_parameter("turn_done_deg", 80.0)
        self.declare_parameter("infer_hz", 3.0)
        self.declare_parameter("lookahead_idx", 8)     # which waypoint to chase (1..8)
        self.declare_parameter("stale_after", 0.8)     # s; older -> stop
        self.declare_parameter("jpeg_quality", 92)
        self.declare_parameter("dry_run", False)
        # Safety bounds only. High by default so the model's own values pass through.
        self.declare_parameter("max_speed", 1.0)       # m/s   (model asks ~0.7)
        self.declare_parameter("max_omega", 2.0)       # rad/s (model asks ~0.5)
        gp = lambda n: self.get_parameter(n).value
        self.auto_turn_done = gp("auto_turn_done")
        self.turn_done_rad = math.radians(gp("turn_done_deg"))
        self.infer_dt = 1.0 / max(0.5, gp("infer_hz"))
        self.look_idx = int(gp("lookahead_idx"))
        self.stale_after = gp("stale_after")
        self.jpeg_q = int(gp("jpeg_quality"))
        self.dry_run = gp("dry_run")
        self.max_speed = gp("max_speed")
        self.max_omega = gp("max_omega")
        self.url = os.environ.get("POLICY_URL", "http://127.0.0.1:8000/predict")
        self.bridge = CvBridge()
        self.session = requests.Session()
        self.decision = gp("decision")
        self.turn_done = False
        self.turned = 0.0          # |net rotation| since the turn prompt armed (rad)
        self._turn_accum = 0.0     # SIGNED net rotation (rad); wobble cancels
        self._last_yaw = None
        self.last_req = 0.0
        self.seq = 0
        self.traj = None
        self.traj_stamp = 0.0
        self.create_subscription(Image, gp("image_topic"), self.on_image,
                                 QoSPresetProfiles.SENSOR_DATA.value)
        self.create_subscription(Odometry, gp("odom_topic"), self.on_odom, 20)
        self.create_subscription(String, "/pepper_vla/decision", self.on_decision, 10)
        self.pub_cmd = self.create_publisher(Twist, gp("cmd_topic"), 10)
        self.pub_path = self.create_publisher(Path, "/pepper_vla/path", 10)
        self.pub_state = self.create_publisher(String, "/pepper_vla/state", 10)
        self.timer = self.create_timer(0.05, self.control_step)   # 20 Hz
        self.get_logger().info(
            f"pepper_vla RAW | decision='{self.decision}' | server={self.url} | "
            f"speed=from model (cap {self.max_speed}) | omega cap {self.max_omega} | "
            f"auto_turn_done={self.auto_turn_done} "
            f"(net yaw, {math.degrees(self.turn_done_rad):.0f}deg) | "
            f"dry_run={self.dry_run}")
    # ---------------- prompt state machine ----------------
    def on_decision(self, msg: String):
        d = msg.data.strip()
        if d not in TURNS + ("straight", "stop"):
            self.get_logger().warn(f"ignoring unknown decision {d!r}")
            return
        self.decision = d
        self.turn_done = False
        self.turned = 0.0
        self._turn_accum = 0.0
        self.get_logger().info(f"decision -> {d} (re-armed)")
    def on_odom(self, msg: Odometry):
        y = yaw_from_quat(msg.pose.pose.orientation)
        if (self.auto_turn_done and self._last_yaw is not None
                and self.active_prompt() in TURNS):
            # SIGNED accumulation: left and right wobble cancel over a straight
            # approach, so `turned` reflects real rotation rather than integrated
            # noise. (Summing signed deltas rather than wrapping one big difference
            # keeps turns > 180 deg correct.)
            self._turn_accum += wrap(y - self._last_yaw)
            self.turned = abs(self._turn_accum)
            if self.turned >= self.turn_done_rad and not self.turn_done:
                self.turn_done = True
                self.get_logger().info(
                    f"TURN COMPLETE after {math.degrees(self.turned):.0f}deg net "
                    f"-> prompt back to 'straight'")
        self._last_yaw = y
    def active_prompt(self) -> str:
        if self.decision in TURNS and not (self.auto_turn_done and self.turn_done):
            return self.decision
        return "straight"
    # ---------------- perception -> waypoints ----------------
    def on_image(self, msg: Image):
        now = time.time()
        if now - self.last_req < self.infer_dt:
            return
        self.last_req = now
        prompt = self.active_prompt()
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
            if not ok:
                return
            payload = {"image_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                       "prompt": prompt, "seq": self.seq}
            self.seq += 1
            r = self.session.post(self.url, json=payload, timeout=1.0)
            r.raise_for_status()
            out = r.json()
            if "waypoints" not in out:
                self.get_logger().warn(f"server: {out.get('error')}")
                return
            raw_wp = np.asarray(out["waypoints"], dtype=float)  # (8, 2) from policy
            refined = self.refine.process(raw_wp.reshape(-1))  # 16 in, 16 out
            self.traj = np.asarray(refined, dtype=float).reshape(8, 2)
            self.traj_stamp = time.time()
            self.publish_path(msg.header)
            st = String()
            st.data = (f"decision={self.decision} prompt={prompt} "
                       f"turned={math.degrees(self.turned):.0f}deg done={self.turn_done}")
            self.pub_state.publish(st)
            self.get_logger().info(
                f"[{prompt}] turned={math.degrees(self._turn_accum):+6.1f}deg "
                f"lat={out.get('latency_ms')}ms "
                f"wp1=({self.traj[0][0]:+.3f},{self.traj[0][1]:+.3f}) "
                f"wp8=({self.traj[-1][0]:+.3f},{self.traj[-1][1]:+.3f})",
                throttle_duration_sec=1.0)
        except requests.exceptions.RequestException as e:
            self.get_logger().warn(f"server unreachable: {e}", throttle_duration_sec=2.0)
        except Exception as e:
            self.get_logger().error(f"on_image: {type(e).__name__}: {e}")
    def publish_path(self, header):
        p = Path()
        p.header.frame_id = "base_link"
        p.header.stamp = header.stamp
        for x, y in self.traj:
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            p.poses.append(ps)
        self.pub_path.publish(p)
    # ---------------- waypoints -> motion : RAW ----------------
    def control_step(self):
        cmd = Twist()
        if self.traj is None or (time.time() - self.traj_stamp) > self.stale_after:
            self.publish(cmd)                      # stale / no trajectory -> STOP
            return
        # speed: straight from the model's own waypoint spacing
        v = float(np.hypot(self.traj[0][0], self.traj[0][1]) / DT)
        # steering: pure pursuit on the model's waypoints, unscaled
        i = min(max(1, self.look_idx), len(self.traj)) - 1
        tx, ty = float(self.traj[i][0]), float(self.traj[i][1])
        if tx <= 0.01:
            self.publish(cmd)
            return
        L2 = tx * tx + ty * ty
        curvature = 2.0 * ty / L2
        omega = v * curvature
        # safety bounds only (high by default; they should not bind)
        cmd.linear.x = float(np.clip(v, 0.0, self.max_speed))
        cmd.angular.z = float(np.clip(omega, -self.max_omega, self.max_omega))
        self.publish(cmd)
    def publish(self, cmd: Twist):
        if self.dry_run:
            self.get_logger().info(
                f"[dry_run] v={cmd.linear.x:+.3f} w={cmd.angular.z:+.3f}",
                throttle_duration_sec=0.5)
            return
        self.pub_cmd.publish(cmd)
    def stop(self):
        self.pub_cmd.publish(Twist())
def main():
    rclpy.init()
    node = PepperVLANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.stop()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == "__main__":
    main()

