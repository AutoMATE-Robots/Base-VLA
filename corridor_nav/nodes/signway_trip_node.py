#!/usr/bin/env python3
r"""
signway_trip_node.py -- the missing link. Runs on the JETSON, alongside
pepper_vla_node.py.

    /image_raw --(HTTP)--> reasoning_server /reason  (MSI :8001)
                              |
                              v  when the gate fires and the VLM answers
                  /pepper_vla/decision  (std_msgs/String)
                              |
                              v
                    pepper_vla_node  --(HTTP)--> policy_server /predict (MSI :8000)
                              |
                              v
                          /cmd_vel

LONG HORIZON: the goal is a STANDING instruction ("room 305") for the whole trip.
At each junction the gate arms on a detected sign, fires, and the VLM decides
whether that sign is relevant to the goal. When pepper_vla_node reports the turn
complete (its own odometry state machine, published on /pepper_vla/state), this
node POSTs /rearm and the cycle repeats at the next sign. No map, no waypoints --
a chain of sign decisions.

WHAT RUNS TODAY vs LATER
  * Without a trained deadline estimator the gate fires on EVIDENCE SUFFICIENCY
    only. That is the reactive baseline -- valid to run, and the control condition
    the deadline work is measured against.
  * Pass --oracle-speed-distance to feed a crude d_q10 (speed x a fixed
    time-to-junction guess) if you want to exercise the deadline path before the
    estimator exists. OFF by default: a made-up distance produces made-up science.

SAFETY: this node only publishes PROMPTS. It never touches /cmd_vel. The worst it
can do is set the wrong standing direction -- pepper_vla_node still decides when.
Human on the e-stop regardless.

Run:
    python3 signway_trip_node.py --ros-args \
        -p goal:="room 305" -p server:="http://127.0.0.1:8001" -p dry_run:=true
"""
import base64
import time

import cv2
import rclpy
import requests
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from std_msgs.msg import String

VALID = ("turn_left", "turn_right", "straight", "stop")


class TripNode(Node):
    def __init__(self):
        super().__init__("signway_trip")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("server", "http://127.0.0.1:8001")
        self.declare_parameter("goal", "room 305")
        self.declare_parameter("reason_hz", 2.0)
        self.declare_parameter("session", "trip")
        self.declare_parameter("jpeg_quality", 92)
        self.declare_parameter("dry_run", True)          # safe default
        self.declare_parameter("auto_rearm", True)
        self.declare_parameter("oracle_speed_distance", 0.0)  # 0 = disabled

        gp = lambda n: self.get_parameter(n).value
        self.server = gp("server").rstrip("/")
        self.goal = gp("goal")
        self.period = 1.0 / max(0.2, gp("reason_hz"))
        self.session = gp("session")
        self.jpeg_q = int(gp("jpeg_quality"))
        self.dry_run = gp("dry_run")
        self.auto_rearm = gp("auto_rearm")
        self.oracle_t = float(gp("oracle_speed_distance"))

        self.bridge = CvBridge()
        self.http = requests.Session()
        self.speed = 0.0
        self.frame_idx = 0
        self.last_call = 0.0
        self.decided = None          # decision currently published
        self.turn_was_done = False
        self.junctions = 0
        self.fired_this_run = False  # guards against acting on stale server state

        self.create_subscription(Image, gp("image_topic"), self.on_image,
                                 QoSPresetProfiles.SENSOR_DATA.value)
        self.create_subscription(Odometry, gp("odom_topic"), self.on_odom, 20)
        self.create_subscription(String, "/pepper_vla/state", self.on_state, 10)
        self.pub_decision = self.create_publisher(String, "/pepper_vla/decision", 10)
        self.pub_trip = self.create_publisher(String, "/signway_trip/status", 10)

        self.get_logger().info(
            f"signway_trip | goal='{self.goal}' | server={self.server} | "
            f"{gp('reason_hz')}Hz | dry_run={self.dry_run} | "
            f"auto_rearm={self.auto_rearm}")
        self._health()
        # Gate state lives on the SERVER, keyed by session id -- restarting this
        # node does not reset it. Without this, a fresh trip inherits the previous
        # run's DECIDED state and replays a stale decision on the first frame,
        # before any sign has been seen.
        self._rearm()
        self.get_logger().info("gate re-armed for a fresh trip")

    # ------------------------------------------------------------------ setup
    def _health(self):
        try:
            h = self.http.get(f"{self.server}/health", timeout=5).json()
            self.get_logger().info(
                f"reasoning server OK | evidence={h.get('evidence_backend')} "
                f"vlm={h.get('vlm')} deadline={h.get('deadline_estimator')} "
                f"memory={h.get('memory_records')}")
            if not h.get("deadline_estimator"):
                self.get_logger().warn(
                    "no deadline estimator loaded -> SUFFICIENCY-ONLY firing "
                    "(this is the reactive baseline condition)")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(
                f"cannot reach {self.server}: {e}  -- is the SSH tunnel up? "
                f"ssh -N -L 8001:<node>:8001 munda057@agate.msi.umn.edu")

    # --------------------------------------------------------------- feedback
    def on_odom(self, msg: Odometry):
        v = msg.twist.twist.linear
        self.speed = float((v.x ** 2 + v.y ** 2) ** 0.5)

    def on_state(self, msg: String):
        """pepper_vla_node publishes '... turned=NNdeg done=True/False'.
        The False->True edge is the turn-complete event: re-arm for the next sign."""
        done = "done=True" in msg.data
        if done and not self.turn_was_done and self.auto_rearm:
            self.junctions += 1
            self.get_logger().info(
                f"turn complete (junction {self.junctions}) -> re-arming gate")
            self._rearm()
        self.turn_was_done = done

    def _rearm(self):
        self.decided = None
        self.fired_this_run = False
        try:
            self.http.post(f"{self.server}/rearm",
                           json={"session": self.session}, timeout=3)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"rearm failed: {e}")

    # ------------------------------------------------------------ the request
    def on_image(self, msg: Image):
        now = time.time()
        if now - self.last_call < self.period:
            return
        self.last_call = now
        self.frame_idx += 1

        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
            if not ok:
                return
            payload = {
                "image_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "goal": self.goal, "speed_mps": self.speed,
                "frame_idx": self.frame_idx, "session": self.session,
            }
            if self.oracle_t > 0 and self.speed > 0.05:
                # crude stand-in ONLY for exercising the deadline path
                payload["d_q10"] = self.speed * self.oracle_t
                payload["p_junction"] = 1.0

            r = self.http.post(f"{self.server}/reason", json=payload, timeout=2.0)
            r.raise_for_status()
            out = r.json()
        except requests.exceptions.RequestException as e:
            self.get_logger().warn(f"reason unreachable: {e}",
                                   throttle_duration_sec=3.0)
            return
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"on_image: {type(e).__name__}: {e}")
            return

        self._handle(out)

    def _handle(self, out: dict):
        state = out.get("state")
        if out.get("fire"):
            self.fired_this_run = True
            self.get_logger().info(
                f"GATE FIRED ({out.get('fire_reason')}) "
                f"score={out.get('best_score')} buffer={out.get('buffer_size')} "
                f"crops={out.get('buffer_texts')} -> VLM running, robot keeps driving")

        txt = out.get("best_text") or ""
        ign = out.get("ignored", 0)
        self.get_logger().info(
            f"[{state}] score={out.get('best_score')} buf={out.get('buffer_size')} "
            f"regions={out.get('regions')}{f'(-{ign} ignored)' if ign else ''} "
            f"read={txt[:32]!r} pending={out.get('vlm_pending')} "
            f"srv={out.get('server_ms')}ms", throttle_duration_sec=2.0)

        d = out.get("decision")
        if d and not self.fired_this_run and out.get("decision_source") == "vlm":
            self.get_logger().warn(
                f"ignoring stale server decision '{d}' -- this run has not fired yet",
                throttle_duration_sec=5.0)
            return
        if d and d in VALID and d != self.decided:
            self.decided = d
            src = out.get("decision_source")
            self.get_logger().info(
                f"DECISION: {d}  (source={src}, vlm_p90="
                f"{out.get('vlm_latency_p90_s')}s)")
            st = String()
            st.data = (f"junction={self.junctions + 1} decision={d} source={src} "
                       f"goal={self.goal}")
            self.pub_trip.publish(st)
            if self.dry_run:
                self.get_logger().warn(f"[dry_run] NOT publishing '{d}' "
                                       f"to /pepper_vla/decision")
            else:
                m = String()
                m.data = d
                self.pub_decision.publish(m)
        elif d == "not_applicable":
            self.get_logger().info("sign not applicable to goal -- holding straight")


def main():
    rclpy.init()
    node = TripNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
