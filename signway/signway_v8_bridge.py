#!/usr/bin/env python3
"""
signway_v8_bridge.py -- connects SignWay sign reasoning (VLM on the MSI) to the v8 VLA +
LiDAR drive stack on the Jetson. It never touches /cmd_vel; it only sets the PROMPT.

    signway_trip_node --/pepper_vla/decision--> [bridge] --/vla/prompt--> v8_drive_node
          ^                                       |  ^  ^
          +-------------/pepper_vla/state---------+  |  +-- /lidar/openings_array (junction gate)
                        ("done=True" -> re-arm)      +----- /odom (turn yaw, distance)

Roles: the VLM decides WHICH WAY. The LiDAR decides WHEN (junction gate) and HOW (the
drive node's balance planner). The VLA draws the path in the image.

STATES
  CRUISE     prompt=straight, waiting for a decision
  COMMITTED  a turn was decided, but prompt stays "straight" until the LiDAR sees a
             robot-wide opening on that side (gate). Stops the VLA turning early into a
             doorway or a gap between obstacles. -p gate:=false = old behaviour
             (turn prompt sent the moment the sign is read).
  TURNING    prompt=turn_left/right. Done when NET odom yaw since the gate opened
             reaches turn_done_deg -> CRUISE, publishes done=True (signway re-arms).
  PASSING    a "straight" decision. Done once a side opening was seen and is now behind
             us, or after straight_done_m metres -> done=True. Without this the old
             pipeline never re-armed after a "straight" decision.
  GOAL       a "stop" decision. prompt=stop (the drive node ramps to zero).

Run (system ROS shell is fine; no torch needed):
    source /opt/ros/humble/setup.bash && unset ROS_NAMESPACE
    python3 ~/signway_v8_bridge.py
    python3 ~/signway_v8_bridge.py --ros-args -p gate:=false        # old behaviour
"""
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

TURNS = ("turn_left", "turn_right")
VALID = TURNS + ("straight", "stop")


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Bridge(Node):
    def __init__(self):
        super().__init__("signway_v8_bridge")
        p = self.declare_parameter
        p("decision_topic", "/pepper_vla/decision")
        p("state_topic", "/pepper_vla/state")
        p("prompt_topic", "/vla/prompt")
        p("odom_topic", "/odom")
        p("openings_topic", "/lidar/openings_array")
        p("gate", True)                 # wait for a LiDAR opening before the turn prompt
        p("gate_min_bearing", 40.0)     # deg; side opening = |bearing| in [min, max]
        p("gate_max_bearing", 150.0)
        p("gate_min_depth", 2.0)        # m; shallower = doorway/alcove, not a corridor
        p("gate_scans", 3)              # consecutive scans (10 Hz) before trusting it
        p("turn_done_deg", 80.0)        # net yaw that ends a turn (same as pepper node)
        p("straight_done_m", 8.0)       # fallback re-arm distance for "straight"
        p("passed_scans", 10)           # scans with no side opening = junction passed
        p("republish_s", 3.0)           # re-send the prompt (late-starting drive node)
        g = lambda n: self.get_parameter(n).value
        self.gate = bool(g("gate"))
        self.b_min, self.b_max = float(g("gate_min_bearing")), float(g("gate_max_bearing"))
        self.min_depth = float(g("gate_min_depth"))
        self.gate_scans = int(g("gate_scans"))
        self.turn_done = math.radians(float(g("turn_done_deg")))
        self.straight_done_m = float(g("straight_done_m"))
        self.passed_scans = int(g("passed_scans"))
        self.republish_s = float(g("republish_s"))

        self.state = "CRUISE"
        self.decision = None
        self.done = False
        self.prompt = "straight"
        self._yaw_accum = 0.0
        self._dist = 0.0
        self._last_pose = None          # (x, y, yaw)
        self._gate_n = 0                # consecutive scans with an opening on our side
        self._side_seen = False         # PASSING: a junction was alongside us
        self._side_gone_n = 0
        self._ops_t = None
        self._last_pub = 0.0

        self.create_subscription(String, g("decision_topic"), self._on_decision, 10)
        self.create_subscription(Odometry, g("odom_topic"), self._on_odom, 20)
        self.create_subscription(Float32MultiArray, g("openings_topic"), self._on_ops, 10)
        self.pub_prompt = self.create_publisher(String, g("prompt_topic"), 10)
        self.pub_state = self.create_publisher(String, g("state_topic"), 10)
        self.pub_status = self.create_publisher(String, "/signway_bridge/status", 10)
        self.create_timer(0.2, self._tick)
        self.get_logger().info(
            f"bridge | gate={self.gate} side=[{self.b_min:.0f},{self.b_max:.0f}]deg "
            f"depth>={self.min_depth}m x{self.gate_scans} | turn_done={g('turn_done_deg')}deg "
            f"| straight_done={self.straight_done_m}m")
        self._set_prompt("straight")

    # ------------------------------------------------------------------ helpers
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _set_prompt(self, d):
        changed = d != self.prompt
        self.prompt = d
        self.pub_prompt.publish(String(data=d))
        self._last_pub = self._now()
        if changed:
            self.get_logger().info(f"prompt -> {d}   [{self.state}]")

    def _finish(self, why):
        self.get_logger().info(f"{self.state} {self.decision} complete ({why}) -> CRUISE, done=True")
        self.state, self.done = "CRUISE", True
        self._set_prompt("straight")

    def _side_openings(self, ops):
        """Robot-wide, deep openings on each side, from [bearing, width, depth]* triples."""
        left = right = False
        for i in range(0, len(ops) - 2, 3):
            b, _w, d = ops[i], ops[i + 1], ops[i + 2]
            if d < self.min_depth:
                continue
            if self.b_min <= b <= self.b_max:
                left = True
            elif -self.b_max <= b <= -self.b_min:
                right = True
        return left, right

    # ---------------------------------------------------------------- callbacks
    def _on_decision(self, msg):
        d = msg.data.strip()
        if d not in VALID:
            self.get_logger().warn(f"ignoring unknown decision {d!r}")
            return
        self.decision, self.done = d, False
        self._yaw_accum = self._dist = 0.0
        self._gate_n = self._side_gone_n = 0
        self._side_seen = False
        if d == "stop":
            self.state = "GOAL"
            self._set_prompt("stop")
        elif d == "straight":
            self.state = "PASSING"
            self._set_prompt("straight")
        elif self.gate:
            self.state = "COMMITTED"
            self.get_logger().info(f"DECISION {d}: committed, holding 'straight' until a "
                                   f"{d.split('_')[1]} opening is seen")
            self._set_prompt("straight")
        else:
            self.state = "TURNING"
            self._set_prompt(d)

    def _on_odom(self, msg):
        q, pos = msg.pose.pose.orientation, msg.pose.pose.position
        yaw = yaw_from_quat(q)
        if self._last_pose is not None:
            lx, ly, lyaw = self._last_pose
            self._dist += math.hypot(pos.x - lx, pos.y - ly)
            # SIGNED net rotation: obstacle swerves left and right cancel out.
            self._yaw_accum += wrap(yaw - lyaw)
        self._last_pose = (pos.x, pos.y, yaw)

        if self.state == "TURNING" and abs(self._yaw_accum) >= self.turn_done:
            self._finish(f"turned {math.degrees(self._yaw_accum):+.0f}deg")
        elif self.state == "PASSING" and self._dist >= self.straight_done_m:
            self._finish(f"{self._dist:.1f} m travelled")

    def _on_ops(self, msg):
        self._ops_t = self._now()
        left, right = self._side_openings(list(msg.data))
        if self.state == "COMMITTED":
            mine = left if self.decision == "turn_left" else right
            self._gate_n = self._gate_n + 1 if mine else 0
            if self._gate_n >= self.gate_scans:
                self.state = "TURNING"
                self._yaw_accum = 0.0       # count rotation from the junction, not the swerves
                self.get_logger().info(f"GATE: opening confirmed after {self._dist:.1f} m")
                self._set_prompt(self.decision)
        elif self.state == "PASSING":
            if left or right:
                self._side_seen, self._side_gone_n = True, 0
            elif self._side_seen:
                self._side_gone_n += 1
                if self._side_gone_n >= self.passed_scans:
                    self._finish("junction passed")

    def _tick(self):
        now = self._now()
        if now - self._last_pub >= self.republish_s:
            self.pub_prompt.publish(String(data=self.prompt))
            self._last_pub = now
        if self.gate and self.state == "COMMITTED" and (
                self._ops_t is None or now - self._ops_t > 1.0):
            self.get_logger().warn("no /lidar/openings_array -- gate cannot open "
                                   "(is lidar_openings_node running?)",
                                   throttle_duration_sec=3.0)
        s = (f"state={self.state} decision={self.decision} prompt={self.prompt} "
             f"turned={math.degrees(self._yaw_accum):.0f}deg dist={self._dist:.1f}m "
             f"done={self.done}")
        self.pub_state.publish(String(data=s))
        self.pub_status.publish(String(data=s))


def main():
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
