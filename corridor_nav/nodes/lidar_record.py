#!/usr/bin/env python3
"""Record /rslidar_points + /odom + the node's own analysis to JSON for the HTML viewer.

    source /opt/ros/humble/setup.bash && source ~/v8env/bin/activate && unset ROS_NAMESPACE
    python3 -u ~/lidar_record.py 60 ~/lidar_session.json     # seconds, output

Reuses LidarOpenings' real methods, so what the viewer shows is what the node decides.
"""
import json
import math
import sys

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import PointCloud2

sys.path.insert(0, "/home/hunter")
from lidar_openings_node import LidarOpenings, N_BINS, TX, TZ   # noqa: E402

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/hunter/lidar_session.json"


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    rclpy.init()
    pol = LidarOpenings()                      # the real analysis, same parameters
    rec = Node("lidar_record")
    frames, odom = [], {"x": 0.0, "y": 0.0, "yaw": 0.0, "seen": False}

    def on_odom(m):
        odom.update(x=m.pose.pose.position.x, y=m.pose.pose.position.y,
                    yaw=yaw_of(m.pose.pose.orientation), seen=True)

    def on_cloud(msg):
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            npts = buf.size // msg.point_step
            pts = buf[: npts * msg.point_step].reshape(npts, msg.point_step)[:, :12].copy().view(np.float32)
            x = pts[:, 0] + TX
            y = pts[:, 1]
            z = pts[:, 2] + TZ
            g = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z >= pol.z_lo) & (z <= pol.z_hi)
            x, y = x[g], y[g]
            r = np.hypot(x, y)
            ok = (r >= pol.min_range) & (r <= pol.max_range)
            x, y, r = x[ok], y[ok], r[ok]
            mins = pol._bin_min_ranges(r, np.degrees(np.arctan2(y, x)))
            has = np.isfinite(mins)
            om = (mins > pol.open_range) & has
            ops = pol._openings_from_mins(mins, om)
            cov = set()
            for o in ops:
                cov |= o["labels"]
            dl = pol._sector_median(mins, 60.0, 120.0)
            dr = pol._sector_median(mins, -120.0, -60.0)
            frames.append({
                "t": round(rec.get_clock().now().nanoseconds * 1e-9, 3),
                # -1 marks "no return" so the viewer can draw unknown differently from free
                "r": [round(float(v), 2) if np.isfinite(v) else -1.0 for v in mins],
                "scene": pol._scene(cov),
                "e": None if (dl is None or dr is None) else round((dr - dl) / 2.0, 3),
                "dl": None if dl is None else round(dl, 2),
                "dr": None if dr is None else round(dr, 2),
                "ops": [{"b": round(o["bearing"], 1), "w": round(o["width"], 1),
                         "d": round(o["depth"], 2), "l": sorted(o["labels"])} for o in ops],
                "odom": {"x": round(odom["x"], 3), "y": round(odom["y"], 3),
                         "yaw": round(odom["yaw"], 4)} if odom["seen"] else None,
            })
        except Exception as e:
            rec.get_logger().warn(f"skip scan: {type(e).__name__}: {e}")

    rec.create_subscription(PointCloud2, "/rslidar_points", on_cloud,
                            QoSPresetProfiles.SENSOR_DATA.value)
    # /odom only publishes when the base driver runs; /pcl_odom comes from the lidar
    # localisation and is up whenever the planner stack is. Take whichever arrives.
    rec.create_subscription(Odometry, "/odom", on_odom, QoSPresetProfiles.SENSOR_DATA.value)
    rec.create_subscription(Odometry, "/pcl_odom", on_odom, QoSPresetProfiles.SENSOR_DATA.value)

    t0 = rec.get_clock().now().nanoseconds * 1e-9
    print(f"recording {DUR:.0f}s -> {OUT}  (push the robot around now)", flush=True)
    while rclpy.ok() and (rec.get_clock().now().nanoseconds * 1e-9 - t0) < DUR:
        rclpy.spin_once(rec, timeout_sec=0.05)
        if len(frames) % 50 == 0 and frames:
            print(f"  {len(frames)} frames, last scene={frames[-1]['scene']}", flush=True)

    meta = {"open_range": pol.open_range, "z_lo": pol.z_lo, "z_hi": pol.z_hi,
            "floor_z": pol.floor_z, "half_width": pol.half_width,
            "robot_width": 0.75, "n_bins": N_BINS,
            "has_odom": any(f["odom"] for f in frames)}
    with open(OUT, "w") as f:
        json.dump({"meta": meta, "frames": frames}, f, separators=(",", ":"))
    print(f"wrote {len(frames)} frames -> {OUT}  (odom: {meta['has_odom']})", flush=True)
    rec.destroy_node()
    pol.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
