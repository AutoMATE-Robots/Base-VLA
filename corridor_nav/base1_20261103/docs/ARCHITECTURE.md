# Architecture

## Hardware and frames

| Item | Detail |
|---|---|
| Base | AgileX **Hunter**. It uses **Ackermann** (car-like) steering, so it **cannot rotate in place**. Min turning radius about 1.6 m. |
| Compute | Jetson Orin (sm_87), ROS 2 Humble |
| Camera | 1920×1280, captured as UYVY. `v4l2_camera` publishes **rgb8** on `/image_raw` at about 25–28 Hz. |
| LiDAR | RoboSense via `rslidar_sdk`. `/rslidar_points` at 10 Hz, about 236k points per scan. |
| Robot frame | `base_link`: +x forward, **+y left**, +ω = turn left |
| LiDAR mount | `base_link → rslidar` = translation (0.247, 0, 0.420), no rotation |
| Floor | z_robot ≈ **−0.66 m**. `base_link` is *not* on the floor (see LIDAR_PERCEPTION.md). |

## The two nodes we wrote

```
 ┌───────────────────────┐         ┌──────────────────────────────────────────────────┐
 │ rslidar_sdk           │         │ lidar_openings_node.py   (perception only)       │
 │ /rslidar_points 10 Hz ├────────►│  height band → 360 one-degree bins (min range)   │
 └───────────────────────┘         │  → clearance, centring, openings, scene type     │
                                   └───────┬──────────────────────────────────────────┘
                                           │ /lidar/clearance     Float32  robot-wide lane ahead
                                           │ /lidar/min_dist      Float32  emergency lane ahead
                                           │ /lidar/center_error  Float32  + = robot LEFT of centre
                                           │ /lidar/junction_type String   corridor / T_junction / ...
                                           │ /lidar/openings_array Float32MultiArray [bearing,width,depth]*
                                           │ /lidar/bins          Float32MultiArray 360 min ranges
                                           ▼
 ┌───────────────────────┐         ┌──────────────────────────────────────────────────┐
 │ v4l2_camera           │         │ v8_drive_node.py                                 │
 │ /image_raw            ├────────►│  inference thread (~2 Hz):                       │
 └───────────────────────┘         │     image + prompt → VLA → 8 waypoints           │
 ┌───────────────────────┐         │  control thread (20 Hz):                         │
 │ /vla/prompt (String)  ├────────►│     pure pursuit → centring → safety/avoid/creep │
 └───────────────────────┘         │     → slew limits → /cmd_vel                     │
                                   └───────┬──────────────────────────────────────────┘
                                           │ /cmd_vel   (only with --live)
                                           │ /vla/path  (predicted waypoints, for rviz)
                                           │ /vla/status
                                           ▼
                                   hunter_base (CAN)
```

Other topics the LiDAR node publishes for debugging: `/lidar/d_left`, `/lidar/d_right`,
`/lidar/openings` (rviz arrows and labels) and `/lidar/openings_text`.

## Why "mapless"

Nothing in this stack has a world frame. There is no map, no AMCL/NDT localization and
no `/odom` subscription. Every decision is made in the robot's **current** frame, from
the **current** camera frame and LiDAR scan:

- **Where to go** comes from the VLA and the human-chosen prompt.
- **Where it is safe to go** comes from the LiDAR, this scan only.
- **Route choice** (which way to turn at a junction) is the operator's job, through `/vla/prompt`.

This keeps the system simple and robust to map drift. The cost is that the robot can't
plan around things it can't currently see. In particular, it can't see junction arms
hidden behind corridor corners (see HISTORY_AND_LESSONS.md, open issues).

## Threading in the drive node

The VLA forward pass takes hundreds of milliseconds. If it shared a thread with control,
the 20 Hz loop would freeze during every inference and the commands would stutter. So:

| Callback group | Runs |
|---|---|
| `cg_infer` | image subscriber (keeps **only the newest** frame), prompt subscriber, inference timer |
| `cg_ctrl` | all `/lidar/*` subscribers, 20 Hz control timer |

These run in a `MultiThreadedExecutor(num_threads=3)`, and a single lock guards the
shared state. The LiDAR safety data keeps flowing at full rate while the model is busy.

## Rates and freshness

| Signal | Rate | What happens if it goes stale |
|---|---|---|
| Camera | ~25–28 Hz in, only the latest frame is kept | n/a |
| VLA inference | `infer_hz` = 2.0 | waypoints older than `stale_after` (1.0 s) → **hard zero** |
| LiDAR scan | 10 Hz | clearance older than `safety_stale` (0.7 s) → **hold still** (fail closed) |
| Control / `/cmd_vel` | 20 Hz | n/a |

## Safety by construction

- **Dry run is the default.** Without `--live`, the `/cmd_vel` publisher **is never
  created**. Even a bug can't publish a Twist.
- `/cmd_vel` is only written in one function, `_send()`. That function also replaces any
  NaN/inf with zero.
- On Ctrl-C, a final zero Twist is sent so the base is left stopped.
- All topic names are **absolute** (`/cmd_vel`, not `cmd_vel`), so the exported
  `ROS_NAMESPACE=robot1` can't move them where nothing is listening.
