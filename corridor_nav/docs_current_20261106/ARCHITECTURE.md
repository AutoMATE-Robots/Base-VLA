# Architecture

## Hardware and frames

| Item | Detail |
|---|---|
| Base | AgileX **Hunter 2.0**. **Ackermann** (car-like) steering, so it **cannot rotate in place**. Wheelbase 0.650 m, track 0.605 m, footprint ~0.98 × 0.75 m. |
| Steering limit | Central (bicycle) angle 0.461 rad (26.4°), inner wheel ~0.574 rad (~33°). Tightest turn ≈ 1.3 m radius at `base_link`. |
| Compute | Jetson Orin (sm_87), ROS 2 Humble, power mode **MAXN** |
| Camera | 1920×1280, captured as UYVY. `v4l2_camera` publishes **rgb8** on `/image_raw`. Run at **10 fps** through `~/camfps/camera_10fps.sh` (see below). |
| LiDAR | RoboSense via `rslidar_sdk`, run standalone. `/rslidar_points` at 10 Hz, about 236k points per scan. |
| Robot frame | `base_link`: +x forward, **+y left**, +ω = turn left. Assumed at the **centre** of the body. |
| Rear axle | ~0.33 m behind `base_link`. This is the point that follows the path on an Ackermann base. |
| LiDAR mount | `base_link → rslidar` = translation (0.247, 0, 0.420), no rotation |
| Floor | z_robot ≈ **−0.66 m**. `base_link` is *not* on the floor (see LIDAR_PERCEPTION.md). |

## The two nodes we wrote

```
 ┌───────────────────────┐         ┌──────────────────────────────────────────────────┐
 │ rslidar_sdk           │         │ lidar_openings_node.py   (perception only)       │
 │ /rslidar_points 10 Hz ├────────►│  height band → 360 one-degree bins (min range)   │
 └───────────────────────┘         │  → clearance, centring, openings, scene type     │
                                   └───────┬──────────────────────────────────────────┘
                                           │ /lidar/bins          Float32MultiArray 360 min ranges ◄── planner
                                           │ /lidar/clearance     Float32  robot-wide lane ahead    ◄── freshness + fallback
                                           │ /lidar/min_dist      Float32  emergency lane ahead
                                           │ /lidar/center_error  Float32  + = robot LEFT of centre (fallback only)
                                           │ /lidar/junction_type String   corridor / T_junction / ...
                                           │ /lidar/openings_array Float32MultiArray [bearing,width,depth]*
                                           ▼
 ┌───────────────────────┐         ┌──────────────────────────────────────────────────┐
 │ v4l2_camera (10 fps)  │         │ v8_drive_node.py                                 │
 │ /image_raw            ├────────►│  inference thread (~2 Hz):                       │
 └───────────────────────┘         │     image + prompt → VLA → 8 waypoints           │
 ┌───────────────────────┐         │  control thread (20 Hz):                         │
 │ /vla/prompt (String)  ├────────►│     pure pursuit → balance planner → safety gate │
 └───────────────────────┘         │     → curvature → wheel angle (slewed) → /cmd_vel│
                                   └───────┬──────────────────────────────────────────┘
                                           │ /cmd_vel   (only with --live)
                                           │ /vla/path  (predicted waypoints, for rviz)
                                           │ /vla/status
                                           ▼
                                   hunter_base (CAN)
```

Other topics the LiDAR node publishes for debugging: `/lidar/d_left`, `/lidar/d_right`,
`/lidar/openings` (rviz arrows and labels) and `/lidar/openings_text`.

## What `/cmd_vel` means on this robot

This is the most important fact added since base 1.

| Field | What the drive node sends | What `hunter_base` does with it |
|---|---|---|
| `linear.x` | speed, m/s | speed |
| `angular.z` | **inner-wheel steering angle, rad** (`steer_mode:=angle`, default) | used **directly** as the steering angle ("Use directly" in `hunter_messenger.hpp`) |

So `angular.z` is **not** a turn rate on this base build. Base 3 and earlier sent
ω = v·κ there, so the wheel angle shrank as the robot slowed, and it steered least
exactly when it was slowing beside an obstacle. The drive node now converts the chosen
curvature κ into the inner-wheel angle with the same formula `hunter_base` uses (see
DRIVE_CONTROL.md §8). `-p steer_mode:=legacy` restores the old ω output.

## Why "mapless"

Nothing in this stack has a world frame. There is no map, no AMCL/NDT localization, no TF
lookup and no `/odom` subscription. Every decision is made in the robot's **current**
frame, from the **current** camera frame and LiDAR scan:

- **Where to go** comes from the VLA and the human-chosen prompt.
- **How to get through** comes from the LiDAR, this scan only (the balance planner).
- **Route choice** (which way to turn at a junction) is the operator's job, through `/vla/prompt`.

This keeps the system simple and robust to map drift. The cost is that the robot can't
plan around things it can't currently see. In particular, it can't see junction arms
hidden behind corridor corners (see HISTORY_AND_LESSONS.md, open issues).

It is also why the localization stack (`full_system.launch.py`: rviz, localizer, octomap,
planner and its own rslidar) **must not** run alongside it: this stack needs only the raw
`/rslidar_points`, and the extra load starved the LiDAR pipeline (see "Power" below).

## Threading in the drive node

The VLA forward pass takes hundreds of milliseconds. If it shared a thread with control,
the 20 Hz loop would freeze during every inference and the commands would stutter. So:

| Callback group | Runs |
|---|---|
| `cg_infer` | image subscriber (keeps **only the newest** frame), prompt subscriber, inference timer |
| `cg_ctrl` | all `/lidar/*` subscribers, 20 Hz control timer (planner + safety + steering) |

These run in a `MultiThreadedExecutor(num_threads=3)`, and a single lock guards the
shared state. The LiDAR safety data keeps flowing at full rate while the model is busy.

The balance planner costs ~25 ms per run, half the control period, on a CPU the VLA is
also using. It therefore re-plans only when a **new scan** arrives (10 Hz), the committed
side changes, or the policy's curvature moves by more than 0.03 1/m. Otherwise the
previous plan is reused.

## Rates and freshness

| Signal | Rate | What happens if it goes stale |
|---|---|---|
| Camera | 10 fps in (shim), only the latest frame is kept | n/a |
| VLA inference | `infer_hz` = 2.0 (one call takes ~410 ms standalone, ~500–600 ms in the node) | waypoints older than `stale_after` (1.0 s) → **hard zero** speed |
| LiDAR scan | 10 Hz | clearance older than `safety_stale` (0.7 s) → **hold still** (fail closed) |
| `/lidar/bins` | 10 Hz | older than 0.7 s → planner off, straight-lane fallback |
| Control / `/cmd_vel` | 20 Hz | n/a |

## Power (why the camera runs at 10 fps)

Measured 2026-11-04 with the VLA running (`oc3_event_cnt` = Jetson over-current events):

| Setup | Over-current events |
|---|---|
| Localization stack (`full_system.launch.py`) running | **111 in 15 s** |
| Camera at 30 fps, no localization stack | 23 in 15 s |
| Camera at 10 fps + standalone LiDAR + openings node + drive node | **11 in 60 s** |

Each event cuts CPU/GPU clocks. That slowed inference and, worse, delayed
`/lidar/clearance` past 0.7 s, which tripped "no clearance → holding still" 35–66 times
per run: the stop-start jerk. The lean setup had 0 such holds.

`v4l2_camera` 0.6.2 ignores `framerate`, `image_width` and `image_height`, and the
camera only offers 30/20/10 fps. `~/camfps/camera_10fps.sh` uses an `LD_PRELOAD` shim
(`v4l2_fps_shim.so`) that sets 10 fps right after the format is set. The images are
byte-identical; camera CPU dropped from ~102 % to ~27–36 %.

## Safety by construction

- **Dry run is the default.** Without `--live`, the `/cmd_vel` publisher **is never
  created**. Even a bug can't publish a Twist.
- `/cmd_vel` is only written in one function, `_send()`. That function also replaces any
  NaN/inf with zero.
- On Ctrl-C, a final zero Twist (speed 0, wheels straight) is sent so the base is left stopped.
- All topic names are **absolute** (`/cmd_vel`, not `cmd_vel`), so the exported
  `ROS_NAMESPACE=robot1` can't move them where nothing is listening.
- Faults (stale waypoints, no inference yet) zero the **speed** immediately. In angle mode
  the wheels are held where they are rather than snapped straight, so the robot can pull
  away already turned.
