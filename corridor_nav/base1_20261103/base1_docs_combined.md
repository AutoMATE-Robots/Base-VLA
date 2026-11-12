# Base 1: Mapless LiDAR + VLA Corridor Driving

**Base 1** is the snapshot of the driving code taken on 2026-11-03. It comes
*before* the side-stop and gentle-arc fixes, which live in `base2_20261103/`.
The code is the two files next to this folder:

```
~/weights/corridor_nav/base1_20261103/
├── v8_drive_node.py        801 lines  camera → VLA → waypoints → steering + safety → /cmd_vel
├── lidar_openings_node.py  599 lines  LiDAR point cloud → clearance, centring, openings, bins
└── docs/                   ← you are here
```

To revert to base 1, copy both `.py` files back to `~`.

---

## The idea in one paragraph

The robot has **no map, no localization and no odometry**. It drives from what it sees
right now. A camera frame and a one-word instruction (`straight`, `turn_left`,
`turn_right`) go into a fine-tuned **vision-language-action model (OpenVLA-OFT, "v8")**.
The model returns **8 waypoints** for the next ~0.8 s. A classic **pure-pursuit**
controller turns those waypoints into a speed and a turn rate. A separate **LiDAR node**
watches the geometry around the robot. The drive node uses it to:

1. keep the robot in the **middle of the corridor**,
2. **slow down and stop** for obstacles,
3. **steer around** obstacles on the clearest arc, and
4. **creep out** of a stop instead of freezing.

The VLA decides *where to go*. The LiDAR decides *whether it is safe*, and always
has the final say.

```
        camera ──► VLA (v8) ──► 8 waypoints ──► pure pursuit ──┐
        prompt ──┘  ~2 Hz                                      │  (v, ω)
                                                               ▼
   LiDAR ──► lidar_openings_node ──► centring ──► safety / avoidance / creep ──► ramps ──► /cmd_vel
             10 Hz                    (PD)         (LiDAR always wins)          20 Hz
```

---

## Documents in this folder

| File | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Nodes, topics, threads, timing. The big picture. |
| [LIDAR_PERCEPTION.md](LIDAR_PERCEPTION.md) | How the point cloud becomes clearance, centring error, openings and junction type. |
| [DRIVE_CONTROL.md](DRIVE_CONTROL.md) | VLA inference, pure pursuit, centring, safety gate, arc avoidance, creep recovery, ramps. |
| [PARAMETERS.md](PARAMETERS.md) | Every parameter and its base 1 default, with notes on tuning. |
| [HISTORY_AND_LESSONS.md](HISTORY_AND_LESSONS.md) | How we got from the Oct 22 baseline to base 1, the bugs we hit, and what's still open. |

---

## How to launch base 1

Use a separate terminal for each step. **Unset `ROS_NAMESPACE` everywhere.** `.bashrc`
exports `robot1`, which hides every topic.

```bash
# 1. Base (CAN + hunter_base; listens on un-namespaced /cmd_vel)
bash ~/Robot_teleop_ws/robot.bash

# 2. LiDAR driver (check `cat /sys/class/net/eno1/carrier` prints 1 first)
source /opt/ros/humble/setup.bash && source ~/Robot_autonomy_ws/install/setup.bash
unset ROS_NAMESPACE && ros2 launch rslidar_sdk start.py

# 3. Camera (use video0; sometimes it is video1. The real one lists UYVY)
ros2 run v4l2_camera v4l2_camera_node --ros-args -p video_device:=/dev/video0 \
  -p image_width:=1920 -p image_height:=1280 -p framerate:=20.0 -p pixel_format:=uyvy

# 4. LiDAR perception. REQUIRED: without it the drive node holds still on purpose
source ~/v8env/bin/activate && unset ROS_NAMESPACE && cd ~ && python3 -u lidar_openings_node.py

# 5. Drive (dry run unless --live)
source ~/v8_env.sh && unset ROS_NAMESPACE && cd ~ && \
  python3 -u v8_drive_node.py --live --ros-args -p infer_hz:=2.0 -p creep_speed:=0.25

# Change the instruction at any time
ros2 topic pub --once /vla/prompt std_msgs/msg/String "{data: 'turn_left'}"
```

Why `creep_speed:=0.25`: the Hunter's minimum turning radius is about 1.6 m. At the
default 0.10 m/s, a 45° escape takes about 12.6 s. That is longer than the 8 s recovery
timeout.

### Three things to remember

- **"VLA is running but the robot won't move"** almost always means the LiDAR is silent.
  The safety gate *fails closed*. Check the LiDAR and `lidar_openings_node` first. Never
  "fix" this with `use_safety:=false`.
- **The `stop` prompt was never trained.** The node honours it by commanding zero, but
  don't rely on it as a safety stop. Use Ctrl-C or the E-stop.
- `~/v8_env.sh` sets `LD_PRELOAD` to Tegra cuBLAS builds. Without it, the installed
  (sbsa) cuBLAS has no sm_87 kernels and CUDA fails on the Orin.


---

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


---

# LiDAR Perception (`lidar_openings_node.py`)

This node is **perception only**. It never publishes `/cmd_vel` or `/vla/prompt`. It
turns every 10 Hz scan into a few small, easy-to-use signals.

```
/rslidar_points
   │  1. move points into base_link            x += 0.247, z += 0.420
   │  2. keep a height band                    0.20–0.80 m above the FLOOR
   │  3. keep a range window                   0.15 m … 30 m
   ├──► 4. safety lanes (raw points)  ──────►  /lidar/clearance, /lidar/min_dist
   │  5. 360 bins of 1°, min range per bin ─►  /lidar/bins
   ├──► 6. corridor centring  ─────────────►  /lidar/center_error, d_left, d_right
   └──► 7. openings → scene  ──────────────►  /lidar/openings_array, /lidar/junction_type
```

---

## 1–2. The height band, and the floor trap

**`base_link` rides about 0.66 m above the floor.** We measured this from real scans,
with three independent checks:

1. No point in the cloud lies below z_robot = −0.662. A flat cluster sits right there
   (std 0.016 m). You can't see through a floor, so that is the floor.
2. The lowest beam points down at −24.8°. With the sensor 1.08 m up, it should first hit
   the floor at 1.08 / tan 24.8° = 2.34 m. That is exactly where the cluster starts.
3. Ceiling (+1.55) minus floor (−0.66) gives a 2.21 m room height, which is normal.

So the band is set as **height above floor**:
`z_robot = floor_z_robot + height_AGL`. The default 0.20–0.80 m AGL becomes
z_robot −0.46 … +0.14. A band of z_robot 0.20–0.80 would have looked at chest-to-head
height and missed every low obstacle.

> Trap: a raw z-histogram has a big hump at about +0.42. That hump is **walls**, not
> floor: most beams are near-horizontal and land on vertical surfaces.

## 4. Safety lanes (computed on raw points, not bins)

Two rectangles straight ahead of the robot. Each reports the **along-track** distance
(x) to the nearest point inside it:

| Topic | Half-width | Used for |
|---|---|---|
| `/lidar/clearance` (`d_front`) | 0.375 + 0.30 = **0.675 m** | slow-down / stop |
| `/lidar/min_dist` (`d_near`) | 0.375 + 0.05 = **0.425 m** | emergency cut |

If nothing is in a lane, it reports `max_range` (30 m). Both topics are published on
**every** scan, so silence is a fault the drive node can detect.

The emergency lane is deliberately **narrower** than the slow lane. Earlier versions got
this wrong in both directions:

- A whole-half-plane *radial* distance tripped on a wall merely **beside** the robot.
- A *wider* lane caught a wall running alongside at x ≈ 0 and would have stopped the
  robot forever.

## 5. Bins

Each point gets a bearing `atan2(y, x)`, which drops it into one of 360 one-degree bins.
Each bin keeps its **minimum** range. A bin with no return is `inf` ("unknown"). The
whole 360-value array goes out on `/lidar/bins`. The drive node rebuilds obstacle points
from it for arc avoidance.

## 6. Corridor centring

```
d_left  = median of bins at bearings  +60° … +120°
d_right = median of bins at bearings −120° … −60°
e       = (d_right − d_left) / 2           e > 0  →  robot is LEFT of centre
```

The median makes it robust to a doorframe or a chair leg. This is only meaningful
**between two parallel walls**. At a junction, one side sees down a branch and `e` can
reach ±1.5 m. That's why the drive node only applies centring while the scene is
`corridor`.

## 7. Openings and scene type

An **opening** is a direction the robot could drive into.

**Step A: candidate runs.** Bins with range > `open_range` (2.5 m) are "open".
Contiguous open bins (wrapping around ±180°) form runs. Runs of 1 bin are dropped.

**Step B: split by prominence.** This fixes the hardest problem. A single
absolute threshold can't tell "corridor continues" from "a wall 2.6 m ahead". At a real
T-junction, the left arm, the facing wall and the right arm fused into one 160° run,
which was classified `four_way`.

The fix is **threshold-free**. Inside a run, find the deepest interior dip. If it's much
lower than the peaks on both sides, cut the run there:

```
split if   dip < split_ratio × min(left_peak, right_peak)        split_ratio = 0.6
```

At the real T: peaks 10.1 m (right arm) and 8.3 m (left arm), dip 2.64 m (the facing
wall). 2.64 < 0.6 × 8.3, so the run splits into two arms. A straight corridor has one
peak and is left whole. Each split piece is then **trimmed** to the part that stays
above 0.6 × its own peak. Without the trim, the shoulder running into the wall dragged
the left arm's bearing from +52° to +24° and turned a T into `right_branch`.

**Step C: does the robot fit?** At depth `d`, the opening must span at least
`2·atan2(0.675, d)` degrees. Narrow slits are discarded.

**Step D: label each opening by its centre bearing.**

| Bearing | Label |
|---|---|
| \|b\| ≤ 25° | FORWARD |
| +25° … +155° | LEFT |
| −25° … −155° | RIGHT |
| otherwise | BEHIND |

**One run means one passage, which gets one label.** Letting a run claim every sector it
touched created phantom branches. The exception is a run wider than 200°, which is open
space, not a passage. For that case each sector is claimed only if its own share is
robot-wide.

**Step E: scene table.**

| Labels present | Scene |
|---|---|
| FORWARD | `corridor` |
| FORWARD + LEFT | `left_branch` |
| FORWARD + RIGHT | `right_branch` |
| LEFT + RIGHT (no FORWARD) | `T_junction` |
| all three | `four_way` |
| nothing (or only LEFT, or only RIGHT) | `dead_end` |

A scene is published on `/lidar/junction_type` only after it holds for **5 consecutive
scans** (0.5 s). This stops it from flickering.

Openings also go out numerically on `/lidar/openings_array` as flat
`[bearing°, width°, depth_m, …]` triples. The creep-recovery fallback in the drive node
uses them.

## Console line

```
corridor     | e=+0.12m dL=1.4 dR=1.6 | clr=4.20 | openings: fwd(+2deg, w=31deg, d=6.3m)
```

This shows the published scene, centring error, side distances, forward clearance and
the openings.

## Known quirks in base 1

- The module docstring still says discontinuity-based splitting is "not implemented".
  It is out of date: prominence splitting **is** implemented and on by default.
- A comment above `d_front`/`d_near` still describes `d_near` as the "forward half-plane".
  The code uses the narrow 0.425 m lane described above.
- LEFT-only or RIGHT-only falls through to `dead_end`. That's probably wrong at a side door.
- `empty_is_open` is false. The driver drops no-return points, so a truly open doorway
  can look "unknown" rather than open.


---

# Drive and Control (`v8_drive_node.py`)

The drive node has two loops.

- **Inference loop (~2 Hz):** newest camera frame + prompt → VLA → 8 waypoints.
- **Control loop (20 Hz):** waypoints → (v, ω), then centring, then safety, avoidance and
  creep, then ramps → `/cmd_vel`.

---

## 1. The VLA (v8)

| | |
|---|---|
| Model | OpenVLA-7B + OFT action head, fine-tuned on our "signway" data. Merged LoRA r32, bf16, step 9000 |
| Weights | `~/weights/v8_vla` (sha256 in `MANIFEST.txt`) |
| Wrapper | `~/v8_policy.py` → `load_policy(...).predict(pil_image, prompt)` |
| Prompts | `straight`, `turn_left`, `turn_right` (`stop` exists but was **never trained**) |
| Output | `(8, 2)` array of **cumulative** waypoints `(x_forward, y_left)` in metres, in the robot frame of the image. wp0 ≈ 0.1 m ahead, wp7 ≈ end of a ~0.8 s horizon |

The text prompt is built exactly as in training:
`"In: What action should the robot take to turn left?\nOut:"`.

The preprocessing reproduces training **without TensorFlow** (TF won't install on the
Jetson): JPEG round-trip (q95) → Lanczos resize to 224 → 90 % centre crop → resize back
to 224. Un-normalisation (BOUNDS_Q99) happens inside `predict_action`, so the output is
already in metres.

The image decoder is numpy only (no cv_bridge). It honours `msg.step`, so padded rows
can't shear the image.

## 2. Pure pursuit: waypoints → (v, ω)

```
pick the first waypoint at least lookahead_dist (0.6 m) from the robot
kappa = 2·y / (x² + y²)          curvature of the arc from the robot to that point
v     = speed (0.4 m/s, clipped to max_speed 0.6)
ω     = v · kappa                (clipped to ±max_omega 1.0 rad/s)
```

Guards: if the target is at or behind the axle (x ≤ 0.05) or the output is NaN, the
result is **(0, 0)**. An Ackermann base can't spin to recover, so stopping is safer
than a wild turn.

The model's implied speed (~0.9 m/s) is deliberately ignored. We use only the **shape**
of the path, at a fixed speed.

## 3. Corridor centring (PD)

Added to ω, only when the scene is `corridor`:

```
e_db = e with a ±0.05 m deadband removed
ω   += clip( −(0.45 · e_db  +  0.35 · de/dt),  ±0.35 rad/s )
```

- The **minus sign**: e > 0 means the robot is left of centre, so it must turn right
  (negative ω).
- The **D term** matters: P alone made the car-like base overshoot and weave from wall to wall.
- `de/dt` is computed in the 10 Hz LiDAR callback (not the 20 Hz loop, which would see
  each value twice) and smoothed with `0.7·old + 0.3·new`.
- It is switched off at junctions, because there `e` measures a branch mouth, not a wall.

## 4. The safety gate: LiDAR always wins

`_apply_safety` runs every control tick and can override the policy completely. It
checks these in order and returns at the first match:

```
 1. No /lidar/clearance yet, or older than 0.7 s ─────────► HOLD STILL (fail closed)
 2. d_near < emergency_distance (0.40 m) ─────────────────► HOLD STILL
 3. /lidar/bins fresh and avoidance on? ──yes──► ARC MODE   (sections 5–6)
                                          no ──► STRAIGHT-LANE MODE (section 7)
```

## 5. Arc avoidance (`_plan_avoid`)

The robot can only drive along circular arcs, so we test arcs directly.

1. Rebuild obstacle points from `/lidar/bins` (one point per bin centre).
2. Make **26 candidate arcs**: 25 curvatures evenly spread over ±0.6 1/m (about the
   1.6 m minimum turning radius), plus the **policy's own arc** `k_pol = ω / v`.
3. For each arc, compute its **free length**: how far the robot can drive along it
   before a band of ±0.625 m (half robot width 0.375 + 0.25 margin) touches an obstacle.
   Only the half-circle ahead counts, and anything ≥ 5 m counts as fully clear.

```
              obstacle
                 ■
     ╲    │    ╱          each arc: free length until the swept band hits something
      ╲   │   ╱           policy arc blocked at 1.4 m → look for a better one
       ╲  │  ╱
        ╲ │ ╱
         ▲  robot
```

4. **Decision:**
   - Policy arc free ≥ 3.0 m → **follow the policy** (no override).
   - Otherwise, score every arc as `free − 1.0 × |kappa − k_pol|`. This prefers clear arcs
     that are close to what the policy wanted.
   - Pick the best. It must beat the policy arc by at least **0.3 m**. If it doesn't, there's
     no better way round, and the result is tagged `blocked`.
   - If it does, steer onto it: **AVOID L** or **AVOID R**.
5. **Commitment:** once the robot picks a side, switching to the other side costs a 1.0 m
   score penalty. Without this, the robot dithered left/right in front of an obstacle
   and hit it. It releases only when the policy arc is free past 3.0 + 0.25 m.

## 6. Arc-mode speed gate (`_apply_arc_safety`)

Using the free length of the **chosen** arc:

| Free length on chosen arc | Action |
|---|---|
| ≥ 3.0 m (`slow_distance`) | full speed; ω = v·kappa if avoiding, else policy ω |
| 1.0 … 3.0 m | speed scaled linearly from 100 % down to 0 % at 1.0 m |
| < 1.0 m (`stop_distance`), **no** clearer arc exists | **hold still** (creeping would only move closer) |
| < 1.0 m, an AVOID arc exists | **creep along that arc** (section 8) |

## 7. Straight-lane fallback (no `/lidar/bins`)

Same idea, using `/lidar/clearance` (the 0.675 m rectangle) instead of arcs:

- `clear` < 1.0 m → creep toward the widest opening from `/lidar/openings_array`.
- 1.0 … 3.0 m → linear slow-down.
- ≥ 3.0 m → policy unchanged.

## 8. Creep recovery: a stop must be escapable

On an Ackermann robot, a hard stop is a trap. ω with v = 0 does nothing, and a
stationary robot never changes its own clearance. Before creep recovery, **every stop
was terminal** until someone pushed the robot.

So when blocked but the 0.40 m emergency lane is still clear, the robot **rolls slowly
(`creep_speed`) while steering out**:

- **Arc mode:** follow the best avoidance arc.
- **Fallback mode:** aim pure pursuit at the widest opening. That direction is
  **latched**, because at a real T-junction the two arms were 29° and 28° wide and
  re-picking each tick flipped left/right/left. It re-targets only if the latched
  opening moves more than 50°.
- **Hysteresis:** enter below 1.0 m, exit only above 1.25 m. Measured `d_front` jittered
  0.74 / 1.16 / 0.74 m between scans, which made recovery toggle every 0.1 s.
- **Gives up** (hard stop) after 8 s, if the emergency lane is breached, or if the best
  arc is free less than 0.5 m.

Run with `creep_speed:=0.25`. At 0.10 m/s the robot can't turn far enough within 8 s.

## 9. Slew-rate limits (ramps)

```
v ramps up at most 0.5 m/s², down at most 1.5 m/s²   (braking is 3× faster than accelerating)
ω changes at most 2.0 rad/s²
```

**Faults bypass the ramps.** Stale waypoints, no inference yet, or NaN all go
straight to (0, 0). Normal slow-downs and a `stop` prompt go through the ramps, so
they are smooth.

## 10. Reading the console

In dry run, every second:

```
[dry_run] v=+0.212 w=-0.140 | AVOID R 1.84m | clr=1.92 e=+0.10
```

The middle field (`_gate`) tells you **why** the command looks the way it does:

| Gate | Meaning |
|---|---|
| `policy` | following the VLA unchanged |
| `slow 2.10m` | slowing for something ahead |
| `AVOID L 1.84m` / `AVOID R …` | steering round an obstacle on the chosen arc |
| `blocked 1.40m` | policy path blocked, no clearly better arc (still slowing) |
| `blocked 0.80m no-way-round` | inside stop distance, nothing better: holding still |
| `CREEP arc k=+0.42 free 0.9m 2.3s` | creep recovery in arc mode |
| `CREEP 0.85m -> +62deg 1.1s` | creep recovery toward an opening (fallback mode) |
| `creep exhausted 8.0s` | recovery timed out: hard stop |
| `EMERG 0.35m` | something in the emergency lane: hard stop |
| `no-lidar` | safety data missing or stale: holding still |
| `stop` | `stop` prompt active |

Each inference also logs a line like
`[turn_left] 480.2ms wp0=(+0.110,+0.004) wp7=(+0.690,+0.180) -> v=+0.400 w=+0.290`.


---

# Parameters (base 1 defaults)

Override any of these with `--ros-args -p name:=value`.

## `v8_drive_node.py`

### Inputs and policy
| Parameter | Default | Meaning |
|---|---|---|
| `image_topic` | `/image_raw` | camera input (absolute name on purpose) |
| `cmd_topic` | `/cmd_vel` | output, only with `--live` |
| `prompt_topic` | `/vla/prompt` | String: `straight` / `turn_left` / `turn_right` / `stop` |
| `checkpoint` | `/home/hunter/weights/v8_vla` | VLA weights |
| `prompt` | `straight` | starting prompt |
| `infer_hz` | 2.0 | VLA inferences per second |
| `control_hz` | 20.0 | control loop rate |
| `jpeg_roundtrip` | true | matches training preprocessing (costs some latency) |

### Pure pursuit and limits
| Parameter | Default | Meaning |
|---|---|---|
| `speed` | 0.4 m/s | fixed cruise speed |
| `lookahead_dist` | 0.6 m | pursuit target distance along the predicted path |
| `max_speed` | 0.6 m/s | hard cap |
| `max_omega` | 1.0 rad/s | hard cap |
| `max_curvature` | 0 (off) | optional steering limit for pursuit (1/m) |
| `accel` / `decel` | 0.5 / 1.5 m/s² | speed ramps (braking faster) |
| `angular_accel` | 2.0 rad/s² | turn-rate ramp |
| `stale_after` | 1.0 s | waypoints older than this → hard zero |

### Safety
| Parameter | Default | Meaning |
|---|---|---|
| `use_safety` | true | **leave on**. Never use `false` as a fix for "won't move". |
| `stop_distance` | **1.00 m** | below this: stop or creep (was 0.70) |
| `slow_distance` | **3.00 m** | start slowing here (was 1.80) |
| `emergency_distance` | 0.40 m | anything in the narrow lane closer → hard zero |
| `safety_stale` | 0.7 s | no LiDAR for this long → hold still |

### Creep recovery
| Parameter | Default | Meaning |
|---|---|---|
| `creep_speed` | 0.10 m/s | **run with 0.25** (see note) |
| `creep_lookahead` | 1.0 m | pursuit distance toward an opening (fallback mode) |
| `recovery_timeout` | 8.0 s | give up after this |
| `recovery_exit_margin` | 0.25 m | exit only at stop_distance + this (hysteresis) |
| `recovery_retarget_deg` | 50° | latched opening must move this far to re-pick |

Note: with a ~1.6 m minimum turning radius, turn rate = v / 1.6. At 0.10 m/s that is
0.062 rad/s, so 45° takes 12.6 s, longer than the 8 s timeout.

### Arc avoidance
| Parameter | Default | Meaning |
|---|---|---|
| `use_avoidance` | true | needs `/lidar/bins` |
| `avoid_distance` | 3.0 m | override the policy once its arc is blocked closer than this |
| `avoid_max_curvature` | 0.6 1/m | widest candidate turn (~1.6 m radius) |
| `avoid_n_arcs` | 25 | candidate arcs across ±max curvature |
| `avoid_margin` | 0.25 m | added to half robot width → 0.625 m swept half-band |
| `avoid_horizon` | 5.0 m | arcs this clear count as fully clear |
| `avoid_dev_weight` | 1.0 | metres of clearance traded per 1/m of deviation from policy |
| `avoid_min_gain` | 0.3 m | new arc must be at least this much clearer |
| `avoid_switch_penalty` | 1.0 m | cost of flipping sides once committed |

### Corridor centring
| Parameter | Default | Meaning |
|---|---|---|
| `use_centering` | true | |
| `center_gain` | 0.45 | P: rad/s per metre of error |
| `center_damp` | 0.35 | D: rad/s per m/s of error rate |
| `center_deadband` | 0.05 m | ignore tiny errors |
| `center_max` | 0.35 rad/s | cap on the centring term |
| `center_only_corridor` | true | off at junctions |

## `lidar_openings_node.py`

| Parameter | Default | Meaning |
|---|---|---|
| `cloud_topic` | `/rslidar_points` | input |
| `floor_z_robot` | −0.66 | measured floor height in base_link. Change it if the robot is re-mounted. |
| `band_min_agl` / `band_max_agl` | 0.20 / 0.80 m | height band **above floor** |
| `open_range` | 2.5 m | a bin farther than this is "open" (works 2.0–3.5 thanks to splitting) |
| `max_range` / `min_range` | 30 / 0.15 m | reject absurd returns and self-hits |
| `robot_width` | 0.75 m | |
| `clearance_margin` | **0.30 m** | per side for `/lidar/clearance` (was 0.20) |
| `stable_scans` | 5 | scans a scene must hold before publishing |
| `min_run_bins` | 2 | ignore 1-bin specks |
| `min_sector_bins` | 5 | for the wide-run fallback |
| `split_runs` | true | prominence splitting on |
| `split_ratio` | 0.6 | dip must be < 0.6 × smaller flanking peak |
| `split_min_bins` | 6 | each side of a split must be this wide |
| `wide_run_deg` | 200° | above this a run is open space, not a passage |
| `empty_is_open` | false | treat no-return bins as open |
| `print_hz` | 2.0 | console rate |


---

# History, Lessons, and Open Issues (up to base 1)

## Timeline

| Date | Version / backup | What changed |
|---|---|---|
| 2026-10-21 | first on-device runs | `v8_policy.py` works on the Jetson without TensorFlow. Camera format, `/cmd_vel` and namespace facts verified. |
| 2026-10-22 | **Baseline**, `~/weights/corridor_nav/nodes/` | VLA + pure pursuit, PD centring, straight-lane safety (stop 0.70 / slow 1.80 / emergency 0.40), ramps. LiDAR node with floor-aware band, prominence splitting, scene classification. |
| 2026-10-27 | `backup_before_avoid_20261028/` | **Creep recovery** added (latched escape bearing, hysteresis, timeout). LiDAR node publishes `/lidar/openings_array`. |
| 2026-10-28 | → becomes **base 1** | **Arc avoidance** (26 arcs, swept band, side commitment). LiDAR node publishes `/lidar/bins`. Stop/slow raised to **1.0 / 3.0 m**. `clearance_margin` raised 0.20 → **0.30 m**. |
| 2026-11-03 | `base1_20261103/` | Frozen and named "base 1" before the side-stop / gentle-arc changes (those are base 2). |

### Line counts
| | Oct 22 | Oct 27 | Base 1 |
|---|---|---|---|
| `v8_drive_node.py` | 516 | 620 | 801 |
| `lidar_openings_node.py` | 587 | 595 | 599 |

---

## What each step fixed

### Oct 22 baseline: drive from the camera, stay safe with LiDAR
- The VLA has **no notion of obstacles**, so a LiDAR gate that can override it was there from day one.
- **Centring started as P-only** and weaved from wall to wall. Adding the D term (and
  differentiating at the LiDAR's 10 Hz, not the 20 Hz loop) fixed it.
- **The floor was in the wrong place.** The first height band would have looked at chest
  height. We measured the floor at z_robot ≈ −0.66 and specified the band above floor.
- **Junction classification** went through several failures before prominence splitting:
  - A T read as `four_way` (branches fused through the facing wall).
  - A corridor read as `dead_end` (threshold above visible depth).
  - A T read as `right_branch` (untrimmed shoulder dragged the bearing).
  - A run claiming several sectors created phantom branches.

### Oct 27: creep recovery
**Problem:** every stop was terminal. The Hunter is Ackermann, so with v = 0 it can't
turn, and its clearance never improves. It sat there until pushed.

**Fix:** when blocked but the emergency lane is clear, creep slowly while steering
toward the widest opening. Two details made it work:
- **Latch the target.** Two near-equal arms (29° vs 28°) made the argmax flip every scan.
- **Hysteresis.** Clearance jittered 0.74 ↔ 1.16 m, so recovery needs a separate enter
  threshold and exit threshold.

### Oct 28: arc avoidance → base 1
**Problem 1:** stopping at 0.70 m and slowing at 1.80 m was too late. With a ~1.6 m
turning radius, by the time it reacted it couldn't steer round anything.
**Fix:** stop 1.0 m, slow 3.0 m.

**Problem 2:** the straight rectangle is the wrong shape while turning. It misses things
on the inside of a turn and over-reports on the outside. Also, "slow down" isn't
"go around".
**Fix:** test real driveable arcs against the LiDAR bins and pick the clearest one close
to the policy's intent. **Commit to a side**, because re-deciding every tick made it
dither and hit the obstacle.

**Problem 3:** with a 0.20 m margin, objects just off the lane edge were missed.
**Fix:** margin 0.30 m.

---

## Lessons worth keeping

1. **Fail closed.** No LiDAR means no motion. A missing safety channel is the one place
   where "probably fine" is unacceptable.
2. **Measure, don't assume.** The floor height, the camera encoding (rgb8, not UYVY),
   the real frame rate (~27 Hz, not 20) and the un-namespaced `/cmd_vel` were all
   different from what we first assumed.
3. **Ackermann changes everything.** Stopping is not a safe resting state if you can't
   rotate out of it. Recovery has to keep moving.
4. **Anything that picks a winner per tick needs latching or hysteresis:** escape bearing,
   avoidance side, recovery enter/exit, scene type (5 scans).
5. **Thresholds are traps. Prefer ratios.** Prominence splitting works whether the wall is
   at 2 m or 20 m. An absolute `open_range` only moves the failure around.
6. **Keep the model in its lane.** The VLA chooses the path shape. Speed and safety
   belong to geometry.

---

## Open issues in base 1

1. **Side clearance / stopping beside things.** Base 2 addresses this ("side-stop"
   and "gentle arc" fixes). Base 1 doesn't include them.
2. **The height band is blind** below 0.20 m and above 0.80 m AGL: kerbs, cables, feet,
   table edges, overhangs.
3. **Junctions are seen late.** Arms are hidden behind corridor corners until the last
   1–2 m. No LiDAR method sees round a corner. Earlier warning needs the camera or a map.
4. **Waypoints go slightly stale between inferences.** They're expressed in the frame of
   the image that produced them, and the robot moves in between. This is fine at 0.4 m/s.
   At higher speed, transform them by odometry (the node deliberately has no `/odom`).
5. **The model's speed is discarded** (it implies ~0.9 m/s, we drive 0.4). Also,
   `lookahead_dist` 0.6 against a ~0.7 m horizon nearly always picks the last waypoint.
   0.35–0.4 would let distance-based selection matter.
6. **`stop` prompt is untrained.**
7. **LEFT-only or RIGHT-only scenes read `dead_end`.** That's probably wrong at a side door.
8. **Stale comments** in `lidar_openings_node.py`: the docstring says splitting isn't
   implemented, and a `d_near` comment says "forward half-plane". The code is right,
   the comments are old.
9. **Is LiDAR junction detection needed at all?** If the VLA learned turn *timing* from
   the image (see `eval/eval_openloop.py` and `data/timing_test.png`), prompts can be
   scripted in advance. Junction type then becomes a veto, not a trigger.


---

