# Corridor Navigation — LiDAR perception + VLA driving

Snapshot taken 2026-10-22 from the Jetson (`hunter@ubuntu`). Everything here is code and
recorded data only — the 15 GB `v8_vla` checkpoint is deliberately **not** included.

```
nodes/   the four things that run on the robot, plus reference nodes
eval/    open-loop evaluation and dataset prep
viz/     offline replay viewer (open viz/index.html, it loads viz/session.json)
data/    a 426-scan recording, a reference camera frame, the timing-test plot
```

---

## 1. Measured geometry — the part that is easy to get wrong

**`base_link` is NOT at floor level. It rides ~0.66 m above the floor.**

TF `base_link -> rslidar` is translation `[0.247, 0.000, 0.420]`, rotation identity, so

```
x_robot = x_sensor + 0.247      y_robot = y_sensor      z_robot = z_sensor + 0.420
```

The floor sits at **z_robot ≈ −0.66**, i.e. the sensor is ~1.08 m above the ground.
Three independent lines of evidence, all from real scans:

1. Nothing in the cloud lies below z_robot = −0.662, and there is a plane-like cluster
   there (n = 851, z std **0.016 m**, spanning r = 2.0–2.9 m). You cannot see through a
   floor, so that plane is the floor.
2. The lowest beam is **−24.8°**, so a floor 1.082 m below the sensor is first struck at
   r = 1.082 / tan(24.8°) = **2.34 m** — exactly where that cluster begins and exactly
   where lowest-beam returns stop (max r = 2.3 m).
3. Ceiling z_max = +1.552 minus floor z_min = −0.662 gives a **2.21 m room height**,
   which is normal. If `base_link` were on the floor the room would be 1.55 m tall.

**The trap:** a raw z-histogram shows a large smooth hump at z_robot ≈ +0.42 and *no*
floor spike. That hump is the beam-elevation density (most beams near horizontal) painted
onto vertical walls. Taking "the biggest pile of points" finds the walls, not the ground.

Consequence: the height band is specified **above floor** and converted, never in raw
z_robot. Default 0.20–0.80 m AGL = z_robot −0.46…+0.14. The originally-specified band of
z_robot 0.20–0.80 would have sampled 0.86–1.46 m above the floor — chest-to-head height,
blind to every low obstacle.

---

## 2. Perception pipeline (`nodes/lidar_openings_node.py`)

`/rslidar_points` (10 Hz, ~236k pts, point_step 16, xyzi float32) → height band →
1° bearing bins, **minimum** range per bin (`inf` = no return = unknown) → openings.

### 2.1 Corridor centring signal
```
d_left  = median of non-empty bins over bearings +60…+120
d_right = median of non-empty bins over bearings −120…−60
e = (d_right − d_left) / 2            # POSITIVE = robot is LEFT of centre
```
Confirmed on hardware: drifting toward the right wall gave dL 3.1→3.8, dR 1.3→1.1,
e −0.91→−1.34. Note `e` is only meaningful **between two parallel walls**; at a junction
it measures the branch mouth, not a wall, and can reach ±1.5 m.

### 2.2 Openings — prominence splitting, not an absolute threshold
Naively, "an opening is a run of bins beyond `open_range`". That fails badly: `open_range`
must sit **above** a junction's facing wall but **below** the corridor's visible depth.
Measured in this building that window was only ~3.0–4.0 m. At 2.5 the facing wall (2.64 m)
counted as open and the left branch + wall + right branch fused into one 160° run →
`four_way` for a plain T. At 4.0 an ordinary corridor read `dead_end` because forward
returns dipped below the threshold.

The fix is **threshold-free**: split a run at an interior dip that is deep relative to its
flanking peaks.
```
dip is a split point if   value(dip) < split_ratio * min(left_peak, right_peak)
split_ratio = 0.6
```
At the real T-junction the merged run peaked at 10.1 m (right arm) and 8.3 m (left arm)
with a 2.64 m trough between them (the facing wall). 2.64 < 0.6 × 8.3, so it splits into
two arms → `T_junction`. A straight corridor has a single peak and no qualifying dip, so
it is left whole. The ratio holds whether the wall is at 2 m or 20 m.

Each split piece is then **trimmed** back to the contiguous span around its peak that
stays above `split_ratio × peak`. Without trimming, a piece keeps a shallow shoulder
running down into the dip, which drags its centre bearing toward the wall — the left arm
reported +24° (FORWARD) instead of ~+52° (LEFT), giving `right_branch` for a plain T.

After this, `T_junction` holds for `open_range` anywhere from 2.0 to 5.0, and `corridor`
from 2.0 to 3.5. Default 2.5 sits mid-window and well clear of the visible-depth ceiling.

### 2.3 Robot-fit test
A run survives only if it subtends enough angle for the robot at its own depth:
```
required_half_angle = atan2(robot_width/2 + margin, depth) = atan2(0.575, depth)
keep if angular_width >= 2 * required_half_angle
```

### 2.4 Classification
Per opening: FORWARD if |bearing| ≤ 25°, LEFT +25…+155, RIGHT −25…−155, else BEHIND.

**One contiguous run claims exactly ONE direction** — the sector of its centre bearing.
Multiple directions must come from genuinely separate runs. Letting a single run claim
every sector it overlapped produced phantom branches: an approach run growing 54°→92°
claimed `fwd+right` → `right_branch`, a confident turn signal into solid wall.

Exception: a run wider than `wide_run_deg` (200°) is open space, not a passage, and falls
back to per-sector coverage requiring each sector's share to be robot-wide. Without this,
a room open on all sides but for one 10° obstacle forms a single ~350° run whose circular
mean lands at its antipode → `dead_end` for a wide-open space.

Scene table: `corridor` = FORWARD only; `T_junction` = LEFT and RIGHT, no FORWARD;
`left_branch` = FORWARD+LEFT; `right_branch` = FORWARD+RIGHT; `four_way` = all three;
`dead_end` = nothing. Requires `stable_scans` (5) consecutive identical scans to publish.

Known gap: LEFT-only or RIGHT-only falls through to `dead_end`, because the spec table
only defines `T_junction` for LEFT **and** RIGHT. Probably wrong at a side door.

### 2.5 Safety channels
```
d_front  = min x over |y| <= 0.575, x > 0        -> /lidar/clearance  (speed ramp)
d_near   = min x over |y| <= 0.425, x > 0        -> /lidar/min_dist   (emergency cut)
```
The emergency lane is deliberately **narrower** than the slow-down lane. Two earlier
versions were wrong and both are instructive:
- the whole forward half-plane at *radial* distance — a wall merely beside the robot gave
  rng ≈ 0.39 and tripped the 0.40 m cut. Seen live as stopping for things not in the way.
- a *widened* lane at along-track distance — the points it caught sat at |y| 0.60–0.71,
  a wall running alongside, reported as x ≈ 0.00. Would have stopped the robot forever.

---

## 3. Control (`nodes/v8_drive_node.py`)

Four stages, in order.

### 3.1 Pure pursuit (classic path tracking)
VLA emits 8 cumulative waypoints `(x forward, y left)` in the robot frame, spanning ~0.8 s.
Pick the first waypoint at least `lookahead_dist` along the path, then take the unique
circular arc through the origin, tangent to the current heading, that reaches it:
```
kappa = 2*y / L^2,   L^2 = x^2 + y^2
omega = v * kappa
```
Lookahead is selected by **distance, not index** — a fixed index is an aggressive target
at low speed and a twitchy one at high speed. Guards: a target at or behind the axle
(x ≤ 0.05) returns (0,0) rather than commanding a spin, because the Hunter is Ackermann
and cannot rotate in place; `max_curvature` clamps to the steering limit when set.

### 3.2 Corridor centring — PD
```
e_db  = e with a +-center_deadband deadzone removed
omega += clip( -(center_gain * e_db + center_damp * de/dt), -center_max, +center_max )
```
`e > 0` means left of centre, so it must steer right = negative omega; hence the minus.
Proportional alone overshoots and weaves on a car-like base (observed live). `de/dt` is
differentiated in the 10 Hz subscriber — not the 20 Hz control loop, which would see each
value twice and read zero rate on every other tick — and low-pass filtered (0.7/0.3)
because the sector medians jitter a few cm per scan.

Held off outside `corridor` scenes by default, since at a junction `e` measures the branch
mouth and would fight the policy exactly when the policy is doing the interesting part.

### 3.3 Obstacle avoidance — overrides the policy unconditionally
```
no clearance data, or older than safety_stale   -> (0, 0)     fail CLOSED
d_near < emergency_distance (0.40)              -> (0, 0)
d_front < stop_distance (0.70)                  -> (0, 0)
d_front < slow_distance (1.80)                  -> v scaled linearly to 0 at stop_distance
```
The VLA has no obstacle notion of its own. A missing safety channel is the one case where
guessing "probably clear" is unacceptable.

### 3.4 Slew-rate limiting
Asymmetric on purpose: `accel` 0.5 m/s² up, `decel` 1.5 m/s² down, `angular_accel`
2.0 rad/s². Faults (stale/missing waypoints) **bypass** the ramp and hard-zero; a
commanded `stop` goes through it so it is smooth.

`/cmd_vel` is only ever written in `_send()`, and the publisher is not created at all
unless `--live` is passed.

---

## 4. Open problems

1. **Every stop is terminal.** Below `stop_distance` the robot zeroes out; it is Ackermann
   so it cannot turn in place; clearance never improves. It sits until pushed. This caps
   run length more than anything else. Recovery needs either creep-and-steer toward the
   widest opening (while the 0.425 m emergency lane is still clear) or reversing.
2. **Clearance is not arc-aware.** The straight rectangle is the wrong swept region while
   turning — it misses obstacles on the inside of a turn and over-reports on the outside.
   Correct version: sweep the footprint along the commanded curvature `kappa`, centre of
   rotation at `(0, 1/kappa)`, in-lane when `||R|-hw| <= dist_to_centre <= |R|+hw`.
3. **The band is blind outside 0.20–0.80 m AGL.** Kerbs, cables, feet, table edges and
   overhangs are invisible. This is not a general obstacle detector.
4. **Junction detection is occlusion-limited, not threshold-limited.** The arms are hidden
   behind the corridor corners until the sensor nears the mouth — measured as the last
   1–2 m. No LiDAR method sees around a corner; earlier warning has to come from the
   camera or a map.
5. **The model's implied speed is discarded.** Waypoint spacing implies ~0.9 m/s; we
   command a fixed 0.4. Path *shape* is preserved (pure pursuit uses geometry only) but
   real information is being thrown away. Also `lookahead_dist` 0.6 against a ~0.71 m
   prediction horizon means the selected index is nearly always the last waypoint, so the
   distance-based selection barely does anything. ~0.35–0.4 would give it room.
6. **Architecture question, unresolved.** `eval/eval_openloop.py` produces a timing plot
   (`data/timing_test.png`) testing whether predicted deflection stays flat while the
   prompt says turn and only swings near the junction. If the model learned timing from
   the image, prompts can be route-scripted in advance and LiDAR junction detection is not
   needed for timing at all — it becomes a veto, not a trigger. That plot decides whether
   the LiDAR pipeline is load-bearing.

---

## 5. Running it

```bash
# 1. LiDAR driver (needs Robot_autonomy_ws; check `cat /sys/class/net/eno1/carrier` == 1)
source /opt/ros/humble/setup.bash && source ~/Robot_autonomy_ws/install/setup.bash
unset ROS_NAMESPACE && ros2 launch rslidar_sdk start.py

# 2. camera — verify which /dev/videoN is real: the camera lists UYVY, the other lists nothing
v4l2-ctl -d /dev/video0 --list-formats
ros2 run v4l2_camera v4l2_camera_node --ros-args -p video_device:=/dev/video0 \
  -p image_width:=1920 -p image_height:=1280 -p framerate:=20.0 -p pixel_format:=uyvy

# 3. perception
source ~/v8env/bin/activate && unset ROS_NAMESPACE && python3 -u lidar_openings_node.py

# 4. driving — DRY RUN unless --live is passed
export LD_PRELOAD=$HOME/cuda126_tegra_cublas/lib/libcublas.so.12:$HOME/cuda126_tegra_cublas/lib/libcublasLt.so.12
python3 -u v8_drive_node.py --ros-args -p infer_hz:=2.0          # add --live to move

# prompt
ros2 topic pub --once /vla/prompt std_msgs/msg/String "{data: 'turn_left'}"

# record a session for the viewer
python3 -u lidar_record.py 60 session.json
```

Notes that cost time to rediscover:
- `ROS_NAMESPACE=robot1` is exported in `.bashrc`; every topic name in these nodes is
  absolute on purpose. `unset ROS_NAMESPACE` before launching anything.
- The `stop` prompt is **untrained**. Never rely on it to halt the robot; use Ctrl-C or
  the E-stop.
- "VLA generating actions but the robot won't move" is almost always the safety gate
  failing closed because the LiDAR is silent. First check: `cat /sys/class/net/eno1/carrier`.
