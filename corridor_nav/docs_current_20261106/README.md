# Mapless LiDAR + VLA Corridor Driving (current version, 2026-11-06)

These docs describe the code that is **running now**:

```
~/v8_drive_node.py         1055 lines  camera → VLA → waypoints → balance planner + safety → steering angle → /cmd_vel
~/lidar_openings_node.py    599 lines  LiDAR point cloud → clearance, centring, openings, bins  (unchanged since base 1)
~/v8_policy.py                         VLA wrapper (unchanged)
```

It is **base 1 + three changes**:

| Change | Date | Saved before it as |
|---|---|---|
| Side-stop and gentle-arc fixes | 2026-11-03 | base 1 → `base1_20261103/` |
| **Balance planner** replaces the switched arc avoidance | 2026-11-03 | base 2 → `base2_20261103/` |
| **Steering fix**: send a real wheel angle, not a turn rate | 2026-11-04 | base 3 → `base3_20261104_before_steer/` |

The older base 1 docs stay in `base1_20261103/docs/` as the record of that version.

---

## The idea in one paragraph

The robot has **no map, no localization and no odometry**. It drives from what it sees
right now. A camera frame and a one-word instruction (`straight`, `turn_left`,
`turn_right`) go into a fine-tuned **vision-language-action model (OpenVLA-OFT, "v8")**.
The model returns **8 waypoints** for the next ~0.8 s. **Pure pursuit** turns them into
a path curvature. A separate **LiDAR node** watches the geometry around the robot. Every
control tick, a **balance planner** tries a few hundred paths near the policy's path and
picks the one that keeps the robot's body **in the middle of the free space**: between
two obstacles, or between an obstacle and a wall. The planner's curvature becomes a real
**inner-wheel steering angle** for the Hunter. The LiDAR also slows and stops the robot,
and lets it **creep out** of a stop instead of freezing.

The VLA decides *where to go*. The LiDAR decides *how to get through safely*, and always
has the final say.

```
   camera ──► VLA (v8) ──► 8 waypoints ──► pure pursuit ──► policy curvature k_pol
   prompt ──┘  ~2 Hz                                                │
                                                                    ▼
   LiDAR ──► lidar_openings_node ──► /lidar/bins ──► balance planner (every tick)
             10 Hz                   /lidar/clearance     │  chosen curvature k, free length, side gap
                                     /lidar/min_dist      ▼
                                                   safety gate: emergency / slow / stop / creep
                                                          │  v, k
                                                          ▼
                                     k → inner-wheel angle (≤ 33°), slewed 2 rad/s ──► /cmd_vel  20 Hz
```

---

## Documents in this folder

| File | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Nodes, topics, threads, timing, the steering output. The big picture. |
| [LIDAR_PERCEPTION.md](LIDAR_PERCEPTION.md) | How the point cloud becomes clearance, centring error, openings, bins and junction type. |
| [DRIVE_CONTROL.md](DRIVE_CONTROL.md) | VLA, pure pursuit, the balance planner, safety gate, creep recovery, steering angle, ramps, console. |
| [PARAMETERS.md](PARAMETERS.md) | Every parameter and its current default. |
| [HISTORY_AND_LESSONS.md](HISTORY_AND_LESSONS.md) | Oct 22 → base 1 → base 2 → base 3 → now: what each step fixed, the bugs, and what's still open. |

The step-by-step launch commands and health checks are in
[`../ROBOT_COMMANDS.md`](../ROBOT_COMMANDS.md).

---

## How to launch (short version)

Each step goes in its own terminal, in this order. **Unset `ROS_NAMESPACE` everywhere.**
`.bashrc` exports `robot1`, which hides every topic.

```bash
# 1. Base (CAN + hunter_base; listens on un-namespaced /cmd_vel)
bash ~/Robot_teleop_ws/robot.bash

# 2. Camera at 10 fps (the shim forces 10 fps; v4l2_camera ignores `framerate`)
~/camfps/camera_10fps.sh                  # or: ~/camfps/camera_10fps.sh /dev/video1

# 3. LiDAR driver, STANDALONE (never full_system.launch.py, never two drivers)
source ~/Robot_autonomy_ws/install/setup.bash && unset ROS_NAMESPACE && ros2 launch rslidar_sdk start.py

# 4. LiDAR perception. REQUIRED: without it the drive node holds still on purpose
source ~/v8_env.sh && unset ROS_NAMESPACE && python3 -u ~/lidar_openings_node.py

# 5. Drive (dry run unless --live)
source ~/v8_env.sh && unset ROS_NAMESPACE && cd ~ && \
  python3 -u v8_drive_node.py --live --ros-args -p infer_hz:=2.0 -p creep_speed:=0.25 2>&1 | tee ~/run.log

# Change the instruction at any time
ros2 topic pub --once /vla/prompt std_msgs/msg/String "{data: 'turn_left'}"
```

### Things to remember

- **"VLA is running but the robot won't move"** almost always means the LiDAR is silent
  or late. The safety gate *fails closed*. Check the LiDAR driver and `lidar_openings_node`
  first. Never "fix" this with `use_safety:=false` (that switch is only for the
  VLA-only demo, see ROBOT_COMMANDS.md).
- **Don't run the localization stack** (`full_system.launch.py`) alongside this one. It
  caused ~10× more Jetson over-current events and the stop-start jerk.
- **The `stop` prompt was never trained.** The node honours it by commanding zero, but
  don't rely on it as a safety stop. Use Ctrl-C or the E-stop.
- `~/v8_env.sh` sets `LD_PRELOAD` to Tegra cuBLAS builds. Without it, the installed
  (sbsa) cuBLAS has no sm_87 kernels and CUDA fails on the Orin.
- Run with `creep_speed:=0.25`. At the default 0.10 m/s the robot can't turn far enough
  before the 8 s recovery timeout.
