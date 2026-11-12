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
