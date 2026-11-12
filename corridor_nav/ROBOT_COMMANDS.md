# Robot quick commands (VLA + LiDAR drive stack)

Updated 2026-11-06. Each step runs in its own terminal, in this order.
How the code works: `~/weights/corridor_nav/docs_current_20261106/README.md`.

## 0. Before you start
- Power mode must be MAXN: `nvpmodel -q`
- Close RViz and the Software Updater pop-up.
- Do NOT launch `full_system.launch.py` (the localization stack). It causes over-current throttling and the stop-start jerk.
- Never run two LiDAR drivers at the same time.

## 1. Robot base
Option A (script, uses can0):
```
bash ~/Robot_teleop_ws/robot.bash
```
Option B (CAN adapter on can2):
```
sudo ip link set can2 down
sudo ip link set can2 type can bitrate 500000
sudo ip link set can2 up
cd ~/Robot_teleop_ws && source install/setup.bash
ros2 launch hunter_base hunter_base.launch.py port_name:=can2
```

## 2. Camera (10 fps)
```
~/camfps/camera_10fps.sh
```
If the camera is on video1: `~/camfps/camera_10fps.sh /dev/video1`
Check the rate (should say 10.000): `v4l2-ctl -d /dev/video0 --get-parm | grep "Frames per"`

## 3. LiDAR driver (standalone)
```
source ~/Robot_autonomy_ws/install/setup.bash && unset ROS_NAMESPACE && ros2 launch rslidar_sdk start.py
```

## 4. LiDAR openings node (required: without it the robot holds still)
```
source ~/v8_env.sh && unset ROS_NAMESPACE && python3 -u ~/lidar_openings_node.py
```

## 5. VLA drive node
Live (robot moves):
```
source ~/v8_env.sh && unset ROS_NAMESPACE && cd ~ && python3 -u v8_drive_node.py --live --ros-args -p infer_hz:=2.0 -p creep_speed:=0.25 2>&1 | tee ~/run.log
```
Dry run (nothing is sent to the wheels; for testing): run the same command without `--live`.
Stop: Ctrl+C. It prints the average latency at the end.

### Optional extra parameters (add after `--ros-args`)
| Parameter | What it does |
|---|---|
| `-p steer_rate:=1.0` | Gentler wheel movement if it weaves (default 2.0) |
| `-p steer_mode:=legacy` | Old steering behaviour (before the 2026-11-04 fix) |
| `-p avoid_max_curvature:=0.75 -p avoid_swerve_kmax:=0.6` | Lets avoidance use nearly the full 33° wheel angle |

### Demo: VLA only (no LiDAR), for the VLA vs VLA + LiDAR comparison
Skip steps 3 and 4 and run the drive node with the LiDAR gate and centring switched off:
```
source ~/v8_env.sh && unset ROS_NAMESPACE && cd ~ && python3 -u v8_drive_node.py --live --ros-args -p infer_hz:=2.0 -p use_safety:=false -p use_centering:=false 2>&1 | tee ~/run_vla_only.log
```
Nothing stops the robot in this mode, not even for obstacles. Use open space, keep someone on the E-stop, and only use it for this demo. For the VLA + LiDAR half, use the normal steps 1–5 (log to `~/run_vla_lidar.log`).

## 5b. SignWay goal reasoning (optional: replaces typing prompts by hand)
Needs the drive node (step 5) running, and `~/.signway_secrets` holding the VLM API key.
```
source ~/v8_env.sh && source ~/.signway_secrets && unset ROS_NAMESPACE && python3 -u ~/weights/corridor_nav/current/signway_reasoner_node.py --ros-args --params-file ~/weights/corridor_nav/current/signway_params.yaml -p goal:="room 6-217" 2>&1 | tee ~/signway.log
```
New goal without restarting: `ros2 topic pub --once /signway/goal std_msgs/msg/String "{data: 'Main Elevators'}"`
Reasoning test only (no prompts sent to the drive node): add `-p publish_prompts:=false`.
Watch it: `ros2 topic echo /signway/status`. Each run logs to `~/signway_logs/<time>/`.
First check odometry is there (the node also checks at start-up and logs `/odom OK` or an error):
`ros2 topic hz /odom` should show a steady rate. Without it, turns are not held for the opening,
not confirmed (30 s timeout), and identical signs cannot be told apart by place.
The robot stops on its own when a sign says it ARRIVED (log: `GOAL REACHED`).

## 6. Health checks
| Check | Command | Good value |
|---|---|---|
| Over-current events (count since boot) | `cat /sys/class/hwmon/hwmon7/oc3_event_cnt` | Grows only slowly while driving |
| Safety stops in the last run | `grep -c "no clearance" ~/run.log` | About 0 |
| Robot battery | `source ~/Robot_teleop_ws/install/setup.bash && ros2 topic echo --once /hunter_status \| grep battery` | Volts |
| Live power and clocks | `tegrastats` | Ctrl+C to stop |

## 7. Restore an older version of the code
| Version | What it is | Restore command |
|---|---|---|
| Base 3 | Before the steering fix (2026-11-04) | `cp ~/weights/corridor_nav/base3_20261104_before_steer/v8_drive_node.py ~/` |
| Base 2 | Before the balance planner (2026-11-03) | `cp ~/weights/corridor_nav/base2_20261103/{v8_drive_node.py,lidar_openings_node.py} ~/` |
| Base 1 | Before the side-stop/arc fixes (2026-11-03) | `cp ~/weights/corridor_nav/base1_20261103/{v8_drive_node.py,lidar_openings_node.py} ~/` |

Save the current version first if you want to keep it, e.g.: `mkdir -p ~/weights/corridor_nav/my_backup && cp ~/v8_drive_node.py ~/weights/corridor_nav/my_backup/`
