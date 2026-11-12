#!/bin/bash
# signway_up.sh -- start the SignWay goal-driven stack, one gnome-terminal tab per process.
#
#   bash ~/signway_up.sh [watch|dry|live] ["goal"]      default: watch "Elevators"
#   bash ~/signway_up.sh stop                            stop everything this script starts
#   bash ~/signway_up.sh status                          what is running + topic rates
#
# Modes (each includes the one above it):
#   watch  base, camera, LiDAR, openings node, SignWay reasoner (prompts NOT sent), viewer.
#          No VLA. Use for sign tests (T7). Nothing can move the wheels.
#   dry    + v8 VLA drive node WITHOUT --live, and SignWay sends its prompts to it.
#          The full chain runs and logs what it would command; the wheels never move.
#   live   + drive node WITH --live: THE ROBOT DRIVES ITSELF. Wheels off the ground for the
#          first run, someone on the E-stop. Asks you to type LIVE first.
#
# Already-running parts are skipped, so re-running is safe (never two LiDAR drivers).
# Change the goal without restarting:
#   ros2 topic pub --once /signway/goal std_msgs/msg/String "{data: 'room 6-217'}"
# Logs: ~/signway_logs/ (per-run events.jsonl, ledger.jsonl, VLM crops, viz_*.mp4), ~/run.log

MODE="${1:-watch}"
GOAL="${2:-Elevators}"
PARAMS=~/weights/corridor_nav/current/signway_params.yaml
# Send a turn decision to the VLA at once (the VLA was trained to turn only when the junction
# arrives; LiDAR handles safety). WAIT_FOR_OPENING=true bash ~/signway_up.sh ... = old hold.
WAIT_FOR_OPENING="${WAIT_FOR_OPENING:-false}"
# Extra drive-node params, e.g. DRIVE_ARGS="-p avoid_policy_priority:=off" (old LiDAR-first steering)
DRIVE_ARGS="${DRIVE_ARGS:-}"
# Sign-reading VLM: "jev" (TypeSafe, used automatically when TYPESAFE_API_KEY is in
# ~/.signway_secrets) or "gemini". Force one with SIGNWAY_VLM=gemini bash ~/signway_up.sh ...
if [ -z "$SIGNWAY_VLM" ]; then
  SIGNWAY_VLM=gemini
  (source ~/.signway_secrets 2>/dev/null; [ -n "$TYPESAFE_API_KEY" ]) && SIGNWAY_VLM=jev
fi
VLM_ARGS=""
[ "$SIGNWAY_VLM" = jev ] && VLM_ARGS="-p vlm_model:=jev-latest -p vlm_reasoning:=none"
DRIVE=~/weights/corridor_nav/current/v8_drive_node.py      # new version with /signway/speed_cap
ROS="source ~/v8_env.sh && unset ROS_NAMESPACE"
SIGNWAY_ENV="$ROS && source ~/signway_handoff/cudnn_tegra_env.sh && source ~/.signway_secrets"

source /opt/ros/humble/setup.bash >/dev/null 2>&1
unset ROS_NAMESPACE

# pattern for each part (used to detect / stop it)
P_BASE="hunter_base_node"
P_CAM="v4l2_camera_node"
P_LIDAR="rslidar_sdk_node"
P_OPEN="lidar_openings_node.py"
P_SIGN="signway_node_viz.py|signway_reasoner_node.py"
P_DRIVE="v8_drive_node.py"
P_VIEW="signway_viewer.py"

running() { pgrep -f "$1" >/dev/null; }
say() { echo -e "\033[1;36m[signway]\033[0m $*"; }
warn() { echo -e "\033[1;33m[signway]\033[0m $*"; }

tab() {   # tab "title" "command"
  gnome-terminal --tab --title="$1" -- bash -c "$2; echo; echo '[$1 exited] press Enter to close this tab'; read"
}

wait_topic() {   # wait_topic /topic seconds
  local t=$1 n=${2:-30}
  for ((i = 0; i < n; i++)); do
    ros2 topic list 2>/dev/null | grep -qx "$t" && return 0
    sleep 1
  done
  warn "$t did not appear after ${n}s"
  return 1
}

camera_dev() {   # first /dev/video* of the GMSL2-USB camera
  v4l2-ctl --list-devices 2>/dev/null | awk '/GMSL2/{f=1;next} f&&/\/dev\/video/{print $1; exit}'
}

stop_all() {
  for p in "$P_VIEW" "$P_DRIVE" "$P_SIGN" "$P_OPEN" "$P_LIDAR" "$P_CAM" "camera_10fps.sh" "$P_BASE"; do
    if running "$p"; then say "stopping $p"; pkill -INT -f "$p"; fi
  done
  sleep 3
  for p in "$P_VIEW" "$P_DRIVE" "$P_SIGN" "$P_OPEN" "$P_LIDAR" "$P_CAM" "$P_BASE"; do
    running "$p" && { warn "$p still up, sending TERM"; pkill -TERM -f "$p"; }
  done
  say "stopped."
}

status() {
  for x in "base:$P_BASE" "camera:$P_CAM" "lidar:$P_LIDAR" "openings:$P_OPEN" "signway:$P_SIGN" \
           "VLA drive:$P_DRIVE" "viewer:$P_VIEW"; do
    name=${x%%:*}; pat=${x#*:}
    if running "$pat"; then echo -e "  \033[32mUP  \033[0m $name"; else echo -e "  \033[31mdown\033[0m $name"; fi
  done
  pgrep -af "$P_DRIVE" | grep -q -- "--live" && echo -e "  \033[1;31mDRIVE NODE IS LIVE (wheels can move)\033[0m"
  for t in /odom /image_raw /lidar/openings_array /signway/debug; do
    r=$(timeout 4 ros2 topic hz "$t" 2>/dev/null | grep -m1 "average rate" | awk '{print $3}')
    printf "  %-24s %s\n" "$t" "${r:+$r Hz}${r:-no messages}"
  done
}

case "$MODE" in
  stop) stop_all; exit 0 ;;
  status) status; exit 0 ;;
  watch|dry|live) ;;
  *) echo "usage: bash ~/signway_up.sh [watch|dry|live] [\"goal\"] | stop | status"; exit 1 ;;
esac

if [ "$MODE" = live ]; then
  warn "LIVE mode: the VLA drive node will send /cmd_vel and THE ROBOT WILL MOVE on its own."
  warn "First run: wheels off the ground. Always: someone on the E-stop, clear corridor."
  read -r -p "Type LIVE to continue: " ok
  [ "$ok" = LIVE ] || { say "aborted"; exit 1; }
fi
[ -f ~/.signway_secrets ] || { warn "~/.signway_secrets missing (VLM API key)"; exit 1; }

# 0. power mode
nvpmodel -q 2>/dev/null | grep -q MAXN || warn "power mode is not MAXN (check: nvpmodel -q)"

# 1. robot base (asks for your sudo password in its tab)
if running "$P_BASE"; then say "base already running"
else
  ip link show can2 >/dev/null 2>&1 || warn "can2 not found: is the USB-CAN adapter plugged in? (the base will fail without it)"
  say "starting base -> type your sudo password in the 'base' tab"
  tab "base" "bash ~/Robot_teleop_ws/robot.bash"
fi

# 2. camera (10 fps)
if running "$P_CAM"; then say "camera already running"
else
  dev=$(camera_dev); dev=${dev:-/dev/video0}
  say "starting camera on $dev"
  tab "camera" "~/camfps/camera_10fps.sh $dev"
fi

# 3. LiDAR driver (standalone; never two at once)
if running "$P_LIDAR"; then say "LiDAR already running"
else
  say "starting LiDAR"
  tab "lidar" "source ~/Robot_autonomy_ws/install/setup.bash && unset ROS_NAMESPACE && ros2 launch rslidar_sdk start.py"
fi

# 4. LiDAR openings node (required by the drive node AND by SignWay's turn hold)
if running "$P_OPEN"; then say "openings node already running"
else
  wait_topic /rslidar_points 30
  say "starting openings node"
  tab "openings" "$ROS && python3 -u ~/lidar_openings_node.py"
fi

wait_topic /image_raw 30
wait_topic /lidar/openings_array 30
wait_topic /odom 40 || warn "no /odom yet: check the 'base' tab (CAN adapter plugged in? sudo password typed?)"

# 5. VLA drive node (dry or live)
if [ "$MODE" != watch ]; then
  if running "$P_DRIVE"; then
    warn "drive node already running -- not starting another (stop it first to change dry/live)"
  else
    LIVEFLAG=""; [ "$MODE" = live ] && LIVEFLAG="--live"
    say "starting VLA drive node ($MODE) -- policy load takes ~15 s"
    tab "VLA drive ($MODE)" "$ROS && cd ~ && python3 -u $DRIVE $LIVEFLAG --ros-args -p infer_hz:=2.0 -p creep_speed:=0.25 $DRIVE_ARGS 2>&1 | tee ~/run.log"
  fi
fi

# 6. SignWay reasoner (instrumented: same node + /signway/debug for the viewer)
PUBLISH=false; [ "$MODE" != watch ] && PUBLISH=true
if running "$P_SIGN"; then
  warn "SignWay already running -- not restarted (stop it first to change goal or mode)"
else
  say "starting SignWay: goal='$GOAL' publish_prompts=$PUBLISH VLM=$SIGNWAY_VLM"
  tab "signway" "$SIGNWAY_ENV && export SIGNWAY_VLM=$SIGNWAY_VLM && python3 -u ~/signway_viz/signway_node_viz.py --ros-args --params-file $PARAMS -p goal:=\"$GOAL\" -p publish_prompts:=$PUBLISH -p wait_for_opening:=$WAIT_FOR_OPENING $VLM_ARGS 2>&1 | tee ~/signway.log"
fi

# 7. viewer
if running "$P_VIEW"; then say "viewer already running"
else
  wait_topic /signway/debug 40
  say "starting viewer"
  tab "viewer" "$ROS && python3 ~/signway_viz/signway_viewer.py"
fi

say "up in '$MODE' mode, goal '$GOAL'. Check with: bash ~/signway_up.sh status   Stop: bash ~/signway_up.sh stop"
