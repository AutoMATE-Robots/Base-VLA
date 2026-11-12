#!/bin/bash
# Start the camera exactly as before, but at 10 fps instead of 30.
# v4l2_camera 0.6.2 ignores framerate:=...; the shim sets the rate right after the node
# configures the format. Images are unchanged (same node, same rgb8 conversion).
# The camera only supports 30/20/10 fps at 1920x1280, so 10 is the lowest possible.
DEV=${1:-/dev/video0}
source /opt/ros/humble/setup.bash
unset ROS_NAMESPACE
CAM_FPS=${CAM_FPS:-10} LD_PRELOAD=$HOME/camfps/v4l2_fps_shim.so \
  ros2 run v4l2_camera v4l2_camera_node --ros-args -p video_device:=$DEV \
  -p image_size:="[1920,1280]" -p pixel_format:=uyvy -p camera_frame_id:=c1_046
