# Mapless VLA + LiDAR Corridor Driving (AgileX Hunter 2.0)

A car-like robot that drives indoor corridors **without a map, localization or odometry**.
A fine-tuned vision-language-action model (**OpenVLA-OFT, "v8"**) looks at the camera and
a one-word instruction (`straight`, `turn_left`, `turn_right`) and predicts where to drive.
A LiDAR node watches the geometry around the robot and **always has the final say**: it
slows, stops, steers the robot through gaps, and lets it creep out of a stop.

> **The VLA decides *where to go*. The LiDAR decides *how to get through safely*.**
>
> **New: goal conditioning (SignWay).** Give the robot a goal ("room 6-217"). It reads the
> corridor signs (docTR), decides *when* a sign is worth reasoning about (adaptive gate), asks a
> VLM about the **cropped sign + goal** only then, and turns the answer into the VLA prompt.
> See [Goal-conditioned driving](#goal-conditioned-driving-signway).

```
 camera ──► signway_reasoner_node: docTR → adaptive gate ──(fire)──► VLM API (cropped sign + goal)
                                            └──────────────── decision ──► /vla/prompt
 camera ──► VLA (v8, ~2 Hz) ──► 8 waypoints ──► pure pursuit ──► policy curvature
 prompt ──┘                                                            │
                                                                       ▼
 LiDAR ──► lidar_openings_node (10 Hz) ──► /lidar/bins, /lidar/clearance ──► balance planner
                                                                       │
                                               safety gate: emergency / slow / stop / creep
                                                                       │
                                    curvature → inner-wheel steering angle ──► /cmd_vel (20 Hz)
```

> ⚠️ **Model weights are not included.** The ~15 GB checkpoint is excluded from this repo
> (see [Getting the weights](#getting-the-weights)). Everything else (code, configs,
> docs, a recorded LiDAR session) is here.

---

## Repository layout

```
signway/                      Jetson tooling: one-command launcher, live viewer, Jev sign reader,
                              turn sims. See signway/README.md
adaptive_reasoning/           SignWay sign reasoning: docTR evidence, gate, VLM reasoner, memory,
                              live.py (per-frame loop), tests/. See adaptive_reasoning/README.md
corridor_nav/
  current/                    ← THE CODE THAT RUNS ON THE ROBOT NOW (start here)
    v8_drive_node.py            camera → VLA → planner + safety → /cmd_vel
    lidar_openings_node.py      point cloud → clearance, centring, openings, bins
    v8_policy.py                loads and runs the v8 VLA
    v8_env.sh                   environment setup (ROS, venv, Tegra cuBLAS preload)
    signway_reasoner_node.py    goal + signs → VLM decision → /vla/prompt
    signway_params.yaml         gate threshold, VLM endpoint/model, maneuver completion
    signway_secrets.example     template for ~/.signway_secrets (API key, never committed)
  docs_current_20261106/      ← full documentation of the current code
    README.md  ARCHITECTURE.md  LIDAR_PERCEPTION.md  DRIVE_CONTROL.md
    PARAMETERS.md  HISTORY_AND_LESSONS.md
  ROBOT_COMMANDS.md           step-by-step launch commands + health checks
  README.md                   original Oct 22 write-up (geometry, perception math)
  nodes/                      Oct 22 snapshot + helper nodes (recorder, logger, server clients)
  base1_20261103/  base2_20261103/  base3_20261104_before_steer/
  backup_before_avoid_20261028/   earlier versions of the code, kept for comparison/rollback
  eval/                       open-loop evaluation + rosbag → episode conversion
  viz/                        offline LiDAR replay viewer (open viz/index.html)
  data/                       426-scan LiDAR recording, sample camera frame, timing plot
v8_vla/                       checkpoint configs, tokenizer, dataset stats, Prismatic model code
                              (everything except the weight files)
v8_vla_code/                  inference-side code: openvla-oft.tar.gz (patched), eval scripts
tools/camfps/                 LD_PRELOAD shim that forces the camera to 10 fps
```

**New here?** Read `corridor_nav/docs_current_20261106/README.md` and then
`ARCHITECTURE.md`. Use `corridor_nav/ROBOT_COMMANDS.md` to run the robot.

---

## Dependencies

### Hardware (as tested)

| Part | Model / notes |
|---|---|
| Robot base | **AgileX Hunter 2.0**: Ackermann steering (cannot turn in place), CAN bus |
| Compute | **NVIDIA Jetson AGX Orin** (sm_87), JetPack / L4T R36 (kernel 5.15-tegra), power mode **MAXN** |
| LiDAR | **RoboSense** 3D LiDAR over Ethernet (`/rslidar_points`, 10 Hz) |
| Camera | USB UVC camera, 1920×1280 UYVY (30/20/10 fps) |
| CAN | onboard CAN or a USB-CAN adapter, 500 kbit/s |
| Memory | ~16 GB free GPU/unified memory for the 7B model in bf16 |

### System software

| Component | Version | Purpose |
|---|---|---|
| Ubuntu | 22.04 (aarch64) | OS |
| ROS 2 | **Humble** | middleware |
| Python | **3.10** | all nodes |
| CUDA | 12.x (Tegra) | GPU inference |
| cuBLAS (Tegra build) | **12.6.4.1 arm64** | needed on Orin, see note below |
| `gcc` | any | builds the camera fps shim |

### ROS 2 packages

| Package | Used for |
|---|---|
| `rclpy`, `std_msgs`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `visualization_msgs` | messages used by the nodes (`ros-humble-desktop` includes them) |
| [`v4l2_camera`](https://gitlab.com/boldhearts/ros2_v4l2_camera) 0.6.x | camera driver → `/image_raw` (rgb8) |
| [`rslidar_sdk`](https://github.com/RoboSense-LiDAR/rslidar_sdk) + `rslidar_msg` | LiDAR driver → `/rslidar_points` |
| [`hunter_ros2`](https://github.com/agilexrobotics/hunter_ros2) (`hunter_base`) + [`ugv_sdk`](https://github.com/agilexrobotics/ugv_sdk) | base driver, listens on `/cmd_vel` |
| `cv_bridge` | only for the helper nodes in `corridor_nav/nodes/` (the main nodes decode images with numpy) |

### Python packages (versions from the working Jetson venv)

```
torch==2.8.0            # Jetson (aarch64 + CUDA) wheel, NOT the PyPI x86 one
torchvision==0.23.0
transformers==4.40.1    # pinned by OpenVLA-OFT; newer versions break the Prismatic code
tokenizers==0.19.1
timm==0.9.10            # pinned by OpenVLA-OFT
peft==0.11.1
accelerate==1.15.0
safetensors==0.8.0
huggingface_hub==0.36.2
sentencepiece==0.2.2
einops==0.8.2
draccus==0.11.6
diffusers==0.30.3
json-numpy==2.1.1
numpy==1.26.4           # keep < 2
pillow
requests
openvla-oft             # installed editable from v8_vla_code/openvla-oft.tar.gz (see setup)
```

`flash-attn` is **not** needed. The model runs with `attn_implementation="sdpa"`.
TensorFlow / `dlimp` are **not** needed for inference; `v8_policy.py` stubs them out.

---

## Setup

```bash
# 1. ROS 2 Humble + drivers (v4l2_camera, rslidar_sdk, hunter_base) built in your workspaces.

# 2. Python venv (ROS's PYTHONPATH still reaches rclpy from inside the venv)
python3.10 -m venv ~/v8env && source ~/v8env/bin/activate
pip install <jetson torch 2.8.0 wheel> <matching torchvision wheel>
pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 peft==0.11.1 \
            accelerate safetensors huggingface_hub sentencepiece einops draccus \
            diffusers json-numpy "numpy<2" pillow requests

# 3. OpenVLA-OFT (patched copy shipped here)
tar -xzf v8_vla_code/openvla-oft.tar.gz -C ~      # creates ~/openvla-oft/
pip install -e ~/openvla-oft --no-deps

# 4. Weights: put the 4 shards + action head into v8_vla/ (see next section).
#    The default path is /home/hunter/weights/v8_vla; on another machine pass
#    -p checkpoint:=/path/to/v8_vla to v8_drive_node.py.

# 5. Camera fps shim
gcc -shared -fPIC -O2 -o tools/camfps/v4l2_fps_shim.so tools/camfps/v4l2_fps_shim.c -ldl

# 6. Copy the runtime files where the launch commands expect them
cp corridor_nav/current/*.py corridor_nav/current/v8_env.sh ~/
mkdir -p ~/camfps && cp tools/camfps/* ~/camfps/
```

**Orin cuBLAS note:** if the system CUDA is an *sbsa* (server ARM) build, its cuBLAS has
no sm_87 kernels and `cublasCreate()` fails. `v8_env.sh` `LD_PRELOAD`s the Tegra
12.6.4.1 `libcublas.so.12` / `libcublasLt.so.12` from `~/cuda126_tegra_cublas/lib/`.
Edit that path for your machine. Install those libraries from NVIDIA's Tegra (arm64)
CUDA redistributables.

---

## Getting the weights

The checkpoint is **OpenVLA-7B fine-tuned with OFT on our corridor dataset**:
step 9000, LoRA r=32 merged, bf16, `ACTION_DIM=16`, `NUM_ACTIONS_CHUNK=1`.

| File | Size |
|---|---|
| `model-00001-of-00004.safetensors` | 4.9 GB |
| `model-00002-of-00004.safetensors` | 4.9 GB |
| `model-00003-of-00004.safetensors` | 4.9 GB |
| `model-00004-of-00004.safetensors` | 263 MB |
| `action_head--9000_checkpoint.pt` | 604 MB |

They are **not** in git. Contact the maintainer for access. <!-- TODO: add a Hugging Face / download link -->
Put them into `v8_vla/` next to the configs. `v8_vla/MANIFEST.txt` has sizes and SHA-256
sums to check them against.

Note: the first model load rewrites `config.json` and leaves `config.json.back.<time>`
files. This is expected (openvla_utils does it); `.gitignore` excludes them.

---

## Running

Short version below; the full checklist is in `corridor_nav/ROBOT_COMMANDS.md`.
Use one terminal per step, in this order. **Always `unset ROS_NAMESPACE`**: all topics are absolute.

```bash
# 1. Base
bash ~/Robot_teleop_ws/robot.bash            # or: ros2 launch hunter_base hunter_base.launch.py port_name:=can0
# 2. Camera at 10 fps
~/camfps/camera_10fps.sh                     # or /dev/video1
# 3. LiDAR driver (standalone; never run two drivers)
ros2 launch rslidar_sdk start.py
# 4. LiDAR perception — REQUIRED (without it the drive node holds still on purpose)
source ~/v8_env.sh && python3 -u ~/lidar_openings_node.py
# 5. Drive — DRY RUN unless --live is given
source ~/v8_env.sh && cd ~ && python3 -u v8_drive_node.py --live \
    --ros-args -p infer_hz:=2.0 -p creep_speed:=0.25

# Change the instruction at any time
ros2 topic pub --once /vla/prompt std_msgs/msg/String "{data: 'turn_left'}"
```

**Without the robot:** open `corridor_nav/viz/index.html` in a browser to replay the
recorded LiDAR session, or run `v8_drive_node.py` without `--live` (dry run, nothing is
sent to the wheels).

---

## Goal-conditioned driving (SignWay)

`signway_reasoner_node.py` is the only new node. It replaces the operator who used to type
`ros2 topic pub /vla/prompt ...`. The drive node, the LiDAR node and the safety gate are unchanged.

**One-time setup on the Jetson** (inside `~/v8env`):
```bash
pip install openai python-doctr        # docTR fetches its OCR weights on first run (needs internet)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # still the Jetson wheel + True?
cp corridor_nav/current/signway_secrets.example ~/.signway_secrets
nano ~/.signway_secrets && chmod 600 ~/.signway_secrets                         # paste the VLM API key
```
If pip tries to replace the Jetson torch wheel with a PyPI one, cancel and install docTR with
`--no-deps` plus its listed dependencies instead.

**Run**: steps 1–5 as usual, then in a new terminal:
```bash
source ~/v8_env.sh && source ~/.signway_secrets && unset ROS_NAMESPACE
python3 -u ~/weights/corridor_nav/current/signway_reasoner_node.py --ros-args \
    --params-file ~/weights/corridor_nav/current/signway_params.yaml -p goal:="room 6-217"
```

**A trip is a chain of signs**, as many junctions as it takes, until a sign says the robot has
arrived. For every sign:

1. **ARMED / ACQUIRE**: docTR (2 Hz) sees a plate that is about the goal. While it is still too
   small to read, the robot keeps approaching, and `/signway/speed_cap` **slows it down according
   to the size of the text**, so the answer can land before the junction:
   `v = (fy·letter_h/text_px − 1 m) / (measured VLM latency + 1 s)`, never below 0.15 m/s.
2. **Fire**: once `q = R·ℓ ≥ tau` and the plate is ≥ `min_plate_h` px, one VLM call is made with
   the cropped sign + goal (or the arrow is read straight from the OCR text: fast path, no call).
3. **Decision**: `turn_left` / `turn_right` is **held**: the prompt stays `straight` until
   `/lidar/openings_array` shows an opening on that side, so an early answer cannot turn the robot
   into the first opening before the junction (released anyway after `max_wait_m`).
   `straight` runs for `straight_done_m`.
4. **Done**: `/odom` net yaw ≥ `turn_done_deg` → prompt back to `straight`, gate re-armed for the
   next sign. Memory stores each sign *with where it was read*, so the same sign is never asked
   twice, but an identical-looking sign at the next junction is.
5. **Arrived**: when the sign marks the goal itself (the room's door plate, the label at the
   elevator doors), the VLM answers `stop` with "ARRIVED". The robot stops and the trip ends.
   Publish a new goal on `/signway/goal` to go on.

`v8_drive_node` follows the speed cap only while it is fresh (≤ 1.5 s) and only to lower its
speed; `-p use_speed_cap:=false` turns it off. The LiDAR safety gate still runs after it.

Logs per run in `~/signway_logs/<time>/`: `events.jsonl` (gate states, prompts),
`ledger.jsonl` (every VLM call), and `payloads/callNN/` (the exact prompt + crop sent).
Live status: `ros2 topic echo /signway/status`.

## Safety

- **Dry run is the default.** The `/cmd_vel` publisher is only created with `--live`.
- The LiDAR safety gate **fails closed**: no fresh clearance means no motion.
- The `stop` prompt was **never trained**. Use Ctrl-C or the hardware E-stop.
- Never disable the safety gate (`use_safety:=false`) except for the supervised VLA-only demo.
- The LiDAR height band (0.20–0.80 m above the floor) is blind to kerbs, cables, feet and overhangs.
- Test new code with the wheels off the ground first, and keep someone on the E-stop.

---

## Known limitations

See `corridor_nav/docs_current_20261106/HISTORY_AND_LESSONS.md` for the full list. Main ones:
the robot cannot reverse out of a dead end yet; junction arms behind corners are invisible
to the LiDAR until the robot is close; there is no map, so route choice is the operator's job.

## Acknowledgements

- [OpenVLA](https://github.com/openvla/openvla) and [OpenVLA-OFT](https://github.com/moojink/openvla-oft) (MIT): base model and fine-tuning recipe
- AgileX Robotics (`hunter_ros2`, `ugv_sdk`) and RoboSense (`rslidar_sdk`)

## License

<!-- TODO: choose a license (e.g. MIT) and add a LICENSE file. -->
