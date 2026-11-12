# SignWay on the Jetson: launcher, live viewer, Jev sign reader

Tooling for running and watching SignWay (goal-conditioned driving, see the main README and
`adaptive_reasoning/README.md`) on the robot. The reasoning code itself is in `adaptive_reasoning/`
and `corridor_nav/current/signway_reasoner_node.py`; nothing here changes its decisions.

## Where the files live on the robot

The scripts use these absolute paths, so copy them there (or edit the paths):

| In this repo | On the Jetson |
|---|---|
| `signway/signway_up.sh` | `~/signway_up.sh` |
| `signway/viz/*` | `~/signway_viz/` |
| `signway/sim/*` | `~/signway_viz/sim/` |
| `signway/jetson/*` | `~/signway_handoff/` |
| the repo root | `~/weights/` |
| `corridor_nav/current/lidar_openings_node.py` | `~/lidar_openings_node.py` |

API keys go in `~/.signway_secrets` (template: `corridor_nav/current/signway_secrets.example`),
never in the repo. `GEMINI_API_KEY` selects Gemini; `TYPESAFE_API_KEY` selects Jev.

## One-command launch

```bash
bash ~/signway_up.sh watch "Stairs"   # base, camera, LiDAR, openings, SignWay, viewer; no VLA, no prompts
bash ~/signway_up.sh dry   "Stairs"   # + VLA drive node WITHOUT --live; prompts are sent, wheels never move
bash ~/signway_up.sh live  "Stairs"   # + drive node WITH --live (type LIVE to confirm; E-stop ready)
bash ~/signway_up.sh status           # what is running + topic rates
bash ~/signway_up.sh stop
```

Each process gets its own gnome-terminal tab. Parts that are already running are skipped.
Env overrides: `SIGNWAY_VLM=gemini|jev`, `WAIT_FOR_OPENING=true` (hold a turn until the LiDAR sees
the opening; the default sends it at once), `DRIVE_ARGS="-p ..."` (extra drive-node params).

## Files

- `viz/signway_node_viz.py`: runs the real reasoner node with read-only hooks and publishes
  `/signway/debug` (JSON: OCR boxes, R/L/q, gate reason, the crop sent to the VLM).
- `viz/signway_viewer.py`: OpenCV window over `/signway/debug`, `/vla/prompt`, LiDAR openings and
  odom. Records to `~/signway_logs/viz_*.mp4`.
- `viz/jev_vlm.py`: TypeSafe Jev as the sign reader. It is a drop-in for the OpenAI-compatible VLM
  client and asks one choice question (turn_left / turn_right / straight / arrived /
  not_applicable). The criteria count an arrow that **heads a group of lines**, not only one printed
  beside the goal's line: building signs list several rooms under one arrow.
- `viz/jev_test.py`, `viz/make_test_frames.py`: one-call offline checks on saved frames.
- `sim/sim_balance.py`, `sim/sim_turn.py`: closed-loop sims of the drive node's balance planner
  turning into a side branch.
- `signway_v8_bridge.py`: older bridge from the MSI-hosted trip node (`/pepper_vla/decision`) to
  `/vla/prompt`. Superseded by `signway_reasoner_node.py`; kept for reference.
- `jetson/`: install notes, `cudnn_tegra_env.sh` and `run_nocudnn.py`. The apt cuDNN 9.25 on this
  Jetson is an SBSA build, so docTR is run with cuDNN off (~370 ms/frame).

## Drive-node changes needed for SignWay (`corridor_nav/current/v8_drive_node.py`)

- `/vla/prompt` and the image subscription moved to the control callback group. With inference
  (~600 ms) longer than the 500 ms timer period, they starved: prompts never arrived and the VLA
  re-ran on its first frame.
- `avoid_policy_priority:=intent` (default): during a turn prompt, the prompt sets the side and the
  balance planner rewards curving into that side's opening instead of overriding the VLA's turn.
  `turns`, `always` and `off` are also available.
- `/signway/speed_cap`: SignWay slows the robot while a sign is still too small to read.
