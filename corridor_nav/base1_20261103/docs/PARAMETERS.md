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
