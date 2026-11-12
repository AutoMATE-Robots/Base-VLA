# Parameters (current defaults, 2026-11-06)

Override any of these with `--ros-args -p name:=value`. **Bold** = changed or new since base 1.

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
| `jpeg_roundtrip` | true | matches training preprocessing (costs ~32 ms) |

### Pure pursuit and limits
| Parameter | Default | Meaning |
|---|---|---|
| `speed` | 0.4 m/s | fixed cruise speed |
| `lookahead_dist` | 0.6 m | pursuit target distance along the predicted path |
| `max_speed` | 0.6 m/s | hard cap |
| `max_omega` | 1.0 rad/s | cap on v·κ (in angle mode this only bounds the policy's κ) |
| `max_curvature` | 0 (off) | optional curvature limit for pursuit and fallback creep (1/m) |
| `accel` / `decel` | 0.5 / 1.5 m/s² | speed ramps (braking faster) |
| `angular_accel` | 2.0 rad/s² | ω ramp, **legacy steer mode only** |
| `stale_after` | 1.0 s | waypoints older than this → hard zero speed |

### Steering output (new 2026-11-04)
| Parameter | Default | Meaning |
|---|---|---|
| **`steer_mode`** | `angle` | `angle`: send inner-wheel angle (rad) in `angular.z`, which is what `hunter_base` expects. `legacy`: send ω = v·κ (base 3 behaviour). |
| **`max_steer_central`** | 0.461 rad | central (bicycle) angle limit; inner wheel ≈ 0.574 rad (~33°) |
| **`steer_rate`** | 2.0 rad/s | slew limit on the wheel angle. Try 1.0 if it weaves. |
| **`wheelbase`** | 0.650 m | Hunter V2 |
| **`track`** | 0.605 m | Hunter V2 |

### Safety
| Parameter | Default | Meaning |
|---|---|---|
| `use_safety` | true | **leave on**. Off = no LiDAR gate and no planner (VLA-only demo *only*). Never use `false` as a fix for "won't move". |
| `stop_distance` | 1.00 m | free length below this: stop or creep |
| `slow_distance` | **4.00 m** | start slowing here (was 3.00 in base 1, 1.80 at Oct 22) |
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

Note: at the avoidance curvature cap (0.6 1/m) the turn rate is 0.6·v. At 0.10 m/s that
is 0.06 rad/s, so 45° takes ~13 s, longer than the 8 s timeout.

### Balance planner (replaces base 1's arc avoidance)
| Parameter | Default | Meaning |
|---|---|---|
| `use_avoidance` | true | needs `/lidar/bins` |
| **`self_filter_margin`** | 0.10 m | drop LiDAR returns within the footprint + this (self-hits at the rear-left) |
| **`avoid_max_offset`** | 1.5 m | furthest sideways shift considered |
| **`avoid_offset_step`** | 0.15 m | spacing of candidate shifts |
| **`avoid_swerve_kmax`** | 0.45 1/m | peak curvature allowed in a swerve (sets the shortest swerve length) |
| `avoid_max_curvature` | 0.6 1/m | hard cap on the planner's curvature (~25.5° inner wheel) |
| **`avoid_bases`** | [−0.3, −0.15, 0.15, 0.3] | base arcs tried besides the policy's and straight |
| **`avoid_lookahead`** | 1.0 m | rear-axle pure-pursuit distance on the chosen path |
| `avoid_margin` | **0.10 m** | added to the 0.425 m body circles **ahead** (was 0.25 band margin) |
| **`avoid_side_margin`** | 0.02 m | added to the body circles **beside** the robot (~7 cm real gap) |
| **`avoid_front_reach`** | 0.50 m | base_link → front bumper; the side margin applies up to here |
| **`avoid_margin_ramp`** | 1.50 m | distance past the bumper over which side margin → full margin |
| `avoid_horizon` | **6.0 m** | how far each path is traced (was 5.0) |
| **`avoid_stations`** | [0.75 … 3.25] every 0.5 m | where mean clearance is scored |
| **`avoid_clear_cap`** | 0.90 m | clearance beyond this counts the same, so the policy drives undisturbed |
| **`avoid_w_clear`** | 3.0 | weight on mean clearance |
| **`avoid_comfort`** | 0.25 m | body gap below which a path is penalised |
| **`avoid_w_tight`** | 10.0 | weight on that penalty: **this is what centres it in a gap** |
| **`avoid_w_free`** | 0.5 | weight on free length / horizon |
| **`avoid_w_off`** | 0.5 | cost per metre of sideways shift |
| **`avoid_w_head`** | 1.0 | cost per 1/m of bending away from the policy's curvature |
| **`avoid_w_curv`** | 0.5 | cost per 1/m of swerve curvature |
| **`avoid_w_side`** | 2.0 | cost of switching sides |
| **`avoid_release`** | 1.0 s | "no shift needed" this long before the side is let go |
| **`avoid_body_gap`** | 0.05 m | side gap below this = no way through |
| **`avoid_gap_slow`** | 0.30 m | side gap below which speed is reduced |
| **`avoid_gap_min_scale`** | 0.35 | slowest speed fraction while squeezing through |

Removed since base 1: `avoid_distance`, `avoid_n_arcs`, `avoid_dev_weight`,
`avoid_min_gain`, `avoid_switch_penalty` (base 1 arc avoidance), and base 2's
`avoid_need`, `avoid_settle`, `avoid_settle_omega`.

Hard-coded in the planner: body circles radius 0.425 m at −0.30/0/+0.30 m
(`ROBOT_COVER`, `BODY_DS`), rear axle 0.33 m behind `base_link` (`REAR_AXLE`), path
sample spacing 0.15 m, minimum useful free length `stop_distance + 0.5` = 1.5 m.

### Corridor centring (fallback only: used when the planner is off or bins are stale)
| Parameter | Default | Meaning |
|---|---|---|
| `use_centering` | true | |
| `center_gain` | 0.45 | P: rad/s per metre of error |
| `center_damp` | 0.35 | D: rad/s per m/s of error rate |
| `center_deadband` | 0.05 m | ignore tiny errors |
| `center_max` | 0.35 rad/s | cap on the centring term |
| `center_only_corridor` | true | off at junctions |

## `lidar_openings_node.py` (unchanged since base 1)

| Parameter | Default | Meaning |
|---|---|---|
| `cloud_topic` | `/rslidar_points` | input |
| `floor_z_robot` | −0.66 | measured floor height in base_link. Change it if the robot is re-mounted. |
| `band_min_agl` / `band_max_agl` | 0.20 / 0.80 m | height band **above floor** |
| `open_range` | 2.5 m | a bin farther than this is "open" (works 2.0–3.5 thanks to splitting) |
| `max_range` / `min_range` | 30 / 0.15 m | reject absurd returns and close self-hits |
| `robot_width` | 0.75 m | |
| `clearance_margin` | 0.30 m | per side for `/lidar/clearance` |
| `stable_scans` | 5 | scans a scene must hold before publishing |
| `min_run_bins` | 2 | ignore 1-bin specks |
| `min_sector_bins` | 5 | for the wide-run fallback |
| `split_runs` | true | prominence splitting on |
| `split_ratio` | 0.6 | dip must be < 0.6 × smaller flanking peak |
| `split_min_bins` | 6 | each side of a split must be this wide |
| `wide_run_deg` | 200° | above this a run is open space, not a passage |
| `empty_is_open` | false | treat no-return bins as open |
| `print_hz` | 2.0 | console rate |

## Common overrides

| Goal | Add after `--ros-args` |
|---|---|
| Normal run | `-p infer_hz:=2.0 -p creep_speed:=0.25` |
| Gentler wheel motion | `-p steer_rate:=1.0` |
| Let avoidance use nearly full lock | `-p avoid_max_curvature:=0.75 -p avoid_swerve_kmax:=0.6` |
| Old steering (base 3) | `-p steer_mode:=legacy` |
| VLA-only demo, LiDAR off (**supervised, slow, open space**) | `-p use_safety:=false -p use_centering:=false` |
