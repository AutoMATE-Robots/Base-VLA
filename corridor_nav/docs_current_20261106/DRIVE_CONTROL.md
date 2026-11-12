# Drive and Control (`v8_drive_node.py`)

The drive node has two loops.

- **Inference loop (~2 Hz):** newest camera frame + prompt → VLA → 8 waypoints.
- **Control loop (20 Hz):** waypoints → policy curvature → balance planner → safety
  gate (emergency / slow / stop / creep) → wheel angle and speed, with ramps → `/cmd_vel`.

```
 waypoints ─► pure pursuit ─► (v, k_pol)
                                 │
              LiDAR fresh? ──no──► HOLD STILL
                                 │
              emergency lane? ─yes─► HOLD STILL
                                 │
              /lidar/bins fresh? ──yes──► BALANCE PLANNER ─► arc speed gate ─┐
                                 │no                                         │
                                 └─► centring + straight-lane gate ──────────┤
                                                                             ▼
                                                 curvature k → inner-wheel angle, slewed
                                                 speed v → accel/decel ramps   ─► /cmd_vel
```

---

## 1. The VLA (v8)

| | |
|---|---|
| Model | OpenVLA-7B + OFT action head, fine-tuned on our "signway" data. Merged LoRA r32, bf16, step 9000 |
| Weights | `~/weights/v8_vla` (sha256 in `MANIFEST.txt`) |
| Wrapper | `~/v8_policy.py` → `load_policy(...).predict(pil_image, prompt)` |
| Prompts | `straight`, `turn_left`, `turn_right` (`stop` exists but was **never trained**) |
| Output | `(8, 2)` array of **cumulative** waypoints `(x_forward, y_left)` in metres, in the robot frame of the image. wp0 ≈ 0.1 m ahead, wp7 ≈ end of a ~0.8 s horizon |

The text prompt is built exactly as in training:
`"In: What action should the robot take to turn left?\nOut:"`.

The preprocessing reproduces training **without TensorFlow** (TF won't install on the
Jetson): JPEG round-trip (q95) → Lanczos resize to 224 → 90 % centre crop → resize back
to 224. Un-normalisation (BOUNDS_Q99) happens inside `predict_action`, so the output is
already in metres. The image decoder is numpy only (no cv_bridge) and honours `msg.step`.

### Where the time goes (measured 2026-11-04, 10 fps camera, MAXN, median of 10)

| Stage | Where | Time |
|---|---|---|
| JPEG round-trip + Lanczos resize of the full 1920×1280 frame | CPU | 59 ms |
| Processor (prompt tokens + tensors) | CPU | 9 ms |
| Vision encoder (DINOv2 + SigLIP) | GPU | 49 ms |
| Llama-7B + action head | GPU | ~308 ms |
| **Whole `predict()`, standalone** | | **~410 ms** (varies 400–550) |
| Inside the running drive node | | ~500–600 ms |

Attention already uses SDPA. Getting reliably under 400 ms would need TensorRT or
quantisation. An optional pre-shrink before the JPEG step saves ~50 ms (waypoints change
≤ 3 mm) but has **not** been applied.

## 2. Pure pursuit: waypoints → policy curvature

```
pick the first waypoint at least lookahead_dist (0.6 m) from the robot
kappa = 2·y / (x² + y²)          curvature of the arc from the robot to that point
v     = speed (0.4 m/s, clipped to max_speed 0.6)
ω     = v · kappa                (clipped to ±max_omega 1.0 rad/s)
k_pol = ω / v                    what the planner treats as "the policy's path"
```

Guards: if the target is at or behind the axle (x ≤ 0.05) or the output is NaN, the
result is **(0, 0)**. An Ackermann base can't spin to recover, so stopping is safer
than a wild turn.

The model's implied speed (~0.9 m/s) is deliberately ignored. We use only the **shape**
of the path, at a fixed speed.

## 3. Corridor centring (PD): fallback only

**Now only used when the balance planner is not available** (avoidance off, or
`/lidar/bins` missing/stale). The planner already centres the robot in the free space,
obstacles included. With both running, the wall-only centring read an obstacle as a
wall and fought the planner beside it.

When it does run (and the scene is `corridor`):

```
e_db = e with a ±0.05 m deadband removed
ω   += clip( −(0.45 · e_db  +  0.35 · de/dt),  ±0.35 rad/s )
```

- The **minus sign**: e > 0 means the robot is left of centre, so it must turn right.
- The **D term** stops the weave a P-only controller had on this car-like base.
- `de/dt` is computed in the 10 Hz LiDAR callback and smoothed `0.7·old + 0.3·new`.

## 4. The safety gate: LiDAR always wins

`_apply_safety` runs every control tick and can override the policy completely:

```
 1. No /lidar/clearance yet, or older than 0.7 s ───────────► HOLD STILL (fail closed)
 2. /lidar/min_dist < emergency_distance (0.40 m) ───────────► HOLD STILL
 3. /lidar/bins fresh and avoidance on? ──yes──► BALANCE PLANNER + arc gate (§5–6)
                                          no ──► STRAIGHT-LANE FALLBACK (§7)
```

## 5. The balance planner (`balance_plan`)

**Goal:** keep the robot's body in the middle of the free space. Between a wall and a
box, it should pass midway, not swerve away from the box and into the wall.

**It runs every tick**, not only when the path is blocked. The base 2 design switched on
when the policy's arc was blocked; its own steering then changed the arc it was judging,
so it flapped on/off every 50 ms, effectively never steered, and hit two objects.

### 5.1 Candidate paths: "swerve, then carry on"

```
   policy arc ──────────────╮         each candidate = a base arc
                             ╲         + a sideways shift dy that ramps in
   shift +0.45 ═════════════╗ ╲        (smoothstep) over a swerve length L
                ╱          ║  ╲       and then holds
   robot ▲════╱  ← swerve L →║
```

- **Base arcs:** the policy's curvature `k_pol`, straight (0), and `avoid_bases`
  (−0.3, −0.15, +0.15, +0.3 1/m). All clipped to ±`avoid_max_curvature` (0.6).
- **Shifts:** −1.5 … +1.5 m in 0.15 m steps (21 values).
- **Swerve lengths:** the shortest the steering allows (peak smoothstep curvature
  6·dy/L² ≤ `avoid_swerve_kmax` 0.45, at least 1 m), plus a gentler 1.6× version.
- That is up to 6 × 41 ≈ **246 candidates**, sampled every 0.15 m out to 6 m.

Plain arcs couldn't express "move over, then carry on": every arc that cleared an
obstacle kept turning into the wall beside it.

### 5.2 Ackermann body model

- The **rear axle** (0.33 m behind `base_link`) traces the path; the body pivots about
  it, so the front swings wide on a turn. Letting the centre follow the path missed that
  (simulation: front corner clipped by 7 cm, then wedged).
- The body is **three circles** of radius 0.425 m at −0.30, 0, +0.30 m along the heading.
  Together they cover the 0.98 × 0.75 m footprint, corners included. One circle at the
  centre missed the front corners by ~0.2 m.
- Obstacles **behind the rear axle** are ignored: going forward the rear axle follows
  the path and the body ahead of it does the sweeping. Counting them stopped the robot
  for clutter just behind its rear corner.

### 5.3 Free length (hard limit)

Walk each path until a body circle gets closer to an obstacle than its allowed radius:

| Where along the path | Allowed circle radius | Real gap to the body |
|---|---|---|
| Beside the robot (arc length < 0.5 m, the bumper) | 0.425 + **0.02** | ~7 cm |
| Ramps linearly over the next 1.5 m | → 0.425 + **0.10** | |
| Ahead (≥ 2.0 m) | 0.425 + 0.10 | ~15 cm |

The narrow side margin is the **side-stop fix**: with the full margin beside the body,
an object 0.5–0.6 m to the side read as "blocked 0.07 m ahead", and the robot stopped
next to things it was not going to touch. The modest ahead margin is deliberate too: at
0.25 m a 1.25 m gap beside a box read as blocked. Keeping a *comfortable* distance is
the score's job, not the hard limit's.

### 5.4 Score

```
score =  3.0 · clearance            mean body-to-obstacle distance at 0.75…3.25 m, capped at 0.90 m
      − 10.0 · shortfall            mean of max(0, 0.425 + 0.25 − d_body) over every sample   ← centres it in gaps
      +  0.5 · free / horizon
      −  0.5 · |shift|              stay near the policy's line
      −  1.0 · |k_base − k_pol|     don't bend away from the policy's curvature
      −  0.5 · swerve curvature     prefer gentle swerves
      −  2.0 · flipped side         no dithering in front of things
```

- **Capped clearance:** once everything is further than 0.9 m from the body, all
  candidates score the same and **the policy drives undisturbed**.
- **The shortfall term is what centres it.** Rewarding only the single tightest point
  didn't work: once alongside an object, that point is where the robot already is, the
  same for every path. It held 11 cm off a box with 38 cm free on the other side.
- **Side commitment:** a candidate whose position 2 m ahead is on the other side of the
  policy path from the committed side costs 2.0. The commitment is released only after
  `avoid_release` (1.0 s) of "no shift needed". A single tick of zero offset used to wipe
  it, and the next pick went the other way.

### 5.5 Selection

1. Drop paths blocked closer than **1.5 m** (`stop_distance + 0.5`), if any path isn't.
2. Of those, keep only paths whose tightest body gap (beyond the first 0.6 m) is
   ≥ `avoid_body_gap` (5 cm), if any fit. `fits` records whether any did.
3. Take the best score.
4. If the winner is the policy's own path (shift 0, base `k_pol`), use `k_pol` unchanged.
   Otherwise steer with **pure pursuit for the rear axle** on the chosen path at
   `avoid_lookahead` (1.0 m), clipped to ±0.6 1/m.

It returns the curvature `k`, the chosen path's `free` length, its tightest gaps
(`cmin` overall, `cnear` within 1.5 m), the sideways deviation `dev` 2 m ahead, and
whether it `leads_away` (free ≥ policy path free + 0.3 m).

## 6. Arc-mode speed gate (`_apply_arc_safety`)

| Situation | Action |
|---|---|
| No path fits through (`fits` false) | **hold still**: `gap too narrow (body clearance …)` |
| Chosen path free < 1.0 m, and it doesn't lead away from the obstacle | **hold still**: `blocked … no-way-round` |
| Chosen path free < 1.0 m, but it leads away | **creep along it** (§7) |
| Otherwise | steer with `k`; speed scaled as below |

Speed scale = the **smaller** of:

- **Distance:** linear from 100 % at `slow_distance` (**4.0 m**) to 0 % at `stop_distance` (1.0 m).
- **Side gap:** when the body passes within 0.30 m of something in the next 1.5 m,
  speed drops in proportion to the gap, but not below 35 %. It slows to squeeze through
  a gap rather than stopping.

## 7. Creep recovery: a stop must be escapable

On an Ackermann robot, a hard stop is a trap: steering with v = 0 does nothing, and a
stationary robot never changes its own clearance. So when blocked but the emergency lane
is clear, the robot **rolls slowly (`creep_speed`) while steering out**:

- **Arc mode:** follow the planner's chosen path. Gives up if its free length is below
  `emergency_distance + 0.10` (0.5 m).
- **Fallback mode:** aim at the widest opening from `/lidar/openings_array`. That
  direction is **latched** (two T-junction arms measured 29° and 28° and flipped every
  scan). It re-targets only if the latched opening moves more than 50°.
- **Hysteresis:** enter below 1.0 m, exit only above 1.25 m (clearance jittered
  0.74 / 1.16 / 0.74 m between scans).
- **Gives up** (hard stop) after 8 s, or if the emergency lane is breached.

Run with `creep_speed:=0.25`. At 0.10 m/s the robot can't turn far enough within 8 s.

### Straight-lane fallback (no `/lidar/bins`)

Same logic on `/lidar/clearance` (the 0.675 m half-width rectangle): below 1.0 m creep
toward the widest opening; 1.0 … 4.0 m linear slow-down; above 4.0 m policy unchanged.
Corridor centring (§3) is active in this mode.

## 8. Steering output: curvature → wheel angle (2026-11-04 fix)

`hunter_base` uses `angular.z` **directly as the inner-wheel steering angle**. So in
`steer_mode:=angle` (the default) the node sends an angle, computed from the curvature
the safety layer chose (the planner's `k`, the creep `k`, or the policy's own):

```
central angle  φ  = clip( atan(wheelbase · κ), ±0.461 rad )          wheelbase 0.650 m
inner angle    φi = atan( 2L·sin|φ| / (2L·cos|φ| − track·sin|φ|) ) · sign(φ)   track 0.605 m
(|φ| < 0.005 → 0)          same as ConvertCentralAngleToInner in hunter_messenger.hpp
```

| Curvature asked for | Old output at 0.4 m/s | Old at 0.25 m/s | **Now** |
|---|---|---|---|
| gentle, 0.3 1/m | 7° | 4° | **12°** |
| swerve, 0.45 1/m | 10° | 6° | **19°** |
| avoidance cap, 0.6 1/m | 14° | 9° | **25.5°** |
| hardware maximum | n/a | n/a | **~33°** |

- The wheel angle is **slewed** at `steer_rate` (2.0 rad/s), the same rate the old ω ramp had.
- While the speed target is zero (holding, `stop` prompt, faults), the wheels **stay where
  they are**, ready to pull away already turned.
- `-p steer_rate:=1.0` gives gentler wheel motion if it weaves.
- `-p avoid_max_curvature:=0.75 -p avoid_swerve_kmax:=0.6` lets avoidance use nearly the
  full 33°.
- `-p steer_mode:=legacy` sends ω = v·κ with the 2.0 rad/s² ramp, exactly as base 3.

## 9. Speed ramps

```
v ramps up at most 0.5 m/s², down at most 1.5 m/s²   (braking is 3× faster than accelerating)
```

**Faults bypass the ramps.** Stale waypoints, no inference yet, or NaN all go straight to
v = 0. Normal slow-downs and a `stop` prompt go through the ramps, so they are smooth.

## 10. Reading the console

In dry run, once per second:

```
[dry_run] v=+0.212 steer=-0.081 (-5deg) | BAL -0.30 k=-0.12 free 4.10m gap 0.22m x0.73 | clr=1.92 e=+0.10
```

(`steer=` is the inner-wheel angle in rad and degrees; in legacy mode it prints `w=`.)

The middle field (`_gate`) tells you **why** the command looks the way it does:

| Gate | Meaning |
|---|---|
| `policy k=+0.05 free 6.00m gap 0.90m` | planner agrees with the VLA; driving its path |
| `BAL +0.45 k=+0.21 free 5.2m gap 0.18m x0.60` | balancing: shifted 0.45 m left of the policy path; `x0.60` = speed scale |
| `gap too narrow (body clearance 0.03m)` | no path fits: holding still |
| `blocked 0.80m no-way-round` | inside stop distance, nothing better: holding still |
| `CREEP arc k=+0.42 free 0.9m 2.3s` | creep recovery along the planner's path |
| `blocked 0.42m no-free-arc` | every path blocked almost at once: holding still |
| `slow 2.10m` | fallback mode, slowing for something ahead |
| `CREEP 0.85m -> +62deg 1.1s` | fallback creep toward an opening |
| `blocked 0.80m no-opening` | fallback, no opening to creep toward |
| `creep exhausted 8.0s` | recovery timed out: hard stop |
| `EMERG 0.35m` | something in the emergency lane: hard stop |
| `no-lidar` | safety data missing or stale: holding still |
| `stop` | `stop` prompt active |
| `init` | no inference yet (usually: no camera frames) |

Other log lines:

- Each inference: `[turn_left] 480.2ms wp0=(+0.110,+0.004) wp7=(+0.690,+0.180) -> v=+0.400 w=+0.290`.
- Planner transitions: `balance SHIFT +0.45m: k=+0.21 (policy +0.02) free 5.20m policy-path free 1.40m centre-clear 0.61m`
  and `balance RELEASE: …`.
- `recovery ENTER / EXIT / ABORT / RETARGET`.
- On Ctrl-C: `latency over N frames: mean … p50 … p95 … -> X Hz sustained`.
- `grep -c "no clearance" ~/run.log` counts fail-closed holds (prints at most every 3 s,
  so it under-counts).
