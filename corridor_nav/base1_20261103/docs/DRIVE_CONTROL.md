# Drive and Control (`v8_drive_node.py`)

The drive node has two loops.

- **Inference loop (~2 Hz):** newest camera frame + prompt → VLA → 8 waypoints.
- **Control loop (20 Hz):** waypoints → (v, ω), then centring, then safety, avoidance and
  creep, then ramps → `/cmd_vel`.

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
already in metres.

The image decoder is numpy only (no cv_bridge). It honours `msg.step`, so padded rows
can't shear the image.

## 2. Pure pursuit: waypoints → (v, ω)

```
pick the first waypoint at least lookahead_dist (0.6 m) from the robot
kappa = 2·y / (x² + y²)          curvature of the arc from the robot to that point
v     = speed (0.4 m/s, clipped to max_speed 0.6)
ω     = v · kappa                (clipped to ±max_omega 1.0 rad/s)
```

Guards: if the target is at or behind the axle (x ≤ 0.05) or the output is NaN, the
result is **(0, 0)**. An Ackermann base can't spin to recover, so stopping is safer
than a wild turn.

The model's implied speed (~0.9 m/s) is deliberately ignored. We use only the **shape**
of the path, at a fixed speed.

## 3. Corridor centring (PD)

Added to ω, only when the scene is `corridor`:

```
e_db = e with a ±0.05 m deadband removed
ω   += clip( −(0.45 · e_db  +  0.35 · de/dt),  ±0.35 rad/s )
```

- The **minus sign**: e > 0 means the robot is left of centre, so it must turn right
  (negative ω).
- The **D term** matters: P alone made the car-like base overshoot and weave from wall to wall.
- `de/dt` is computed in the 10 Hz LiDAR callback (not the 20 Hz loop, which would see
  each value twice) and smoothed with `0.7·old + 0.3·new`.
- It is switched off at junctions, because there `e` measures a branch mouth, not a wall.

## 4. The safety gate: LiDAR always wins

`_apply_safety` runs every control tick and can override the policy completely. It
checks these in order and returns at the first match:

```
 1. No /lidar/clearance yet, or older than 0.7 s ─────────► HOLD STILL (fail closed)
 2. d_near < emergency_distance (0.40 m) ─────────────────► HOLD STILL
 3. /lidar/bins fresh and avoidance on? ──yes──► ARC MODE   (sections 5–6)
                                          no ──► STRAIGHT-LANE MODE (section 7)
```

## 5. Arc avoidance (`_plan_avoid`)

The robot can only drive along circular arcs, so we test arcs directly.

1. Rebuild obstacle points from `/lidar/bins` (one point per bin centre).
2. Make **26 candidate arcs**: 25 curvatures evenly spread over ±0.6 1/m (about the
   1.6 m minimum turning radius), plus the **policy's own arc** `k_pol = ω / v`.
3. For each arc, compute its **free length**: how far the robot can drive along it
   before a band of ±0.625 m (half robot width 0.375 + 0.25 margin) touches an obstacle.
   Only the half-circle ahead counts, and anything ≥ 5 m counts as fully clear.

```
              obstacle
                 ■
     ╲    │    ╱          each arc: free length until the swept band hits something
      ╲   │   ╱           policy arc blocked at 1.4 m → look for a better one
       ╲  │  ╱
        ╲ │ ╱
         ▲  robot
```

4. **Decision:**
   - Policy arc free ≥ 3.0 m → **follow the policy** (no override).
   - Otherwise, score every arc as `free − 1.0 × |kappa − k_pol|`. This prefers clear arcs
     that are close to what the policy wanted.
   - Pick the best. It must beat the policy arc by at least **0.3 m**. If it doesn't, there's
     no better way round, and the result is tagged `blocked`.
   - If it does, steer onto it: **AVOID L** or **AVOID R**.
5. **Commitment:** once the robot picks a side, switching to the other side costs a 1.0 m
   score penalty. Without this, the robot dithered left/right in front of an obstacle
   and hit it. It releases only when the policy arc is free past 3.0 + 0.25 m.

## 6. Arc-mode speed gate (`_apply_arc_safety`)

Using the free length of the **chosen** arc:

| Free length on chosen arc | Action |
|---|---|
| ≥ 3.0 m (`slow_distance`) | full speed; ω = v·kappa if avoiding, else policy ω |
| 1.0 … 3.0 m | speed scaled linearly from 100 % down to 0 % at 1.0 m |
| < 1.0 m (`stop_distance`), **no** clearer arc exists | **hold still** (creeping would only move closer) |
| < 1.0 m, an AVOID arc exists | **creep along that arc** (section 8) |

## 7. Straight-lane fallback (no `/lidar/bins`)

Same idea, using `/lidar/clearance` (the 0.675 m rectangle) instead of arcs:

- `clear` < 1.0 m → creep toward the widest opening from `/lidar/openings_array`.
- 1.0 … 3.0 m → linear slow-down.
- ≥ 3.0 m → policy unchanged.

## 8. Creep recovery: a stop must be escapable

On an Ackermann robot, a hard stop is a trap. ω with v = 0 does nothing, and a
stationary robot never changes its own clearance. Before creep recovery, **every stop
was terminal** until someone pushed the robot.

So when blocked but the 0.40 m emergency lane is still clear, the robot **rolls slowly
(`creep_speed`) while steering out**:

- **Arc mode:** follow the best avoidance arc.
- **Fallback mode:** aim pure pursuit at the widest opening. That direction is
  **latched**, because at a real T-junction the two arms were 29° and 28° wide and
  re-picking each tick flipped left/right/left. It re-targets only if the latched
  opening moves more than 50°.
- **Hysteresis:** enter below 1.0 m, exit only above 1.25 m. Measured `d_front` jittered
  0.74 / 1.16 / 0.74 m between scans, which made recovery toggle every 0.1 s.
- **Gives up** (hard stop) after 8 s, if the emergency lane is breached, or if the best
  arc is free less than 0.5 m.

Run with `creep_speed:=0.25`. At 0.10 m/s the robot can't turn far enough within 8 s.

## 9. Slew-rate limits (ramps)

```
v ramps up at most 0.5 m/s², down at most 1.5 m/s²   (braking is 3× faster than accelerating)
ω changes at most 2.0 rad/s²
```

**Faults bypass the ramps.** Stale waypoints, no inference yet, or NaN all go
straight to (0, 0). Normal slow-downs and a `stop` prompt go through the ramps, so
they are smooth.

## 10. Reading the console

In dry run, every second:

```
[dry_run] v=+0.212 w=-0.140 | AVOID R 1.84m | clr=1.92 e=+0.10
```

The middle field (`_gate`) tells you **why** the command looks the way it does:

| Gate | Meaning |
|---|---|
| `policy` | following the VLA unchanged |
| `slow 2.10m` | slowing for something ahead |
| `AVOID L 1.84m` / `AVOID R …` | steering round an obstacle on the chosen arc |
| `blocked 1.40m` | policy path blocked, no clearly better arc (still slowing) |
| `blocked 0.80m no-way-round` | inside stop distance, nothing better: holding still |
| `CREEP arc k=+0.42 free 0.9m 2.3s` | creep recovery in arc mode |
| `CREEP 0.85m -> +62deg 1.1s` | creep recovery toward an opening (fallback mode) |
| `creep exhausted 8.0s` | recovery timed out: hard stop |
| `EMERG 0.35m` | something in the emergency lane: hard stop |
| `no-lidar` | safety data missing or stale: holding still |
| `stop` | `stop` prompt active |

Each inference also logs a line like
`[turn_left] 480.2ms wp0=(+0.110,+0.004) wp7=(+0.690,+0.180) -> v=+0.400 w=+0.290`.
