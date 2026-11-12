# History, Lessons, and Open Issues (up to 2026-11-06)

## Timeline

| Date | Version / backup | What changed |
|---|---|---|
| 2026-10-21 | first on-device runs | `v8_policy.py` works on the Jetson without TensorFlow. Camera format, `/cmd_vel` and namespace facts verified. |
| 2026-10-22 | **Baseline**, `nodes/` | VLA + pure pursuit, PD centring, straight-lane safety (stop 0.70 / slow 1.80 / emergency 0.40), ramps. LiDAR node with floor-aware band, prominence splitting, scene classification. |
| 2026-10-27 | `backup_before_avoid_20261028/` | **Creep recovery** (latched escape bearing, hysteresis, timeout). `/lidar/openings_array`. |
| 2026-10-28 | → **base 1** | **Arc avoidance** (26 arcs, swept band, side commitment). `/lidar/bins`. Stop/slow 1.0 / 3.0 m. `clearance_margin` 0.30 m. |
| 2026-11-03 | `base1_20261103/` | Base 1 frozen. |
| 2026-11-03 | → **base 2**, `base2_20261103/` | **Side-stop fix** (narrow margin beside the body, ramping up ahead), **gentle-arc** (take the gentlest arc clear 3.5 m, not the clearest), settle window after avoiding, slow 4.0 m, horizon 6 m. |
| 2026-11-03 | → **base 3**, `base3_20261104_before_steer/` | **Balance planner** replaces arc avoidance: runs every tick, swerve-then-carry-on paths, Ackermann 3-circle body, clearance + shortfall score. Self-hit filter. Centring only as fallback. Gap-based slow-down. Good live run 17:12: 6 min, 442 ms/inference, 0 holds. |
| 2026-11-04 | lean launch | Diagnosed over-current + stop-start: localization stack and 30 fps camera. **10 fps camera shim**, standalone rslidar. No code change. |
| 2026-11-04 | → **current** (`~/v8_drive_node.py`) | **Steering fix**: `angular.z` is the inner-wheel angle, not ω. `steer_mode`, `steer_rate`, wheels held while stopped. |

### Line counts (`v8_drive_node.py`)
| Oct 22 | Oct 27 | Base 1 | Base 2 | Base 3 | Current |
|---|---|---|---|---|---|
| 516 | 620 | 801 | 844 | 999 | 1055 |

`lidar_openings_node.py` has been 599 lines and unchanged since Oct 28.

---

## What each step fixed

### Oct 22 baseline: drive from the camera, stay safe with LiDAR
- The VLA has **no notion of obstacles**, so a LiDAR gate that overrides it was there from day one.
- **Centring started as P-only** and weaved from wall to wall. The D term (differentiated
  at the LiDAR's 10 Hz) fixed it.
- **The floor was in the wrong place.** We measured it at z_robot ≈ −0.66 and specified the
  band above floor.
- **Junction classification** went through several failures before prominence splitting
  (T read as `four_way`, corridor as `dead_end`, T as `right_branch`, phantom branches).

### Oct 27: creep recovery
**Problem:** every stop was terminal. Ackermann can't turn with v = 0.
**Fix:** creep slowly toward the widest opening. Latch the target; add enter/exit hysteresis.

### Oct 28: arc avoidance → base 1
- Stop/slow at 0.70 / 1.80 m was too late for a car-like turning radius → 1.0 / 3.0 m.
- A straight rectangle is the wrong shape while turning → test real arcs against LiDAR bins.
- Commit to a side, because re-deciding every tick made it dither and hit the obstacle.

### Nov 3: side-stop + gentle arc → base 2
**Problem 1: stopping beside things.** The full avoidance margin applied right beside the
body. An object 0.5–0.6 m to the side (a 12–22 cm real gap) read as "blocked 0.07 m
ahead" and the robot stopped next to things it was never going to touch.
**Fix:** narrow margin beside the body, growing to the full margin over 1.5 m past the bumper.

**Problem 2: hard swerves.** Picking the *clearest* arc swung it into turns it didn't
need, then hard back. **Fix:** take the gentlest arc that is clear far enough.

**What was still wrong:** avoidance switched on only when the policy's arc was blocked.
Its own override changed the arc it was judging, so it **flapped on/off every 50 ms**,
effectively never steered, and the robot **hit two objects**.

### Nov 3: the balance planner → base 3
**Goal:** go *between* obstacles, not just away from the nearest one. Developed against a
simulator first (that simulator lived in a session scratchpad and is gone; rebuild one
before retuning).

| Problem found | Fix |
|---|---|
| Switched avoidance flapped | Plan **every tick**; the policy path is just one candidate |
| Arcs that cleared an obstacle kept turning into the wall beside it | Candidates are **"swerve by dy, then carry on"** on several base arcs |
| Front corner clipped by 7 cm, then wedged | **Rear axle** follows the path; body = **3 circles** placed rigidly at the heading |
| Stopped for clutter just behind the rear corner | Ignore returns behind the rear axle |
| Held 11 cm off a box with 38 cm free on the other side | Penalise **every** sample closer than comfort (shortfall term), not just the tightest point |
| 1.25 m gap beside a box read as blocked | Ahead margin 0.25 → 0.10 m; comfort handled by the score |
| Dithered after a one-tick zero offset | Side released only after 1.0 s of "no shift" |
| Wall-centring fought the planner beside an object | Centring runs only when the planner is unavailable |
| Refused to move in a clear corridor | **Self-hit filter**: permanent returns inside the footprint at the rear-left |
| Plan cost ~25 ms on a shared CPU | Re-plan only on a new scan, side change, or policy change > 0.03 |

### Nov 4: over-current and stop-start (no code change)
**Symptoms:** "system throttled / over current" pop-ups, robot jerky and slow.
**Cause:** the localization stack (`full_system.launch.py`) plus a 30 fps camera pushed
the Jetson into constant over-current throttling (111 events in 15 s). That delayed
`/lidar/clearance` past 0.7 s, and the fail-closed gate held the robot 35–66 times per run.
An earlier bad run (Nov 3 14:59) was **two LiDAR drivers** at once.
**Fix:** standalone rslidar, no localization stack, camera at 10 fps via an `LD_PRELOAD`
shim (`v4l2_camera` ignores `framerate`). Result: 11 events in 60 s, 0 holds.

### Nov 4: steering fix → current
**Symptom:** "the wheels can't turn enough": it saw the object and wanted to avoid it,
but the wheels barely moved.
**Cause:** `hunter_base` uses `angular.z` **directly as the wheel angle**. The node sent
ω = v·κ, so the angle shrank with speed: a 0.45 1/m swerve got ~6° at 0.25 m/s instead
of ~19°, worst exactly when it slowed beside an obstacle.
**Fix:** convert κ → inner-wheel angle with Hunter geometry (same formula as the base
driver), clamp at the hardware limit, slew 2 rad/s, hold the wheels while stopped.
Verified in dry run only. Normal driving now steers ~1.6× more at 0.4 m/s.

---

## Lessons worth keeping

1. **Fail closed.** No LiDAR means no motion. And because it fails closed, anything that
   *delays* the LiDAR (CPU load, power throttling) shows up as stop-start motion.
2. **Measure, don't assume.** Floor height, camera encoding (rgb8), the real frame rate,
   un-namespaced `/cmd_vel`, and that `angular.z` is a **wheel angle** were all different
   from what we first assumed. Read the driver source.
3. **Ackermann changes everything.** Stopping isn't a safe resting state if you can't
   rotate out of it, the rear axle (not the centre) follows the path, and the front swings wide.
4. **Anything that picks a winner per tick needs latching or hysteresis:** escape bearing,
   avoidance side (with a timed release), recovery enter/exit, scene type.
5. **Don't switch a controller on based on a quantity it changes.** The base 2 avoidance
   judged the very arc it was overriding, and flapped. Running every tick fixed it.
6. **Score what you want, not a proxy.** "Maximise the tightest gap" didn't centre the
   robot; penalising every close sample did.
7. **Thresholds are traps. Prefer ratios and soft costs.** Hard margins only move the
   failure around; keep hard limits small and let the score handle comfort.
8. **Keep the model in its lane.** The VLA chooses the path shape. Speed, clearance and
   safety belong to geometry.
9. **Keep the compute lean.** One LiDAR driver, no unused stacks, camera at the lowest rate.

---

## Open issues

1. **Steering fix only dry-run tested** at the time it was written. Confirm in open space
   with someone on the E-stop; try `steer_rate:=1.0` if it weaves.
2. **Latency** is ~500–600 ms in the node (~410 ms standalone). Under 400 ms needs
   TensorRT or quantisation. The 50 ms preprocessing speed-up is available but not applied.
3. **The height band is blind** below 0.20 m and above 0.80 m AGL: kerbs, cables, feet,
   table edges, overhangs. The planner can't avoid what it can't see.
4. **Junctions are seen late.** Arms are hidden behind corridor corners until the last 1–2 m.
5. **Waypoints go slightly stale between inferences** (expressed in the frame of the image
   that produced them). Fine at 0.4 m/s; at higher speed transform them by odometry.
6. **The model's speed is discarded**, and `lookahead_dist` 0.6 against a ~0.7 m horizon
   nearly always picks the last waypoint.
7. **`stop` prompt is untrained.**
8. **Planner assumptions to re-check if hardware changes:** `base_link` at the body centre,
   0.98 × 0.75 m footprint, rear axle 0.33 m back, the rear-left self-hits.
9. **No planner simulator in the repo.** The one used to develop it was in a wiped
   scratchpad. Rebuild one before retuning the weights.
10. **LEFT-only or RIGHT-only scenes read `dead_end`**, and stale comments remain in
    `lidar_openings_node.py` (see LIDAR_PERCEPTION.md).
