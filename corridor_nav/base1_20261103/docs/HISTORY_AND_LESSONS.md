# History, Lessons, and Open Issues (up to base 1)

## Timeline

| Date | Version / backup | What changed |
|---|---|---|
| 2026-10-21 | first on-device runs | `v8_policy.py` works on the Jetson without TensorFlow. Camera format, `/cmd_vel` and namespace facts verified. |
| 2026-10-22 | **Baseline**, `~/weights/corridor_nav/nodes/` | VLA + pure pursuit, PD centring, straight-lane safety (stop 0.70 / slow 1.80 / emergency 0.40), ramps. LiDAR node with floor-aware band, prominence splitting, scene classification. |
| 2026-10-27 | `backup_before_avoid_20261028/` | **Creep recovery** added (latched escape bearing, hysteresis, timeout). LiDAR node publishes `/lidar/openings_array`. |
| 2026-10-28 | → becomes **base 1** | **Arc avoidance** (26 arcs, swept band, side commitment). LiDAR node publishes `/lidar/bins`. Stop/slow raised to **1.0 / 3.0 m**. `clearance_margin` raised 0.20 → **0.30 m**. |
| 2026-11-03 | `base1_20261103/` | Frozen and named "base 1" before the side-stop / gentle-arc changes (those are base 2). |

### Line counts
| | Oct 22 | Oct 27 | Base 1 |
|---|---|---|---|
| `v8_drive_node.py` | 516 | 620 | 801 |
| `lidar_openings_node.py` | 587 | 595 | 599 |

---

## What each step fixed

### Oct 22 baseline: drive from the camera, stay safe with LiDAR
- The VLA has **no notion of obstacles**, so a LiDAR gate that can override it was there from day one.
- **Centring started as P-only** and weaved from wall to wall. Adding the D term (and
  differentiating at the LiDAR's 10 Hz, not the 20 Hz loop) fixed it.
- **The floor was in the wrong place.** The first height band would have looked at chest
  height. We measured the floor at z_robot ≈ −0.66 and specified the band above floor.
- **Junction classification** went through several failures before prominence splitting:
  - A T read as `four_way` (branches fused through the facing wall).
  - A corridor read as `dead_end` (threshold above visible depth).
  - A T read as `right_branch` (untrimmed shoulder dragged the bearing).
  - A run claiming several sectors created phantom branches.

### Oct 27: creep recovery
**Problem:** every stop was terminal. The Hunter is Ackermann, so with v = 0 it can't
turn, and its clearance never improves. It sat there until pushed.

**Fix:** when blocked but the emergency lane is clear, creep slowly while steering
toward the widest opening. Two details made it work:
- **Latch the target.** Two near-equal arms (29° vs 28°) made the argmax flip every scan.
- **Hysteresis.** Clearance jittered 0.74 ↔ 1.16 m, so recovery needs a separate enter
  threshold and exit threshold.

### Oct 28: arc avoidance → base 1
**Problem 1:** stopping at 0.70 m and slowing at 1.80 m was too late. With a ~1.6 m
turning radius, by the time it reacted it couldn't steer round anything.
**Fix:** stop 1.0 m, slow 3.0 m.

**Problem 2:** the straight rectangle is the wrong shape while turning. It misses things
on the inside of a turn and over-reports on the outside. Also, "slow down" isn't
"go around".
**Fix:** test real driveable arcs against the LiDAR bins and pick the clearest one close
to the policy's intent. **Commit to a side**, because re-deciding every tick made it
dither and hit the obstacle.

**Problem 3:** with a 0.20 m margin, objects just off the lane edge were missed.
**Fix:** margin 0.30 m.

---

## Lessons worth keeping

1. **Fail closed.** No LiDAR means no motion. A missing safety channel is the one place
   where "probably fine" is unacceptable.
2. **Measure, don't assume.** The floor height, the camera encoding (rgb8, not UYVY),
   the real frame rate (~27 Hz, not 20) and the un-namespaced `/cmd_vel` were all
   different from what we first assumed.
3. **Ackermann changes everything.** Stopping is not a safe resting state if you can't
   rotate out of it. Recovery has to keep moving.
4. **Anything that picks a winner per tick needs latching or hysteresis:** escape bearing,
   avoidance side, recovery enter/exit, scene type (5 scans).
5. **Thresholds are traps. Prefer ratios.** Prominence splitting works whether the wall is
   at 2 m or 20 m. An absolute `open_range` only moves the failure around.
6. **Keep the model in its lane.** The VLA chooses the path shape. Speed and safety
   belong to geometry.

---

## Open issues in base 1

1. **Side clearance / stopping beside things.** Base 2 addresses this ("side-stop"
   and "gentle arc" fixes). Base 1 doesn't include them.
2. **The height band is blind** below 0.20 m and above 0.80 m AGL: kerbs, cables, feet,
   table edges, overhangs.
3. **Junctions are seen late.** Arms are hidden behind corridor corners until the last
   1–2 m. No LiDAR method sees round a corner. Earlier warning needs the camera or a map.
4. **Waypoints go slightly stale between inferences.** They're expressed in the frame of
   the image that produced them, and the robot moves in between. This is fine at 0.4 m/s.
   At higher speed, transform them by odometry (the node deliberately has no `/odom`).
5. **The model's speed is discarded** (it implies ~0.9 m/s, we drive 0.4). Also,
   `lookahead_dist` 0.6 against a ~0.7 m horizon nearly always picks the last waypoint.
   0.35–0.4 would let distance-based selection matter.
6. **`stop` prompt is untrained.**
7. **LEFT-only or RIGHT-only scenes read `dead_end`.** That's probably wrong at a side door.
8. **Stale comments** in `lidar_openings_node.py`: the docstring says splitting isn't
   implemented, and a `d_near` comment says "forward half-plane". The code is right,
   the comments are old.
9. **Is LiDAR junction detection needed at all?** If the VLA learned turn *timing* from
   the image (see `eval/eval_openloop.py` and `data/timing_test.png`), prompts can be
   scripted in advance. Junction type then becomes a veto, not a trigger.
