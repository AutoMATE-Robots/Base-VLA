# LiDAR Perception (`lidar_openings_node.py`)

> **Unchanged since base 1.** `~/lidar_openings_node.py` is byte-identical to
> `base1_20261103/lidar_openings_node.py`. What changed is how the drive node *uses* its
> outputs: `/lidar/bins` now feeds the balance planner every tick, and centring and the
> straight-lane gate are only the fallback when bins are missing. See the end of this file.

This node is **perception only**. It never publishes `/cmd_vel` or `/vla/prompt`. It
turns every 10 Hz scan into a few small, easy-to-use signals.

```
/rslidar_points
   │  1. move points into base_link            x += 0.247, z += 0.420
   │  2. keep a height band                    0.20–0.80 m above the FLOOR
   │  3. keep a range window                   0.15 m … 30 m
   ├──► 4. safety lanes (raw points)  ──────►  /lidar/clearance, /lidar/min_dist
   │  5. 360 bins of 1°, min range per bin ─►  /lidar/bins
   ├──► 6. corridor centring  ─────────────►  /lidar/center_error, d_left, d_right
   └──► 7. openings → scene  ──────────────►  /lidar/openings_array, /lidar/junction_type
```

---

## 1–2. The height band, and the floor trap

**`base_link` rides about 0.66 m above the floor.** We measured this from real scans,
with three independent checks:

1. No point in the cloud lies below z_robot = −0.662. A flat cluster sits right there
   (std 0.016 m). You can't see through a floor, so that is the floor.
2. The lowest beam points down at −24.8°. With the sensor 1.08 m up, it should first hit
   the floor at 1.08 / tan 24.8° = 2.34 m. That is exactly where the cluster starts.
3. Ceiling (+1.55) minus floor (−0.66) gives a 2.21 m room height, which is normal.

So the band is set as **height above floor**:
`z_robot = floor_z_robot + height_AGL`. The default 0.20–0.80 m AGL becomes
z_robot −0.46 … +0.14. A band of z_robot 0.20–0.80 would have looked at chest-to-head
height and missed every low obstacle.

> Trap: a raw z-histogram has a big hump at about +0.42. That hump is **walls**, not
> floor: most beams are near-horizontal and land on vertical surfaces.

## 4. Safety lanes (computed on raw points, not bins)

Two rectangles straight ahead of the robot. Each reports the **along-track** distance
(x) to the nearest point inside it:

| Topic | Half-width | Used for |
|---|---|---|
| `/lidar/clearance` (`d_front`) | 0.375 + 0.30 = **0.675 m** | slow-down / stop |
| `/lidar/min_dist` (`d_near`) | 0.375 + 0.05 = **0.425 m** | emergency cut |

If nothing is in a lane, it reports `max_range` (30 m). Both topics are published on
**every** scan, so silence is a fault the drive node can detect.

The emergency lane is deliberately **narrower** than the slow lane. Earlier versions got
this wrong in both directions:

- A whole-half-plane *radial* distance tripped on a wall merely **beside** the robot.
- A *wider* lane caught a wall running alongside at x ≈ 0 and would have stopped the
  robot forever.

## 5. Bins

Each point gets a bearing `atan2(y, x)`, which drops it into one of 360 one-degree bins.
Each bin keeps its **minimum** range. A bin with no return is `inf` ("unknown"). The
whole 360-value array goes out on `/lidar/bins`. The drive node rebuilds one obstacle
point per bin centre from it and feeds them to the **balance planner** every tick.

## 6. Corridor centring

```
d_left  = median of bins at bearings  +60° … +120°
d_right = median of bins at bearings −120° … −60°
e       = (d_right − d_left) / 2           e > 0  →  robot is LEFT of centre
```

The median makes it robust to a doorframe or a chair leg. This is only meaningful
**between two parallel walls**. At a junction, one side sees down a branch and `e` can
reach ±1.5 m. That's why the drive node only applies centring while the scene is
`corridor`, and now **only when the balance planner is unavailable** (no fresh bins).

## 7. Openings and scene type

An **opening** is a direction the robot could drive into.

**Step A: candidate runs.** Bins with range > `open_range` (2.5 m) are "open".
Contiguous open bins (wrapping around ±180°) form runs. Runs of 1 bin are dropped.

**Step B: split by prominence.** This fixes the hardest problem. A single
absolute threshold can't tell "corridor continues" from "a wall 2.6 m ahead". At a real
T-junction, the left arm, the facing wall and the right arm fused into one 160° run,
which was classified `four_way`.

The fix is **threshold-free**. Inside a run, find the deepest interior dip. If it's much
lower than the peaks on both sides, cut the run there:

```
split if   dip < split_ratio × min(left_peak, right_peak)        split_ratio = 0.6
```

At the real T: peaks 10.1 m (right arm) and 8.3 m (left arm), dip 2.64 m (the facing
wall). 2.64 < 0.6 × 8.3, so the run splits into two arms. A straight corridor has one
peak and is left whole. Each split piece is then **trimmed** to the part that stays
above 0.6 × its own peak. Without the trim, the shoulder running into the wall dragged
the left arm's bearing from +52° to +24° and turned a T into `right_branch`.

**Step C: does the robot fit?** At depth `d`, the opening must span at least
`2·atan2(0.675, d)` degrees. Narrow slits are discarded.

**Step D: label each opening by its centre bearing.**

| Bearing | Label |
|---|---|
| \|b\| ≤ 25° | FORWARD |
| +25° … +155° | LEFT |
| −25° … −155° | RIGHT |
| otherwise | BEHIND |

**One run means one passage, which gets one label.** Letting a run claim every sector it
touched created phantom branches. The exception is a run wider than 200°, which is open
space, not a passage. For that case each sector is claimed only if its own share is
robot-wide.

**Step E: scene table.**

| Labels present | Scene |
|---|---|
| FORWARD | `corridor` |
| FORWARD + LEFT | `left_branch` |
| FORWARD + RIGHT | `right_branch` |
| LEFT + RIGHT (no FORWARD) | `T_junction` |
| all three | `four_way` |
| nothing (or only LEFT, or only RIGHT) | `dead_end` |

A scene is published on `/lidar/junction_type` only after it holds for **5 consecutive
scans** (0.5 s). This stops it from flickering.

Openings also go out numerically on `/lidar/openings_array` as flat
`[bearing°, width°, depth_m, …]` triples. The fallback creep recovery in the drive node
(no bins) uses them.

## Console line

```
corridor     | e=+0.12m dL=1.4 dR=1.6 | clr=4.20 | openings: fwd(+2deg, w=31deg, d=6.3m)
```

This shows the published scene, centring error, side distances, forward clearance and
the openings.

## Known quirks (still present)

- The module docstring still says discontinuity-based splitting is "not implemented".
  It is out of date: prominence splitting **is** implemented and on by default.
- A comment above `d_front`/`d_near` still describes `d_near` as the "forward half-plane".
  The code uses the narrow 0.425 m lane described above.
- LEFT-only or RIGHT-only falls through to `dead_end`. That's probably wrong at a side door.
- `empty_is_open` is false. The driver drops no-return points, so a truly open doorway
  can look "unknown" rather than open.

## How the drive node uses these outputs now

| Topic | Used for |
|---|---|
| `/lidar/clearance` | **freshness check** (older than 0.7 s → hold still) and the straight-lane fallback gate |
| `/lidar/min_dist` | emergency stop below 0.40 m, in every mode |
| `/lidar/bins` | balance planner obstacle points (main path) |
| `/lidar/center_error` | PD centring, fallback mode only |
| `/lidar/junction_type` | gates fallback centring to `corridor` scenes |
| `/lidar/openings_array` | fallback creep target |

### Self-hits are filtered in the drive node, not here

Measured 2026-11-03: every scan holds returns **inside the robot's own footprint** at the
rear-left, e.g. (−0.47, +0.10) and (−0.30, +0.31) in `base_link`. That is the robot
itself or something mounted on it. The forward-only lanes never saw them, but the planner
checks beside the body too, read them as touching, and refused to move in a clear
corridor. The drive node therefore drops bin points within the 0.98 × 0.75 m footprint
plus `self_filter_margin` (0.10 m) before planning. If something is re-mounted on the
robot, re-check this.

### Height band reminder

The planner only sees what is inside the **0.20–0.80 m above floor** band. Anything lower
(kerbs, cables, feet) or higher (table tops, overhangs) is invisible to it.
