#!/usr/bin/env python3
"""
lidar_openings_node.py -- corridor centering + opening detection from /rslidar_points.

PERCEPTION ONLY. Publishes diagnostics; never touches /cmd_vel or /vla/prompt.

    /rslidar_points (PointCloud2, 10 Hz, ~236k pts)
        -> height band -> 1-deg bearing bins (min range per bin)
        -> /lidar/center_error   std_msgs/Float32   (+ = robot is LEFT of centre)
           /lidar/d_left,/lidar/d_right  std_msgs/Float32
           /lidar/openings      visualization_msgs/MarkerArray  (arrow + text per opening)
           /lidar/openings_text std_msgs/String
           /lidar/junction_type std_msgs/String     (hysteresis: N consecutive scans)

Run:
    source /opt/ros/humble/setup.bash && source ~/v8env/bin/activate && unset ROS_NAMESPACE
    python3 -u ~/lidar_openings_node.py

HEIGHT BAND -- measured, not assumed (2026-10-22, 5 scans, 1.19M points)
  base_link is NOT at floor level. Evidence:
    * Nothing anywhere in the cloud lies below z_robot = -0.662 m, and there is a
      plane-like cluster there (n=851, z std 0.016 m, spanning r = 2.0..2.9 m).
      You cannot see through a floor, so that plane IS the floor.
    * Beam geometry agrees: the lowest beam is -24.8 deg, so a floor 1.082 m below
      the sensor is first struck at r = 1.082/tan(24.8) = 2.34 m -- exactly where
      that cluster begins and exactly where lowest-beam returns stop (max r = 2.3 m).
    * Cross-check: z_max +1.552 (ceiling) minus z_min -0.662 = 2.21 m room height,
      which is normal. Were base_link on the floor, the room would be 1.55 m tall.
  So the floor is at z_robot ~= -0.66, i.e. base_link rides ~0.66 m above it and the
  sensor ~1.08 m above it. The originally-requested band (z_robot 0.20..0.80) would
  therefore have sampled 0.86..1.46 m above the floor -- chest-to-head height, blind
  to every low obstacle. The band is instead specified ABOVE FLOOR and converted:
      z_robot = floor_z_robot + height_above_floor
  Defaults 0.20..0.80 m AGL => z_robot -0.46..+0.14. Override floor_z_robot if the
  robot is re-mounted or the TF changes.

  NOTE the broad hump at z_robot ~ +0.42 in a raw z-histogram is NOT the floor: it is
  the beam-elevation density (most beams near horizontal) painted onto vertical walls.
  Looking for the "biggest pile of points" finds the walls, not the ground.

THE WEAK POINT: open_range is an ABSOLUTE threshold, so any wall farther away than it
reads as an opening. Measured at a real T-junction (2026-10-22): facing wall at 2.64 m,
open_range 2.5 -> the wall counted as open and the left branch, the wall and the right
branch fused into a single 160-deg "forward" run, classified four_way. No per-sector rule
can repair that, because by then the three are one contiguous run. Raising open_range fixes
this instance but only moves the failure to junctions with a more distant facing wall.
The robust fix is to detect branches by range DISCONTINUITY rather than absolute range: in
that same scan the right branch is bounded by a 10.13 -> 1.26 m step and the left by
8.30 -> 1.20 m, and those corner steps are obvious at any absolute distance. Not implemented
here; worth doing before this drives anything.

EMPTY BINS. A bin with no return is treated as UNKNOWN and breaks an opening run, per
spec. Be aware this driver already discards no-return points (is_dense=True, zero
all-zero points, 242377 of 242377 finite), so a genuinely open doorway often yields NO
returns at all -- i.e. empty bins are in practice evidence OF openness here. Set
empty_is_open:=true to treat them as open instead; left false to match the spec.
"""
import math
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA, Float32, Float32MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

# TF base_link -> rslidar: translation [0.247, 0.000, 0.420], rotation identity.
TX, TZ = 0.247, 0.420
N_BINS = 360                      # 1 degree
BIN_DEG = 360.0 / N_BINS


class LidarOpenings(Node):
    def __init__(self):
        super().__init__("lidar_openings")
        p = self.declare_parameter
        p("cloud_topic", "/rslidar_points")      # absolute: ROS_NAMESPACE=robot1 is exported
        p("floor_z_robot", -0.66)                # measured; see module docstring
        p("band_min_agl", 0.20)                  # metres above floor
        p("band_max_agl", 0.80)
        p("open_range", 2.5)                     # m; bins beyond this count as open.
        # History, because the value is not arbitrary. With a plain absolute threshold it
        # had to sit ABOVE a junction's facing wall (measured 2.64 m) but BELOW the
        # corridor's visible depth (measured 4.5 m) -- a window of only ~3.0..4.0 m. 2.5
        # merged the branches into one 160-deg run (four_way for a plain T); 4.0 made an
        # ordinary corridor read dead_end as the forward bins dipped under it. Prominence
        # splitting removed that squeeze: T_junction now holds from 2.0 to 5.0 and corridor
        # from 2.0 to 3.5, so 2.5 sits mid-window and well clear of the visible-depth
        # ceiling that caused the dead_end. See _split_by_prominence.
        p("max_range", 30.0)                     # m; reject absurd returns
        p("min_range", 0.15)                     # m; reject self-hits
        p("robot_width", 0.75)
        p("clearance_margin", 0.20)              # per side
        p("stable_scans", 5)                     # hysteresis before publishing a change (0.5 s @ 10 Hz)
        p("min_run_bins", 2)                     # ignore 1-bin specks
        p("min_sector_bins", 5)                  # absolute floor; see _openings_from_mins
        p("split_runs", True)                    # split an opening at a deep interior dip
        p("split_ratio", 0.6)                    # dip counts if < ratio * smaller flanking peak
        p("split_min_bins", 6)                   # each side of a split must be this wide
        p("wide_run_deg", 200.0)                 # above this a run is open space, not a passage.
        # Swept on the real T-junction scan: at 150 a merged 160-deg run (robot still too
        # far back for the facing wall to register) tripped the exception and reported
        # four_way; at 200 it reports corridor, which is the honest "cannot tell yet",
        # while a genuine 350-deg open room still reports four_way. Above ~300 the merged
        # cases decay into confident single-branch answers again, so do not raise it far.
        p("empty_is_open", False)                # see docstring
        p("print_hz", 2.0)
        g = lambda n: self.get_parameter(n).value

        self.floor_z = float(g("floor_z_robot"))
        self.z_lo = self.floor_z + float(g("band_min_agl"))
        self.z_hi = self.floor_z + float(g("band_max_agl"))
        self.open_range = float(g("open_range"))
        self.max_range = float(g("max_range"))
        self.min_range = float(g("min_range"))
        self.half_width = float(g("robot_width")) / 2.0 + float(g("clearance_margin"))
        self.stable_scans = max(1, int(g("stable_scans")))
        self.min_run_bins = max(1, int(g("min_run_bins")))
        self.min_sector_bins = max(1, int(g("min_sector_bins")))
        self.split_runs = bool(g("split_runs"))
        self.split_ratio = float(g("split_ratio"))
        self.split_min_bins = max(2, int(g("split_min_bins")))
        self.wide_run_deg = float(g("wide_run_deg"))
        self.empty_is_open = bool(g("empty_is_open"))
        self.print_period = 1.0 / max(0.1, float(g("print_hz")))

        # bin centres in degrees, -180..+179 (+ = left)
        self.bin_deg = (np.arange(N_BINS) * BIN_DEG) - 180.0 + BIN_DEG / 2.0

        self._cand = None            # candidate classification awaiting confirmation
        self._cand_n = 0
        self._published = None       # last published classification
        self._last_print = 0.0
        self._warned = set()         # log-once keys
        self._n_scans = 0
        self._d_front = float('nan')
        self._d_near = float('nan')

        self.pub_err = self.create_publisher(Float32, "/lidar/center_error", 10)
        self.pub_dl = self.create_publisher(Float32, "/lidar/d_left", 10)
        self.pub_dr = self.create_publisher(Float32, "/lidar/d_right", 10)
        self.pub_mark = self.create_publisher(MarkerArray, "/lidar/openings", 10)
        self.pub_text = self.create_publisher(String, "/lidar/openings_text", 10)
        self.pub_junc = self.create_publisher(String, "/lidar/junction_type", 10)
        # Safety channels, consumed by the drive node. Published EVERY scan (10 Hz) so a
        # consumer can treat silence as a fault and stop.
        self.pub_clear = self.create_publisher(Float32, "/lidar/clearance", 10)
        self.pub_near = self.create_publisher(Float32, "/lidar/min_dist", 10)
        # Flat [bearing_deg, width_deg, depth_m] triples, one per surviving opening, so a
        # controller can consume openings numerically instead of parsing the human string.
        # Empty array = no openings this scan (a real state, distinct from no message).
        self.pub_ops = self.create_publisher(Float32MultiArray, "/lidar/openings_array", 10)

        self.create_subscription(PointCloud2, g("cloud_topic"), self._on_cloud,
                                 QoSPresetProfiles.SENSOR_DATA.value)
        self.get_logger().info(
            f"lidar_openings | band {g('band_min_agl'):.2f}..{g('band_max_agl'):.2f} m AGL "
            f"= z_robot {self.z_lo:+.2f}..{self.z_hi:+.2f} (floor {self.floor_z:+.2f}) | "
            f"open_range {self.open_range} m | fit half-width {self.half_width:.3f} m | "
            f"stable_scans {self.stable_scans} | empty_is_open {self.empty_is_open}")

    # ---------------- helpers ----------------
    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().warn(msg)

    def _bin_min_ranges(self, rng, bearing_deg):
        """-> (N_BINS,) min range per bin, np.inf where no return. Vectorised."""
        b = ((bearing_deg + 180.0) / BIN_DEG).astype(np.int32)
        np.clip(b, 0, N_BINS - 1, out=b)
        mins = np.full(N_BINS, np.inf, dtype=np.float64)
        # sort by range so the FIRST occurrence of each bin is its minimum; cheaper and
        # far more predictable than np.minimum.at on 236k points.
        order = np.argsort(rng, kind="stable")
        bs, rs = b[order], rng[order]
        first = np.unique(bs, return_index=True)[1]
        mins[bs[first]] = rs[first]
        return mins

    @staticmethod
    def _circular_runs(mask, min_len):
        """Maximal runs of True in a circular boolean array -> [(start, length), ...]."""
        n = len(mask)
        if not mask.any():
            return []
        if mask.all():
            return [(0, n)]
        # rotate so index 0 begins a run, making runs contiguous (no wrap special-case)
        start0 = int(np.argmax((~np.roll(mask, 1)) & mask))
        rot = np.roll(mask, -start0)
        runs, i = [], 0
        while i < n:
            if rot[i]:
                j = i
                while j < n and rot[j]:
                    j += 1
                if j - i >= min_len:
                    runs.append(((i + start0) % n, j - i))
                i = j
            else:
                i += 1
        return runs

    def _classify_one(self, bearing):
        a = abs(bearing)
        if a <= 25.0:
            return "FORWARD"
        if 25.0 < bearing <= 155.0:
            return "LEFT"
        if -155.0 <= bearing < -25.0:
            return "RIGHT"
        return "BEHIND"

    @staticmethod
    def _scene(labels):
        f, l, r = ("FORWARD" in labels), ("LEFT" in labels), ("RIGHT" in labels)
        if f and l and r:
            return "four_way"
        if f and l:
            return "left_branch"
        if f and r:
            return "right_branch"
        if l and r:
            return "T_junction"
        if f:
            return "corridor"
        return "dead_end"

    # ---------------- main callback ----------------
    def _on_cloud(self, msg):
        try:
            self._process(msg)
        except Exception as e:                      # never throw inside a callback
            self._warn_once(f"cb:{type(e).__name__}",
                            f"scan processing failed ({type(e).__name__}: {e}); skipping")

    def _process(self, msg):
        if msg.point_step < 12 or not msg.data:
            self._warn_once("malformed", f"malformed cloud (point_step={msg.point_step}); skipping")
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        stride = msg.point_step
        npts = buf.size // stride
        if npts == 0:
            self._warn_once("empty", "empty cloud; skipping")
            return
        # Read x,y,z honouring point_step rather than assuming a tight (N,4) float32 block,
        # so a driver that adds fields (ring/timestamp) does not shear the parse.
        pts = buf[: npts * stride].reshape(npts, stride)[:, :12].copy().view(np.float32)

        x = pts[:, 0] + TX
        y = pts[:, 1]
        z = pts[:, 2] + TZ

        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        good &= (z >= self.z_lo) & (z <= self.z_hi)
        if not good.any():
            self._warn_once("band_empty", "no points survived the height band; skipping")
            return
        x, y = x[good], y[good]
        rng = np.hypot(x, y)
        ok = (rng >= self.min_range) & (rng <= self.max_range)
        if not ok.any():
            self._warn_once("range_empty", "no points in range window; skipping")
            return
        x, y, rng = x[ok], y[ok], rng[ok]

        # ---- safety: clearance inside the robot's own swept width, and raw proximity ----
        # d_front: nearest return inside a robot-wide corridor straight ahead -- what limits
        # forward speed. d_near: nearest return anywhere in the forward half-plane, a blunt
        # proximity guard that also catches obstacles the swept rectangle misses while
        # turning. Both are deliberately computed on raw points, not on the binned minima,
        # so nothing is lost to bin quantisation.
        inlane = (np.abs(y) <= self.half_width) & (x > 0.0)
        d_front = float(np.min(x[inlane])) if np.any(inlane) else float(self.max_range)
        # d_near gates the EMERGENCY cut, so it must answer "is something about to be hit",
        # not "is something near". Two earlier versions got this wrong:
        #   1. the whole forward half-plane at RADIAL distance -- a wall merely beside the
        #      robot gave rng ~0.39 and tripped the 0.40 m cut. Seen live as stopping for
        #      objects plainly not in the way.
        #   2. a WIDENED lane at along-track distance -- measured here, the points it
        #      caught sat at |y| 0.60..0.71, i.e. a wall running alongside, reported as
        #      x ~= 0.0. That would have stopped the robot permanently.
        # The lane must therefore be NARROWER than the slow-down lane, not wider: only a
        # genuine body-collision risk belongs on the emergency path. Slowing keeps the full
        # 0.20 m margin; the emergency cut uses the robot's own width plus 0.05 m.
        emerg_half = 0.5 * float(self.get_parameter("robot_width").value) + 0.05
        near_lane = (np.abs(y) <= emerg_half) & (x > 0.0)
        d_near = float(np.min(x[near_lane])) if np.any(near_lane) else float(self.max_range)
        self.pub_clear.publish(Float32(data=d_front))
        self.pub_near.publish(Float32(data=d_near))
        self._d_front, self._d_near = d_front, d_near

        bearing = np.degrees(np.arctan2(y, x))
        mins = self._bin_min_ranges(rng, bearing)
        self._n_scans += 1

        # ---- corridor centering ----
        d_left = self._sector_median(mins, 60.0, 120.0)
        d_right = self._sector_median(mins, -120.0, -60.0)
        if d_left is not None and d_right is not None:
            e = (d_right - d_left) / 2.0           # + => robot is LEFT of centre
        else:
            e = float("nan")
        self.pub_err.publish(Float32(data=float(e)))
        self.pub_dl.publish(Float32(data=float(d_left if d_left is not None else float("nan"))))
        self.pub_dr.publish(Float32(data=float(d_right if d_right is not None else float("nan"))))

        # ---- openings ----
        has = np.isfinite(mins)
        openmask = (mins > self.open_range) if not self.empty_is_open else \
                   (~has | (mins > self.open_range))
        openmask &= (has | self.empty_is_open)     # empty = unknown unless opted in

        # Diagnose an open_range set above what the sensor can actually SEE. Measured
        # 2026-10-22: in a corridor whose furthest forward return was 4.47 m, running with
        # open_range:=5.0 left no bin anywhere above threshold, so the node reported
        # dead_end for an ordinary corridor. open_range must sit BELOW the visible depth
        # (and above the junction's facing wall) -- here a window of roughly 3.0..4.0 m.
        # Judge this on the FORWARD/LEFT/RIGHT arc only: a long return BEHIND (back down
        # the corridor you came from) is irrelevant to where you can go, and would
        # otherwise mask the warning -- which is exactly what happened on first test.
        ahead = mins[np.abs(self.bin_deg) <= 155.0]
        finite = ahead[np.isfinite(ahead)]
        # Not "above the ceiling" but "too close to it": with a visible depth of 5.04 m,
        # open_range=5.0 left just 5 qualifying bins -- too few to form a passable run --
        # so the corridor still read dead_end while the threshold was technically under
        # the max. Warn once open_range reaches 85% of what can be seen ahead.
        if finite.size and self.open_range >= 0.85 * finite.max():
            self._warn_once(
                "open_range_too_high",
                f"open_range={self.open_range:.1f} m is close to or above the furthest "
                f"return ahead of you ({finite.max():.1f} m), so little or nothing can "
                f"register as open and the scene will tend to read dead_end. open_range "
                f"must sit BELOW the visible depth (and above a junction's facing wall).")

        openings = self._openings_from_mins(mins, openmask)
        self._publish_openings(openings, msg.header.stamp)

        # ---- classification with hysteresis ----
        # Union of the sectors the runs actually COVER, not the single centre-bearing
        # label -- see _openings_from_mins for why the centre bearing cannot carry this.
        covered = set()
        for o in openings:
            covered |= o["labels"]
        scene = self._scene(covered)
        if scene == self._cand:
            self._cand_n += 1
        else:
            self._cand, self._cand_n = scene, 1
        if self._cand_n >= self.stable_scans and self._cand != self._published:
            self._published = self._cand
            self.pub_junc.publish(String(data=self._published))
            self.get_logger().info(f"junction_type -> {self._published}")

        self._console(e, d_left, d_right, openings)

    def _trim_to_peak(self, prof, a, b):
        """Shrink [a,b) to the contiguous span around its peak that stays deep. -> (a, b)."""
        seg = prof[a:b]
        if seg.size == 0:
            return a, b
        peak = float(np.max(seg))
        floor = self.split_ratio * peak
        k = int(np.argmax(seg))
        lo = k
        while lo > 0 and seg[lo - 1] >= floor:
            lo -= 1
        hi = k
        while hi < seg.size - 1 and seg[hi + 1] >= floor:
            hi += 1
        return a + lo, a + hi + 1

    def _split_by_prominence(self, vals):
        """Split one run's range profile at deep interior dips. -> list of (start, stop) slices.

        An absolute open_range cannot tell "corridor continues" from "wall 2.6 m ahead",
        and tuning it is a trap: it has to sit above a junction's facing wall but below the
        corridor's visible depth, a window measured at only ~3.0-4.0 m in this building.
        Set it too high and an ordinary corridor reads dead_end.

        A dip's PROMINENCE is threshold-free. At the real T-junction the merged run peaked
        at 10.1 m (right arm) and 8.3 m (left arm) with a 2.64 m trough between them -- the
        facing wall. 2.64 < 0.6 * 8.3, so the run splits into two arms and classifies
        T_junction. A straight corridor has a single peak and no qualifying dip, so it is
        left whole. The ratio holds whether the wall is at 2 m or 20 m.
        """
        out = []
        stack = [(0, len(vals))]
        while stack:
            a, b = stack.pop()
            n = b - a
            if n < 2 * self.split_min_bins:
                out.append((a, b))
                continue
            seg = vals[a:b]
            # only consider interior dips that leave both sides wide enough to be passages
            lo, hi = self.split_min_bins, n - self.split_min_bins
            if hi <= lo:
                out.append((a, b))
                continue
            m = lo + int(np.argmin(seg[lo:hi]))
            left_peak = float(np.max(seg[:m + 1]))
            right_peak = float(np.max(seg[m:]))
            if float(seg[m]) < self.split_ratio * min(left_peak, right_peak):
                stack.append((a, a + m))
                stack.append((a + m, b))
            else:
                out.append((a, b))
        return sorted(out)

    def _openings_from_mins(self, mins, openmask):
        """-> list of opening dicts. Split out from _process so it is unit-testable.

        Each opening carries BOTH a centre bearing (for display) and `labels`, the set of
        sectors the run genuinely covers. The two are not interchangeable: the circular
        mean of a run wider than 180 deg is dominated by the *blocked* arc and lands at
        its antipode. A room open left, right and forward except for one 10-deg obstacle
        dead ahead forms a single 350-deg run whose circular mean is +-180 deg -- labelled
        BEHIND, scene "dead_end", for a wide-open four-way. One angle cannot represent a
        run spanning several sectors, so classification uses per-bin coverage instead.

        A sector must contain at least min_sector_bins of the run's bins to be claimed.
        Without that floor, a narrow opening straddling a boundary (say bins at +23..+28
        deg) would claim both FORWARD and LEFT and report a left_branch where there is
        only a corridor; the threshold keeps one-or-two-bin spillover from flipping the
        scene.
        """
        openings = []
        for start, length in self._circular_runs(openmask, self.min_run_bins):
            full_idx = (start + np.arange(length)) % N_BINS
            prof = np.where(np.isfinite(mins[full_idx]), mins[full_idx], self.max_range)
            pieces = self._split_by_prominence(prof) if self.split_runs else [(0, length)]
            for a, b in pieces:
                if len(pieces) > 1:
                    # A split piece keeps a shallow shoulder running down into the dip (the
                    # facing wall). Left in, that shoulder drags the piece's centre bearing
                    # toward the wall: at the real T-junction the left arm reported +24 deg
                    # (FORWARD) instead of ~+52 (LEFT), giving right_branch for a plain T.
                    # Trim each piece back to the contiguous deep part around its peak.
                    a, b = self._trim_to_peak(prof, a, b)
                if b - a < self.min_run_bins:
                    continue
                idx = full_idx[a:b]
                self._emit_opening(openings, mins, idx, (b - a) * BIN_DEG)
        openings.sort(key=lambda o: abs(o["bearing"]))
        return openings

    def _emit_opening(self, openings, mins, idx, width_deg):
            length = len(idx)
            vals = mins[idx]
            vals = vals[np.isfinite(vals)]
            depth = float(np.median(vals)) if vals.size else float(self.max_range)
            # fit test: the gap must subtend enough angle at its own depth
            req_half = math.degrees(math.atan2(self.half_width, max(depth, 1e-3)))
            if width_deg < 2.0 * req_half:
                return
            # circular mean bearing (a plain mean breaks across +-180); display only
            ang = np.radians(self.bin_deg[idx])
            centre = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean()))
            # A sector is claimed only if the run's share of THAT sector is itself wide
            # enough for the robot to fit at this depth -- the same criterion the run as a
            # whole just passed, applied per sector. A bare bin-count floor is too crude:
            # a forward opening centred -8 deg with width 53 deg spans -34.5..+18.5, so it
            # spills ~10 bins past the -25 deg RIGHT boundary and would manufacture a
            # phantom right branch, turning a left_branch into a four_way. Requiring the
            # spill to be robot-wide (2*req_half = 13.1 deg at 5 m) rejects it.
            # ONE CONTIGUOUS RUN IS ONE PASSAGE, so it claims exactly ONE direction: the
            # sector its centre bearing falls in. Separate branches must come from
            # separate runs. Measured 2026-10-22 approaching a real T-junction: while the
            # facing wall was still beyond open_range it read as open and fused with the
            # right arm into a single run that grew 54->92 deg. Letting that one run claim
            # the sectors it overlapped produced "fwd+right" -> right_branch: a confident
            # turn signal for a direction that is solid wall. Claiming only the centre
            # sector reports "corridor" there, which is the honest answer until the arms
            # actually separate into their own runs.
            #
            # The exception is a run so wide it is not a passage at all but open space
            # around the robot. A room open left, right and forward but for one 10-deg
            # obstacle forms a single ~350-deg run whose circular mean lands at its
            # antipode; calling that one direction gives "dead_end" for a wide-open space.
            # Above wide_run_deg we therefore fall back to per-sector coverage, requiring
            # each sector's share to be robot-wide at this depth.
            if width_deg >= self.wide_run_deg:
                labels = set()
                counts = {}
                for b in self.bin_deg[idx]:
                    lab_b = self._classify_one(float(b))
                    counts[lab_b] = counts.get(lab_b, 0) + 1
                for lab, cnt in counts.items():
                    if cnt * BIN_DEG >= 2.0 * req_half and cnt >= self.min_sector_bins:
                        labels.add(lab)
                if not labels:
                    labels = {self._classify_one(centre)}
            else:
                labels = {self._classify_one(centre)}
            openings.append({"bearing": centre, "width": width_deg, "depth": depth,
                             "label": self._classify_one(centre), "labels": labels})

    def _sector_median(self, mins, lo_deg, hi_deg):
        sel = (self.bin_deg >= lo_deg) & (self.bin_deg <= hi_deg)
        vals = mins[sel]
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if vals.size else None

    # ---------------- outputs ----------------
    def _fmt(self, openings):
        if not openings:
            return "openings: none"
        tag = {"FORWARD": "fwd", "LEFT": "left", "RIGHT": "right", "BEHIND": "behind"}
        order = {"FORWARD": 0, "LEFT": 1, "RIGHT": 2, "BEHIND": 3}
        def name(o):
            # Name an opening by the sectors it actually CLAIMS, not by its centre bearing.
            # Those differ for a run straddling a boundary, and when they do the old text
            # ("fwd(...)") flatly contradicted a junction_type of right_branch, which reads
            # as a bug. Showing "fwd+right(...)" makes the classification self-explanatory.
            labs = sorted(o.get("labels") or {o["label"]}, key=lambda k: order[k])
            return "+".join(tag[k] for k in labs)
        return "openings: " + " ".join(
            f"{name(o)}({o['bearing']:+.0f}deg, w={o['width']:.0f}deg, d={o['depth']:.1f}m)"
            for o in openings)

    def _publish_openings(self, openings, stamp):
        self.pub_text.publish(String(data=self._fmt(openings)))
        flat = []
        for o in openings:
            flat.extend([float(o["bearing"]), float(o["width"]), float(o["depth"])])
        self.pub_ops.publish(Float32MultiArray(data=flat))
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = "base_link"
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL       # drop last scan's arrows before redrawing
        arr.markers.append(clear)
        for i, o in enumerate(openings):
            th = math.radians(o["bearing"])
            a = Marker()
            a.header.frame_id = "base_link"
            a.header.stamp = stamp
            a.ns, a.id, a.type, a.action = "openings", i * 2, Marker.ARROW, Marker.ADD
            a.scale.x, a.scale.y, a.scale.z = 0.08, 0.16, 0.0   # shaft, head dia
            a.color = ColorRGBA(r=0.2, g=0.8, b=0.3, a=0.9)
            p0, p1 = Marker().pose.position.__class__(), Marker().pose.position.__class__()
            p0.x = p0.y = p0.z = 0.0
            p1.x = o["depth"] * math.cos(th)
            p1.y = o["depth"] * math.sin(th)
            p1.z = 0.0
            a.points = [p0, p1]
            a.pose.orientation.w = 1.0
            arr.markers.append(a)

            t = Marker()
            t.header.frame_id = "base_link"
            t.header.stamp = stamp
            t.ns, t.id, t.type, t.action = "openings", i * 2 + 1, Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position.x = p1.x
            t.pose.position.y = p1.y
            t.pose.position.z = 0.3
            t.pose.orientation.w = 1.0
            t.scale.z = 0.25
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            t.text = f"{o['label']} {o['bearing']:+.0f}d w{o['width']:.0f}d {o['depth']:.1f}m"
            arr.markers.append(t)
        self.pub_mark.publish(arr)

    def _console(self, e, dl, dr, openings):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_print < self.print_period:
            return
        self._last_print = now
        f = lambda v: "n/a" if v is None else f"{v:.1f}"
        es = "  n/a" if not np.isfinite(e) else f"{e:+.2f}m"
        print(f"{self._published or '(settling)':<12} | e={es} dL={f(dl)} dR={f(dr)} | "
              f"clr={getattr(self, '_d_front', float('nan')):.2f} | "
              f"{self._fmt(openings)}", flush=True)


def main():
    rclpy.init(args=sys.argv)
    node = LidarOpenings()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
