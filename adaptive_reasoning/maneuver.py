"""
maneuver.py — two small rules that connect a sign decision to the robot's motion.
Pure functions (no ROS), used by corridor_nav/current/signway_reasoner_node.py.

1. speed_cap_from_text: slow down while a relevant sign is being read, matched to
   the size of its text.

   The VLM needs `latency` seconds to answer.  The answer is only useful if it lands
   before the robot reaches the sign (the junction is next to it).  With a pinhole
   camera, a text line of real height H that appears h pixels tall is at distance

        d ≈ fy · H / h

   so the speed that still leaves `latency + margin` seconds before the robot is within
   `d_buffer` of the sign is

        v_cap = (d − d_buffer) / (latency + margin)          clipped to [v_min, ∞)

   Small text in the image (far sign) → large d → no slowdown.  As the sign grows the
   cap drops, down to v_min.  H is not known per sign: `letter_h_m` assumes a typical
   letter height.  A sign with bigger real letters looks "closer" than it is, so the
   robot slows a little more than strictly needed (the safe side).  `latency` should be
   the MEASURED median call latency, not a constant.

2. opening_on_side: does /lidar/openings_array show a passage on the turn's side?
   Same sector convention as lidar_openings_node._classify_one: LEFT is bearing
   (+25°, +155°], RIGHT is [−155°, −25°).  Used to HOLD a turn decision until the
   robot is actually at the junction, instead of turning into the first opening it sees
   after the decision lands far from the junction.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence


def distance_from_text(text_h_px: float, fy_px: float, letter_h_m: float) -> Optional[float]:
    """Pinhole distance estimate (m) to a text line that appears text_h_px tall."""
    if text_h_px <= 0 or fy_px <= 0 or letter_h_m <= 0:
        return None
    return fy_px * letter_h_m / text_h_px


def speed_cap_from_text(text_h_px: float, *, fy_px: float, letter_h_m: float,
                        latency_s: float, margin_s: float, d_buffer_m: float,
                        v_min: float) -> Optional[float]:
    """Speed (m/s) that lets the VLM answer before the robot reaches the sign, or None
    if the text height is unknown (caller keeps its previous cap)."""
    d = distance_from_text(text_h_px, fy_px, letter_h_m)
    if d is None:
        return None
    v = (d - d_buffer_m) / max(latency_s + margin_s, 1e-3)
    return max(v, v_min)


LEFT_SECTOR = (25.0, 155.0)
RIGHT_SECTOR = (-155.0, -25.0)


def opening_on_side(openings: Sequence[float], turn: str) -> bool:
    """openings = flat [bearing_deg, width_deg, depth_m, ...] from /lidar/openings_array
    (bearing + = left).  turn = 'turn_left' | 'turn_right'."""
    lo, hi = LEFT_SECTOR if turn == "turn_left" else RIGHT_SECTOR
    for i in range(0, len(openings) - 2, 3):
        b = float(openings[i])
        if math.isfinite(b) and ((lo < b <= hi) if turn == "turn_left" else (lo <= b < hi)):
            return True
    return False
