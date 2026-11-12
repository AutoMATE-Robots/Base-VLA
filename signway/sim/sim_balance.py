"""Run the drive node's real balance_plan() on a synthetic corridor with a left branch."""
import importlib.util, sys, numpy as np
spec = importlib.util.spec_from_file_location("drv", sys.argv[1]); drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
import os
HALF = float(os.environ.get("HALF", 0.85)); BR = float(os.environ.get("BR", 1.7))
def scene(branch_x0, branch_x1):
    pts = []
    for x in np.arange(-0.5, 8.0, 0.05):
        pts.append((x, -HALF))                    # right wall, continuous
        if not (branch_x0 <= x <= branch_x1): pts.append((x, HALF))   # left wall with the opening
    for y in np.arange(HALF, 5.0, 0.05):          # the branch corridor's two walls
        pts += [(branch_x0, y), (branch_x1, y)]
    a = np.array(pts); return a[:, 0], a[:, 1]
KW = dict(bases=[-0.3, -0.15, 0.15, 0.3], max_offset=1.5, offset_step=0.15, swerve_kmax=0.45,
          side_hw=drv.ROBOT_COVER + 0.02, full_hw=drv.ROBOT_COVER + 0.10, front_reach=0.5, margin_ramp=1.5,
          horizon=6.0, stations=[0.75, 1.25, 1.75, 2.25, 2.75, 3.25], clear_cap=0.9, w_clear=3.0, w_tight=10.0,
          w_free=0.5, w_off=0.5, w_curv=0.5, w_side=2.0, need=1.5, lookahead=1.0, kmax=0.6,
          min_clear=drv.ROBOT_COVER + 0.05, comfort=0.25)
extra = eval(sys.argv[2]) if len(sys.argv) > 2 else {}
w_head = extra.pop("w_head", 1.0)
print(f"{'opening ahead':>14s} {'VLA k':>6s} -> {'cmd k':>6s} {'shift':>6s}  policy-path free / cmin")
for d0 in (4.0, 2.0, 1.0, 0.3):              # distance from robot to the opening's near edge
    ox, oy = scene(d0, d0 + BR)
    for k_pol in (0.06, 0.2, 0.45):
        p = drv.balance_plan(ox, oy, k_pol, 0, w_head=w_head, **KW, **extra)
        tag = "OVERRIDE" if abs(p["k"] - k_pol) > 0.05 else "follows VLA"
        print(f"{d0:12.1f} m {k_pol:+6.2f} -> {p['k']:+6.2f} {p['dev']:+6.2f}  {tag:12s} VLA path: free {p['free_pol']:.1f} m, gap {p.get('cmin_pol', 0) - drv.ROBOT_COVER:+.2f} m")
