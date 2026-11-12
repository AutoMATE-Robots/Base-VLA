"""Closed-loop junction sim: Ackermann robot + the drive node's real balance_plan (from a module
file), VLA modelled as the weak turn intent seen live. Reports min body gap and outcome."""
import importlib.util, math, sys, numpy as np
spec = importlib.util.spec_from_file_location("drv", sys.argv[1]); drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
MODE = sys.argv[2]; HALF = float(sys.argv[3]); BRW = float(sys.argv[4]); VLA = sys.argv[5] if len(sys.argv) > 5 else "weak"
X0 = 8.0                                         # opening near edge (m along the corridor)
def walls():
    P = []
    for x in np.arange(-2, 16, 0.05):
        P.append((x, -HALF))
        if not (X0 <= x <= X0 + BRW): P.append((x, HALF))
    for y in np.arange(HALF, 12, 0.05): P += [(X0, y), (X0 + BRW, y)]
    return np.array(P)
W = walls()
KW = dict(bases=[-0.3, -0.15, 0.15, 0.3], max_offset=1.5, offset_step=0.15, swerve_kmax=0.45,
          side_hw=drv.ROBOT_COVER + 0.02, full_hw=drv.ROBOT_COVER + 0.10, front_reach=0.5, margin_ramp=1.5,
          horizon=6.0, stations=[0.75, 1.25, 1.75, 2.25, 2.75, 3.25], clear_cap=0.9, w_clear=3.0, w_tight=10.0,
          w_free=0.5, w_off=0.5, w_curv=0.5, w_side=2.0, need=1.5, lookahead=1.0, kmax=0.6,
          min_clear=drv.ROBOT_COVER + 0.05, comfort=0.25)
L, PHI_MAX, RATE = 0.65, 0.461, 2.0
def body_gap(x, y, th):                           # min distance walls -> 0.98 x 0.75 box (neg = inside)
    c, s = math.cos(th), math.sin(th); dx, dy = W[:, 0] - x, W[:, 1] - y
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    qx, qy = np.abs(lx) - 0.49, np.abs(ly) - 0.375
    out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)); inside = np.minimum(np.maximum(qx, qy), 0)
    i = int(np.argmin(out + inside)); return float((out + inside)[i]), W[i]
def vla_k(x, y, th):                              # VLA intent model (from the live log)
    yaw = math.degrees(th); mouth = X0 + BRW / 2 - x
    if yaw > 75: return 0.0                       # in the branch: drive straight down it
    if VLA == "weak":
        if yaw > 20: return 0.39
        return 0.14 if mouth < 2.0 else 0.06
    if VLA == "good":
        if yaw > 20: return 0.45
        return 0.45 if mouth < 1.2 else 0.04
pol_over = False; x, y, th = 0.0, 0.0, 0.0; phi = 0.0; dt = 0.1; worst = (9, None); log = []
for step in range(900):
    rx, ry = x - drv.REAR_AXLE * math.cos(th), y - drv.REAR_AXLE * math.sin(th)
    d = W - (x, y); c, s = math.cos(th), math.sin(th)
    ox, oy = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]; m = np.hypot(ox, oy) < 10; ox, oy = ox[m], oy[m]
    kp = vla_k(x, y, th); kw = {}; wh = 1.0
    if MODE == "vla_first":
        wh = 10.0; kw = dict(policy_clear=drv.ROBOT_COVER + 0.15 + (0.05 if pol_over else 0), policy_horizon=2.0)
    elif MODE.startswith("intent"):
        wh = float(MODE.split(":")[1]); import os; kw = dict(turn_side=int(os.environ.get("SIDE", 1)), w_turn=float(MODE.split(":")[2]))
    p = drv.balance_plan(ox, oy, kp, 0, w_head=wh, **KW, **kw)
    if MODE == "vla_first": pol_over = not p["policy_first"]
    k = p["k"]; free = p["free"]
    sc = 1.0 if free >= 4 else max(0.0, (free - 1.0) / 3.0)
    gap = p["cnear"] - drv.ROBOT_COVER
    if gap < 0.30: sc = min(sc, max(0.35, gap / 0.30))
    v = 0.4 * sc if free >= 1.0 else 0.10
    tgt = max(-PHI_MAX, min(PHI_MAX, math.atan(L * k)))
    phi += max(-RATE * dt, min(RATE * dt, tgt - phi))
    kc = math.tan(phi) / L
    th += v * kc * dt; rx += v * math.cos(th) * dt; ry += v * math.sin(th) * dt
    x, y = rx + drv.REAR_AXLE * math.cos(th), ry + drv.REAR_AXLE * math.sin(th)
    g, wp = body_gap(x, y, th)
    if g < worst[0]: worst = (g, wp, x, y, math.degrees(th))
    log.append((x, y, math.degrees(th), kp, k, v))
    if g <= 0 or abs(y) > HALF + 3.0 or x > X0 + BRW + 4: break
res = ("TURNED RIGHT?!" if y < -HALF - 0.5 else "IN BRANCH") if abs(y) > HALF + 2.0 else None
res = res or ("PASSED junction" if x > X0 + BRW + 2 else ("HIT" if worst[0] <= 0 else "stuck"))
first_turn = next((i * dt for i, r in enumerate(log) if r[4] > 0.25), None)
fl = ", ".join(f"{r[0]:.1f}:{r[4]:+.2f}" for r in log[::15])
print(f"{MODE:16s} VLA={VLA:4s} corr {2*HALF:.1f} branch {BRW:.1f}: {res:15s} min body gap {worst[0]:+.2f} m at wall {np.round(worst[1],2)} (robot x {worst[2]:.1f} yaw {worst[4]:.0f}) t={len(log)*dt:.0f}s")
if "-v" in sys.argv:
    for r in log[::10]:
        if r[0] > 5.0: print(f"   x {r[0]:5.2f} y {r[1]:+5.2f} yaw {r[2]:4.0f}  VLA k {r[3]:+.2f} cmd k {r[4]:+.2f} v {r[5]:.2f}")
