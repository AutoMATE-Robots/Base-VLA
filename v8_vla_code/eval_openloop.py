#!/usr/bin/env python3
"""
eval_openloop.py -- evaluate a trained trajectory VLA WITHOUT a simulator.
Why this works with no sim: the bag already contains BOTH the inputs (images) and the correct
answers (the trajectory the human actually drove, from odometry). So we do OPEN-LOOP eval:
for every frame, feed the model (image + prompt) and compare its predicted trajectory to the
actual future path from odometry. No driving, no compounding errors, no simulator.
    predicted trajectory  = model(image, prompt)          <- what the model wants to do
    actual trajectory     = future odometry poses          <- what was correct here (ground truth)
Two things to read off the output:
  1. Do predicted and actual paths match?  (per-frame L1/endpoint error; a summary number.)
  2. THE KEY PLOT for our design: before the junction, does the predicted path stay STRAIGHT
     even though the prompt says "turn_right", and only curve as the junction approaches?
     That is the visual test of "the model learned timing from the image, not from the prompt".
This script is model-agnostic on purpose. Wire your OpenVLA-OFT checkpoint into `load_policy`
and `predict` (two small functions marked TODO). Everything else -- data, matching, plotting,
metrics -- is done and testable with a stub policy (`--stub`) so you can validate the eval
harness before the real model is ready.
Usage:
    # dry-run the harness with a stub that returns the ground truth (sanity of the plumbing):
    python eval_openloop.py --bag /path/to/bag --flip-frame 0 --decision turn_right \
        --turn-done-frame 21 --stub --out eval_out
    # with a real checkpoint (after you wire load_policy/predict):
    python eval_openloop.py --bag /path/to/bag --flip-frame 0 --decision turn_right \
        --turn-done-frame 21 --checkpoint /path/to/oft_ckpt --out eval_out
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
# CRITICAL: prismatic/vla/constants.py picks the robot platform by scanning sys.argv for a
# keyword ("signway", "libero", ...). Our eval command doesn't naturally contain "signway",
# so without this it defaults to LIBERO -> ACTION_DIM=7 and the 2-dim action head fails to
# load (size mismatch [7,4096] vs our [2,4096]). Inject the keyword BEFORE any prismatic
# import so SIGNWAY constants (ACTION_DIM=2) are selected.
if not any("signway" in a.lower() for a in sys.argv):
    sys.argv.append("--signway")
import numpy as np
# This file lives in signway_dataset/ but bag_to_episode.py and bag_to_mp4.py live in tools/.
# Add the repo root and tools/ to the path so the imports resolve wherever this is run from.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_ROOT, _ROOT / "tools", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from bag_to_episode import build_steps, read_bag
# ============================================================================================
# WIRE YOUR MODEL HERE. Two functions. Keep the trajectory convention: (N,2) float32, (fwd,left)
# in the robot's current frame, metres. Everything else already matches this.
# ============================================================================================
def load_policy(checkpoint: str):
    """Load the OpenVLA-OFT checkpoint using the repo's own inference helpers.
    Returns a dict with everything `predict` needs: the vla model, processor, action head, and a
    GenerateConfig-like cfg carrying unnorm_key + flags. Mirrors the reference LIBERO eval setup.
    """
    import sys, types
    # the repo's helpers live under experiments/robot; make them importable
    from experiments.robot.openvla_utils import get_vla, get_processor, get_action_head
    from experiments.robot.robot_utils import get_action  # noqa: F401 (used in predict)
    # minimal cfg object matching what get_* expect (see GenerateConfig fields)
    cfg = types.SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=1,
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=1,           # NUM_ACTIONS_CHUNK=1 (one-shot 16-dim trajectory)
        unnorm_key="signway_dataset",    # the stats saved with our dataset
        lora_rank=32,
    )
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    return {"cfg": cfg, "vla": vla, "processor": processor, "action_head": action_head}
def predict(policy, image: np.ndarray, prompt: str, horizon: int) -> np.ndarray:
    """One forward pass -> (horizon,2) trajectory (fwd,left) in the robot frame.
    Uses the repo's `get_action`, which builds the VLA prompt from the task label, runs
    predict_action with the action head, and returns a list of per-step actions (each (2,)).
    Our prompt string IS the task label (e.g. "turn right")."""
    from experiments.robot.robot_utils import get_action
    cfg = policy["cfg"]
    # get_action expects an observation dict with 'full_image'; our prompt maps to the task label.
    task_label = prompt.replace("_", " ")            # "turn_right" -> "turn right"
    obs = {"full_image": image}
    actions = get_action(
        cfg, policy["vla"], obs, task_label,
        processor=policy["processor"],
        action_head=policy["action_head"],
        proprio_projector=None,
        noisy_action_projector=None,
        use_film=cfg.use_film,
    )
    a = np.asarray(actions, dtype=np.float32).reshape(-1)   # (16,) one-shot trajectory
    traj = a.reshape(-1, 2)[:horizon]                        # (horizon,2) cumulative waypoints
    return traj
def _attach_gt_trajectory(steps, horizon):
    """Action is now the (16,) flattened cumulative-waypoint trajectory in the step's own
    frame (ViNT-style) -- no chaining needed; just reshape to (horizon,2)."""
    for st in steps:
        st["trajectory"] = np.asarray(st["action"], dtype=np.float32).reshape(-1, 2)[:horizon]
    return steps
def _stub_predict(step, horizon, noise=0.0):
    """Return the ground-truth trajectory (optionally + noise). Lets us verify the eval
    plumbing end-to-end: with noise=0 the error must be ~0 and the overlay must coincide."""
    tr = step["trajectory"].copy()
    if noise:
        tr = tr + np.random.default_rng(0).normal(0, noise, tr.shape).astype(np.float32)
    return tr
# ---- metrics -------------------------------------------------------------------------------
def traj_errors(pred: np.ndarray, actual: np.ndarray):
    """Per-frame errors between predicted and actual trajectories."""
    l1 = float(np.mean(np.abs(pred - actual)))
    endpoint = float(np.linalg.norm(pred[-1] - actual[-1]))
    # 'turn-ness' = signed left-deflection of the endpoint (for the timing plot)
    return {"l1": l1, "endpoint_err": endpoint,
            "pred_left": float(pred[-1][1]), "actual_left": float(actual[-1][1])}
# ---- raw-data dump -------------------------------------------------------------------------
def save_series(out_dir, steps, preds, errs, mean_l1, mean_ep, args):
    """Dump everything needed to replot / re-analyze WITHOUT re-running the model.
    Lesson learned the hard way (v6 3k-vs-6k night): PNGs can't be restyled; if only
    images survive, cosmetic replots cost an hour of GPU instead of 5s on a login node.
    Saves endpoint series (the timing plot), FULL trajectories (overlays + viz_ar.py AR
    renders offline), per-frame errors, summary metrics, and the run args.
    Deliberately wrapped in try/except: a dump failure must never kill an eval run
    after the expensive GPU work is already done."""
    try:
        payload = {
            # per-frame series (what timing_test.png plots)
            "frames": np.array([s["frame_index"] for s in steps]),
            "prompts": np.array([s["prompt"] for s in steps]),
            "pred_lat": np.array([p[-1][1] for p in preds], dtype=np.float32),
            "actual_lat": np.array([s["trajectory"][-1][1] for s in steps], dtype=np.float32),
            "pred_fwd": np.array([p[-1][0] for p in preds], dtype=np.float32),
            "actual_fwd": np.array([s["trajectory"][-1][0] for s in steps], dtype=np.float32),
            # full (N, horizon, 2) trajectories -> offline overlays / AR renders
            "pred_traj": np.stack(preds).astype(np.float32),
            "actual_traj": np.stack([s["trajectory"] for s in steps]).astype(np.float32),
            # per-frame + summary errors (report reads these; no log-grepping needed)
            "l1": np.array([e["l1"] for e in errs], dtype=np.float32),
            "endpoint_err": np.array([e["endpoint_err"] for e in errs], dtype=np.float32),
            "mean_l1": float(mean_l1),
            "mean_endpoint": float(mean_ep),
            # run identity: which bag/checkpoint/args produced this
            "flip_frame": args.flip_frame,
            "turn_done_frame": args.turn_done_frame,
            "stride": args.stride,
            "decision": args.decision,
            "horizon": args.horizon,
            "bag": str(args.bag),
            "checkpoint": str(args.checkpoint),
        }
        path = os.path.join(out_dir, "series.npy")
        np.save(path, payload)
        print(f"series     -> {path}")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: series.npy dump failed ({e}) -- plots/metrics unaffected")
# ---- plots ---------------------------------------------------------------------------------
def plot_overlays(steps, preds, out_dir, every=6):
    """Grid of per-frame overlays: predicted (blue) vs actual (black) trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = steps[::every]
    selp = preds[::every]
    ncol = 4
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (s, p) in enumerate(zip(sel, selp)):
        ax = axes[k // ncol][k % ncol]
        ax.axis("on")
        a = np.vstack([[0, 0], s["trajectory"]])
        pp = np.vstack([[0, 0], p])
        # raw left-value on x: +x = left, -x = right. No hidden negation.
        ax.plot(a[:, 1], a[:, 0], "k-", lw=2, label="actual")
        ax.plot(pp[:, 1], pp[:, 0], color="#4C9BE8", lw=2, ls="--", label="predicted")
        ax.invert_xaxis()   # +y=left mirrored so physical right shows on the screen's right
        ax.scatter([0], [0], marker="^", c="k", s=40)
        ax.set_title(f"frame {s['frame_index']} · {s['prompt']}", fontsize=8)
        ax.axis("equal")
        if k == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Predicted (blue dashed) vs actual (black) trajectory, per frame", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "overlays.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p
def plot_timing(steps, preds, out_dir):
    """THE design-test plot: endpoint left-deflection vs frame, predicted vs actual.
    For our concern: with prompt 'turn_right' held, both should stay ~0 while the junction is
    far and swing negative (right) only near it. If the PREDICTED curve swings early while the
    actual stays flat, the model is turning on the prompt, not the image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fi = [s["frame_index"] for s in steps]
    a_left = [s["trajectory"][-1][1] for s in steps]
    p_left = [p[-1][1] for p in preds]
    prompts = [s["prompt"] for s in steps]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(fi, a_left, "k-", lw=2, label="actual (odometry)")
    ax.plot(fi, p_left, color="#4C9BE8", lw=2, ls="--", label="predicted (model)")
    # shade where the prompt is a turn
    turn_frames = [f for f, pr in zip(fi, prompts) if pr not in ("straight",)]
    if turn_frames:
        ax.axvspan(min(turn_frames), max(turn_frames), color="#E8994C", alpha=0.12,
                   label="prompt = turn")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xlabel("frame")
    ax.set_ylabel("endpoint y  (+ = left,  - = right)   [m]")
    ax.set_title("Timing test: does the model stay straight until the junction, then turn?\n"
                 "predicted should track actual — turning only when the junction is near, "
                 "not when the prompt is set")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "timing_test.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--flip-frame", type=int, required=True)
    ap.add_argument("--decision", required=True, choices=["turn_left", "turn_right", "stop", "straight"])
    ap.add_argument("--turn-done-frame", type=int, default=None)
    ap.add_argument("--image-topic", default="/c1/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--signway", action="store_true", default=False,
                    help="(internal) platform keyword for constants detection; auto-injected")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--checkpoint", default=None, help="OFT checkpoint dir")
    ap.add_argument("--stub", action="store_true",
                    help="use a ground-truth stub policy to test the eval harness itself")
    ap.add_argument("--stub-noise", type=float, default=0.0,
                    help="add noise to the stub to see non-zero error in the plots")
    ap.add_argument("--out", default="eval_out")
    args = ap.parse_args()
    frames, odom = read_bag(Path(args.bag), args.image_topic, args.odom_topic, args.ros_distro)
    steps = build_steps(frames, odom, args.flip_frame, args.decision,
                        args.turn_done_frame, args.horizon, args.stride)
    _attach_gt_trajectory(steps, args.horizon)
    print(f"{len(steps)} frames to evaluate")
    if args.stub:
        preds = [_stub_predict(s, args.horizon, args.stub_noise) for s in steps]
    else:
        if not args.checkpoint:
            raise SystemExit("give --checkpoint or use --stub")
        policy = load_policy(args.checkpoint)
        preds = [predict(policy, s["image"], s["prompt"], args.horizon) for s in steps]
    # metrics
    errs = [traj_errors(p, s["trajectory"]) for p, s in zip(preds, steps)]
    mean_l1 = np.mean([e["l1"] for e in errs])
    mean_ep = np.mean([e["endpoint_err"] for e in errs])
    print(f"mean L1 traj error   : {mean_l1:.4f} m")
    print(f"mean endpoint error  : {mean_ep:.4f} m")
    if args.stub and args.stub_noise == 0:
        assert mean_l1 < 1e-6, "stub with no noise must have ~0 error -- harness bug"
        print("harness check OK: zero-noise stub reproduces ground truth exactly")
    os.makedirs(args.out, exist_ok=True)
    save_series(args.out, steps, preds, errs, mean_l1, mean_ep, args)
    p1 = plot_overlays(steps, preds, args.out, every=max(1, len(steps) // 16))
    p2 = plot_timing(steps, preds, args.out)
    print(f"overlays   -> {p1}")
    print(f"timing plot-> {p2}   (the design test: predicted should turn only near the junction)")
if __name__ == "__main__":
    main()