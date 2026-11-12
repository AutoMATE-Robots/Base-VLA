# adaptive_reasoning/: when to read a sign, and what it means for the goal

This is the **inference copy** of the SignWay adaptive-reasoning scheduler (paper Sec. III-B/C).
The pure modules were copied unchanged from the SignWay research repo
(`adaptive_reasoning/`, commit `d37f6f4`). The offline replay/figure scripts (`replay/`)
were left behind. `live.py` is new: it runs the same gate on live frames instead of a
pre-dumped rosbag jsonl.

```
camera frame ──► evidence/detect.py      docTR → text lines → plates (union-find) → crops
             ──► evidence/buffer.py      same physical sign across frames (digit signature)
             ──► evidence/relevance.py   R: is this plate about my goal? (room ranges / V_g)
             ──► evidence/resolve.py     fast path: goal line has exactly one arrow → no call
             ──► evidence/features.py    φ: text height, sharpness, OCR conf, ...
             ──► evidence/legibility.py  ℓ = σ(w·φ)   (calib/w.json, placeholder if absent)
             ──► gate.py                 fire iff N_t · max R·ℓ ≥ τ   (NO_SIGN/ARMED/PENDING/DECIDED)
             ──► reasoner.py             VLM call: cropped sign + goal + parser hint → JSON direction
             ──► memory.py               signs already used for this goal are never asked again, stamped with the
                                     /odom pose they were read from (same text elsewhere = new sign)
live.py      = all of the above per frame, VLM call in a background thread, arrival ("ARRIVED")
maneuver.py  = speed cap from sign text size; "is the LiDAR opening on the turn side?"
```

The ROS 2 wrapper is `corridor_nav/current/signway_reasoner_node.py`. It publishes the decision
on `/vla/prompt`, which `v8_drive_node.py` already listens to.

## Test (laptop, no GPU, no network)

```bash
pip install numpy pillow pytest
python -m pytest adaptive_reasoning/tests -q        # 110 tests
```

## Try it on still frames (real docTR + real VLM, no ROS)

```bash
pip install python-doctr openai
export GEMINI_API_KEY=...                            # never commit this
python -m adaptive_reasoning.live --goal "room 6-217" --tau 0.45 --wait \
    frames/000.png frames/001.png frames/002.png
```
Every call's exact prompt and crop are saved under `signway_logs/cli/payloads/`.

## Calibration status: read before reporting numbers

- `calib/w.json` (legibility weights) is **not here**, so `ℓ` uses placeholder weights
  and warns `UNCALIBRATED`. Until it is fitted, `min_plate_h` (60 px) handles most of the
  "too far to read" decision. That is the same stand-in the submission replay videos used.
- `tau` = 0.45 comes from those replay videos (`signway_params.yaml`). It is not a fitted value.
- If you fit `w.json` on MSI (`replay/calibrate.py` in the research repo), copy it into `calib/`.

## Not ported (yet)

- `evidence/direct.py`: the contour-based arrow-icon detector. docTR reads text, not arrow
  icons, so the fast path rarely fires without it and the VLM reads the arrow instead.
- The full odometry-anchored memory of paper Eq. (4) (`junction_memory/`: reproject a stored
  sign's bearing on a revisit and reuse its direction without a call). `memory.py` here only uses
  the pose to tell signs apart (text + within 8 m + same heading ±45°), not to reuse answers.
