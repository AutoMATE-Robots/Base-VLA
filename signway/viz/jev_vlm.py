"""
jev_vlm.py -- TypeSafe Jev (System One decision model) as SignWay's sign reader.

Drop-in for adaptive_reasoning.reasoner.OpenAICompatVLM: same complete(prompt, images)
-> (raw_text, latency_s), and raw_text is the JSON VLMAnswer.parse() already reads
({"applicable", "direction", "confidence", "summary"}). Jev does not generate text: it
scores the options of ONE choice question and returns a probability for each, so the
"confidence" is Jev's probability of the chosen direction.

API (docs.typesafe.ai, read 2026-11-11): POST https://api.typesafe.ai/v1/systemone,
Bearer TYPESAFE_API_KEY, multipart (part "request" = JSON, one file part per image),
header TypeSafe-Danger-Zone-Images: true (image input is "dreamer-ready", i.e. NOT
production-ready per TypeSafe). Images: PNG/JPEG, each side 280..1568 px.

Test tooling for the Jetson, outside the repo. Enabled by signway_node_viz.py when
SIGNWAY_VLM=jev (signway_up.sh sets it when TYPESAFE_API_KEY is in ~/.signway_secrets).
"""
import io
import json
import os
import time
import uuid
from typing import Sequence

import numpy as np

URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/") + "/v1/systemone"
MIN_SIDE, MAX_SIDE = 280, 1568

# option -> description Jev scores. "arrived" maps to SignWay's stop + "ARRIVED" summary.
# Wayfinding signs usually print ONE arrow at the top of a GROUP of lines (arrow, then several
# rooms/places beneath it, then the next arrow group). The goal's direction is the arrow heading
# its group, even when it is several lines away. (Before 2026-11-12 the criteria said "arrow next
# to the goal's line", so Jev answered not_applicable for "Stairs" five lines under a left arrow.)
_GROUP = (" (the arrow may be beside that line, or head the group/block of lines the goal is "
          "listed in, possibly several lines above it)")
CRITERIA = {
    "turn_left": "The goal is named on this sign and the arrow for the goal's line points LEFT" + _GROUP + ".",
    "turn_right": "The goal is named on this sign and the arrow for the goal's line points RIGHT" + _GROUP + ".",
    "straight": "The goal is named on this sign and the arrow for the goal's line points UP / straight ahead" + _GROUP + ".",
    "arrived": "This sign marks the goal itself (the goal room's door plate, or the label at the "
               "goal, e.g. at the elevator doors); it does not point somewhere else.",
    "not_applicable": "The goal is not named on this sign at all, or the sign has no arrow that "
                      "applies to the goal's line or its group.",
}
TO_DIRECTION = {"turn_left": "turn_left", "turn_right": "turn_right", "straight": "straight",
                "arrived": "stop"}


def _fit_png(img: np.ndarray) -> bytes:
    """RGB array -> PNG whose sides are within Jev's 280..1568 px (scale, then pad)."""
    from PIL import Image
    arr = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    im = Image.fromarray(arr[..., :3])
    w, h = im.size
    s = 896 / max(w, h)                    # long side ~896 px (small arrows need pixels)
    if min(w, h) * s < MIN_SIDE:           # short side too small: grow it if the long side allows
        s = min(MIN_SIDE / min(w, h), MAX_SIDE / max(w, h))   # (else padded below)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    w, h = im.size
    W, H = max(w, MIN_SIDE), max(h, MIN_SIDE)                   # pad a very wide/tall crop
    if (W, H) != (w, h):
        canvas = Image.new("RGB", (W, H), (128, 128, 128))
        canvas.paste(im, ((W - w) // 2, (H - h) // 2))
        im = canvas
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class JevVLM:
    def __init__(self, model: str = None, timeout: float = 15.0, max_retries: int = 1, **_ignored):
        self.api_key = os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError("No TYPESAFE_API_KEY in the environment (source ~/.signway_secrets)")
        self.model = os.environ.get("SIGNWAY_JEV_MODEL") or "jev-latest"
        self.timeout, self.max_retries = timeout, max_retries

    def complete(self, prompt: str, images: Sequence[np.ndarray]) -> tuple[str, float]:
        import requests
        if not images:
            # Jev answers fixed-choice questions; it cannot write the goal-synonym list that
            # expand_goal asks for. Raising makes the caller fall back to the goal text alone.
            raise RuntimeError("Jev client: text-only generation is not supported")
        ref = f"sign-{uuid.uuid4().hex[:12]}"
        body = {
            "model": self.model,
            # the full SignWay prompt (goal, OCR text, parser hint, arrival rule) as context
            "state": {"sign": ref, "context": prompt},
            "assets": {ref: {"type": "image", "part": "sign-crop"}},
            "questions": {"direction": {
                "type": "choice",
                "instructions": "The robot is navigating to the goal stated in `context`. Look at the "
                                "sign in `sign`: which way does it direct the robot to the goal? "
                                "Signs group several lines under one arrow: a line belongs to the "
                                "nearest arrow printed above it in the same block, so 'beside the "
                                "line' in `context` includes the arrow heading the line's group.",
                "criteria": CRITERIA}},
        }
        files = {"request": (None, json.dumps(body), "application/json"),
                 "sign-crop": ("sign.png", _fit_png(images[0]), "image/png")}
        headers = {"Authorization": f"Bearer {self.api_key}", "TypeSafe-Danger-Zone-Images": "true"}
        last = None
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                r = requests.post(URL, headers=headers, files=files, timeout=self.timeout)
            except requests.RequestException as e:
                last = RuntimeError(f"Jev transport error: {type(e).__name__}: {e}")
            else:
                lat = time.monotonic() - t0
                if r.status_code == 200:
                    return self._to_answer_json(r.json()), lat
                detail = r.text[:300]
                last = RuntimeError(f"Jev HTTP {r.status_code}: {detail}")
                if r.status_code not in (429, 500, 502, 503, 529):
                    break                                   # 4xx = our request is wrong: no retry
            if attempt < self.max_retries:
                time.sleep(1.0 * (attempt + 1))
        raise last

    @staticmethod
    def _to_answer_json(resp: dict) -> str:
        a = resp["answers"]["direction"]
        choice, probs = a["choice"], a.get("probabilities", {})
        p = float(probs.get(choice, a.get("confidence") or 0.0))
        top = ", ".join(f"{k} {v:.2f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3])
        direction = TO_DIRECTION.get(choice)
        summary = ("ARRIVED: " if choice == "arrived" else "") + f"Jev {resp.get('model', '')}: {top}"
        return json.dumps({"applicable": direction is not None, "direction": direction,
                           "confidence": round(p, 3), "summary": summary,
                           "jev": {"choice": choice, "probabilities": probs,
                                   "confidence": a.get("confidence"), "usage": resp.get("usage")}})
