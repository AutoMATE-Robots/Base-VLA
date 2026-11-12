"""
reasoner.py — the (expensive) call, asked well and never asked twice.

One job: given plate crops + the scene + the goal + a memory summary, return a
structured answer

    VLMAnswer(applicable, direction, confidence, summary, raw)

with direction ∈ {turn_left, turn_right, straight, stop, None}.

Two design rules:
  * CACHE EVERYTHING.  Every response is stored on disk keyed by a hash of
    (model, goal, memory summary, crop bytes).  Replays, the τ sweep, and all
    baselines then share one set of paid calls — E5 becomes free after E2.
  * TRANSPORT-AGNOSTIC.  `OpenAICompatVLM` speaks the OpenAI chat-completions
    schema, which Gemini also exposes (base_url
    https://generativelanguage.googleapis.com/v1beta/openai/), so one client
    covers both.  Key comes from GEMINI_API_KEY / OPENAI_API_KEY env
    (~/.secrets on MSI).  Tests use FakeVLM.

The prompt grounds the model in the recognized text (the IROS augmentation we
cite and use) and demands JSON only.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

DIRECTIONS = ("turn_left", "turn_right", "straight", "stop")

# JSON-only prompt (kept for the prompt ablation; it is what a single-shot
# schema-constrained request looks like).  Measured Sept 11 on Qwen2.5-VL-7B:
# ~0% on goal signs — the model refuses the range arithmetic in one shot.
PROMPT_JSON_ONLY = """You are the sign-reading module of an indoor robot.
Goal: "{goal}"
Recognized text on the candidate sign(s) (may contain OCR errors):
{plate_texts}
Previously used signs/decisions for this goal: {memory_summary}

From the attached image crops (sign close-ups), decide whether any sign tells
the robot which way to go NOW to reach the goal.
Answer with JSON ONLY, no prose, exactly:
{{"applicable": true/false, "direction": "turn_left"|"turn_right"|"straight"|"stop"|null,
  "confidence": 0.0-1.0, "summary": "<one short sentence>"}}
If no sign addresses the goal, applicable=false and direction=null."""

# DEPLOYED PROMPT: rules + short reasoning before the JSON.  Small VLMs need to
# enumerate the lines in text before they can match a room range; JSON-only
# suppresses that and collapses to "not applicable"/"straight".
PROMPT = """You are the sign-reading module of an indoor robot.
Goal: "{goal}"
Recognized text on the candidate sign (may contain OCR errors):
{plate_texts}
Previously used signs/decisions for this goal: {memory_summary}

{hint}
Rules:
1. The goal is a room number or a place name.
2. A line like "Rooms 43-58" means EVERY room from 43 to 58, so goal 49 IS on
   that line. "5-117 to 5-196" includes 5-182. Check each line this way.
3. Find the ONE line that contains the goal (as a number inside its range, or
   as a name). If no line contains it, answer applicable=false.
4. Read the arrow printed next to THAT line in the image: \u2190 = turn_left,
   \u2192 = turn_right, \u2191 = straight, \u2197/\u2196 = straight (then turn later).
5. First write 2-4 short lines of reasoning: list each sign line with its arrow,
   and state which line contains the goal. Then, on a new line, output the JSON:
{{"applicable": true/false, "direction": "turn_left"|"turn_right"|"straight"|"stop"|null,
  "confidence": 0.0-1.0, "summary": "<which line matched and which arrow it has>"}}"""


@dataclass
class VLMAnswer:
    applicable: bool
    direction: Optional[str]
    confidence: float
    summary: str
    raw: str = ""
    latency_s: float = 0.0
    cached: bool = False

    @classmethod
    def parse(cls, text: str, latency_s: float = 0.0) -> "VLMAnswer":
        t = text.strip()
        if "```" in t:  # strip code fences
            t = t.split("```")[1]
            t = t[4:] if t.startswith("json") else t
        try:
            start, end = t.index("{"), t.rindex("}") + 1
            d = json.loads(t[start:end])
        except (ValueError, json.JSONDecodeError):
            return cls(False, None, 0.0, "unparseable", raw=text, latency_s=latency_s)
        direction = d.get("direction")
        if direction not in DIRECTIONS:
            direction = None
        applicable = bool(d.get("applicable")) and direction is not None
        conf = float(d.get("confidence", 0.0) or 0.0)
        return cls(applicable, direction, max(0.0, min(1.0, conf)),
                   str(d.get("summary", ""))[:300], raw=text, latency_s=latency_s)


class VLMClient(Protocol):
    def complete(self, prompt: str, images: Sequence[np.ndarray]) -> tuple[str, float]:
        """Return (raw_text, latency_seconds)."""
        ...


# ----------------------------------------------------------------------------
# Transports
# ----------------------------------------------------------------------------

def _jpeg_b64(img: np.ndarray, quality: int = 90, max_dim: int = 896, min_dim: int = 768) -> str:
    from PIL import Image
    arr = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h, w = arr.shape[:2]
    if max(h, w) > max_dim:                    # cap vision tokens/latency
        sc = max_dim / max(h, w)
        arr = np.asarray(Image.fromarray(arr).resize((int(w * sc), int(h * sc)), Image.LANCZOS))
    elif max(h, w) < min_dim:
        # UPSCALE small sign crops.  Directional arrows are small glyphs (~20 px
        # in a 300 px crop); at ~800 px the vision encoder spends many more
        # patches on them.  Measured Sept 12: arrow misreads were the dominant
        # residual error on room-range directories.
        sc = min_dim / max(h, w)
        arr = np.asarray(Image.fromarray(arr).resize((int(w * sc), int(h * sc)), Image.LANCZOS))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


class OpenAICompatVLM:
    """OpenAI chat-completions transport; covers Gemini via its compat endpoint."""

    def __init__(self, model: str,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
                 api_key: Optional[str] = None, timeout: float = 60.0,
                 reasoning_effort: Optional[str] = "low",
                 rpm: float = 15.0, max_retries: int = 6,
                 hedge_after: Optional[float] = None):
        # Gemini 3 models THINK by default (seconds of hidden deliberation);
        # reasoning_effort="low" is the practical floor via the compat layer.
        self.model, self.base_url, self.timeout = model, base_url, timeout
        self.reasoning_effort = reasoning_effort
        # Free tier is 15 requests/minute/model: pace client-side (a 429 costs
        # more than the wait) and retry with backoff, honouring the server's
        # retryDelay when it gives one.  Batch jobs (E2) are throughput-bound,
        # not latency-bound, so waiting is free there.
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self.max_retries = max_retries
        self._last_call = 0.0
        # Observed (flash-lite, text-only, n=20): median ~1.6s but ~20% of calls
        # land in an 8-17s tail — server-side routing luck, not payload size.
        # Hedging: if no answer by `hedge_after`, fire a SECOND identical
        # request and take whichever returns first.  Costs an extra call on the
        # slow fraction only; converts the tail into ~hedge_after + median.
        self.hedge_after = hedge_after
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("No API key: set GEMINI_API_KEY or OPENAI_API_KEY (MSI: source ~/.secrets)")

    def _once(self, client, content, kw) -> tuple[str, float]:
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}],
            temperature=0.0, **kw)
        return resp.choices[0].message.content or "", time.monotonic() - t0

    def _hedged(self, client, content, kw) -> tuple[str, float]:
        """Fire one request; if slow, fire a twin and take the first answer."""
        import concurrent.futures as cf
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(self._once, client, content, kw)]
            try:
                txt, _ = futs[0].result(timeout=self.hedge_after)
                return txt, time.monotonic() - t0
            except cf.TimeoutError:
                pass
            futs.append(ex.submit(self._once, client, content, kw))
            for fut in cf.as_completed(futs):
                try:
                    txt, _ = fut.result()
                    return txt, time.monotonic() - t0
                except Exception:                        # noqa: BLE001
                    continue
            raise RuntimeError("both hedged requests failed")

    def complete(self, prompt: str, images: Sequence[np.ndarray]) -> tuple[str, float]:
        from openai import OpenAI  # lazy
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(img)}"}})
        kw = {}
        if self.reasoning_effort:
            kw["extra_body"] = {"reasoning_effort": self.reasoning_effort}

        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            # latency is timed AFTER pacing: it measures the API, not our queue
            try:
                out = (self._hedged(client, content, kw) if self.hedge_after
                       else self._once(client, content, kw))
                self._last_call = time.monotonic()
                return out
            except Exception as e:                      # noqa: BLE001
                self._last_call = time.monotonic()
                msg = str(e)
                is_rate = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate" in msg.lower()
                # a single slow request must not kill a 7-hour batch: timeouts and
                # transient connection errors are retried like rate limits
                is_transient = any(k in type(e).__name__ for k in ("Timeout", "APIConnection", "InternalServer")) \
                    or "timed out" in msg.lower() or "connection" in msg.lower()
                if not (is_rate or is_transient) or attempt == self.max_retries:
                    raise
                if is_transient and not is_rate:
                    delay = min(2.0 ** attempt, 20.0)
                    print(f"  transient API error ({type(e).__name__}), retrying in {delay:.1f}s "
                          f"(attempt {attempt + 1}/{self.max_retries})", flush=True)
                    time.sleep(delay)
                    continue
                m = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)", msg)
                delay = float(m.group(1)) if m else min(2.0 ** attempt, 30.0)
                delay = max(delay, self.min_interval) + 0.5
                print(f"  rate-limited, retrying in {delay:.1f}s "
                      f"(attempt {attempt + 1}/{self.max_retries})", flush=True)
                time.sleep(delay)
        raise RuntimeError("unreachable")


class FakeVLM:
    """Scripted transport for tests/dry-runs: answers from a dict keyed by goal text."""

    def __init__(self, script: dict[str, dict]):
        self.script, self.calls = script, 0

    def complete(self, prompt: str, images: Sequence[np.ndarray]) -> tuple[str, float]:
        self.calls += 1
        for key, ans in self.script.items():
            if key in prompt:
                return json.dumps(ans), 0.01
        return json.dumps({"applicable": False, "direction": None,
                           "confidence": 0.2, "summary": "no relevant sign"}), 0.01


# ----------------------------------------------------------------------------
# The reasoner (cache + payload assembly)
# ----------------------------------------------------------------------------

@dataclass
class Reasoner:
    client: VLMClient
    model_tag: str                              # part of the cache key
    cache_dir: Path = Path("vlm_cache")
    top_k_crops: int = 3
    save_payload: bool = True                   # dump exactly-what-was-sent for inspection
    payload_dir: Optional[Path] = None          # human-friendly dump location (else next to cache)
    ledger_path: Optional[Path] = None          # append one JSON line per ask() — the audit trail
    eval_tag: str = ""                          # which experiment this ask belongs to
    prompt_template: str = PROMPT               # strategy variants override this (heatmap rows)
    _call_n: int = 0

    def _key(self, goal: str, memory_summary: str, plate_texts: Sequence[str],
             encoded: Sequence[str], hint: str = "") -> str:
        """Cache key over the model, prompt, hint, texts and the ENCODED images —
        i.e. exactly the bytes sent.  A payload-policy change (resize, quality,
        crop) therefore produces new keys; it can never replay stale answers."""
        h = hashlib.sha256()
        h.update(self.model_tag.encode())
        h.update(self.prompt_template.encode())
        h.update(goal.encode()); h.update(memory_summary.encode()); h.update(hint.encode())
        for t in plate_texts:
            h.update(t.encode())
        for b64 in encoded:
            h.update(b64.encode())
        return h.hexdigest()[:32]

    def _ledger(self, row: dict) -> None:
        if self.ledger_path is None:
            return
        Path(self.ledger_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def ask(self, goal: str, plate_texts: Sequence[str],
            crops: Sequence[np.ndarray], scene: Optional[np.ndarray] = None,
            memory_summary: str = "none", meta: Optional[dict] = None,
            hint: str = "") -> VLMAnswer:
        """`hint` = structural grounding from our parser, e.g. which sign line
        contains the goal.  Part of the payload (and the cache key)."""
        images = list(crops[: self.top_k_crops]) + ([scene] if scene is not None else [])
        encoded = [_jpeg_b64(im) for im in images]
        key = self._key(goal, memory_summary, plate_texts, encoded, hint)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        f = self.cache_dir / f"{key}.json"
        tmpl = self.prompt_template
        if "{hint}" not in tmpl:
            tmpl = tmpl.replace("Rules:", "{hint}\nRules:", 1) if "Rules:" in tmpl else tmpl
        prompt = tmpl.format(
            goal=goal, memory_summary=memory_summary, hint=hint,
            plate_texts="\n".join(f"- {t}" for t in plate_texts) or "- (none)") if "{hint}" in tmpl \
            else tmpl.format(goal=goal, memory_summary=memory_summary,
                             plate_texts="\n".join(f"- {t}" for t in plate_texts) or "- (none)")
        pd = None
        if self.save_payload:
            self._call_n += 1
            pd = (Path(self.payload_dir) / f"call{self._call_n:02d}" if self.payload_dir
                  else self.cache_dir / f"{key}_payload")
            pd.mkdir(parents=True, exist_ok=True)
            (pd / "prompt.txt").write_text(prompt)
            for k, b64 in enumerate(encoded):
                # byte-identical to what the model receives
                (pd / f"img{k}.jpg").write_bytes(base64.b64decode(b64))
        base_row = {"ts": time.time(), "eval": self.eval_tag, "model": self.model_tag,
                    "key": key, "goal": goal, "plate_texts": list(plate_texts), "hint": hint,
                    "n_images": len(images), "payload_dir": str(pd) if pd else None,
                    **(meta or {})}
        if f.exists():
            d = json.loads(f.read_text())
            if pd is not None:
                (pd / "answer.json").write_text(json.dumps(
                    {**d, "model": self.model_tag, "cached": True}, indent=2))
            ans = VLMAnswer(**{**d, "cached": True})
            self._ledger({**base_row, "cached": True, "applicable": ans.applicable,
                          "direction": ans.direction, "confidence": ans.confidence,
                          "latency_s": ans.latency_s, "summary": ans.summary})
            return ans
        raw, lat = self.client.complete(prompt, images)
        ans = VLMAnswer.parse(raw, latency_s=lat)
        record = {"applicable": ans.applicable, "direction": ans.direction,
                  "confidence": ans.confidence, "summary": ans.summary,
                  "raw": ans.raw, "latency_s": ans.latency_s}
        f.write_text(json.dumps(record))
        if pd is not None:
            (pd / "answer.json").write_text(json.dumps({**record, "model": self.model_tag}, indent=2))
        self._ledger({**base_row, "cached": False, "applicable": ans.applicable,
                      "direction": ans.direction, "confidence": ans.confidence,
                      "latency_s": ans.latency_s, "summary": ans.summary, "raw": ans.raw[:400]})
        return ans


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="latency probe: tiny text-only calls")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--rpm", type=float, default=15.0, help="client-side pacing (free tier = 15)")
    ap.add_argument("--hedge", type=float, default=None,
                    help="seconds before firing a duplicate request (e.g. 3.0)")
    ap.add_argument("--base-url",
                    default="https://generativelanguage.googleapis.com/v1beta/openai/",
                    help="e.g. http://localhost:8000/v1 for a local vLLM server")
    a = ap.parse_args()
    reasoning = None if (a.reasoning or "").lower() in ("", "none", "off") else a.reasoning
    client = OpenAICompatVLM(model=a.model, base_url=a.base_url, reasoning_effort=reasoning,
                             rpm=a.rpm, hedge_after=a.hedge)
    lats = []
    for i in range(a.n):
        _, lat = client.complete("Reply with exactly: OK", [])
        lats.append(lat)
        print(f"call {i}: {lat:.2f}s")
    import statistics
    lats_sorted = sorted(lats)
    p90 = lats_sorted[max(int(0.9 * len(lats_sorted)) - 1, 0)]
    print(f"text-only floor ({a.model}, reasoning={a.reasoning}, n={len(lats)}): "
          f"median {statistics.median(lats):.2f}s  min {min(lats):.2f}s  "
          f"p90 {p90:.2f}s  max {max(lats):.2f}s")


def _digit_deletion_variants(line: str) -> list[str]:
    """OCR sometimes INSERTS a digit ('2-1011 to2-140'); try every single-digit
    deletion inside numeric tokens.  Cheap, and only used when nothing matched."""
    import re as _re
    out = []
    for m in _re.finditer(r"\d{3,}", line):
        for k in range(len(m.group(0))):
            tok = m.group(0)[:k] + m.group(0)[k + 1:]
            out.append(line[:m.start()] + tok + line[m.end():])
    return out


OVERRIDE_TOKENS = ("CLOSED", "CIOSED", "COSED", "NO ENTRY", "NOENTRY", "DO NOT ENTER",
                   "DETOUR", "USE OTHER", "OUT OF SERVICE", "BLOCKED")


def has_override(plate_text: str) -> bool:
    """Does this plate carry a temporary notice that can CHANGE the action?"""
    t = plate_text.upper().replace("|", " ")
    return any(tok in t for tok in OVERRIDE_TOKENS)


def override_clause(plate_text: str) -> str:
    """Added when a notice is present: the permanent arrow is no longer sufficient."""
    if not has_override(plate_text):
        return ""
    return ("\nIMPORTANT — this sign carries a TEMPORARY NOTICE (e.g. a closure or detour). "
            "The notice OVERRIDES the permanent directory. After reading the arrow, check the "
            "notice: if it closes the direction the arrow gives, report the direction the notice "
            "sends you instead; if it closes the route with no alternative, answer direction "
            "\"stop\". State in your reasoning which notice you applied.")


def structural_hint(plate_text: str, goal_text: str) -> str:
    """What OUR parser knows: which line contains the goal (or that none does).
    Goes into the prompt so the model reads the arrow instead of redoing the
    range arithmetic (which small VLMs get wrong).  Quotes the line TEXT rather
    than a line number: OCR order interleaves the columns of two-column signs,
    so 'line 3' can name a different line than the one the model counts."""
    from .evidence.relevance import Goal, relevance_lines
    lines = [t.strip() for t in plate_text.split("|") if t.strip()]
    if not lines:
        return ""
    goal = Goal.parse(goal_text)
    rel = relevance_lines(lines, goal)
    matched, how, repaired = None, "", False
    if rel.score >= 0.5 and rel.best_line is not None:
        matched = lines[rel.best_line]
        how = "the room number falls inside its range" if rel.lines[rel.best_line].struct >= 1.0 \
            else "the name matches"
    else:
        # OCR-tolerant retry: exactly one single-digit-deletion variant must match
        from .evidence.relevance import parse_rooms_and_ranges
        hits = []
        for i, ln in enumerate(lines):
            for v in _digit_deletion_variants(ln):
                r2 = relevance_lines([v], goal)
                if not (r2.score >= 0.5 and r2.lines[0].struct >= 1.0):
                    continue
                # plausibility: a printed range has endpoints with equal digit counts
                # ("2-101 to 2-140"); "2-70 to 2-175" is not a repair, it's a coincidence
                rngs, _ = parse_rooms_and_ranges(v)
                if any(len(str(r.lo.number)) == len(str(r.hi.number)) for r in rngs):
                    hits.append((i, v))
        if len({i for i, _ in hits}) == 1:
            # quote the ORIGINAL OCR text (that is what the model will see on the sign)
            matched, how, repaired = lines[hits[0][0]], "the room number falls inside its range", True
    if matched:
        return (f'Parser check: the goal matches the sign line reading "{matched}" ({how}'
                f'{"; OCR inserted a spurious digit, the printed range does contain the goal" if repaired else ""}); the text match is '
                f"already verified, so do NOT re-check it.\n"
                f"Your only task is the ARROW. In your reasoning, first list EVERY arrow glyph you can "
                f"see and which line of text it sits beside (say 'points left', 'points right', "
                f"'points up'), then report the arrow beside the line quoted above. "
                f"If no arrow is printed beside that line, answer applicable=false — never infer a "
                f"direction from the corridor, the layout, or the room numbering."
                + override_clause(plate_text))
    return ("Parser check: the OCR text did not yield a numeric match for the goal, but OCR may have "
            "misread digits. Read the sign lines from the IMAGE yourself, decide whether any line's "
            "range or name contains the goal, and if one does, report the arrow beside it. If none "
            "does, answer applicable=false." + override_clause(plate_text))