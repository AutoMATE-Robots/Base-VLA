"""
expand_goal.py — V_g: generated, not curated.

The rule this file enforces: NO HUMAN writes the descriptor list.  Descriptors
(the semantic vocabulary for a goal) come from ONE text-only LLM call made
when the goal is set — blind to any specific sign in any bag — and are cached
to goals/<goal>.json with provenance metadata:

    {"goal": "...", "descriptors": [...],
     "meta": {"source": "llm", "model": "...", "context": "...", "date": "..."}}

Why a file still exists at all: it is a *cache and an audit trail*, not a
knob.  The per-goal call costs nothing per frame, the cached json makes every
run reproducible, and the meta block lets the validator (and a reviewer)
verify that relevance was never tuned by reading the evaluation signs.
validate_annotations warns on any descriptor file whose meta.source != "llm".

The prompt asks what signs *in this kind of building* might print for the
goal.  It never sees bag data, sign transcriptions, or building-specific
phrasing beyond the context string the operator would genuinely know
(e.g. "university campus building").

Usage (MSI, `ar` env, after `source ~/.secrets`):
  python -m adaptive_reasoning.evidence.expand_goal "Main Elevators" \
      --model gemini-3.6-flash --out-dir adaptive_reasoning/goals
Room-code goals don't need expansion (the structural channel handles them);
running on one anyway just records building/floor phrasings.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from ..reasoner import VLMClient

PROMPT = """You know how indoor signage refers to destinations.
Destination: "{goal}" — somewhere inside a {context}.
List 5-12 SHORT phrases (1-4 words each) that signs in such a building might
literally print to direct people toward this destination: the name itself,
common variants, category names, and typical sign phrasings.
Do NOT invent room numbers or building names. Do NOT explain.
Answer with a JSON array of strings only."""


def parse_descriptor_list(raw: str, goal: str, max_items: int = 12) -> list[str]:
    t = raw.strip()
    if "```" in t:
        t = t.split("```")[1]
        t = t[4:] if t.startswith("json") else t
    try:
        arr = json.loads(t[t.index("["): t.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        arr = []
    out, seen = [], set()
    for item in [goal, *arr]:                       # goal itself always first
        s = str(item).strip()
        key = s.lower()
        if s and len(s) <= 60 and key not in seen:
            seen.add(key)
            out.append(s)
        if len(out) >= max_items:
            break
    return out


def expand_goal(goal: str, client: VLMClient,
                context: str = "university campus building") -> list[str]:
    raw, _ = client.complete(PROMPT.format(goal=goal, context=context), [])
    return parse_descriptor_list(raw, goal)


def write_goal_file(goal: str, descriptors: list[str], out_dir: str | Path,
                    model: str, context: str) -> Path:
    p = Path(out_dir) / f"{goal}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "goal": goal, "descriptors": descriptors,
        "meta": {"source": "llm", "model": model, "context": context,
                 "date": datetime.date.today().isoformat()},
    }, indent=2))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("goal")
    ap.add_argument("--context", default="university campus building")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--base-url",
                    default="https://generativelanguage.googleapis.com/v1beta/openai/")
    ap.add_argument("--out-dir", default="adaptive_reasoning/goals")
    a = ap.parse_args()
    from ..reasoner import OpenAICompatVLM
    client = OpenAICompatVLM(model=a.model, base_url=a.base_url)
    desc = expand_goal(a.goal, client, a.context)
    path = write_goal_file(a.goal, desc, a.out_dir, a.model, a.context)
    print(f"{path}:")
    for d in desc:
        print(f"  - {d}")
