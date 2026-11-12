"""Tests for expand_goal.py: parsing, dedup, provenance file, blind prompt."""
import json

from adaptive_reasoning.evidence.expand_goal import (
    PROMPT, expand_goal, parse_descriptor_list, write_goal_file,
)
from adaptive_reasoning.evidence.relevance import Goal
from adaptive_reasoning.reasoner import FakeVLM


def test_parse_plain_and_fenced_and_garbage():
    assert parse_descriptor_list('["Elevators", "Lifts"]', "Main Elevators") == \
        ["Main Elevators", "Elevators", "Lifts"]
    fenced = '```json\n["Restroom", "Toilets"]\n```'
    assert parse_descriptor_list(fenced, "Restrooms") == ["Restrooms", "Restroom", "Toilets"]
    assert parse_descriptor_list("no json here", "Exit") == ["Exit"]   # goal always survives


def test_dedup_case_insensitive_and_cap():
    raw = json.dumps(["Main Elevators", "main elevators", "Elevators"] + [f"x{i}" for i in range(20)])
    out = parse_descriptor_list(raw, "Main Elevators")
    assert out[0] == "Main Elevators" and "Elevators" in out
    assert len(out) <= 12 and len([d for d in out if d.lower() == "main elevators"]) == 1


def test_expand_goal_via_fake_client_and_prompt_is_blind():
    fake = FakeVLM({"Main Elevators": ["ignored"]})  # FakeVLM returns not-applicable JSON…
    # …so use a script-free check of the prompt content instead:
    assert "sign_text" not in PROMPT and "bag" not in PROMPT
    class Client:
        def complete(self, prompt, images):
            assert "Main Elevators" in prompt and images == []
            return '["Main Elevators, Stairs", "Elevators", "Elevator Lobby"]', 0.01
    desc = expand_goal("Main Elevators", Client())
    assert desc[0] == "Main Elevators" and "Elevator Lobby" in desc


def test_write_goal_file_provenance_and_goal_load(tmp_path):
    p = write_goal_file("Main Elevators", ["Main Elevators", "Elevators"],
                        tmp_path, model="gemini-3.6-flash", context="university campus building")
    d = json.loads(p.read_text())
    assert d["meta"]["source"] == "llm" and d["meta"]["model"] == "gemini-3.6-flash"
    g = Goal.load(p)                      # Goal.load tolerates the meta block
    assert g.descriptors == ["Main Elevators", "Elevators"]
