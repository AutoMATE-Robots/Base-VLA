"""Tests for evidence/resolve.py — the fast path."""
import warnings

import pytest

from adaptive_reasoning.evidence.relevance import Goal
from adaptive_reasoning.evidence.resolve import Plate, PlateLine, resolve, resolve_all


def _g(text, *desc):
    return Goal.parse(text, desc)


# ---------------- parsing ----------------

def test_plate_from_text_tokens_and_glyphs():
    p = Plate.from_text("6-201 to 6-250 > | Main Elevators \u2190 | Restrooms")
    assert p.texts == ["6-201 to 6-250", "Main Elevators", "Restrooms"]
    assert [ln.arrows for ln in p.lines] == [["right"], ["left"], []]


def test_plate_from_text_does_not_eat_the_v_in_words():
    p = Plate.from_text("Elevators v")
    assert p.lines[0].text == "Elevators" and p.lines[0].arrows == ["down"]


def test_plate_from_text_arrow_only_line():
    p = Plate.from_text("Amundson Hall | 2-101 to 2-140 | <")
    assert p.texts == ["Amundson Hall", "2-101 to 2-140", ""]
    assert p.all_directions == {"left"}


# ---------------- the fast path ----------------

def test_resolves_goal_line_with_one_arrow():
    r = resolve(Plate.from_text("6-201 to 6-250 > | Main Elevators <"), _g("6-217"))
    assert r.resolved and r.direction == "right" and r.vla_prompt == "turn_right"
    assert r.line_index == 0 and r.source == "line" and r.channel == "struct"


def test_up_arrow_means_straight():
    r = resolve(Plate.from_text("6-201 to 6-250 ^"), _g("6-217"))
    assert r.resolved and r.vla_prompt == "straight"


def test_down_arrow_is_left_to_the_vlm():
    r = resolve(Plate.from_text("6-201 to 6-250 v"), _g("6-217"))
    assert r.status == "unresolved" and r.direction == "down" and r.vla_prompt is None


def test_two_arrows_on_goal_line_is_ambiguous():
    r = resolve(Plate.from_text("6-201 to 6-250 < >"), _g("6-217"))
    assert r.status == "ambiguous"


def test_no_arrow_anywhere_is_unresolved():
    r = resolve(Plate.from_text("6-201 to 6-250 | Restrooms"), _g("6-217"))
    assert r.status == "unresolved" and "no arrow" in r.reason


def test_irrelevant_plate_is_unresolved_even_with_arrows():
    r = resolve(Plate.from_text("COMPOSTO > | Recycling <"), _g("6-217"))
    assert r.status == "unresolved" and "no line relevant" in r.reason


def test_plate_level_single_arrow_serves_goal_line():
    r = resolve(Plate.from_text("Amundson Hall | 2-101 to 2-140 | <"), _g("2-125"))
    assert r.resolved and r.direction == "left" and r.source == "plate" and r.line_index == 1


def test_plate_level_two_arrows_does_not_serve_goal_line():
    # goal line has no arrow; plate has two different arrows → we can't tell which → ambiguous
    r = resolve(Plate.from_text("Amundson Hall < | 2-101 to 2-140 | Restrooms >"), _g("2-125"))
    assert r.status == "ambiguous"


def test_semantic_only_match_does_not_resolve_by_default():
    g = _g("2-125", "Amundson Hall", "second floor")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strict = resolve(Plate.from_text("Amundson Hall <"), g)
        loose = resolve(Plate.from_text("Amundson Hall <"), g, require_struct=False)
    assert strict.status == "unresolved" and strict.channel == "sem"
    assert loose.resolved and loose.direction == "left" and loose.channel == "sem"


def test_struct_line_preferred_over_semantic_line_when_both_present():
    g = _g("2-125", "Amundson Hall")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = resolve(Plate.from_text("Amundson Hall < | 2-101 to 2-140 >"), g)
    assert r.resolved and r.direction == "right" and r.line_index == 1 and r.channel == "struct"


# ---------------- resolve_all (for memory) ----------------

def test_resolve_all_per_line():
    p = Plate.from_text("6-201 to 6-250 > | Main Elevators < | Restrooms | Exit < >")
    assert resolve_all(p) == ["right", "left", None, None]


def test_resolve_all_shared_arrow():
    p = Plate.from_text("Amundson Hall | 2-101 to 2-140 | <")
    assert resolve_all(p) == ["left", "left", "left"]


def test_plate_dataclass_direct_construction():
    # detect.py will build plates this way at runtime
    p = Plate([PlateLine("6-201 to 6-250", ["right"]), PlateLine("Main Elevators", ["left"])])
    assert resolve(p, _g("6-217")).vla_prompt == "turn_right"
