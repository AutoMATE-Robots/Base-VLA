"""Tests for evidence/relevance.py.

Every example here is a sign format we expect on campus or a row from the
annotation protocol.  Add a test whenever an annotator finds a format that
misparses — that's how the parser grows.
"""
import warnings

import pytest

from adaptive_reasoning.evidence.relevance import (
    Goal, Room, RoomRange, SemanticCalibration,
    normalize, split_lines, parse_room, parse_rooms_and_ranges,
    structural_relevance, relevance,
)


# ---------------- normalisation ----------------

def test_normalize_unifies_dashes_and_strips_arrows():
    assert normalize("6\u2013201 to 6\u2014250 \u2192") == "6-201 TO 6-250"
    assert normalize("Main Elevators <") == "MAIN ELEVATORS"
    assert normalize("Restrooms v") == "RESTROOMS"          # annotation 'v' arrow token


def test_split_lines_on_pipe_and_newline():
    assert split_lines("6-201 to 6-250 > | Main Elevators <") == ["6-201 to 6-250 >", "Main Elevators <"]
    assert split_lines("Amundson Hall <\nRestrooms >") == ["Amundson Hall <", "Restrooms >"]


# ---------------- room codes ----------------

@pytest.mark.parametrize("text, expected", [
    ("6-210", Room("6", 210)),
    ("210", Room("", 210)),
    ("2-111A", Room("2", 111, "A")),
    ("B12", Room("B", 12)),
    ("B-12", Room("B", 12)),
    ("6210", Room("", 6210)),       # no dash → not "62"+"01"
    ("elevator", None),
])
def test_parse_room(text, expected):
    assert parse_room(text) == expected


@pytest.mark.parametrize("line, ranges, rooms", [
    ("6-201 to 6-250",      [RoomRange(Room("6", 201), Room("6", 250))], []),
    ("6-201 – 6-250",       [RoomRange(Room("6", 201), Room("6", 250))], []),
    ("6-201-6-250",         [RoomRange(Room("6", 201), Room("6", 250))], []),
    ("Rooms 201-250",       [RoomRange(Room("", 201), Room("", 250))], []),
    ("6-201 to 250",        [RoomRange(Room("6", 201), Room("6", 250))], []),   # prefix inherited
    ("6-201 thru 6-250",    [RoomRange(Room("6", 201), Room("6", 250))], []),
    ("12-201",              [], [Room("12", 201)]),                            # floor 12, NOT 12..201
    ("6-210, 6-212",        [], [Room("6", 210), Room("6", 212)]),
    ("Main Elevators",      [], []),
    ("6-201 to 6-250 | Restrooms", [RoomRange(Room("6", 201), Room("6", 250))], []),
])
def test_parse_rooms_and_ranges(line, ranges, rooms):
    assert parse_rooms_and_ranges(line) == (ranges, rooms)


# ---------------- structural relevance ----------------

@pytest.mark.parametrize("line, goal, expected", [
    ("6-201 to 6-250 >", "6-217", 1.0),
    ("6-201 to 6-250 >", "6-250", 1.0),     # endpoint inclusive
    ("6-201 to 6-250 >", "6-201", 1.0),
    ("6-201 to 6-250 >", "6-251", 0.0),
    ("6-201 to 6-250 >", "7-217", 0.0),     # right number, wrong floor
    ("Rooms 201-250",   "6-217", 1.0),      # sign omits floor → number-only match
    ("6-201 to 6-250",  "217",   1.0),      # goal omits floor
    ("6-110",           "6-210", 0.0),      # door plate, not our room
    ("6-210",           "6-210", 1.0),      # door plate, our room
    ("Main Elevators <", "6-217", 0.0),
    ("COMPOSTO",        "6-217", 0.0),
    ("Main Elevators <", "elevator", 0.0),  # typed goals have NO structural channel — R_sem via V_g
])
def test_structural_relevance(line, goal, expected):
    score, why = structural_relevance(line, Goal.parse(goal))
    assert score == expected, why


# ---------------- whole plate ----------------

def test_plate_picks_best_line_and_explains():
    g = Goal.parse("6-217")
    r = relevance("6-201 to 6-250 > | Main Elevators < | Restrooms >", g)
    assert r.score == 1.0 and r.best_line == 0
    assert "range" in r.lines[0].reason
    g_el = Goal.parse("elevator", descriptors=["elevator", "elevators", "lift"])  # V_g for a typed goal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = relevance("6-201 to 6-250 > | Main Elevators < | Restrooms >", g_el)
    # stub embedder: "MAIN ELEVATORS" vs "elevators" shares one of two tokens → cos 0.71; a real encoder scores higher
    assert r2.score > 0.5 and r2.best_line == 1 and r2.lines[1].struct == 0.0 and "semantic" in r2.lines[1].reason


def test_typed_goal_without_descriptors_matches_its_own_word():
    g = Goal.parse("exit")
    assert g.room is None and g.descriptors == ["exit"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert relevance("Exit >", g).score == 1.0
        assert relevance("Main Elevators <", g).score == 0.0


def test_relevance_lines_keeps_indices_for_empty_lines():
    from adaptive_reasoning.evidence.relevance import relevance_lines
    r = relevance_lines(["Amundson Hall", "", "2-101 to 2-140"], Goal.parse("2-125"))
    assert len(r.lines) == 3 and r.best_line == 2 and r.lines[1].reason == "empty line"


def test_irrelevant_plate_is_zero_without_descriptors():
    assert relevance("COMPOSTO | Please recycle", Goal.parse("6-217")).score == 0.0
    assert relevance("", Goal.parse("6-217")).score == 0.0


def test_semantic_catches_building_sign():
    # The R_sem case: sign names the building, goal is a room inside it.
    g = Goal.parse("2-125", descriptors=["Amundson Hall", "Amundson", "second floor", "Chemical Engineering"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # identity calibration warns — expected until T6
        r = relevance("Amundson Hall <", g)
        r_bad = relevance("COMPOSTO", g)
    assert r.score > 0.5 and r.lines[0].struct == 0.0 and "semantic" in r.lines[0].reason
    assert r_bad.score == 0.0


def test_structural_beats_semantic_when_both_present():
    g = Goal.parse("6-217", descriptors=["Keller Hall"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = relevance("6-201 to 6-250 > | Keller Hall ^", g)
    assert r.best_line == 0 and r.lines[0].struct == 1.0


def test_semantic_calibration_logistic():
    c = SemanticCalibration(kind="logistic", a=10.0, b=-5.0)
    assert c(0.5) == pytest.approx(0.5)
    assert c(0.9) > 0.95 and c(0.1) < 0.05


def test_goal_roundtrip(tmp_path):
    g = Goal.parse("6-217", ["Keller Hall", "sixth floor"])
    g.save(tmp_path / "g.json")
    g2 = Goal.load(tmp_path / "g.json")
    assert g2.room == Room("6", 217) and g2.descriptors == ["Keller Hall", "sixth floor"]


def test_small_two_digit_ranges_parse_as_ranges_not_floor_codes():
    """Rapson prints 'Rooms 43-58' and 'Rooms 1-37, 63-71'; goal '49' must hit."""
    import warnings
    warnings.simplefilter("ignore")
    from adaptive_reasoning.evidence.relevance import Goal, relevance, parse_rooms_and_ranges
    line = "Rooms 43-58 < | Rooms 1-37, 63-71 >"
    assert relevance(line, Goal.parse("49")).best_line == 0
    assert relevance(line, Goal.parse("20")).best_line == 1
    assert relevance(line, Goal.parse("65")).best_line == 1
    assert relevance(line, Goal.parse("40")).score == 0.0
    # the floor-room reading must survive for the 3-digit case
    rngs, rooms = parse_rooms_and_ranges("12-201")
    assert not rngs and rooms[0].prefix == "12" and rooms[0].number == 201
