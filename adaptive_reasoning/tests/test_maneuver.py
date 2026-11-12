from adaptive_reasoning.maneuver import distance_from_text, opening_on_side, speed_cap_from_text

KW = dict(fy_px=2060.0, letter_h_m=0.05, latency_s=4.0, margin_s=1.0, d_buffer_m=1.0, v_min=0.15)


def test_distance_shrinks_as_text_grows():
    assert abs(distance_from_text(20, 2060, 0.05) - 5.15) < 1e-6
    assert distance_from_text(40, 2060, 0.05) < distance_from_text(20, 2060, 0.05)
    assert distance_from_text(0, 2060, 0.05) is None


def test_speed_cap_far_sign_no_slowdown_near_sign_creeps():
    far = speed_cap_from_text(15, **KW)       # ~6.9 m away
    mid = speed_cap_from_text(40, **KW)       # ~2.6 m
    near = speed_cap_from_text(90, **KW)      # ~1.1 m
    assert far > 1.0 > mid > near
    assert near == 0.15                       # never below v_min
    assert speed_cap_from_text(0, **KW) is None


def test_slower_vlm_means_slower_approach():
    fast = speed_cap_from_text(40, **{**KW, "latency_s": 1.5})
    slow = speed_cap_from_text(40, **{**KW, "latency_s": 6.0})
    assert fast > slow


def test_opening_on_side_uses_lidar_sector_convention():
    corridor = [0.0, 30.0, 8.0]                         # forward only
    right_branch = [0.0, 30.0, 8.0, -80.0, 40.0, 5.0]
    left_t = [70.0, 40.0, 6.0]
    assert not opening_on_side(corridor, "turn_right")
    assert opening_on_side(right_branch, "turn_right")
    assert not opening_on_side(right_branch, "turn_left")
    assert opening_on_side(left_t, "turn_left")
    assert not opening_on_side([], "turn_left")
    assert not opening_on_side([-170.0, 20.0, 3.0], "turn_right")   # behind, not right
