"""Tests for evidence/features.py and evidence/legibility.py — synthetic crops."""
import warnings

import numpy as np
import pytest

from adaptive_reasoning.evidence.features import (
    FEATURES, OcrLine, PlateObservation, StringAgreement,
    completeness, foreshortening, ocr_conf, phi, phi_vector, sharpness, text_height_px,
)
from adaptive_reasoning.evidence.legibility import Legibility


def _checker(n=64, cell=4):
    """High-frequency checkerboard — very sharp."""
    y, x = np.indices((n, n))
    return (((y // cell) + (x // cell)) % 2 * 255).astype(np.float64)


def _blur(img, r=3):
    """Box blur via cumulative sums (no scipy)."""
    out = img.astype(np.float64)
    for ax in (0, 1):
        k = 2 * r + 1
        pad = np.take(out, np.clip(np.arange(-r, out.shape[ax] + r), 0, out.shape[ax] - 1), axis=ax)
        c = np.cumsum(pad, axis=ax)
        out = (np.take(c, np.arange(k - 1, k - 1 + out.shape[ax]), axis=ax)
               - np.concatenate([np.take(c, [0], axis=ax) * 0,
                                 np.take(c, np.arange(0, out.shape[ax] - 1), axis=ax)], axis=ax)) / k
    return out


def _plate(lines=None, crop=None, quad=None):
    lines = lines if lines is not None else [OcrLine("6-201 to 6-250", 0.9, 30.0, ["right"])]
    return PlateObservation(lines, crop, quad)


# ---------------- individual features ----------------

def test_text_height_median_of_nonempty_lines():
    p = _plate([OcrLine("A", 0.9, 20.0), OcrLine("B", 0.9, 30.0), OcrLine("", 0.9, 999.0)])
    assert text_height_px(p) == 25.0
    assert text_height_px(_plate([])) == 0.0


def test_sharpness_sharp_beats_blurred_and_flat_is_zero():
    sharp = _checker()
    assert sharpness(_plate(crop=sharp)) > sharpness(_plate(crop=_blur(sharp))) * 2
    assert sharpness(_plate(crop=np.full((32, 32), 128.0))) == 0.0
    assert sharpness(_plate(crop=None)) == 0.0


def test_sharpness_accepts_rgb():
    rgb = np.stack([_checker()] * 3, axis=-1)
    assert sharpness(_plate(crop=rgb)) > 0


def test_ocr_conf_mean():
    p = _plate([OcrLine("A", 0.8, 20), OcrLine("B", 0.6, 20)])
    assert ocr_conf(p) == pytest.approx(0.7)


def test_completeness_needs_text_and_arrow():
    assert completeness(_plate([OcrLine("6-110", 0.9, 20, [])])) == 0.0        # door plate
    assert completeness(_plate([OcrLine("", 0.9, 20, ["left"])])) == 0.0       # arrow only
    assert completeness(_plate([OcrLine("6-201 to 6-250", 0.9, 20, ["left"])])) == 1.0


def test_foreshortening_frontal_vs_oblique():
    frontal = np.array([[0, 0], [100, 0], [100, 40], [0, 40]])
    oblique = np.array([[0, 0], [100, 10], [100, 30], [0, 40]])   # right edge half as tall
    assert foreshortening(_plate(quad=frontal)) == pytest.approx(1.0)
    assert foreshortening(_plate(quad=oblique)) == pytest.approx(0.5)
    assert foreshortening(_plate(quad=None)) == 1.0


def test_string_agreement_window():
    agr = StringAgreement(k=4)
    assert agr.update("6-201 to 6-250") == pytest.approx(1 / 4)
    assert agr.update("6-201 to 6-250") == pytest.approx(2 / 4)
    assert agr.update("6-2O1 to 6-25O") == pytest.approx(1 / 4)   # OCR flicker: O vs 0
    assert agr.update("6-201 to 6-250") == pytest.approx(3 / 4)   # window [A,A,flicker,A]
    assert agr.update("") == 0.0


def test_phi_assembles_all_features_in_order():
    d = phi(_plate(crop=_checker()), agreement=0.75)
    assert list(d) == FEATURES and d["agreement_k"] == 0.75
    assert phi_vector(d).shape == (len(FEATURES),)


# ---------------- legibility ----------------

def test_placeholder_warns_and_is_monotone_in_evidence():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        leg = Legibility.placeholder()
        good = phi(_plate([OcrLine("6-201 to 6-250", 0.95, 60.0, ["right"])], crop=_checker()), 1.0)
        bad = phi(_plate([OcrLine("6-2", 0.2, 6.0, [])], crop=_blur(_checker(), 6)), 0.25)
        assert leg(good) > leg(bad)
        assert 0.0 < leg(bad) < leg(good) <= 1.0 and leg(bad) < 0.5 < leg(good)
        assert any("UNCALIBRATED" in str(x.message) for x in w)


def test_load_save_roundtrip_and_feature_mismatch(tmp_path):
    leg = Legibility(np.zeros(6), np.ones(6), np.arange(6, dtype=float), -1.0)
    leg.save(tmp_path / "w.json", meta={"fit": "test"})
    leg2 = Legibility.load(tmp_path / "w.json")
    x = phi(_plate(crop=_checker()), 0.5)
    assert leg2(x) == pytest.approx(leg(x)) and leg2.calibrated
    # corrupt the feature list -> must refuse to load
    import json
    d = json.loads((tmp_path / "w.json").read_text()); d["features"] = ["a", "b"]
    (tmp_path / "w.json").write_text(json.dumps(d))
    with pytest.raises(ValueError):
        Legibility.load(tmp_path / "w.json")


def test_load_missing_file_gives_placeholder():
    leg = Legibility.load("/nonexistent/w.json")
    assert not leg.calibrated


# ---------------- payload crop (shared live/calibration construction) ----------

def test_payload_box_grows_to_swallow_panel_fragments():
    from adaptive_reasoning.evidence.detect import payload_box
    frag = (1000, 560, 1060, 585)                   # the bare "B50" line
    panel = [(995, 530, 1080, 555), (998, 590, 1075, 615), (1000, 620, 1090, 650)]
    far = [(300, 800, 360, 820)]
    grown = payload_box(frag, panel + far)
    assert grown[0] < 995 and grown[2] > 1090       # covers the whole panel
    assert grown[1] < 530 and grown[3] > 650
    assert grown[0] > 400                            # but NOT the far-away notice


def test_payload_crop_enforces_minimum_size():
    import numpy as np
    from adaptive_reasoning.evidence.detect import payload_crop, DetectConfig
    img = np.zeros((1080, 1920, 3), np.uint8)
    tiny = (1000, 560, 1030, 575)
    c = payload_crop(img, tiny, [])
    assert max(c.shape[:2]) >= DetectConfig().min_crop_px


def test_payload_crop_clips_to_frame():
    import numpy as np
    from adaptive_reasoning.evidence.detect import payload_crop
    img = np.zeros((200, 300, 3), np.uint8)
    c = payload_crop(img, (10, 10, 40, 30), [])
    assert c.shape[0] <= 200 and c.shape[1] <= 300 and c.size > 0
