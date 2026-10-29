"""OCR 距離テキストの解析（単位 m なし・末尾ノイズ）"""

import numpy as np
import pytest

from src.ocr_utils import (
    OCRConfig,
    OCRTextNotFoundError,
    _parse_distance_text,
    extract_distance_from_frame,
    is_ocr_digit_confusion,
    ocr_confusion_alternates_mm,
    resolve_ocr_distance_with_range,
)


def _parse(text: str) -> float:
    cfg = OCRConfig()
    return _parse_distance_text(
        text, cfg.distance_pattern_decimal, cfg.distance_pattern_integer
    )


def test_parse_accepts_unitless_decimal_with_trailing_dot():
    assert _parse("0.00.\n") == pytest.approx(0.0)
    assert _parse("0.00.") == pytest.approx(0.0)
    assert _parse("0.00") == pytest.approx(0.0)


def test_parse_still_accepts_meters_suffix():
    assert _parse("4.10m") == pytest.approx(4100.0)
    assert _parse("12.34m") == pytest.approx(12340.0)
    assert _parse("410m") == pytest.approx(4100.0)


def test_parse_unitless_decimal_is_meters():
    assert _parse("4.10") == pytest.approx(4100.0)


def test_parse_rejects_empty():
    with pytest.raises(OCRTextNotFoundError):
        _parse("")
    with pytest.raises(OCRTextNotFoundError):
        _parse("...")


def test_confusion_repairs_960_and_760_to_260():
    alts = ocr_confusion_alternates_mm(960.0)
    assert 260.0 in alts
    assert resolve_ocr_distance_with_range(960.0, (220.0, 310.0)) == pytest.approx(260.0)
    assert resolve_ocr_distance_with_range(760.0, (220.0, 310.0)) == pytest.approx(260.0)
    assert resolve_ocr_distance_with_range(260.0, (220.0, 310.0)) == pytest.approx(260.0)
    assert resolve_ocr_distance_with_range(9600.0, (220.0, 310.0)) is None


def test_extract_with_preprocessing_repairs_960(monkeypatch):
    def fake_extract(*_args, **_kwargs):
        return "0.96"

    monkeypatch.setattr("src.ocr_utils._extract_with_threshold", fake_extract)
    cfg = OCRConfig(
        preprocessing_enabled=True,
        preprocessing_method="otsu",
        retry_thresholds=[100, 160, 180, 200],
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    dist, _ = extract_distance_from_frame(
        frame,
        (0.0, 0.5, 0.0, 0.5),
        cfg,
        expected_range_mm=(220.0, 310.0),
    )
    assert dist == pytest.approx(260.0)


def test_extract_with_preprocessing_uses_retry_threshold_180(monkeypatch):
    def fake_extract(frame, roi, threshold, psm_mode, preprocessing_method="fixed"):
        if preprocessing_method == "fixed" and threshold == 180:
            return "0.26"
        raise OCRTextNotFoundError("fail")

    monkeypatch.setattr("src.ocr_utils._extract_with_threshold", fake_extract)
    cfg = OCRConfig(
        preprocessing_enabled=True,
        preprocessing_method="otsu",
        retry_thresholds=[100, 160, 180, 200],
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    dist, used = extract_distance_from_frame(
        frame,
        (0.0, 0.5, 0.0, 0.5),
        cfg,
        expected_range_mm=(220.0, 310.0),
    )
    assert dist == pytest.approx(260.0)
    assert used == 180


def test_confusion_220_270_and_410_470():
    assert 270.0 in ocr_confusion_alternates_mm(220.0)
    assert 220.0 in ocr_confusion_alternates_mm(270.0)
    assert is_ocr_digit_confusion(220.0, 270.0)
    assert is_ocr_digit_confusion(410.0, 470.0)
    assert is_ocr_digit_confusion(440.0, 470.0)
    assert not is_ocr_digit_confusion(220.0, 260.0)


def test_confusion_repairs_1400_to_1300_even_when_in_range():
    assert 1300.0 in ocr_confusion_alternates_mm(1400.0)
    assert is_ocr_digit_confusion(1300.0, 1400.0)
    assert resolve_ocr_distance_with_range(1400.0, (1270.0, 1420.0)) == pytest.approx(1300.0)
    assert resolve_ocr_distance_with_range(1400.0, (1270.0, 1330.0)) == pytest.approx(1300.0)
    assert resolve_ocr_distance_with_range(1300.0, (1270.0, 1420.0)) == pytest.approx(1300.0)


def test_extract_prefers_plateau_220_over_270(monkeypatch):
    def fake_extract(frame, roi, threshold, psm_mode, preprocessing_method="fixed"):
        if preprocessing_method == "fixed" and threshold == 180:
            return "0.22"
        return "0.27"

    monkeypatch.setattr("src.ocr_utils._extract_with_threshold", fake_extract)
    cfg = OCRConfig(
        preprocessing_enabled=True,
        preprocessing_method="otsu",
        retry_thresholds=[100, 160, 180, 200],
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    dist, _ = extract_distance_from_frame(
        frame,
        (0.0, 0.5, 0.0, 0.5),
        cfg,
        expected_range_mm=(220.0, 280.0),
    )
    assert dist == pytest.approx(220.0)


def test_extract_snaps_lone_270_back_to_plateau_220(monkeypatch):
    def fake_extract(*_args, **_kwargs):
        return "0.27"

    monkeypatch.setattr("src.ocr_utils._extract_with_threshold", fake_extract)
    cfg = OCRConfig(
        preprocessing_enabled=True,
        preprocessing_method="otsu",
        retry_thresholds=[100, 160, 180, 200],
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    dist, _ = extract_distance_from_frame(
        frame,
        (0.0, 0.5, 0.0, 0.5),
        cfg,
        expected_range_mm=(220.0, 280.0),
    )
    assert dist == pytest.approx(220.0)


def test_extract_accepts_real_countup_220_to_260(monkeypatch):
    def fake_extract(*_args, **_kwargs):
        return "0.26"

    monkeypatch.setattr("src.ocr_utils._extract_with_threshold", fake_extract)
    cfg = OCRConfig(
        preprocessing_enabled=True,
        preprocessing_method="otsu",
        retry_thresholds=[100, 160, 180, 200],
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    dist, _ = extract_distance_from_frame(
        frame,
        (0.0, 0.5, 0.0, 0.5),
        cfg,
        expected_range_mm=(220.0, 280.0),
    )
    assert dist == pytest.approx(260.0)


def test_extract_repairs_1400_to_1300_when_plateau_is_1270(monkeypatch):
    def fake_extract(*_args, **_kwargs):
        return "1.40"

    monkeypatch.setattr("src.ocr_utils._extract_with_threshold", fake_extract)
    cfg = OCRConfig(
        preprocessing_enabled=True,
        preprocessing_method="otsu",
        retry_thresholds=[100, 160, 180, 200],
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    dist, _ = extract_distance_from_frame(
        frame,
        (0.0, 0.5, 0.0, 0.5),
        cfg,
        expected_range_mm=(1270.0, 1420.0),
    )
    assert dist == pytest.approx(1300.0)
