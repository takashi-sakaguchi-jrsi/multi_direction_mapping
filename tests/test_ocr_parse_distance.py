"""OCR 距離テキストの解析（単位 m なし・末尾ノイズ）"""

import pytest

from src.ocr_utils import OCRConfig, OCRTextNotFoundError, _parse_distance_text


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
