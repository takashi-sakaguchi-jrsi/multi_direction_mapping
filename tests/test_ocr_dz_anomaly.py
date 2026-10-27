"""推定 dz と OCR 窓平均の比による異常判定"""

from src.config import FrontConfig
from src.main_twopass import is_abnormally_small_dz


def test_abnormally_small_when_ocr_moving():
    assert is_abnormally_small_dz(
        0.02, 6.0, ratio=0.3, min_ocr_mm=1.0
    )
    assert is_abnormally_small_dz(
        0.0, 6.0, ratio=0.3, min_ocr_mm=1.0
    )


def test_not_anomaly_when_dz_comparable_to_ocr():
    assert not is_abnormally_small_dz(
        5.0, 6.0, ratio=0.3, min_ocr_mm=1.0
    )
    assert not is_abnormally_small_dz(
        1.9, 6.0, ratio=0.3, min_ocr_mm=1.0
    )


def test_skip_when_ocr_also_stopped():
    assert not is_abnormally_small_dz(
        0.0, 0.2, ratio=0.3, min_ocr_mm=1.0
    )


def test_skip_when_ocr_missing_or_disabled():
    assert not is_abnormally_small_dz(0.0, None, ratio=0.3, min_ocr_mm=1.0)
    assert not is_abnormally_small_dz(0.0, 6.0, ratio=0.0, min_ocr_mm=1.0)
    assert not is_abnormally_small_dz(None, 6.0, ratio=0.3, min_ocr_mm=1.0)
    assert not is_abnormally_small_dz(
        0.0, float("nan"), ratio=0.3, min_ocr_mm=1.0
    )


def test_front_config_defaults():
    front = FrontConfig()
    assert front.ocr_dz_anomaly_ratio == 0.3
    assert front.ocr_dz_anomaly_min_ocr_mm == 1.0


def test_boundary_equals_ratio_is_not_anomaly():
    assert not is_abnormally_small_dz(
        1.8, 6.0, ratio=0.3, min_ocr_mm=1.0
    )
    assert is_abnormally_small_dz(
        1.79, 6.0, ratio=0.3, min_ocr_mm=1.0
    )
