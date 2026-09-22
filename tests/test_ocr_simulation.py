"""10mm 遅れ OCR と 4.5mm/frame 走行のシミュレーション"""

import numpy as np
import pytest

from src.validation.ocr_simulation import (
    DEFAULT_Z_STEP_MM,
    OCR_STEP_MM,
    ocr_metadata_fields,
    simulate_ocr_distance_mm,
)


def test_ocr_does_not_count_up_on_exact_tick():
    z = np.array([40.0, 45.0, 50.0, 55.0])
    ocr = simulate_ocr_distance_mm(z)
    np.testing.assert_allclose(ocr, [40.0, 40.0, 40.0, 50.0])


def test_ocr_counts_up_only_after_crossing():
    z = np.array([10.0, 14.5, 19.0, 23.5, 28.0, 32.5, 37.0, 41.5, 46.0, 50.5])
    ocr = simulate_ocr_distance_mm(z)
    np.testing.assert_allclose(
        ocr, [10.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 50.0]
    )
    assert 50.0 not in z
    assert ocr[-1] == 50.0


def test_four_point_five_misses_early_10mm_ticks():
    """5mm/frame だと 20,30,...50 に乗るが、4.5mm ではすぐには同期しない。"""
    assert DEFAULT_Z_STEP_MM == pytest.approx(4.5)
    z = 10.0 + np.arange(20) * DEFAULT_Z_STEP_MM
    for t in (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0):
        assert not np.any(np.abs(z - t) < 1e-9), f"z が {t}mm に乗っている"


def test_five_mm_step_would_sync_with_ocr_ticks():
    z = 10.0 + np.arange(10) * 5.0
    assert 50.0 in set(z.tolist())


def test_ocr_metadata_fields_include_true_and_ocr():
    z = 10.0 + np.arange(8) * 4.5
    meta = ocr_metadata_fields(z, z_step_mm=4.5)
    assert meta["ocr_step_mm"] == 10.0
    assert meta["z_step_mm"] == 4.5
    assert len(meta["ocr_z_mm"]) == 8
    assert meta["z_values_mm"][0] == pytest.approx(10.0)
    assert meta["ocr_z_mm"][0] == pytest.approx(10.0)
    assert all(v % 10 == 0 for v in meta["ocr_z_mm"])
