"""T4: OCR-only の z_positions が既存 OCR 系列と整合すること"""

from unittest.mock import patch

import numpy as np
import pytest

from src.camera_estimation import (
    CameraEstimator,
    compute_distance_constraints_only,
    ocr_z_ocr_max_mm,
    ocr_allowed_inclusive_hi_mm,
    ocr_expected_range_mm,
    validate_ocr_distance_value,
)
from src.config import Config
from src.ocr_utils import estimate_high_precision_z_positions, compute_average_speed


def test_ocr_only_z_matches_high_precision_helper(transformer):
    cfg = Config.from_defaults()
    cfg.estimation.use_offset_moving_average = False
    cfg.estimation.use_ocr_z_constraints = True
    estimator = CameraEstimator(cfg.estimation, transformer)
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)]
    distances = [100.0, 108.0, 116.0, 124.0, 132.0]

    def fake_extract(frame, roi, ocr_cfg, **kwargs):
        idx = next(i for i, f in enumerate(frames) if f is frame)
        return distances[idx], 90

    with patch("src.ocr_utils.extract_distance_from_frame", side_effect=fake_extract):
        constraints, z_pos, ocr_dist, success = compute_distance_constraints_only(
            frames=frames,
            estimator=estimator,
            config=cfg.estimation,
        )

    assert np.all(success)
    normalized = ocr_dist - ocr_dist[0]
    avg = compute_average_speed(normalized, success)
    expected = estimate_high_precision_z_positions(normalized, success, avg)
    np.testing.assert_allclose(z_pos, expected, atol=1e-6)
    assert "dz" in constraints[1]


def _run_ocr_only_with_distances(
    transformer, distances, max_increment=10.0, enforce_upper=True,
    countup_mean_dz_max=10.0,
):
    cfg = Config.from_defaults()
    cfg.estimation.use_offset_moving_average = False
    estimator = CameraEstimator(cfg.estimation, transformer)
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in distances]

    def fake_extract(frame, roi, ocr_cfg, **kwargs):
        idx = next(i for i, f in enumerate(frames) if f is frame)
        return distances[idx], 90

    with patch("src.ocr_utils.extract_distance_from_frame", side_effect=fake_extract):
        return compute_distance_constraints_only(
            frames=frames,
            estimator=estimator,
            config=cfg.estimation,
            max_distance_increment_mm=max_increment,
            enforce_ocr_increment_upper_bound=enforce_upper,
            countup_mean_dz_max_mm=countup_mean_dz_max,
        )


def test_ocr_only_rejects_reverse(transformer):
    _, _, ocr_dist, success = _run_ocr_only_with_distances(
        transformer, [100.0, 110.0, 90.0, 120.0]
    )
    assert list(success) == [True, True, False, True]
    np.testing.assert_allclose(ocr_dist[success], [100.0, 110.0, 120.0])


def test_ocr_only_rejects_digit_and_period_jumps(transformer):
    _, _, ocr_dist, success = _run_ocr_only_with_distances(
        transformer, [1100.0, 4100.0, 1110.0, 111000.0, 1120.0]
    )
    assert list(success) == [True, False, True, False, True]
    np.testing.assert_allclose(ocr_dist[success], [1100.0, 1110.0, 1120.0])


def test_ocr_only_passes_expected_range_to_extract(transformer):
    cfg = Config.from_defaults()
    estimator = CameraEstimator(cfg.estimation, transformer)
    frames = [np.zeros((12, 12, 3), dtype=np.uint8) for _ in range(2)]
    seen = []

    def fake_extract(frame, roi, ocr_cfg, expected_range_mm=None, **kwargs):
        seen.append(expected_range_mm)
        idx = next(i for i, f in enumerate(frames) if f is frame)
        return [100.0, 108.0][idx], 90

    with patch("src.ocr_utils.extract_distance_from_frame", side_effect=fake_extract):
        compute_distance_constraints_only(
            frames=frames,
            estimator=estimator,
            config=cfg.estimation,
            max_distance_increment_mm=10.0,
        )
    assert seen[0] is None
    assert seen[1] == (100.0, 110.0)


def test_ocr_only_accepts_fast_jumps_without_upper_bound(transformer):
    _, _, ocr_dist, success = _run_ocr_only_with_distances(
        transformer, [100.0, 120.0, 150.0], max_increment=10.0, enforce_upper=False,
    )
    assert list(success) == [True, True, True]
    np.testing.assert_allclose(ocr_dist, [100.0, 120.0, 150.0])


def test_ocr_only_still_rejects_reverse_without_upper_bound(transformer):
    _, _, ocr_dist, success = _run_ocr_only_with_distances(
        transformer, [100.0, 120.0, 90.0], max_increment=10.0, enforce_upper=False,
    )
    assert list(success) == [True, True, False]
    np.testing.assert_allclose(ocr_dist[success], [100.0, 120.0])


def test_ocr_only_expected_range_has_no_plus_cap_when_disabled(transformer):
    cfg = Config.from_defaults()
    estimator = CameraEstimator(cfg.estimation, transformer)
    frames = [np.zeros((12, 12, 3), dtype=np.uint8) for _ in range(2)]
    seen = []

    def fake_extract(frame, roi, ocr_cfg, expected_range_mm=None, **kwargs):
        seen.append(expected_range_mm)
        idx = next(i for i, f in enumerate(frames) if f is frame)
        return [100.0, 130.0][idx], 90

    with patch("src.ocr_utils.extract_distance_from_frame", side_effect=fake_extract):
        compute_distance_constraints_only(
            frames=frames,
            estimator=estimator,
            config=cfg.estimation,
            max_distance_increment_mm=10.0,
            enforce_ocr_increment_upper_bound=False,
        )
    assert seen[0] is None
    assert seen[1][0] == 100.0
    assert seen[1][1] == float("inf")


def test_ocr_countup_mean_dz_max_is_independent_of_motion_max_dz():
    cfg = Config.from_defaults()
    cfg.estimation.motion.max_dz = 3.0
    cfg.ocr.countup_mean_dz_max_mm = 10.0
    assert cfg.estimation.motion.max_dz == pytest.approx(3.0)
    assert cfg.ocr.countup_mean_dz_max_mm == pytest.approx(10.0)


def test_ocr_z_ocr_max_follows_delayed_count_up_physics():
    assert ocr_z_ocr_max_mm(100.0, 1, 10.0) == 120.0
    assert ocr_z_ocr_max_mm(10.0, 1, 10.0) == 30.0
    assert ocr_z_ocr_max_mm(10.0, 10, 10.0) == 120.0
    assert ocr_allowed_inclusive_hi_mm(100.0, 1, 10.0, 10.0) == pytest.approx(110.0)
    assert ocr_allowed_inclusive_hi_mm(10.0, 1, 10.0, 60.0) == pytest.approx(30.0 - 1e-9)
    assert ocr_expected_range_mm(100.0, 1, 10.0, countup_mean_dz_max=10.0) == (
        100.0,
        pytest.approx(110.0),
    )


def test_ocr_validate_allows_tick_and_rejects_faster_than_countup_mean_dz_max():
    ok_tick = validate_ocr_distance_value(
        frame_idx=1,
        ocr_distance=110.0,
        prev_ocr_distance=100.0,
        prev_ocr_success=True,
        base_ocr_distance=100.0,
        frame_gap=1,
        max_increment=10.0,
        countup_mean_dz_max=10.0,
    )
    assert ok_tick == 110.0
    too_fast = validate_ocr_distance_value(
        frame_idx=1,
        ocr_distance=40.0,
        prev_ocr_distance=10.0,
        prev_ocr_success=True,
        base_ocr_distance=10.0,
        frame_gap=1,
        max_increment=60.0,
        countup_mean_dz_max=10.0,
    )
    assert too_fast is None
    delayed = validate_ocr_distance_value(
        frame_idx=10,
        ocr_distance=40.0,
        prev_ocr_distance=10.0,
        prev_ocr_success=True,
        base_ocr_distance=10.0,
        frame_gap=10,
        max_increment=60.0,
        countup_mean_dz_max=10.0,
    )
    assert delayed == 40.0


def test_ocr_only_allows_delayed_count_up_jump(transformer):
    distances = [10.0] * 10 + [40.0]
    _, _, ocr_dist, success = _run_ocr_only_with_distances(
        transformer, distances, max_increment=60.0, countup_mean_dz_max=10.0,
    )
    assert np.all(success)
    np.testing.assert_allclose(ocr_dist, distances)


def test_ocr_only_rejects_jump_too_fast_for_countup_mean_dz_max(transformer):
    _, _, ocr_dist, success = _run_ocr_only_with_distances(
        transformer, [10.0, 40.0], max_increment=60.0, countup_mean_dz_max=10.0,
    )
    assert list(success) == [True, False]
    np.testing.assert_allclose(ocr_dist[success], [10.0])


def test_high_precision_splits_actual_delta_not_fixed_10mm():
    ocr = np.array([10.0] * 10 + [40.0], dtype=float)
    success = np.ones(ocr.size, dtype=bool)
    z = estimate_high_precision_z_positions(ocr, success, 3.0)
    np.testing.assert_allclose(z[:10], 10.0 + np.arange(10) * 3.0)
    assert z[10] == pytest.approx(40.0)


def test_high_precision_10mm_ticks_still_split_by_next_delta():
    ocr = np.array([3000.0, 3010.0, 3010.0, 3020.0, 3020.0, 3030.0])
    success = np.ones(ocr.size, dtype=bool)
    z = estimate_high_precision_z_positions(ocr, success, 6.0)
    np.testing.assert_allclose(z, [3000.0, 3010.0, 3015.0, 3020.0, 3025.0, 3030.0])


def test_high_precision_last_plateau_holds_display_value():
    ocr = np.array([10.0, 10.0, 10.0], dtype=float)
    success = np.ones(3, dtype=bool)
    z = estimate_high_precision_z_positions(ocr, success, 1.0)
    np.testing.assert_allclose(z, [10.0, 10.0, 10.0])


def test_high_precision_last_plateau_uses_previous_countup_speed():
    ocr = np.array([10.0] * 5 + [40.0] * 5, dtype=float)
    success = np.ones(ocr.size, dtype=bool)
    z = estimate_high_precision_z_positions(ocr, success, 99.0)
    speed = (40.0 - 10.0) / 5.0
    expected = np.concatenate([
        10.0 + np.arange(5) * speed,
        40.0 + np.arange(5) * speed,
    ])
    np.testing.assert_allclose(z, expected)
    assert z[-1] == pytest.approx(40.0 + speed * 4)


def test_high_precision_last_plateau_continues_through_trailing_failures():
    ocr = np.array([10.0, 10.0, 40.0, 40.0, np.nan])
    success = np.array([True, True, True, True, False])
    z = estimate_high_precision_z_positions(ocr, success, 1.0)
    speed = (40.0 - 10.0) / 2.0
    np.testing.assert_allclose(z, [10.0, 25.0, 40.0, 55.0, 70.0])
    assert z[-1] == pytest.approx(40.0 + speed * 2)
