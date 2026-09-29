"""T4: OCR-only の z_positions が既存 OCR 系列と整合すること"""

from unittest.mock import patch

import numpy as np

from src.camera_estimation import CameraEstimator, compute_distance_constraints_only
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


def _run_ocr_only_with_distances(transformer, distances, max_increment=10.0):
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
