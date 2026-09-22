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

    def fake_extract(frame, roi, ocr_cfg):
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
