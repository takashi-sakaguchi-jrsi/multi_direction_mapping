"""T3: PoC 経路から VP / 暗部 / 輝度判定が呼ばれないこと"""

from unittest.mock import patch

import numpy as np

from src.camera_estimation import CameraEstimator, compute_distance_constraints_only
from src.config import Config
from src.coordinate_transform import CoordinateTransformer, FisheyeCamera


def test_ocr_only_does_not_call_vp_or_dark(transformer):
    cfg = Config.from_defaults()
    estimator = CameraEstimator(cfg.estimation, transformer)
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]

    with patch.object(estimator, "detect_dark_region_centroid") as dark, \
         patch("src.camera_estimation.collect_vanishing_points") as vp, \
         patch("src.ocr_utils.extract_distance_from_frame", side_effect=lambda *a, **k: (10.0 * (a and 0 or 1), 80)):
        # extract_distance_from_frame is imported inside the function
        pass

    with patch("src.ocr_utils.extract_distance_from_frame", return_value=(100.0, 80)) as ocr, \
         patch.object(estimator, "detect_dark_region_centroid") as dark:
        constraints, z_pos, ocr_dist, success = compute_distance_constraints_only(
            frames=frames,
            estimator=estimator,
            config=cfg.estimation,
        )
        assert dark.call_count == 0
        assert ocr.call_count == len(frames)
        assert len(constraints) == len(frames)
        assert z_pos is not None
