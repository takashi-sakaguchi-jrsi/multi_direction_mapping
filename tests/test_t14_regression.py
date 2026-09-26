"""T14: 既存 main_twopass 経路が残っていること"""

import inspect

from src.camera_estimation import CameraEstimator, compute_frame_constraints
from src import main_twopass


def test_main_twopass_importable():
    assert hasattr(main_twopass, "main")
    assert hasattr(main_twopass, "ColorMapPipelineTwoPass")


def test_legacy_vp_path_still_present():
    src = inspect.getsource(compute_frame_constraints)
    assert "use_vanishing_point_constraints" in src
    assert "collect_vanishing_points" in src or "collect_vanishing_points_and_ocr" in src


def test_estimate_motion_flexible_keeps_yaw_base_without_reference():
    src = inspect.getsource(CameraEstimator.estimate_motion_flexible)
    assert "yaw_base" in src
    assert "run_reference" in src
