"""開始・終了フレーム範囲 [start, end)"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.camera_estimation import CameraEstimator
from src.config import Config, ConfigValidationError
from src.validation.frame_analyzer import FrameAnalyzer, resolve_frame_window


def test_resolve_frame_window_defaults():
    td = SimpleNamespace(start_frame=0, end_frame=None, max_frames=None)
    assert resolve_frame_window(td) == (0, None)


def test_resolve_frame_window_max_frames_from_start():
    td = SimpleNamespace(start_frame=10, end_frame=40, max_frames=5)
    assert resolve_frame_window(td) == (10, 15)


def test_resolve_frame_window_end_clips_to_total():
    td = SimpleNamespace(start_frame=240, end_frame=400, max_frames=None)
    assert resolve_frame_window(td, n_total=320) == (240, 320)


def test_resolve_frame_window_rejects_empty():
    td = SimpleNamespace(start_frame=10, end_frame=10, max_frames=None)
    with pytest.raises(ValueError):
        resolve_frame_window(td)


def test_config_rejects_end_not_after_start():
    cfg = Config.from_defaults()
    cfg.two_direction.start_frame = 8
    cfg.two_direction.end_frame = 8
    with pytest.raises(ConfigValidationError):
        cfg.validate_two_direction_config()


def test_analyze_run_slices_known_z_and_keeps_original_frame_num(transformer, config):
    config.two_direction.start_frame = 2
    config.two_direction.end_frame = 5
    config.estimation.use_offset_moving_average = False
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(8)]
    known = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    run_cfg = config.two_direction.run_A
    estimator = CameraEstimator(config.estimation, transformer)
    recs, extra = FrameAnalyzer(config, estimator, mode="A").analyze_run(
        run_cfg, frames=frames, fps=20.0, known_z_mm=known,
    )
    assert extra["start_frame"] == 2
    assert extra["end_frame"] == 5
    assert len(recs) == 3
    assert [r.frame_num for r in recs] == [2, 3, 4]
    assert recs[0].timestamp == pytest.approx(2 / 20.0)
    assert recs[0].z_ocr == pytest.approx(30.0)
    assert recs[-1].z_ocr == pytest.approx(50.0)
    np.testing.assert_allclose(extra["ocr_dist"], [30.0, 40.0, 50.0])
