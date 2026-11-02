"""工程4: 側視の dz_only / OCR比異常 / FAILED 前進を個別に切替できる"""

import numpy as np
import pytest

from src.camera_estimation import CameraEstimator
from src.config import Config, ConfigValidationError
from src.validation.frame_analyzer import (
    FrameAnalyzer,
    is_abnormally_small_dz,
    normalize_sideview_attitude_source,
    ocr_local_avg_dz_mm,
)


def _blank_frames(n=4, h=32, w=32):
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


def _patch_motion(analyzer, sequence):
    """sequence: list of (status, dz, dpitch) for each pair."""
    seq = list(sequence)

    def _fake(prev, curr, local_i, run_id, position, orientation, reference,
              camera_params, mapper_roi, constraints):
        status, dz, dpitch = seq[min(local_i - 1, len(seq) - 1)]
        raw = np.zeros(6)
        final = np.zeros(6)
        raw[2] = dz
        raw[5] = dpitch
        final[2] = dz
        final[5] = dpitch
        return raw, final, 8, {}, 0.0, {}, status

    analyzer._motion_between = _fake


def test_normalize_sideview_attitude_rejects_discard():
    assert normalize_sideview_attitude_source("match") == "match"
    assert normalize_sideview_attitude_source("dz_only") == "dz_only"
    with pytest.raises(ValueError, match="dz_only"):
        normalize_sideview_attitude_source("discard")


def test_config_defaults_keep_step2_motion():
    cfg = Config.from_defaults()
    td = cfg.two_direction
    assert td.physical_attitude_source == "match"
    assert td.ocr_dz_anomaly_ratio == pytest.approx(0.0)
    assert td.fail_advance_median_dz is False


def test_config_rejects_discard_attitude_source():
    cfg = Config.from_defaults()
    cfg.two_direction.physical_attitude_source = "discard"
    with pytest.raises(ConfigValidationError, match="match/dz_only"):
        cfg.validate_two_direction_config()


def test_is_abnormally_small_dz_respects_ratio_and_floor():
    assert is_abnormally_small_dz(0.2, 10.0, ratio=0.3, min_ocr_mm=1.0) is True
    assert is_abnormally_small_dz(4.0, 10.0, ratio=0.3, min_ocr_mm=1.0) is False
    assert is_abnormally_small_dz(0.2, 10.0, ratio=0.0, min_ocr_mm=1.0) is False
    assert is_abnormally_small_dz(0.2, 0.5, ratio=0.3, min_ocr_mm=1.0) is False


def test_ocr_local_avg_dz_clamps_negative():
    avg = ocr_local_avg_dz_mm([0.0, 10.0, 20.0, 15.0], window=2)
    assert avg is not None
    assert avg[1] == pytest.approx(10.0)
    assert avg[3] >= 0.0


def test_dz_only_does_not_accumulate_pitch(transformer, config):
    config.two_direction.physical_attitude_source = "dz_only"
    config.two_direction.fail_advance_median_dz = False
    config.two_direction.ocr_dz_anomaly_ratio = 0.0
    config.estimation.use_offset_moving_average = False
    analyzer = FrameAnalyzer(
        config, CameraEstimator(config.estimation, transformer), mode="A"
    )
    _patch_motion(analyzer, [("OK", 5.0, 0.05), ("OK", 5.0, 0.05)])
    recs, extra = analyzer.analyze_run(
        config.two_direction.run_A, frames=_blank_frames(3), fps=10.0,
        known_z_mm=np.array([0.0, 10.0, 20.0]),
    )
    assert extra["physical_attitude_source"] == "dz_only"
    pitch0 = recs[0].orientation[2]
    assert recs[-1].orientation[2] == pytest.approx(pitch0)
    assert recs[-1].motion_final[5] == pytest.approx(0.0)
    assert recs[-1].position[2] == pytest.approx(recs[0].position[2] + 10.0)


def test_match_attitude_still_accumulates_pitch(transformer, config):
    config.two_direction.physical_attitude_source = "match"
    config.estimation.use_offset_moving_average = False
    analyzer = FrameAnalyzer(
        config, CameraEstimator(config.estimation, transformer), mode="A"
    )
    _patch_motion(analyzer, [("OK", 5.0, 0.05), ("OK", 5.0, 0.05)])
    recs, _extra = analyzer.analyze_run(
        config.two_direction.run_A, frames=_blank_frames(3), fps=10.0,
        known_z_mm=np.array([0.0, 10.0, 20.0]),
    )
    pitch0 = recs[0].orientation[2]
    assert recs[-1].orientation[2] == pytest.approx(pitch0 + 0.10, abs=1e-9)


def test_fail_freeze_is_default(transformer, config):
    config.two_direction.fail_advance_median_dz = False
    config.estimation.use_offset_moving_average = False
    analyzer = FrameAnalyzer(
        config, CameraEstimator(config.estimation, transformer), mode="A"
    )
    _patch_motion(analyzer, [("OK", 4.0, 0.0), ("FAILED", 0.0, 0.0)])
    recs, extra = analyzer.analyze_run(
        config.two_direction.run_A, frames=_blank_frames(3), fps=10.0,
        known_z_mm=np.array([0.0, 10.0, 20.0]),
    )
    assert recs[-1].status == "FAILED"
    assert recs[-1].position[2] == pytest.approx(recs[1].position[2])
    assert extra["fail_feature_count"] == 1
    assert extra["fail_advance_median_dz"] is False


def test_fail_advance_median_dz_updates_prev_and_z(transformer, config):
    config.two_direction.fail_advance_median_dz = True
    config.estimation.use_offset_moving_average = False
    analyzer = FrameAnalyzer(
        config, CameraEstimator(config.estimation, transformer), mode="A"
    )
    _patch_motion(analyzer, [("OK", 4.0, 0.0), ("OK", 6.0, 0.0), ("FAILED", 0.0, 0.0)])
    recs, extra = analyzer.analyze_run(
        config.two_direction.run_A, frames=_blank_frames(4), fps=10.0,
        known_z_mm=np.array([0.0, 10.0, 20.0, 30.0]),
    )
    assert recs[-1].status == "FAILED"
    assert extra["fail_advance_median_dz"] is True
    z0 = recs[0].position[2]
    assert recs[-1].position[2] == pytest.approx(z0 + 4.0 + 6.0 + 5.0)
    assert recs[-1].motion_final[2] == pytest.approx(5.0)
    assert recs[-1].residual_stats.get("fail_kind") == "feature"


def test_ocr_dz_anomaly_is_independent_switch(transformer, config):
    config.two_direction.ocr_dz_anomaly_ratio = 0.3
    config.two_direction.ocr_dz_anomaly_min_ocr_mm = 1.0
    config.two_direction.fail_advance_median_dz = True
    config.estimation.use_offset_moving_average = False
    analyzer = FrameAnalyzer(
        config, CameraEstimator(config.estimation, transformer), mode="A"
    )
    _patch_motion(analyzer, [("OK", 5.0, 0.0), ("OK", 0.1, 0.0)])
    recs, extra = analyzer.analyze_run(
        config.two_direction.run_A, frames=_blank_frames(3), fps=10.0,
        known_z_mm=np.array([0.0, 10.0, 20.0]),
    )
    assert recs[-1].status == "FAILED"
    assert recs[-1].residual_stats.get("fail_kind") == "z_inconsistency"
    assert extra["fail_z_inconsistency_count"] == 1
    assert extra["fail_feature_count"] == 0
    z0 = recs[0].position[2]
    assert recs[-1].position[2] == pytest.approx(z0 + 5.0 + 5.0)


def test_ocr_dz_anomaly_off_keeps_small_dz(transformer, config):
    config.two_direction.ocr_dz_anomaly_ratio = 0.0
    config.estimation.use_offset_moving_average = False
    analyzer = FrameAnalyzer(
        config, CameraEstimator(config.estimation, transformer), mode="A"
    )
    _patch_motion(analyzer, [("OK", 5.0, 0.0), ("OK", 0.1, 0.0)])
    recs, extra = analyzer.analyze_run(
        config.two_direction.run_A, frames=_blank_frames(3), fps=10.0,
        known_z_mm=np.array([0.0, 10.0, 20.0]),
    )
    assert recs[-1].status == "OK"
    assert extra["fail_z_inconsistency_count"] == 0
