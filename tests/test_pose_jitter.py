"""生成動画の dz/dpitch/dyaw 合成振動と OCR の累積 z 整合"""

import numpy as np
import pytest

from src.generate_two_direction_test_videos import generate_two_direction_videos
from src.validation.fisheye_sideview_renderer import (
    FisheyeSideviewRenderer,
    make_demo_layout_colormap,
)
from src.validation.geometry import pose_R_c2w
from src.validation.ocr_simulation import simulate_ocr_distance_mm
from src.validation.pose_jitter import (
    PoseJitterConfig,
    build_motion_series,
    motion_metadata_fields,
)


def test_two_tone_dz_stays_positive_and_near_nominal():
    cfg = PoseJitterConfig(seed=1)
    series = build_motion_series(80, z_start_mm=10.0, z_step_mm=4.5, z_max_mm=2000.0, config=cfg)
    dz = series["dz_mm"]
    assert dz[0] == pytest.approx(0.0)
    assert np.all(dz[1:] >= cfg.dz_min_mm - 1e-9)
    assert np.mean(dz[1:]) == pytest.approx(4.5, abs=0.35)
    assert np.std(dz[1:]) > 0.2


def test_angle_offsets_oscillate_without_drift():
    series = build_motion_series(90, 10.0, 4.5, 2000.0, PoseJitterConfig(seed=2))
    yaw = series["yaw_offset_deg"]
    pitch = series["pitch_offset_deg"]
    assert abs(float(np.mean(yaw))) < 0.6
    assert abs(float(np.mean(pitch))) < 0.6
    assert float(np.max(np.abs(yaw))) < 2.5
    assert float(np.max(np.abs(pitch))) < 2.5
    np.testing.assert_allclose(series["dyaw_deg"][1:], np.diff(yaw))
    np.testing.assert_allclose(series["dpitch_deg"][1:], np.diff(pitch))


def test_ocr_follows_accumulated_true_z():
    series = build_motion_series(40, 10.0, 4.5, 2000.0, PoseJitterConfig(seed=3))
    meta = motion_metadata_fields(series, z_step_mm=4.5, config=PoseJitterConfig(seed=3))
    ocr = simulate_ocr_distance_mm(series["z_mm"])
    np.testing.assert_allclose(meta["ocr_z_mm"], ocr)
    np.testing.assert_allclose(meta["z_values_mm"], series["z_mm"])
    assert meta["motion_jitter"] is True
    assert all(v % 10 == 0 for v in meta["ocr_z_mm"])


def test_program_pose_matches_demo_optical_axis():
    """U/R のプログラム姿勢は Demo 回転の光軸と一致する。"""
    for roll_car, rpy_deg in (
        (0.0, (-90.0, 90.0, 90.0)),
        (120.0, (-90.0, 90.0, -30.0)),
    ):
        R_demo = FisheyeSideviewRenderer.demo_rotation(roll_car, 90.0)
        roll, yaw, pitch = np.radians(rpy_deg)
        R_prog = pose_R_c2w(roll, yaw, pitch).T
        axis_d = R_demo @ np.array([0.0, 0.0, 1.0])
        axis_p = R_prog @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(axis_p, axis_d, atol=1e-6)


def test_generate_jitter_writes_motion_and_ocr(tmp_path):
    import cv2

    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=120.0, pixels_per_mm=0.35)
    cmap = tmp_path / "map.png"
    cv2.imwrite(str(cmap), tex)
    result = generate_two_direction_videos(
        colormap=cmap,
        output_dir=tmp_path / "videos",
        runs=["U", "R"],
        width=48,
        height=48,
        radius_mm=125.0,
        fov_deg=181.0,
        pitch_deg=90.0,
        n_frames=12,
        fps=5.0,
        name="jitter_side",
        z_start_mm=8.0,
        z_step_mm=4.5,
        z_margin_mm=1.0,
        shading=False,
        reconstruct=False,
        save_png=False,
        jitter=True,
        jitter_seed=4,
    )
    assert result["motion_jitter"] is True
    n = len(result["z_values_mm"])
    assert n == 12
    assert len(result["ocr_z_mm"]) == n
    assert len(result["dz_mm"]) == n
    assert len(result["dyaw_deg"]) == n
    assert len(result["dpitch_deg"]) == n
    ocr = simulate_ocr_distance_mm(result["z_values_mm"])
    np.testing.assert_allclose(result["ocr_z_mm"], ocr)
    z = np.asarray(result["z_values_mm"])
    dz = np.asarray(result["dz_mm"])
    np.testing.assert_allclose(z, z[0] + np.cumsum(dz))
    assert abs(z[-1] - (8.0 + 4.5 * 11)) > 0.05
