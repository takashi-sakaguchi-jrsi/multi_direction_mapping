"""正面: 前フレーム姿勢での管 (θ,Z) マッチと姿勢トレンド補正"""

from pathlib import Path

import numpy as np
import pytest

from src.main_twopass import pre_deroll_colormap_path

from src.colormap_correction import (
    apply_front_attitude_trend_warp,
    blend_orientation_toward_vp,
    deroll_colormap_to_zero,
    deroll_roll_per_column,
    front_attitude_trend_dtheta_rad,
    remap_column_eta_shift,
)
from src.feature_matching_cylindrical import (
    CylindricalFeatureMatcher,
    CylindricalMatchingConfig,
    estimate_front_physical_option2,
    filter_physical_motion_vectors,
)


def _synthetic_flow(theta_deg, z_mm, radius, dz, dyaw, dpitch):
    theta = np.deg2rad(theta_deg)
    z = np.maximum(np.asarray(z_mm, dtype=float), 1e-3)
    c = np.cos(theta)
    s = np.sin(theta)
    kZ = radius + (z * z) / radius
    kth = z / radius
    dZ = -dz - kZ * (dyaw * c + dpitch * s)
    dth = kth * (dpitch * c - dyaw * s)
    return dZ, np.degrees(dth)


def test_local_motion_recovers_dz_dyaw_dpitch():
    rng = np.random.default_rng(0)
    n = 80
    radius = 125.0
    theta = rng.uniform(0.0, 360.0, n)
    z = rng.uniform(80.0, 250.0, n)
    dz, dyaw, dpitch = 5.0, np.radians(0.4), np.radians(-0.25)
    dZ, dth = _synthetic_flow(theta, z, radius, dz, dyaw, dpitch)
    prev = np.column_stack([theta, z])
    curr = np.column_stack([theta + dth, z + dZ])
    result = estimate_front_physical_option2(prev, curr, pipe_radius_mm=radius)
    assert result["success"]
    assert result["droll"] == 0.0
    assert result["dz"] == pytest.approx(dz, rel=0.05, abs=0.2)
    assert np.degrees(result["dyaw"]) == pytest.approx(np.degrees(dyaw), abs=0.08)
    assert np.degrees(result["dpitch"]) == pytest.approx(np.degrees(dpitch), abs=0.08)


def test_iqr_keeps_first_harmonic_dz():
    theta = np.linspace(0.0, 360.0, 36, endpoint=False)
    z = np.full(theta.shape, 150.0)
    dZ = -4.0 + 1.5 * np.cos(np.deg2rad(theta))
    dth = 0.3 * np.sin(np.deg2rad(theta))
    prev = np.column_stack([theta, z])
    curr = np.column_stack([theta + dth, z + dZ])
    curr[0, 1] += 40.0
    fprev, fcurr = filter_physical_motion_vectors(prev, curr, iqr_multiplier=1.5)
    assert len(fprev) >= 20
    assert float(np.max(np.abs(fcurr[:, 1] - fprev[:, 1]))) < 10.0


def test_blend_orientation_moves_toward_vp():
    yaw, pitch = np.radians(12.89), np.radians(-4.57)
    yaw_vp, pitch_vp = np.radians(8.0), np.radians(-2.0)
    yaw2, pitch2 = blend_orientation_toward_vp(yaw, pitch, yaw_vp, pitch_vp, 0.2)
    assert abs(yaw2 - yaw_vp) < abs(yaw - yaw_vp)
    assert abs(pitch2 - pitch_vp) < abs(pitch - pitch_vp)
    yaw0, pitch0 = blend_orientation_toward_vp(yaw, pitch, yaw_vp, pitch_vp, 0.0)
    assert yaw0 == pytest.approx(yaw)
    assert pitch0 == pytest.approx(pitch)


def test_attitude_trend_undoes_slow_yaw_ramp():
    z = np.linspace(0.0, 4000.0, 200)
    yaw_vp = np.full(z.shape, np.radians(12.89))
    pitch_vp = np.full(z.shape, np.radians(-4.57))
    yaw = yaw_vp + np.linspace(0.0, np.radians(6.0), z.size)
    pitch = pitch_vp.copy()
    dth = front_attitude_trend_dtheta_rad(
        z, yaw, pitch, yaw_vp, pitch_vp, window_mm=800.0
    )
    assert float(np.nanmax(np.abs(dth))) > 0.02
    assert abs(dth[-1]) > abs(dth[0])


def test_attitude_trend_skips_missing_vp_without_zero_spike():
    z = np.linspace(0.0, 1000.0, 51)
    yaw_vp = np.full(z.shape, np.radians(12.89))
    pitch_vp = np.full(z.shape, np.radians(-4.57))
    yaw = yaw_vp + np.radians(3.0)
    pitch = pitch_vp.copy()
    yaw_vp[20:24] = np.nan
    dth = front_attitude_trend_dtheta_rad(
        z, yaw, pitch, yaw_vp, pitch_vp, window_mm=200.0
    )
    assert np.all(np.isnan(dth[20:24]))
    good = dth[np.isfinite(dth)]
    assert good.size >= 20
    assert float(np.min(np.abs(good))) > 0.01
    assert float(np.max(np.abs(good)) - np.min(np.abs(good))) < 0.05


def test_attitude_warp_shifts_horizontal_band():
    h, w = 180, 80
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[80:88, :, :] = 200
    z = np.linspace(0.0, 80.0, 20)
    yaw_vp = np.full(z.shape, np.radians(12.89))
    pitch_vp = np.full(z.shape, np.radians(-4.57))
    yaw = yaw_vp + np.linspace(0.0, np.radians(20.0), z.size)
    pitch = pitch_vp.copy()
    out = apply_front_attitude_trend_warp(
        image, z, yaw, pitch, yaw_vp, pitch_vp, pixels_per_mm=1.0, window_mm=40.0
    )
    assert out.shape == image.shape
    src_row = int(np.argmax(image.mean(axis=(1, 2))))
    dst_row = int(np.argmax(out.mean(axis=(1, 2))))
    assert src_row != dst_row or float(np.mean(np.abs(out.astype(float) - image))) > 0.5


def test_deroll_restores_band_after_roll_shear():
    h, w = 360, 80
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[100:108, :, :] = 200
    ppm = 1.0
    z = np.arange(w, dtype=float)
    roll = np.linspace(0.0, np.radians(20.0), w)
    rolled = np.empty_like(image)
    for c in range(w):
        k = int(round(roll[c] * h / (2.0 * np.pi)))
        rolled[:, c] = np.roll(image[:, c], k, axis=0)
    src_row = int(np.argmax(image.mean(axis=(1, 2))))
    rolled_row = int(np.argmax(rolled[:, -1].mean(axis=1)))
    assert rolled_row != src_row
    out = deroll_colormap_to_zero(rolled, z, roll, pixels_per_mm=ppm, z_offset_mm=0.0)
    dst_row = int(np.argmax(out[:, -1].mean(axis=1)))
    assert abs(dst_row - src_row) <= 2


def test_deroll_smooth_reduces_column_jumps():
    w = 400
    z = np.linspace(0.0, 400.0, 40)
    roll = np.linspace(0.0, np.radians(12.0), z.size)
    roll[10] += np.radians(8.0)
    roll[11] -= np.radians(6.0)
    raw = deroll_roll_per_column(w, z, roll, pixels_per_mm=1.0, smooth=False)
    sm = deroll_roll_per_column(
        w, z, roll, pixels_per_mm=1.0, smooth=True, smooth_window_mm=80.0
    )
    assert raw is not None and sm is not None
    assert float(np.max(np.abs(np.diff(sm)))) < float(np.max(np.abs(np.diff(raw))))
    raw_img = deroll_colormap_to_zero(
        np.zeros((40, w, 3), dtype=np.uint8), z, roll, 1.0, smooth=False
    )
    sm_img = deroll_colormap_to_zero(
        np.zeros((40, w, 3), dtype=np.uint8), z, roll, 1.0,
        smooth=True, smooth_window_mm=80.0,
    )
    assert raw_img.shape == sm_img.shape


def test_deroll_smooth_off_matches_default():
    h, w = 60, 80
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[20:24, :, :] = 180
    z = np.linspace(0.0, 80.0, 16)
    roll = np.linspace(0.0, np.radians(10.0), z.size)
    a = deroll_colormap_to_zero(image, z, roll, pixels_per_mm=1.0)
    b = deroll_colormap_to_zero(image, z, roll, pixels_per_mm=1.0, smooth=False)
    assert np.array_equal(a, b)


def test_pre_deroll_colormap_path_is_beside_temp():
    temp = Path("data/output/colormap_temp_20261023_120000.png")
    out = pre_deroll_colormap_path(temp)
    assert out.parent == temp.parent
    assert out.name == "colormap_temp_20261023_120000_before_deroll.png"


def test_eta_shift_chunks_match_single_block():
    rng = np.random.default_rng(0)
    h, w = 40, 90
    image = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    d_rows = np.linspace(0.0, 7.3, w)
    full = remap_column_eta_shift(image, d_rows, chunk_width=w)
    chunked = remap_column_eta_shift(image, d_rows, chunk_width=11)
    assert np.array_equal(full, chunked)


def test_deroll_skips_near_zero_roll():
    h, w = 40, 20
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[10, 5] = 255
    z = np.linspace(0.0, 20.0, 8)
    roll = np.zeros(8)
    out = deroll_colormap_to_zero(image, z, roll, pixels_per_mm=1.0)
    assert np.array_equal(out, image)


def test_pose_unwrap_maps_differ_from_zero_pose(transformer):
    cfg = CylindricalMatchingConfig(
        z_axis="physical", theta_resolution=90, z_resolution=40
    )
    matcher = CylindricalFeatureMatcher(transformer, cfg)
    params0 = {
        "pipe_diameter": 250.0,
        "center": (transformer.camera.cx, transformer.camera.cy),
        "image_width": transformer.camera.image_width,
        "image_height": transformer.camera.image_height,
        "cam_pos": np.array([0.0, 0.0, 0.0]),
        "roll": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
    }
    params1 = dict(params0)
    params1["yaw"] = np.radians(12.89)
    params1["pitch"] = np.radians(-4.57)
    u0, v0, _, _ = matcher._build_remap_maps((108, 192, 3), params0)
    u1, v1, _, _ = matcher._build_remap_maps((108, 192, 3), params1)
    valid = (u0 >= 0) & (u1 >= 0)
    assert np.any(valid)
    assert float(np.mean(np.abs(u1[valid] - u0[valid]))) > 1.0
