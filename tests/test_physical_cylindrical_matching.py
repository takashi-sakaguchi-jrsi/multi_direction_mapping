"""物理 (θ, Z) マッチングと正面案②（roll 固定、dz は ΔZ の DC）。"""

import numpy as np
import pytest

from src.config import Config
from src.coordinate_transform import CoordinateTransformer, FisheyeCamera
from src.feature_matching_cylindrical import (
    CylindricalFeatureMatcher,
    CylindricalMatchingConfig,
    clip_front_physical_estimate,
    estimate_front_physical_option2,
    filter_physical_motion_vectors,
)
from src.product_runtime import apply_front_profile


def test_physical_grid_uses_atan2_R_over_Z():
    cam = FisheyeCamera(
        f=344.0, cx=610.0, cy=540.0, image_width=1220, image_height=1080
    )
    matcher = CylindricalFeatureMatcher(
        CoordinateTransformer(cam, 125.0),
        CylindricalMatchingConfig(
            z_axis="physical", theta_resolution=36, z_resolution=20
        ),
    )
    params = {
        "f": 344.0,
        "center": (610.0, 540.0),
        "pipe_diameter": 250.0,
        "image_width": 1220,
        "image_height": 1080,
    }
    _theta, z_grid, _eta, zeta = matcher._create_cylindrical_grid(params)
    assert matcher._z_far_mm > matcher._z_near_mm
    assert z_grid[0, 0] > z_grid[-1, 0]
    R = 125.0
    np.testing.assert_allclose(zeta, np.arctan2(R, np.maximum(z_grid, 1e-3)), atol=1e-9)


def test_pure_dz_is_uniform_and_recovers_dz():
    z = np.array([80.0, 120.0, 160.0, 200.0])
    theta = np.array([0.0, 90.0, 180.0, 270.0])
    dz_true = 4.5
    prev = np.column_stack([theta, z])
    curr = np.column_stack([theta, z - dz_true])
    dZ = curr[:, 1] - prev[:, 1]
    np.testing.assert_allclose(dZ, -dz_true)
    est = estimate_front_physical_option2(prev, curr)
    assert est["success"]
    assert est["droll"] == 0.0
    assert est["dz"] == pytest.approx(dz_true, abs=1e-6)


def test_yaw_like_harmonic_does_not_go_to_droll_or_dz():
    """Δθ ∝ cosθ、ΔZ ∝ sinθ は yaw/pitch。dz の DC と droll に載せない。"""
    theta = np.linspace(0.0, 360.0, 24, endpoint=False)
    z = np.full(theta.shape, 150.0)
    prev = np.column_stack([theta, z])
    dtheta = 1.2 * np.cos(np.deg2rad(theta))
    dZ = 3.0 * np.sin(np.deg2rad(theta))
    curr = np.column_stack([theta + dtheta, z + dZ])
    est = estimate_front_physical_option2(prev, curr)
    assert est["droll"] == 0.0
    assert est["dz"] == pytest.approx(0.0, abs=0.2)
    assert abs(est["dtheta_cos"]) > 0.5


def test_reverse_dc_does_not_drop_all_points():
    theta = np.linspace(0.0, 360.0, 24, endpoint=False)
    z = np.full(theta.shape, 140.0)
    prev = np.column_stack([theta, z])
    curr = np.column_stack([theta, z + 2.0])
    filt_p, filt_c = filter_physical_motion_vectors(prev, curr)
    assert len(filt_p) == len(prev)
    assert len(filt_c) == len(curr)


def test_small_harmonic_dtheta_is_kept():
    theta = np.linspace(0.0, 360.0, 36, endpoint=False)
    z = np.full(theta.shape, 150.0)
    dth = 4.0 * np.cos(np.deg2rad(theta))
    dZ = -4.0 + 2.0 * np.sin(np.deg2rad(theta))
    prev = np.column_stack([theta, z])
    curr = np.column_stack([theta + dth, z + dZ])
    filt_p, filt_c = filter_physical_motion_vectors(prev, curr)
    assert len(filt_p) == len(prev)


def test_implausible_dtheta_is_dropped_before_iqr():
    theta = np.linspace(0.0, 360.0, 36, endpoint=False)
    z = np.full(theta.shape, 150.0)
    prev = np.column_stack([theta, z])
    curr = np.column_stack([theta, z - 4.0])
    curr[0, 0] += 25.0
    filt_p, filt_c = filter_physical_motion_vectors(prev, curr)
    assert len(filt_p) == len(prev) - 1


def test_implausible_dz_is_dropped_before_iqr():
    theta = np.linspace(0.0, 360.0, 36, endpoint=False)
    z = np.full(theta.shape, 150.0)
    prev = np.column_stack([theta, z])
    curr = np.column_stack([theta, z - 4.0])
    curr[0, 1] -= 30.0
    filt_p, filt_c = filter_physical_motion_vectors(prev, curr)
    assert len(filt_p) == len(prev) - 1


def test_clip_physical_estimate_rejects_beyond_limits():
    over_dz = clip_front_physical_estimate(
        {"success": True, "dz": 50.0, "dyaw": 0.0, "dpitch": 0.0, "n_points": 20},
        max_dz_mm=20.0,
    )
    assert over_dz["success"] is False
    assert over_dz["failure_reason"] == "dz_exceeded_clip"

    over_yaw = clip_front_physical_estimate(
        {"success": True, "dz": 4.0, "dyaw": np.radians(10.0), "dpitch": 0.0, "n_points": 20},
        max_dyaw_deg=2.0,
        max_dpitch_deg=2.0,
    )
    assert over_yaw["success"] is False
    assert over_yaw["failure_reason"] == "attitude_exceeded_clip"

    reverse = clip_front_physical_estimate(
        {"success": True, "dz": -3.0, "dyaw": 0.0, "dpitch": 0.0, "n_points": 20},
        reject_reverse=True,
        reverse_tol_mm=0.5,
    )
    assert reverse["success"] is False
    assert reverse["failure_reason"] == "reverse_travel"

    few = clip_front_physical_estimate(
        {"success": True, "dz": 4.0, "dyaw": 0.0, "dpitch": 0.0, "n_points": 3},
        min_points=4,
    )
    assert few["success"] is False
    assert few["failure_reason"] == "too_few_points"

    ok = clip_front_physical_estimate(
        {"success": True, "dz": 4.0, "dyaw": np.radians(0.4), "dpitch": np.radians(-0.2), "n_points": 20},
        max_dz_mm=20.0,
        max_dyaw_deg=2.0,
        max_dpitch_deg=2.0,
        reject_reverse=True,
        reverse_tol_mm=0.5,
    )
    assert ok["success"] is True
    assert ok["dz"] == pytest.approx(4.0)
    assert np.degrees(ok["dyaw"]) == pytest.approx(0.4)

    small_reverse = clip_front_physical_estimate(
        {"success": True, "dz": -0.3, "dyaw": 0.0, "dpitch": 0.0, "n_points": 20},
        reject_reverse=True,
        reverse_tol_mm=0.5,
    )
    assert small_reverse["success"] is True
    assert small_reverse["dz"] == pytest.approx(-0.3)


def test_clip_suppresses_large_attitude_keeps_dz():
    low_n = clip_front_physical_estimate(
        {"success": True, "dz": 5.0, "dyaw": np.radians(1.5), "dpitch": np.radians(-1.6), "n_points": 9},
        max_dyaw_deg=2.0,
        max_dpitch_deg=2.0,
        min_points_for_attitude=20,
        untrusted_attitude_deg=0.8,
        max_applied_attitude_deg=1.2,
    )
    assert low_n["success"] is True
    assert low_n["dz"] == pytest.approx(5.0)
    assert low_n["dyaw"] == pytest.approx(0.0)
    assert low_n["dpitch"] == pytest.approx(0.0)
    assert low_n["attitude_suppressed"] is True
    assert low_n["attitude_suppress_reason"] == "attitude_too_large"

    low_n_mid = clip_front_physical_estimate(
        {"success": True, "dz": 4.0, "dyaw": np.radians(0.95), "dpitch": 0.0, "n_points": 12},
        min_points_for_attitude=20,
        untrusted_attitude_deg=0.8,
        max_applied_attitude_deg=1.2,
    )
    assert low_n_mid["success"] is True
    assert low_n_mid["dyaw"] == pytest.approx(0.0)
    assert low_n_mid["attitude_suppress_reason"] == "low_n_large_attitude"

    tiny = clip_front_physical_estimate(
        {"success": True, "dz": 4.0, "dyaw": np.radians(0.3), "dpitch": np.radians(-0.2), "n_points": 9},
        min_points_for_attitude=20,
        untrusted_attitude_deg=0.8,
        max_applied_attitude_deg=1.2,
    )
    assert tiny["success"] is True
    assert tiny.get("attitude_suppressed") is False
    assert np.degrees(tiny["dyaw"]) == pytest.approx(0.3)


def test_harmonic_residual_iqr_drops_false_match():
    theta = np.linspace(0.0, 360.0, 20, endpoint=False)
    z = np.full(theta.shape, 140.0)
    prev = np.column_stack([theta, z])
    curr = np.column_stack([theta, z - 5.0])
    curr[0, 0] += 40.0
    curr[0, 1] -= 30.0
    filt_p, filt_c = filter_physical_motion_vectors(
        prev, curr, min_points=4
    )
    assert len(filt_p) < len(prev)
    assert len(filt_p) >= 8


def test_apply_front_profile_enables_option2():
    config = Config()
    config.estimation.feature_matching.use_cylindrical_matching = False
    config.estimation.motion.estimate_droll = True
    apply_front_profile(config)
    cyl = config.estimation.feature_matching.cylindrical
    assert config.estimation.feature_matching.use_cylindrical_matching is True
    assert cyl.z_axis == "physical"
    assert cyl.spatial_filter_enabled is False
    assert config.estimation.feature_filtering.magnitude_filter_enabled is False
    assert config.estimation.motion.estimate_dx is False
    assert config.estimation.motion.estimate_droll is False
