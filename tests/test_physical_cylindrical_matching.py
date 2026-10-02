"""物理 (θ, Z) マッチングと正面案②（roll 固定、dz は ΔZ の DC）。"""

import numpy as np
import pytest

from src.config import Config
from src.coordinate_transform import CoordinateTransformer, FisheyeCamera
from src.feature_matching_cylindrical import (
    CylindricalFeatureMatcher,
    CylindricalMatchingConfig,
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


def test_harmonic_residual_iqr_drops_false_match():
    theta = np.linspace(0.0, 360.0, 20, endpoint=False)
    z = np.full(theta.shape, 140.0)
    prev = np.column_stack([theta, z])
    curr = np.column_stack([theta, z - 5.0])
    curr[0, 0] += 40.0
    curr[0, 1] -= 30.0
    filt_p, filt_c = filter_physical_motion_vectors(
        prev, curr, min_points=4, max_dtheta_deg=50.0, max_dz_mm=40.0
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
