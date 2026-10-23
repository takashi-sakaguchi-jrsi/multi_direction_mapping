"""直視タイルの残差 (η, z) 微調整"""

import numpy as np
import pytest

from src.config import Config, FrontConfig
from src.feature_matching_cylindrical import estimate_front_physical_option2
from src.front_residual_warp import (
    COLORMAP_ETA_OFFSET_RAD,
    apply_residual_to_grids,
    build_residual_warp_field,
    point_residuals,
)
from src.product_runtime import apply_front_profile


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


def _circle_points(n=36, z=150.0):
    theta = np.linspace(0.0, 360.0, n, endpoint=False)
    return np.column_stack([theta, np.full(n, z)])


def test_uniform_dtheta_moves_eta_grid():
    radius = 125.0
    prev = _circle_points()
    extra_deg = 1.0
    curr = prev.copy()
    curr[:, 0] = prev[:, 0] + extra_deg
    field = build_residual_warp_field(
        prev, curr,
        dz=0.0, dyaw=0.0, dpitch=0.0, pipe_radius_mm=radius,
        theta_bins=12, z_bins=1, min_bin_points=2,
    )
    assert field is not None
    eta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False) + COLORMAP_ETA_OFFSET_RAD
    z = np.full(eta.shape, 150.0)
    eta_g, z_g = np.meshgrid(eta, z[:1], indexing="ij")
    eta_s, z_s = apply_residual_to_grids(eta_g, z_g, field)
    d_eta = eta_s - eta_g
    assert np.median(d_eta) == pytest.approx(np.radians(extra_deg), abs=np.radians(0.15))
    assert np.max(np.abs(z_s - z_g)) < 0.2


def test_uniform_dz_moves_z_grid():
    radius = 125.0
    prev = _circle_points()
    extra_mm = 3.0
    curr = prev.copy()
    curr[:, 1] = prev[:, 1] + extra_mm
    field = build_residual_warp_field(
        prev, curr,
        dz=0.0, dyaw=0.0, dpitch=0.0, pipe_radius_mm=radius,
        theta_bins=12, z_bins=1, min_bin_points=2,
    )
    assert field is not None
    eta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False) + COLORMAP_ETA_OFFSET_RAD
    z = np.full(eta.shape, 150.0)
    eta_g, z_g = np.meshgrid(eta, z[:1], indexing="ij")
    eta_s, z_s = apply_residual_to_grids(eta_g, z_g, field)
    assert np.median(z_s - z_g) == pytest.approx(extra_mm, abs=0.4)
    assert np.median(np.abs(eta_s - eta_g)) < np.radians(0.2)


def test_threedof_explained_residual_is_zero():
    radius = 125.0
    prev = _circle_points(n=48)
    dz, dyaw, dpitch = 5.0, np.radians(0.4), np.radians(-0.25)
    dZ, dth = _synthetic_flow(prev[:, 0], prev[:, 1], radius, dz, dyaw, dpitch)
    curr = np.column_stack([prev[:, 0] + dth, prev[:, 1] + dZ])
    est = estimate_front_physical_option2(prev, curr, pipe_radius_mm=radius)
    assert est["success"]
    _th, _z, r_th, r_z = point_residuals(
        prev, curr, est["dz"], est["dyaw"], est["dpitch"], radius
    )
    assert float(np.max(np.abs(np.degrees(r_th)))) < 0.05
    assert float(np.max(np.abs(r_z))) < 0.2
    field = build_residual_warp_field(
        prev, curr,
        dz=est["dz"], dyaw=est["dyaw"], dpitch=est["dpitch"],
        pipe_radius_mm=radius, theta_bins=12, z_bins=1,
    )
    assert field is None


def test_local_residual_does_not_fill_empty_bins():
    radius = 125.0
    theta = np.linspace(-12.0, 12.0, 16)
    prev = np.column_stack([theta, np.full(theta.shape, 150.0)])
    curr = prev.copy()
    curr[:, 0] = prev[:, 0] + 1.0
    field = build_residual_warp_field(
        prev, curr,
        dz=0.0, dyaw=0.0, dpitch=0.0, pipe_radius_mm=radius,
        theta_bins=12, z_bins=1, min_bin_points=2,
    )
    assert field is not None
    eta_local = np.deg2rad(0.0) + COLORMAP_ETA_OFFSET_RAD
    eta_far = np.deg2rad(180.0) + COLORMAP_ETA_OFFSET_RAD
    z = np.array([[150.0]])
    d_eta, _dz = field.deltas_on_grid(np.array([[eta_local]]), z)
    d_eta_far, _dz_far = field.deltas_on_grid(np.array([[eta_far]]), z)
    assert float(d_eta[0, 0]) > np.radians(0.4)
    assert abs(float(d_eta_far[0, 0])) < np.radians(0.15)


def test_too_few_points_returns_none():
    prev = np.array([[10.0, 150.0]])
    curr = np.array([[11.0, 151.0]])
    field = build_residual_warp_field(
        prev, curr, dz=0.0, dyaw=0.0, dpitch=0.0, pipe_radius_mm=125.0
    )
    assert field is None


def test_clip_rejects_large_residual():
    prev = _circle_points()
    curr = prev.copy()
    curr[:, 1] = prev[:, 1] + 20.0
    field = build_residual_warp_field(
        prev, curr,
        dz=0.0, dyaw=0.0, dpitch=0.0, pipe_radius_mm=125.0,
        max_dz_mm=8.0, theta_bins=12, z_bins=1,
    )
    assert field is None


def test_none_field_leaves_grids():
    eta = np.array([[1.0, 2.0]])
    z = np.array([[10.0, 11.0]])
    eta_s, z_s = apply_residual_to_grids(eta, z, None)
    assert eta_s is eta
    assert z_s is z


def test_front_config_residual_defaults_and_profile():
    front = FrontConfig()
    assert front.residual_paste_warp is True
    assert front.residual_theta_bins == 12
    assert front.residual_z_bins == 4
    assert front.residual_max_dtheta_deg == pytest.approx(2.0)
    assert front.residual_max_dz_mm == pytest.approx(8.0)
    config = Config()
    apply_front_profile(config)
    assert config.front.residual_paste_warp is True
    assert config.estimation.motion.estimate_droll is False
