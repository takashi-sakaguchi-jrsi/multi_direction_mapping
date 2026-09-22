"""T8-T10: モードA/C の拘束"""

import numpy as np
import pytest

from src.camera_estimation import CameraEstimator
from src.validation.geometry import build_run_reference, mode_a_estimate_yaw_pitch


def _known_correspondences(transformer, physical_roll_deg=0.0, n=20):
    """光軸付近の円筒点を2姿勢へ投影して対応点を作る"""
    from src.validation.geometry import optical_axis_world
    radius = transformer.pipe_radius
    ref = build_run_reference(physical_roll_deg)
    ori0 = np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]])
    axis = optical_axis_world(*ori0)
    nxy = np.linalg.norm(axis[:2]) + 1e-12
    base = np.array([axis[0] / nxy * radius, axis[1] / nxy * radius, 0.0])
    tang = np.array([-axis[1], axis[0], 0.0])
    tang = tang / (np.linalg.norm(tang) + 1e-12)
    pts_w = []
    for i in range(n):
        s = (i / max(n - 1, 1) - 0.5) * 40.0
        z = (i / max(n - 1, 1) - 0.5) * 30.0
        p = base + tang * s * 0.3
        p[:2] *= radius / (np.linalg.norm(p[:2]) + 1e-12)
        p[2] = z
        pts_w.append(p)
    pts_w = np.stack(pts_w, axis=0)
    pos0 = np.array([0.0, 0.0, 0.0])
    pos1 = np.array([0.0, 0.0, 10.0])
    p0 = transformer.world_to_pixel(pts_w, pos0, *ori0)
    p1 = transformer.world_to_pixel(pts_w, pos1, *ori0)
    cam = transformer.camera
    inside = (
        (p0[:, 0] > 1) & (p0[:, 0] < cam.image_width - 1)
        & (p0[:, 1] > 1) & (p0[:, 1] < cam.image_height - 1)
        & (p1[:, 0] > 1) & (p1[:, 0] < cam.image_width - 1)
        & (p1[:, 1] > 1) & (p1[:, 1] < cam.image_height - 1)
    )
    return p0[inside], p1[inside], ref


def _cam_params(transformer):
    cam = transformer.camera
    return {
        "center": (cam.cx, cam.cy),
        "radius": min(cam.image_width, cam.image_height) / 2,
        "f": cam.f,
        "model": "fisheye",
    }


def test_mode_a_free_angles_by_run():
    from src.validation.geometry import mode_a_use_frame_xy_residual
    assert mode_a_estimate_yaw_pitch("U") == (True, False)
    assert mode_a_estimate_yaw_pitch("R") == (False, True)
    assert mode_a_estimate_yaw_pitch("L") == (False, True)
    assert mode_a_use_frame_xy_residual("U") is True
    assert mode_a_use_frame_xy_residual("R") is False


def test_mode_a_u_maps_frame_dy_to_dz(transformer, config):
    estimator = CameraEstimator(config.estimation, transformer)
    p0, p1, ref = _known_correspondences(transformer, 0.0)
    if len(p0) < 4:
        pytest.skip("投影点が不足")
    state = {
        "position": np.array([0.0, 0.0, 0.0]),
        "orientation": np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]]),
    }
    motion = estimator.estimate_motion_flexible(
        p0, p1, state, _cam_params(transformer),
        run_reference=ref,
        center_prior=config.two_direction.center_prior,
        estimation_mode="A",
        hard_bounds=config.two_direction.hard_bounds,
    )
    assert abs(motion["dx"]) < 1e-6
    assert abs(motion["dy"]) < 1e-6
    assert abs(motion["droll"]) < 1e-6
    assert abs(motion["dpitch"]) < 1e-9
    assert motion["dz"] > 1.0
    assert abs(np.degrees(motion["dyaw"])) < 0.5


def test_u_pixel_roundtrip_matches_world_to_pixel(transformer):
    """U 姿勢で R_c2w 逆投影は world_to_pixel と往復する。"""
    ref = build_run_reference(0.0)
    r, y, p = ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]
    pos = np.array([0.0, 0.0, 0.0])
    cam = transformer.camera
    rng = np.random.default_rng(1)
    pts = np.stack([
        rng.uniform(cam.cx - 40, cam.cx + 40, 30),
        rng.uniform(cam.cy - 40, cam.cy + 40, 30),
    ], axis=1)
    world = transformer.pixel_to_world_R_c2w(pts, pos, r, y, p)
    back = transformer.world_to_pixel(world, pos, r, y, p)
    err = np.linalg.norm(back - pts, axis=1)
    assert float(np.mean(err)) < 1.0
    assert float(np.max(err)) < 3.0


def test_mode_a_r_estimates_pitch_freezes_yaw(transformer, config):
    estimator = CameraEstimator(config.estimation, transformer)
    p0, p1, ref = _known_correspondences(transformer, 120.0)
    if len(p0) < 4:
        pytest.skip("投影点が不足")
    state = {
        "position": np.array([0.0, 0.0, 0.0]),
        "orientation": np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]]),
    }
    motion = estimator.estimate_motion_flexible(
        p0, p1, state, _cam_params(transformer),
        run_reference=ref,
        center_prior=config.two_direction.center_prior,
        estimation_mode="A",
        hard_bounds=config.two_direction.hard_bounds,
    )
    assert abs(motion["dx"]) < 1e-6
    assert abs(motion["dy"]) < 1e-6
    assert abs(motion["droll"]) < 1e-6
    assert abs(motion["dyaw"]) < 1e-6
    assert motion["dz"] > 1.0


def test_mode_a_l_estimates_pitch_freezes_yaw(transformer, config):
    estimator = CameraEstimator(config.estimation, transformer)
    p0, p1, ref = _known_correspondences(transformer, 240.0)
    if len(p0) < 4:
        pytest.skip("投影点が不足")
    state = {
        "position": np.array([0.0, 0.0, 0.0]),
        "orientation": np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]]),
    }
    motion = estimator.estimate_motion_flexible(
        p0, p1, state, _cam_params(transformer),
        run_reference=ref,
        center_prior=config.two_direction.center_prior,
        estimation_mode="A",
        hard_bounds=config.two_direction.hard_bounds,
    )
    assert abs(motion["dyaw"]) < 1e-6
    assert motion["dz"] > 1.0


def test_mode_a_keeps_xy_and_roll(transformer, config):
    estimator = CameraEstimator(config.estimation, transformer)
    p0, p1, ref = _known_correspondences(transformer)
    if len(p0) < 4:
        pytest.skip("投影点が不足")
    state = {
        "position": np.array([0.0, 0.0, 0.0]),
        "orientation": np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]]),
    }
    motion = estimator.estimate_motion_flexible(
        p0, p1, state, _cam_params(transformer),
        run_reference=ref,
        center_prior=config.two_direction.center_prior,
        estimation_mode="A",
        hard_bounds=config.two_direction.hard_bounds,
    )
    assert abs(motion["dx"]) < 1e-6
    assert abs(motion["dy"]) < 1e-6
    assert abs(motion["droll"]) < 1e-6
    assert motion["dz"] > 1.0


def test_mode_c_joint_prior_does_not_avalanche(transformer, config):
    estimator = CameraEstimator(config.estimation, transformer)
    p0, p1, ref = _known_correspondences(transformer)
    if len(p0) < 4:
        pytest.skip("投影点が不足")
    state = {
        "position": np.array([0.2, -0.2, 0.0]),
        "orientation": np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]]),
    }
    motion = estimator.estimate_motion_flexible(
        p0, p1, state, _cam_params(transformer),
        run_reference=ref,
        center_prior=config.two_direction.center_prior,
        estimation_mode="C",
        hard_bounds=config.two_direction.hard_bounds,
    )
    x_new = state["position"][0] + motion["dx"]
    y_new = state["position"][1] + motion["dy"]
    assert abs(x_new) < abs(state["position"][0]) + 0.5
    assert abs(y_new) < 5.0
    assert "bound_hit" in motion
