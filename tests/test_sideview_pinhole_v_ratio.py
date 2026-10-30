"""工程2: ピンホール縦割合。主点=最近壁、帯幅 R tan β、反対壁はマスク外。"""

import numpy as np

from src.camera_utils import compute_pinhole_focal_length
from src.config import Config
from src.coordinate_transform import CoordinateTransformer, PinholeCamera
from src.validation.geometry import (
    build_run_reference,
    optical_axis_world,
    world_to_eta,
)
from src.validation.sideview_projection_mapper import SideviewProjectionMapper


def _pinhole_mapper(h=1080, w=1920, fov=60.0, radius=125.0, v_fwd=0.6):
    cfg = Config.from_defaults()
    cfg.pipe.diameter_mm = radius * 2.0
    cfg.two_direction.capture.image_width_px = w
    cfg.two_direction.capture.image_height_px = h
    cfg.two_direction.projection.v_forward_half_ratio = v_fwd
    cfg.two_direction.projection.allow_signed_relative_z = True
    cfg.colormap.alpha = 2.0
    f = compute_pinhole_focal_length(fov, (h, w))
    cam = PinholeCamera(
        f=f, cx=w / 2.0, cy=h / 2.0, image_width=w, image_height=h
    )
    xf = CoordinateTransformer(camera=cam, pipe_radius=radius, calibration=None)
    mapper = SideviewProjectionMapper(cfg, xf)
    return cfg, xf, mapper


def test_principal_point_hits_nearest_wall():
    _cfg, xf, _mapper = _pinhole_mapper()
    cam = xf.camera
    ref = build_run_reference(0.0)
    ori = (
        float(ref["roll_ref_rad"]),
        float(ref["yaw_ref_rad"]),
        float(ref["pitch_ref_rad"]),
    )
    pos = np.array([0.0, 0.0, 0.0])
    world = xf.pixel_to_world(
        np.array([[cam.cx, cam.cy]], dtype=float), pos, *ori
    )
    radial = float(np.hypot(world[0, 0], world[0, 1]))
    assert abs(radial - xf.pipe_radius) < 1.0
    assert abs(float(world[0, 2])) < 2.0
    axis = optical_axis_world(*ori)
    hit_dir = world[0] / (np.linalg.norm(world[0]) + 1e-12)
    assert float(hit_dir @ axis) > 0.95


def test_forward_strip_width_is_r_tan_beta_not_max_dz():
    cfg, xf, mapper = _pinhole_mapper(v_fwd=0.6)
    cam = xf.camera
    pos = np.array([0.0, 0.0, 40.0])
    ori = np.zeros(3)
    z0, z1 = mapper.calculate_dynamic_z_range(
        (cam.image_height, cam.image_width), pos, ori, max_dz=1.0, alpha=99.0
    )
    dv = 0.6 * (cam.image_height / 2.0)
    expected = xf.pipe_radius * (dv / cam.f)
    assert abs((z1 - z0) - expected) < 1e-6
    z2, z3 = mapper.calculate_dynamic_z_range(
        (cam.image_height, cam.image_width), pos, ori, max_dz=50.0, alpha=0.1
    )
    assert abs((z3 - z2) - expected) < 1e-6
    assert abs(z0 - 40.0) < 1e-9


def test_opposite_wall_is_outside_pinhole_mask():
    _cfg, xf, mapper = _pinhole_mapper()
    cam = xf.camera
    ref = build_run_reference(0.0)
    ori = np.array([
        ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]
    ])
    pos = np.array([0.0, 0.0, 0.0])
    frame = np.full((cam.image_height, cam.image_width, 3), 40, dtype=np.uint8)
    strip = mapper.sample_strip(
        frame, pos, ori,
        theta_bins=180,
        pixels_per_mm=0.5,
        max_dz=10.0,
        alpha=2.0,
    )
    mask = strip["mask_2d"]
    assert np.any(mask)
    eta = strip["eta_grid"][mask]
    axis = optical_axis_world(*ori)
    eta0 = float(world_to_eta(np.array([axis[0]]), np.array([axis[1]]))[0])
    dth = np.abs((eta - eta0 + np.pi) % (2.0 * np.pi) - np.pi)
    assert float(np.max(dth)) < np.radians(50.0)
    opposite = np.abs(dth - np.pi) < np.radians(25.0)
    assert not np.any(opposite)
