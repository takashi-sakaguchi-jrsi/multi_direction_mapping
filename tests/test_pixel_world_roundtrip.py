"""pixel_to_world は world_to_pixel と同じ壁へ戻る。"""

import numpy as np
import pytest

from src.coordinate_transform import CoordinateTransformer, FisheyeCamera, R_w2c


def _front_transformer():
    cam = FisheyeCamera(
        f=343.94, cx=630.3, cy=552.7, image_width=1220, image_height=1080
    )
    return CoordinateTransformer(cam, 125.0)


def _wrap_rad(delta):
    return (np.asarray(delta) + np.pi) % (2.0 * np.pi) - np.pi


def test_pixel_to_world_matches_r_c2w_on_yawed_front():
    xf = _front_transformer()
    pos = np.array([0.0, 0.0, 0.0])
    roll, yaw, pitch = 0.0, np.radians(11.63), np.radians(-4.71)
    rng = np.random.default_rng(0)
    pts = np.stack(
        [
            rng.uniform(xf.camera.cx - 80, xf.camera.cx + 80, 40),
            rng.uniform(xf.camera.cy - 80, xf.camera.cy + 80, 40),
        ],
        axis=1,
    )
    a = xf.pixel_to_world(pts, pos, roll, yaw, pitch)
    b = xf.pixel_to_world_R_c2w(pts, pos, roll, yaw, pitch)
    assert np.max(np.linalg.norm(a - b, axis=1)) < 1e-6


def test_world_pixel_roundtrip_keeps_eta_not_opposite_wall():
    xf = _front_transformer()
    pos = np.array([0.0, 0.0, 0.0])
    roll, yaw, pitch = 0.0, np.radians(11.63), np.radians(-4.71)
    radius = xf.pipe_radius
    eta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    z = np.full(eta.shape, 80.0)
    world = np.stack([radius * np.cos(eta), radius * np.sin(eta), z], axis=-1)
    pix = xf.world_to_pixel(world, pos, roll, yaw, pitch)
    pts_cam = (world - pos) @ R_w2c(roll, pitch, yaw)
    in_front = pts_cam[:, 2] > 1e-3
    in_img = (
        (pix[:, 0] >= 0)
        & (pix[:, 0] < xf.camera.image_width)
        & (pix[:, 1] >= 0)
        & (pix[:, 1] < xf.camera.image_height)
    )
    keep = in_front & in_img
    assert int(np.sum(keep)) >= 8
    back = xf.pixel_to_world(pix[keep], pos, roll, yaw, pitch)
    eta_back = np.arctan2(back[:, 1], back[:, 0])
    deta = np.abs(_wrap_rad(eta_back - eta[keep]))
    assert float(np.max(deta)) < np.radians(3.0)
    assert float(np.max(np.abs(back[:, 2] - z[keep]))) < 1.0


def test_yawed_donut_does_not_fall_on_far_side_near_z():
    """Yaw で手前の3時画素が、9時・z=0–50 へ落ちない。"""
    xf = _front_transformer()
    pos = np.array([0.0, 0.0, 0.0])
    roll, yaw, pitch = 0.0, np.radians(11.63), np.radians(-4.71)
    cx, cy = xf.camera.cx, xf.camera.cy
    radius_px = min(cx, cy)
    ys, xs = np.mgrid[0:1080:3, 0:1220:3]
    rr = np.hypot(xs - cx, ys - cy)
    mask = (rr >= radius_px * 0.40) & (rr <= radius_px * 0.80)
    pts = np.stack([xs[mask], ys[mask]], axis=-1).astype(float)
    world = xf.pixel_to_world(pts, pos, roll, yaw, pitch)
    eta_deg = (np.degrees(np.arctan2(world[:, 1], world[:, 0])) + 360.0) % 360.0
    z = world[:, 2]
    opposite_near = (eta_deg >= 160.0) & (eta_deg <= 200.0) & (z >= 0.0) & (z <= 50.0)
    assert int(np.sum(opposite_near)) == 0


def test_kannala_distortion_does_not_paint_yawed_left_near_slit():
    """順歪みがドーナツ外のすれすれ点をリングへ折り込んでも塗らない。"""
    from src.colormap_generator import ColorMapGenerator
    from src.config import load_config
    from src.product_runtime import apply_front_profile

    cfg = load_config()
    apply_front_profile(cfg)
    cfg.colormap.inner_radius_ratio = 0.4
    cfg.colormap.outer_radius_ratio = 0.8
    cfg.colormap.reject_z_behind_camera = True
    xf = _front_transformer()
    xf.calibration = cfg.camera._lens_calibration
    gen = ColorMapGenerator(cfg, xf)
    eta = np.radians(np.linspace(160.0, 200.0, 21))
    z = np.linspace(8.0, 50.0, 15)
    eta_g, z_g = np.meshgrid(eta, z, indexing="ij")
    frame = np.full((1080, 1220, 3), 255, dtype=np.uint8)
    state = {
        "position": np.array([0.0, 0.0, 0.0]),
        "orientation": [0.0, np.radians(11.63), np.radians(-4.71)],
    }
    camera_params = {"center": (630.3, 552.7), "radius": min(630.3, 552.7)}
    _colors, mask, _r = gen.sample_colors_from_frame(
        frame, eta_g, z_g, state, camera_params
    )
    assert int(np.sum(mask)) == 0


def test_kannala_image_donut_still_paints_valid_left_wall():
    """実画像ドーナツ内かつピーク以前の9時帯は塗る。"""
    from src.colormap_generator import ColorMapGenerator
    from src.config import load_config
    from src.product_runtime import apply_front_profile

    cfg = load_config()
    apply_front_profile(cfg)
    xf = _front_transformer()
    xf.calibration = cfg.camera._lens_calibration
    gen = ColorMapGenerator(cfg, xf)
    eta_g = np.full((1, 25), np.pi)
    z_g = np.linspace(70.0, 140.0, 25).reshape(1, -1)
    frame = np.full((1080, 1220, 3), 255, dtype=np.uint8)
    state = {
        "position": np.array([0.0, 0.0, 0.0]),
        "orientation": [0.0, np.radians(11.63), np.radians(-4.71)],
    }
    camera_params = {"center": (630.3, 552.7), "radius": min(630.3, 552.7)}
    _colors, mask, _r = gen.sample_colors_from_frame(
        frame, eta_g, z_g, state, camera_params
    )
    assert int(np.sum(mask)) > 0


def test_kannala_rising_branch_mask_splits_at_peak():
    from src.config import load_config

    xf = _front_transformer()
    xf.calibration = load_config().camera._lens_calibration
    cam = xf.camera
    r70 = cam.f * np.radians(70.0)
    r100 = cam.f * np.radians(100.0)
    pts = np.array(
        [[cam.cx + r70, cam.cy], [cam.cx + r100, cam.cy]], dtype=float
    )
    mask = xf.kannala_rising_branch_mask(pts)
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False


def test_cylindrical_invert_rejects_folded_near_left_wall():
    from src.config import load_config
    from src.feature_matching_cylindrical import (
        CylindricalFeatureMatcher,
        CylindricalMatchingConfig,
    )

    xf = _front_transformer()
    xf.calibration = load_config().camera._lens_calibration
    matcher = CylindricalFeatureMatcher(
        xf, CylindricalMatchingConfig(z_axis="physical")
    )
    params = {
        "pipe_diameter": 250.0,
        "center": (xf.camera.cx, xf.camera.cy),
        "image_width": xf.camera.image_width,
        "image_height": xf.camera.image_height,
        "cam_pos": np.array([0.0, 0.0, 0.0]),
        "roll": 0.0,
        "yaw": np.radians(11.63),
        "pitch": np.radians(-4.71),
    }
    pts = np.array([[180.0, 5.0], [180.0, 100.0]], dtype=float)
    _xy, valid = matcher._invert_to_frame_coordinates(
        pts, np.zeros((1, 1)), np.zeros((1, 1)), params
    )
    assert bool(valid[0]) is False
    assert bool(valid[1]) is True


def test_kannala_theta_peak_is_near_89deg():
    xf = _front_transformer()
    k = np.array(
        [0.020811732077673593, -0.07862798993691063,
         0.0550157842354669, -0.013917812949331134]
    )
    peak = xf.camera.kannala_theta_peak(k, np.radians(110.0))
    assert np.degrees(peak) == pytest.approx(89.0, abs=1.0)
