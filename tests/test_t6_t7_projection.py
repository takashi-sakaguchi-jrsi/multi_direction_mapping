"""T6/T7: 側方投影の幾何と z_cam 前後"""

import numpy as np
import pytest

from src.validation.geometry import build_run_reference, optical_axis_world


def test_center_pixel_d_wall_is_pipe_radius(mapper, transformer, config):
    ref = build_run_reference(0.0)
    ori = np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]])
    pos = np.array([0.0, 0.0, 0.0])
    cam = transformer.camera
    frame = np.zeros((cam.image_height, cam.image_width, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)
    proj = mapper.project_frame(frame, pos, ori, roi_rect=[0, 0, cam.image_width, cam.image_height])
    # 画像中心に最も近い有効画素
    dist = np.hypot(proj["u"] - cam.cx, proj["v"] - cam.cy)
    valid = proj["valid"]
    assert np.any(valid)
    idx = np.argmin(np.where(valid, dist, 1e9))
    radius = config.pipe.diameter_mm / 2.0
    assert abs(proj["d_wall"][idx] - radius) < 8.0
    axis = optical_axis_world(*ori)
    hit = proj["p_wall"][idx]
    # 中心画素はほぼ光軸方向の壁
    dir_hit = hit / (np.linalg.norm(hit) + 1e-9)
    assert float(dir_hit @ axis) > 0.85


def test_same_frame_has_z_both_sides(mapper, transformer):
    ref = build_run_reference(0.0)
    ori = np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]])
    z_cam = 50.0
    pos = np.array([0.0, 0.0, z_cam])
    cam = transformer.camera
    frame = np.zeros((cam.image_height, cam.image_width, 3), dtype=np.uint8)
    frame[:] = 40
    proj = mapper.project_frame(frame, pos, ori, roi_rect=[0, 0, cam.image_width, cam.image_height])
    z = proj["z"][proj["valid"]]
    assert z.min() < z_cam - 1.0
    assert z.max() > z_cam + 1.0


def test_r_generated_frame_unwraps_as_strip(config):
    """Demo 生成の R フレームは、変換後姿勢で帯状に展開される（楕円に潰れない）。"""
    from src.coordinate_transform import CoordinateTransformer, FisheyeCamera
    from src.validation.fisheye_sideview_renderer import (
        FisheyeSideviewRenderer,
        make_demo_layout_colormap,
    )
    from src.validation.sideview_projection_mapper import SideviewProjectionMapper

    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=120.0, pixels_per_mm=0.5)
    renderer = FisheyeSideviewRenderer(
        tex, output_size=(96, 72), radius_mm=125.0, fov_deg=181.0
    )
    cam = FisheyeCamera(
        f=renderer.f, cx=renderer.cx, cy=renderer.cy,
        image_width=96, image_height=72,
    )
    transformer = CoordinateTransformer(camera=cam, pipe_radius=125.0)
    config.two_direction.capture.image_width_px = 96
    config.two_direction.capture.image_height_px = 72
    mapper = SideviewProjectionMapper(config, transformer)
    mapper.sample_stride = 2
    z_cam = 50.0
    frame = renderer.render_view((0.0, 0.0, z_cam), (120.0, 90.0))
    ref = build_run_reference(120.0)
    ori = np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]])
    pos = np.array([0.0, 0.0, z_cam])
    proj = mapper.project_frame(frame, pos, ori, roi_rect=[0, 0, 96, 72])
    valid = proj["valid"]
    assert np.mean(valid) > 0.2
    th = proj["theta"][valid]
    th0 = float(np.median(th))
    dth = np.abs((th - th0 + np.pi) % (2.0 * np.pi) - np.pi)
    assert float(np.percentile(dth, 90)) < np.radians(70.0)
    z = proj["z"][valid]
    assert z.max() - z.min() > 10.0


def test_inverse_strip_is_dense_without_stride_holes(mapper, transformer, config):
    ref = build_run_reference(0.0)
    ori = np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]])
    pos = np.array([0.0, 0.0, 50.0])
    cam = transformer.camera
    frame = np.zeros((cam.image_height, cam.image_width, 3), dtype=np.uint8)
    frame[:] = (0, 0, 200)
    config.colormap.pixels_per_mm = 1.0
    strip = mapper.sample_strip(
        frame, pos, ori,
        theta_bins=64,
        pixels_per_mm=1.0,
        max_dz=8.0,
        alpha=1.2,
    )
    mask = strip["mask_2d"]
    assert np.any(mask)
    # 有効領域は格子穴なく連続（stride=2 の市松模様にならない）
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    core = mask[y0:y1 + 1, x0:x1 + 1]
    assert float(np.mean(core)) > 0.5
    rgb = strip["colors_2d"][mask]
    assert np.median(rgb[:, 0]) > 150


def test_inverse_bilinear_uses_neighbor_pixels(mapper, transformer):
    ref = build_run_reference(0.0)
    ori = np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]])
    pos = np.array([0.0, 0.0, 40.0])
    cam = transformer.camera
    frame = np.zeros((cam.image_height, cam.image_width, 3), dtype=np.uint8)
    # 水平方向に 0→255 のグラデーション（B チャネル）
    for u in range(cam.image_width):
        frame[:, u, 0] = int(round(255.0 * u / max(cam.image_width - 1, 1)))
    eta = np.full((1, 1), np.pi)
    z = np.full((1, 1), 40.0)
    colors, mask = mapper.sample_colors_from_frame(frame, eta, z, pos, ori)
    assert bool(mask[0, 0])
    world = mapper.eta_z_to_world(eta, z).reshape(1, 3)
    px = transformer.world_to_pixel(world, pos, float(ori[0]), float(ori[1]), float(ori[2]))
    u = float(px[0, 0])
    expected = 255.0 * u / max(cam.image_width - 1, 1)
    assert abs(float(colors[0, 0, 2]) - expected) < 3.0


def test_strip_z_span_is_max_dz_times_alpha(mapper, transformer):
    ref = build_run_reference(0.0)
    ori = np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]])
    z_cam = 40.0
    pos = np.array([0.0, 0.0, z_cam])
    cam = transformer.camera
    frame = np.zeros((cam.image_height, cam.image_width, 3), dtype=np.uint8)
    max_dz, alpha, ppm = 10.0, 1.5, 2.546
    strip = mapper.sample_strip(
        frame, pos, ori,
        theta_bins=32,
        pixels_per_mm=ppm,
        max_dz=max_dz,
        alpha=alpha,
    )
    span = max_dz * alpha
    assert float(strip["z_min_strip"]) == pytest.approx(z_cam)
    assert float(strip["z_max_strip"]) == pytest.approx(z_cam + span)
    assert strip["colors_2d"].shape[1] == int(np.ceil(span * ppm))


def test_eta0_world_is_minus_y(mapper):
    eta = np.array([[0.0]])
    z = np.array([[10.0]])
    p = mapper.eta_z_to_world(eta, z)[0, 0]
    np.testing.assert_allclose(p[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(p[1], -mapper.pipe_radius, atol=1e-9)
