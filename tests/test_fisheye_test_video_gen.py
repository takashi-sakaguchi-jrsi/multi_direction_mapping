"""既存展開図から魚眼テスト動画を生成する（Phase1省略）

再現テストの正本は original_colormap/phi250tenkaizu.png（φ250mm）。
"""

from pathlib import Path

import numpy as np
import pytest

from src.generate_two_direction_test_videos import generate_two_direction_videos
from src.validation.fisheye_sideview_renderer import (
    DEFAULT_FOV_DEG,
    DEFAULT_RADIUS_MM,
    FisheyeSideviewRenderer,
    default_colormap_path,
    make_demo_layout_colormap,
    resolve_colormap_path,
    z_values_for_segment,
)

PHI250 = default_colormap_path()


def test_fisheye_fov_181_does_not_use_pinhole_tan():
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=80.0, pixels_per_mm=0.4)
    renderer = FisheyeSideviewRenderer(
        tex, output_size=(80, 80), radius_mm=125.0, fov_deg=DEFAULT_FOV_DEG
    )
    assert np.isfinite(renderer.f)
    assert renderer.f > 0
    frame = renderer.render_view((0.0, 0.0, 40.0), (0.0, 90.0))
    assert frame.shape == (80, 80, 3)
    assert frame.max() > 0


def test_u_and_r_frames_differ():
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=80.0, pixels_per_mm=0.4)
    renderer = FisheyeSideviewRenderer(
        tex, output_size=(64, 64), radius_mm=125.0, fov_deg=181.0
    )
    z = [40.0]
    u, _ = renderer.render_sequence(z, roll_deg=0.0)
    r, _ = renderer.render_sequence(z, roll_deg=120.0)
    assert not np.array_equal(u[0], r[0])


def test_roll_right_shows_bottom_on_the_right():
    """カメラカー roll=+120（R）は右下、つまり光軸が +X かつ -Y。"""
    rotation = FisheyeSideviewRenderer.demo_rotation(120.0, 90.0)
    axis = rotation @ np.array([0.0, 0.0, 1.0])
    assert axis[0] > 0.4
    assert axis[1] < -0.3
    assert abs(axis[2]) < 0.1


def test_roll_left_shows_bottom_on_the_left():
    rotation = FisheyeSideviewRenderer.demo_rotation(-120.0, 90.0)
    axis = rotation @ np.array([0.0, 0.0, 1.0])
    assert axis[0] < -0.4
    assert axis[1] < -0.3
    assert abs(axis[2]) < 0.1


def test_u_center_is_not_a_black_streak():
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=400.0, pixels_per_mm=0.4)
    tex[0, :, :] = (0, 0, 255)
    tex[-1, :, :] = (0, 255, 0)
    renderer = FisheyeSideviewRenderer(
        tex, output_size=(64, 64), radius_mm=125.0, fov_deg=181.0
    )
    frame = renderer.render_view((0.0, 0.0, 200.0), (0.0, 90.0))
    center = frame[32, 32]
    assert int(center.max()) > 40
    mid = frame[16:48, 32]
    black_frac = float(np.mean(np.all(mid < 12, axis=1)))
    assert black_frac < 0.25


def test_roundtrip_skip_phase1_matches_source_sector():
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=120.0, pixels_per_mm=0.5)
    renderer = FisheyeSideviewRenderer(
        tex, output_size=(96, 96), radius_mm=125.0, fov_deg=181.0
    )
    zs = [30.0, 50.0, 70.0]
    frames = []
    positions = []
    orientations = []
    for roll in (0.0, 120.0):
        seq, _ = renderer.render_sequence(zs, roll_deg=roll)
        for frame, z in zip(seq, zs):
            frames.append(frame)
            positions.append((0.0, 0.0, z))
            orientations.append((roll, 90.0))
    canvas, filled = renderer.reconstruct_from_hits(frames, positions, orientations)
    assert float(np.mean(filled)) > 0.15
    mae = float(np.mean(np.abs(canvas[filled].astype(np.int16) - tex[filled].astype(np.int16))))
    assert mae < 35.0


def test_cli_writes_mp4_and_skips_phase1(tmp_path: Path):
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=60.0, pixels_per_mm=0.35)
    cmap = tmp_path / "map.png"
    import cv2

    cv2.imwrite(str(cmap), tex)
    out = tmp_path / "videos"
    result = generate_two_direction_videos(
        colormap=cmap,
        output_dir=out,
        runs=["U", "R"],
        width=48,
        height=48,
        radius_mm=125.0,
        fov_deg=181.0,
        pitch_deg=90.0,
        n_frames=3,
        fps=5.0,
        name="demo_fisheye_side",
        z_start_mm=5.0,
        z_step_mm=8.0,
        z_margin_mm=1.0,
        shading=False,
        reconstruct=True,
        save_png=False,
    )
    assert result["distance_overlay"] is False
    assert result["phase1"] is False
    assert Path(result["runs"]["U"]["video_path"]).is_file()
    assert Path(result["runs"]["R"]["video_path"]).is_file()
    assert Path(result["reconstruct"]["path"]).is_file()
    assert result["reconstruct"]["filled_ratio"] > 0.05
    assert abs(result["z_start_mm"] - 5.0) < 1.0
    assert len(result["z_values_mm"]) == 3
    assert "ocr_z_mm" in result
    assert result["ocr_step_mm"] == 10.0
    assert result["z_step_mm"] == 8.0
    assert len(result["ocr_z_mm"]) == 3
    # z=5 は刻みに乗らない。次の 10mm を超えてからカウントアップ。
    assert result["ocr_z_mm"][0] == 0.0 or result["ocr_z_mm"][0] == 10.0


def test_generate_streams_mp4_without_holding_all_frames(tmp_path: Path):
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=80.0, pixels_per_mm=0.4)
    cmap = tmp_path / "map.png"
    import cv2

    cv2.imwrite(str(cmap), tex)
    out = tmp_path / "videos"
    result = generate_two_direction_videos(
        colormap=cmap,
        output_dir=out,
        runs=["U"],
        width=32,
        height=32,
        radius_mm=125.0,
        fov_deg=181.0,
        pitch_deg=90.0,
        n_frames=8,
        fps=5.0,
        name="stream_side",
        z_start_mm=8.0,
        z_step_mm=6.0,
        z_margin_mm=1.0,
        shading=False,
        reconstruct=False,
        save_png=False,
    )
    video = Path(result["runs"]["U"]["video_path"])
    assert video.is_file()
    assert result["runs"]["U"]["n_frames"] == 8
    cap = cv2.VideoCapture(str(video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    cap.release()


def test_z_start_selects_different_segment():
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=400.0, pixels_per_mm=0.4)
    renderer = FisheyeSideviewRenderer(
        tex, output_size=(48, 48), radius_mm=125.0, fov_deg=181.0
    )
    a = z_values_for_segment(renderer, z_start_mm=20.0, n_frames=5, z_step_mm=10.0, z_margin_mm=0.0)
    b = z_values_for_segment(renderer, z_start_mm=200.0, n_frames=5, z_step_mm=10.0, z_margin_mm=0.0)
    assert len(a) == 5
    assert len(b) == 5
    assert abs(a[0] - 20.0) < 1e-6
    assert abs(b[0] - 200.0) < 1e-6
    assert a[-1] < b[0]
    fa, _ = renderer.render_sequence([a[0]], roll_deg=0.0)
    fb, _ = renderer.render_sequence([b[0]], roll_deg=0.0)
    assert not np.array_equal(fa[0], fb[0])


def test_calibration_json_sets_principal_point_and_distortion():
    from src.calibration import load_calibration, fisheye_intrinsics_for_size

    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=80.0, pixels_per_mm=0.4)
    calib = load_calibration("data/calibration/fisheye_recalibrated_20260217.json")
    f, cx, cy, dist, fov = fisheye_intrinsics_for_size(calib, 64, 36)
    renderer = FisheyeSideviewRenderer(
        tex,
        output_size=(64, 36),
        radius_mm=125.0,
        fov_deg=fov,
        f_px=f,
        cx=cx,
        cy=cy,
        dist_coeffs=dist,
    )
    assert renderer.dist_coeffs is not None
    assert abs(renderer.cx - 64 * calib.cx / calib.image_width) < 1e-6
    assert abs(renderer.cy - 36 * calib.cy / calib.image_height) < 1e-6
    ideal = FisheyeSideviewRenderer(
        tex, output_size=(64, 36), radius_mm=125.0, fov_deg=181.0
    )
    z = 40.0
    a = renderer.render_view((0.0, 0.0, z), (0.0, 90.0))
    b = ideal.render_view((0.0, 0.0, z), (0.0, 90.0))
    assert a.max() > 0
    assert not np.array_equal(a, b)


def test_zero_distortion_calib_matches_explicit_intrinsics():
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=80.0, pixels_per_mm=0.4)
    renderer = FisheyeSideviewRenderer(
        tex,
        output_size=(48, 48),
        radius_mm=125.0,
        fov_deg=181.0,
        f_px=200.0,
        cx=20.0,
        cy=28.0,
        dist_coeffs=[0.0, 0.0, 0.0, 0.0],
    )
    assert renderer.f == pytest.approx(200.0)
    assert renderer.cx == pytest.approx(20.0)
    assert renderer.cy == pytest.approx(28.0)
    assert renderer.dist_coeffs is None
    iu = int(round(renderer.cx))
    iv = int(round(renderer.cy))
    ray = renderer._view_rays[iv, iu]
    assert abs(ray[0]) < 0.02
    assert abs(ray[1]) < 0.02
    assert ray[2] > 0.99


def test_generate_writes_calibration_metadata(tmp_path: Path):
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=60.0, pixels_per_mm=0.35)
    cmap = tmp_path / "map.png"
    import cv2

    cv2.imwrite(str(cmap), tex)
    out = tmp_path / "videos"
    calib = "data/calibration/fisheye_recalibrated_20260217.json"
    result = generate_two_direction_videos(
        colormap=cmap,
        output_dir=out,
        runs=["U"],
        width=48,
        height=32,
        radius_mm=125.0,
        fov_deg=181.0,
        pitch_deg=90.0,
        n_frames=2,
        fps=5.0,
        name="calib_side",
        z_start_mm=8.0,
        z_step_mm=6.0,
        z_margin_mm=1.0,
        shading=False,
        reconstruct=False,
        save_png=False,
        calibration_path=calib,
    )
    assert result["camera_model"] == "fisheye_calibrated"
    assert result["lens_calibration_file"] == calib
    assert result["cx"] != 24.0
    assert any(abs(k) > 1e-8 for k in result["distortion_k"])
    assert Path(result["runs"]["U"]["video_path"]).is_file()


def test_default_colormap_is_phi250():
    path = resolve_colormap_path()
    assert path.name == "phi250tenkaizu.png"
    assert path.is_file()
    assert "original_colormap" in path.as_posix()


@pytest.mark.skipif(not PHI250.is_file(), reason="original_colormap/phi250tenkaizu.png が無い")
def test_phi250_roundtrip_skip_phase1():
    """φ250mm 正本展開図での往復（距離overlay / Phase1 なし）。"""
    renderer = FisheyeSideviewRenderer(
        PHI250,
        output_size=(128, 128),
        radius_mm=DEFAULT_RADIUS_MM,
        fov_deg=181.0,
    )
    assert abs(renderer.radius - 125.0) < 1e-9
    z_mid = 0.5 * renderer.z_extent_mm
    zs = [max(10.0, z_mid - 40.0), z_mid, min(renderer.z_extent_mm - 10.0, z_mid + 40.0)]
    frames = []
    positions = []
    orientations = []
    for roll in (0.0, 120.0):
        seq, _ = renderer.render_sequence(zs, roll_deg=roll, pitch_deg=90.0)
        assert seq[0].max() > 0
        for frame, z in zip(seq, zs):
            frames.append(frame)
            positions.append((0.0, 0.0, z))
            orientations.append((roll, 90.0))
    canvas, filled = renderer.reconstruct_from_hits(frames, positions, orientations)
    n_filled = int(np.sum(filled))
    assert n_filled > 200
    src = renderer.equirect_img
    mae = float(np.mean(np.abs(canvas[filled].astype(np.int16) - src[filled].astype(np.int16))))
    assert mae < 45.0


def test_kannala_invert_roundtrips_distort():
    from src.coordinate_transform import FisheyeCamera

    cam = FisheyeCamera(f=343.936, cx=980.29, cy=552.66, image_width=1920, image_height=1080)
    k = np.array([0.02081173, -0.07862799, 0.05501578, -0.01391781])
    pts = np.array(
        [
            [980.29, 552.66],
            [1080.29, 552.66],
            [1180.29, 552.66],
            [980.29, 652.66],
        ],
        dtype=np.float64,
    )
    dist = cam.distort_points(pts, k)
    r_d = np.hypot(dist[:, 0] - cam.cx, dist[:, 1] - cam.cy)
    th = cam.invert_kannala_theta(r_d / cam.f, k, np.radians(92.5))
    r_u = np.hypot(pts[:, 0] - cam.cx, pts[:, 1] - cam.cy)
    assert np.all(np.isfinite(th))
    assert th == pytest.approx(r_u / cam.f, abs=1e-5)
    beyond = cam.invert_kannala_theta(
        np.array([1.55, 2.0, 10.0]), k, np.radians(92.5)
    )
    assert np.all(np.isnan(beyond))


def test_calib_rim_is_black_not_wrapped():
    from src.calibration import load_calibration, fisheye_intrinsics_for_size

    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=200.0, pixels_per_mm=0.4)
    calib = load_calibration("data/calibration/fisheye_recalibrated_20260217.json")
    f, cx, cy, dist, fov = fisheye_intrinsics_for_size(calib, 128, 72)
    renderer = FisheyeSideviewRenderer(
        tex,
        output_size=(128, 72),
        radius_mm=125.0,
        fov_deg=fov,
        f_px=f,
        cx=cx,
        cy=cy,
        dist_coeffs=dist,
    )
    frame = renderer.render_view((0.0, 0.0, 80.0), (0.0, 90.0))
    gray = frame.mean(axis=2)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - renderer.cx, yy - renderer.cy)
    rmax = min(w, h) / 2.0
    inner = r < 0.5 * rmax
    rim = (r >= 0.9 * rmax) & (r <= 1.0 * rmax)
    assert float(gray[inner].mean()) > 20.0
    assert float((gray[rim] > 250).mean()) < 0.05
    vz = renderer._view_rays[..., 2]
    finite = np.isfinite(vz)
    gamma_deg = np.degrees(np.arccos(np.clip(vz[finite], -1.0, 1.0)))
    half_fov = float(renderer.fov_deg) / 2.0
    assert float(np.max(gamma_deg)) <= half_fov + 0.5
    assert float(np.max(gamma_deg)) < 180.0
    assert float(np.min(vz[finite])) > float(np.cos(np.radians(half_fov))) - 1e-3


def test_calib_rays_never_wrap_behind_the_lens():
    from src.calibration import load_calibration, fisheye_intrinsics_for_size

    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=80.0, pixels_per_mm=0.4)
    calib = load_calibration("data/calibration/fisheye_recalibrated_20260217.json")
    f, cx, cy, dist, fov = fisheye_intrinsics_for_size(calib, 192, 108)
    renderer = FisheyeSideviewRenderer(
        tex,
        output_size=(192, 108),
        radius_mm=125.0,
        fov_deg=fov,
        f_px=f,
        cx=cx,
        cy=cy,
        dist_coeffs=dist,
    )
    rays = renderer._view_rays
    finite = np.isfinite(rays).all(axis=-1)
    vz = rays[..., 2]
    gamma = np.arccos(np.clip(vz[finite], -1.0, 1.0))
    assert float(np.degrees(np.max(gamma))) < 180.0
    assert float(np.degrees(np.max(gamma))) <= float(fov) / 2.0 + 0.5
    # 周回すると vz が強く負になる。185° FOV なら赤道よりわずかに後ろまで。
    assert float(np.min(vz[finite])) > -0.12
