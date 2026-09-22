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
