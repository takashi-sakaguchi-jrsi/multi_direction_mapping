"""ピンホール 60°・8 方向の仮想動画生成"""

from pathlib import Path

import numpy as np
import pytest

from src.camera_utils import compute_pinhole_focal_length
from src.generate_two_direction_test_videos import (
    default_pinhole_run_ids,
    generate_two_direction_videos,
    parse_runs,
)
from src.validation.fisheye_sideview_renderer import (
    FisheyeSideviewRenderer,
    make_demo_layout_colormap,
)


def test_parse_runs_accepts_three_digit_ids():
    assert parse_runs("000,045,090") == ["000", "045", "090"]
    assert parse_runs("U,R") == ["U", "R"]


def test_default_pinhole_run_ids_are_eight_45deg():
    ids = default_pinhole_run_ids()
    assert ids == [
        "000", "045", "090", "135", "180", "225", "270", "315",
    ]


def test_pinhole_renderer_uses_atan_not_equidistant():
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=80.0, pixels_per_mm=0.4)
    renderer = FisheyeSideviewRenderer(
        tex, output_size=(80, 80), radius_mm=125.0, fov_deg=60.0,
        camera_model="pinhole",
    )
    expected_f = compute_pinhole_focal_length(60.0, (80, 80))
    assert renderer.f == pytest.approx(expected_f)
    iu = int(round(renderer.cx))
    iv = int(round(renderer.cy))
    ray = renderer._view_rays[iv, iu]
    assert abs(ray[0]) < 0.02
    assert abs(ray[1]) < 0.02
    assert ray[2] > 0.99
    edge = renderer._view_rays[iv, 0]
    r_px = abs(0.0 - renderer.cx)
    gamma = np.arctan(r_px / renderer.f)
    assert edge[2] == pytest.approx(np.cos(gamma), abs=0.02)


def test_pinhole_fills_rectangular_frame_not_inscribed_circle():
    """ピンホール 60° は長方形全域。魚眼のような内接円の黒は出さない。"""
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=400.0, pixels_per_mm=0.4)
    width, height = 80, 48
    pin = FisheyeSideviewRenderer(
        tex, output_size=(width, height), radius_mm=125.0, fov_deg=60.0,
        camera_model="pinhole",
    )
    fish = FisheyeSideviewRenderer(
        tex, output_size=(width, height), radius_mm=125.0, fov_deg=181.0,
        camera_model="fisheye",
    )
    assert np.isfinite(pin._view_rays[0, 0]).all()
    assert np.isfinite(pin._view_rays[height // 2, 0]).all()
    assert np.isfinite(pin._view_rays[height // 2, width - 1]).all()
    assert not np.isfinite(fish._view_rays[0, 0]).all()
    r_left = abs(0.0 - pin.cx)
    gamma_left = np.arctan(r_left / pin.f)
    assert gamma_left > np.radians(30.0)
    edge = pin._view_rays[height // 2, 0]
    assert edge[2] == pytest.approx(np.cos(gamma_left), abs=0.02)

    frame = pin.render_view((0.0, 0.0, 200.0), (0.0, 90.0))
    assert float(np.mean(np.all(frame == 0, axis=2))) < 0.01
    assert frame[height // 2, 0].max() > 20
    assert frame[height // 2, width - 1].max() > 20
    assert frame[1, 1].max() > 20
    assert frame[1, width - 2].max() > 20
    yy, xx = np.mgrid[0:height, 0:width]
    outside = np.hypot(xx - pin.cx, yy - pin.cy) > (min(width, height) / 2.0 + 1.0)
    assert frame[outside].reshape(-1, 3).max(axis=1).mean() > 20
    fish_frame = fish.render_view((0.0, 0.0, 200.0), (0.0, 90.0))
    assert fish_frame[1, 1].max() == 0


def test_pinhole_generation_rejects_calibration(tmp_path: Path):
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=60.0, pixels_per_mm=0.35)
    cmap = tmp_path / "map.png"
    import cv2

    cv2.imwrite(str(cmap), tex)
    with pytest.raises(ValueError, match="校正"):
        generate_two_direction_videos(
            colormap=cmap,
            output_dir=tmp_path / "videos",
            runs=["000"],
            width=32,
            height=32,
            radius_mm=125.0,
            fov_deg=60.0,
            pitch_deg=90.0,
            n_frames=2,
            fps=5.0,
            name="hp8",
            z_start_mm=8.0,
            z_step_mm=6.0,
            z_margin_mm=1.0,
            shading=False,
            reconstruct=False,
            save_png=False,
            camera_model="pinhole",
            calibration_path="data/calibration/fisheye_recalibrated_20260217.json",
        )


def test_pinhole_8dir_metadata_omits_calibration(tmp_path: Path):
    tex = make_demo_layout_colormap(radius_mm=125.0, z_max_mm=80.0, pixels_per_mm=0.4)
    cmap = tmp_path / "map.png"
    import cv2

    cv2.imwrite(str(cmap), tex)
    out = tmp_path / "videos"
    runs = default_pinhole_run_ids()
    result = generate_two_direction_videos(
        colormap=cmap,
        output_dir=out,
        runs=runs,
        width=32,
        height=32,
        radius_mm=125.0,
        fov_deg=60.0,
        pitch_deg=90.0,
        n_frames=2,
        fps=5.0,
        name="hp8",
        z_start_mm=8.0,
        z_step_mm=6.0,
        z_margin_mm=1.0,
        shading=False,
        reconstruct=False,
        save_png=False,
        camera_model="pinhole",
        n_directions=8,
    )
    assert result["camera_model"] == "pinhole"
    assert result["fov_deg"] == pytest.approx(60.0)
    assert result["n_directions"] == 8
    assert "lens_calibration_file" not in result
    assert list(result["runs"].keys()) == runs
    assert result["runs"]["000"]["physical_roll_deg"] == pytest.approx(0.0)
    assert result["runs"]["045"]["physical_roll_deg"] == pytest.approx(45.0)
    assert result["runs"]["315"]["physical_roll_deg"] == pytest.approx(315.0)
    video = Path(result["runs"]["000"]["video_path"])
    assert video.name == "hp8_000.mp4"
    assert video.is_file()
    meta = Path(result["metadata_path"])
    text = meta.read_text(encoding="utf-8")
    assert "lens_calibration_file" not in text
    assert '"camera_model": "pinhole"' in text
    run_meta = Path(result["runs"]["000"]["metadata_path"])
    assert run_meta.name == "hp8_000_metadata.json"
    assert run_meta.is_file()
    assert "lens_calibration_file" not in run_meta.read_text(encoding="utf-8")
