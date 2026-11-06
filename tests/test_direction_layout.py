"""工程1: n 方向幾何・Euler・名前解決（0° 単体、R/L を 8 方向に載せない）"""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from src.config import Config, RunConfig, build_two_direction_config
from src.product_runtime import (
    apply_product_cli,
    apply_sideview_named_input,
    build_direction_progress_plan,
    resolve_named_run_videos,
)
from src.validation.geometry import (
    assigned_sector_contains_eta0,
    build_direction_layout,
    build_run_reference,
    canonical_run_id,
    capture_to_program_orientation_deg,
    infer_suffix_from_path,
    mode_a_estimate_yaw_pitch,
    mode_a_use_frame_xy_residual,
    optical_axis_near_pipe_vertical,
    optical_axis_world,
    pose_R_c2w,
    sector_camera_car_bounds,
    theta_in_camera_car_sector,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_canonical_run_id_is_three_digits_without_quantizing():
    assert canonical_run_id(0.0) == "000"
    assert canonical_run_id(45.0) == "045"
    assert canonical_run_id(315.0) == "315"
    assert canonical_run_id(47.0) == "047"
    assert canonical_run_id(-120.0) == "240"


def test_eight_dir_layout_cuts_and_adjacent_seams():
    layout = build_direction_layout(8, 45.0, 60.0)
    assert layout.n_directions == 8
    assert layout.assigned_width_deg == pytest.approx(45.0)
    assert layout.rolls_deg == tuple(i * 45.0 for i in range(8))
    assert layout.run_ids == (
        "000", "045", "090", "135", "180", "225", "270", "315"
    )
    assert layout.cut_camera_car_deg[0] == pytest.approx(22.5)
    assert layout.cut_camera_car_deg[1] == pytest.approx(67.5)
    assert ("000", "045") in layout.seam_pairs
    assert ("315", "000") in layout.seam_pairs
    assert ("000", "090") not in layout.seam_pairs
    lo, hi = layout.sectors["000"]
    assert lo == pytest.approx(-22.5)
    assert hi == pytest.approx(22.5)
    assert layout.overlap_center_deg[("000", "045")] == pytest.approx(22.5)
    assert layout.boundary_partners["000"] == ("315", "045")


def test_three_dir_layout_matches_legacy_120_sectors():
    layout = build_direction_layout(3, 120.0, 120.0)
    assert layout.run_ids == ("000", "120", "240")
    assert layout.assigned_width_deg == pytest.approx(120.0)
    assert layout.sectors["000"] == pytest.approx((-60.0, 60.0))
    assert layout.sectors["120"] == pytest.approx((60.0, 180.0))
    assert layout.sectors["240"] == pytest.approx((180.0, 300.0))
    lo, hi = sector_camera_car_bounds("U")
    assert (lo, hi) == pytest.approx((-60.0, 60.0))


def test_capture_to_program_matches_legacy_u_and_eight_optical_axes():
    assert capture_to_program_orientation_deg(90.0, 0.0) == (-90.0, 90.0, 90.0)
    assert capture_to_program_orientation_deg(90.0, 120.0) == (-90.0, 90.0, -30.0)
    assert capture_to_program_orientation_deg(90.0, -120.0) == (90.0, -90.0, -30.0)
    assert capture_to_program_orientation_deg(90.0, 240.0) == (90.0, -90.0, -30.0)
    for car in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
        rpy = capture_to_program_orientation_deg(90.0, car)
        axis = optical_axis_world(*np.radians(rpy))
        want = np.array([
            np.sin(np.radians(car)),
            np.cos(np.radians(car)),
            0.0,
        ])
        assert abs(axis[2]) < 1e-6, car
        assert float(axis @ want) > 0.999, f"{car}: {axis} vs {want}"
        R = pose_R_c2w(*np.radians(rpy))
        np.testing.assert_allclose(R[1], [0.0, 0.0, -1.0], atol=1e-6)


def test_zero_degree_run_reference_stays_u_for_three_dir():
    ref = build_run_reference(0.0)
    assert ref["run_id"] == "U"
    assert (ref["roll_ref_deg"], ref["yaw_ref_deg"], ref["pitch_ref_deg"]) == (
        -90.0, 90.0, 90.0
    )


def test_mode_a_residual_uses_optical_axis_not_letter():
    assert mode_a_use_frame_xy_residual("U") is True
    assert mode_a_use_frame_xy_residual("000") is True
    assert mode_a_use_frame_xy_residual("180") is True
    assert mode_a_use_frame_xy_residual("045") is False
    assert mode_a_use_frame_xy_residual("R") is False
    assert optical_axis_near_pipe_vertical("000")
    assert not optical_axis_near_pipe_vertical("090")
    for rid in ("000", "045", "180", "315", "U", "R", "L"):
        assert mode_a_estimate_yaw_pitch(rid) == (False, True)


def test_theta_sector_000_wraps_across_zero():
    eta = np.array([np.pi])  # カメラカー 0°
    mask = theta_in_camera_car_sector(eta, "000")
    assert bool(mask[0])
    eta_right = np.array([np.radians(180.0 - 22.4)])
    eta_out = np.array([np.radians(180.0 - 30.0)])
    assert bool(theta_in_camera_car_sector(eta_right, "000")[0])
    assert not bool(theta_in_camera_car_sector(eta_out, "000")[0])
    # 3 方向 U は 120° 帯のまま
    assert bool(theta_in_camera_car_sector(eta_out, "U")[0])
    layout = build_direction_layout(8, 45.0, 60.0)
    assert assigned_sector_contains_eta0("180", layout=layout)
    assert not assigned_sector_contains_eta0("000", layout=layout)
    assert assigned_sector_contains_eta0("L")


def test_infer_suffix_three_digit_and_u_alias():
    assert infer_suffix_from_path("data/input/videos/test01_hp250_U.mp4") == "U"
    assert infer_suffix_from_path("/tmp/pipe_000.mp4") == "000"
    assert infer_suffix_from_path("/tmp/pipe_045.avi") == "045"
    assert infer_suffix_from_path("/tmp/pipe_R.mp4") == "R"


def test_named_eight_dir_loads_u_as_000_and_ignores_rl(tmp_path: Path):
    folder = tmp_path / "videos"
    for suf in ("U", "R", "L"):
        _touch(folder / f"test01_hp250_{suf}.mp4")
    found = resolve_named_run_videos(folder, "test01_hp250")
    assert set(found) == {"000"}
    assert found["000"].name == "test01_hp250_U.mp4"


def test_named_eight_dir_prefers_000_over_u(tmp_path: Path):
    folder = tmp_path / "videos"
    _touch(folder / "sess_U.mp4")
    _touch(folder / "sess_000.mp4")
    found = resolve_named_run_videos(folder, "sess")
    assert found["000"].name == "sess_000.mp4"


def test_named_eight_dir_picks_all_ccc(tmp_path: Path):
    folder = tmp_path / "videos"
    for rid in ("000", "045", "090", "135", "180", "225", "270", "315"):
        _touch(folder / f"pipe_{rid}.mp4")
    found = resolve_named_run_videos(folder, "pipe")
    assert set(found) == {
        "000", "045", "090", "135", "180", "225", "270", "315"
    }


def test_named_u_only_is_allowed(tmp_path: Path):
    folder = tmp_path / "videos"
    _touch(folder / "xxx_U.mp4")
    found = resolve_named_run_videos(folder, "xxx")
    assert set(found) == {"000"}


def test_named_missing_all_raises(tmp_path: Path):
    folder = tmp_path / "videos"
    folder.mkdir()
    _touch(folder / "xxx_R.mp4")
    with pytest.raises(FileNotFoundError):
        resolve_named_run_videos(folder, "xxx")


def test_three_dir_named_still_loads_rl_when_n_is_3(tmp_path: Path):
    folder = tmp_path / "videos"
    for suf in ("U", "R", "L"):
        _touch(folder / f"pipe_{suf}.mp4")
    found = resolve_named_run_videos(
        folder, "pipe", n_directions=3, optical_axis_spacing_deg=120.0
    )
    assert found["000"].name == "pipe_U.mp4"
    assert found["120"].name == "pipe_R.mp4"
    assert found["240"].name == "pipe_L.mp4"


def test_apply_sideview_named_does_not_put_rl_on_runs(tmp_path: Path, config):
    folder = tmp_path / "videos"
    for suf in ("U", "R", "L"):
        _touch(folder / f"test02_hp250_{suf}.mp4")
    config.two_direction.run_C.video_path = "stale_L.mp4"
    apply_sideview_named_input(config, str(folder), "test02_hp250")
    ids = [r.run_id for r in config.two_direction.runs]
    assert ids == ["000"]
    tags = [t for t, _ in config.two_direction.active_run_slots()]
    assert tags == ["000"]
    assert config.two_direction.run_C.video_path == ""
    assert config.two_direction.run_B.video_path == ""
    assert Path(config.two_direction.run_A.video_path).name == "test02_hp250_U.mp4"
    assert config.two_direction.run_A.run_id == "000"


def test_product_cli_sideview_name_one_run(tmp_path: Path, config):
    folder = tmp_path / "videos"
    _touch(folder / "test01_hp250_U.mp4")
    _touch(folder / "test01_hp250_R.mp4")
    args = Namespace(
        pipeline="sideview",
        input=None,
        input_dir=str(folder),
        name="test01_hp250",
        output=None,
        start=None,
        end=None,
        debug=False,
        aov=None,
        pi=None,
        ppm=None,
        use_average_speed=False,
        z_source=None,
        save_ocr_metadata=False,
    )
    apply_product_cli(config, args)
    assert [r.run_id for r in config.two_direction.runs] == ["000"]


def test_runs_json_is_source_of_truth():
    cfg = build_two_direction_config({
        "n_directions": 8,
        "fov_deg": 60,
        "optical_axis_spacing_deg": 45,
        "runs": [
            {"run_id": "000", "physical_roll_deg": 0.0, "video_path": "a.mp4"},
            {"run_id": "045", "physical_roll_deg": 45.0, "video_path": "b.mp4"},
        ],
        "run_A": {"run_id": "U", "physical_roll_deg": 0.0, "video_path": "ignore.mp4"},
    })
    tags = [t for t, r in cfg.active_run_slots()]
    assert tags == ["000", "045"]
    assert cfg.n_directions == 8
    assert cfg.fov_deg == pytest.approx(60.0)


def test_legacy_abc_active_slots_unchanged_when_runs_empty():
    cfg = Config.from_defaults().two_direction
    assert cfg.runs == []
    tags = [t for t, _ in cfg.active_run_slots()]
    assert tags == ["A", "B"]


def test_progress_plan_one_and_eight_runs():
    one = build_direction_progress_plan(["000"])
    assert one.n_runs == 1
    names = [s["name_ja"] for s in one.steps]
    assert names == [
        "初期化",
        "フレーム解析（1/1）",
        "部分図生成（1/1）",
        "接合・補正",
    ]
    assert one.analyze_id("000") == 2
    assert one.analyze_id("U") == 2
    eight = build_direction_progress_plan(
        ["000", "045", "090", "135", "180", "225", "270", "315"]
    )
    assert eight.n_runs == 8
    assert eight.steps[-1]["name_ja"] == "接合・補正"
    assert eight.analyze_id("315") == 9
    assert eight.accumulate_id("000") == 10
    assert abs(sum(eight.weights.values()) - 100.0) < 1e-6


def test_one_run_config_is_active():
    td = Config.from_defaults().two_direction
    td.runs = [
        RunConfig(run_id="000", physical_roll_deg=0.0, video_path="u.mp4")
    ]
    jobs = td.active_run_slots()
    assert len(jobs) == 1
    assert jobs[0][0] == "000"
