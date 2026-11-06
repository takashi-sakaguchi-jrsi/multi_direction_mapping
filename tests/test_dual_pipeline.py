"""正面 / 側視の入口分岐・CLI・既知z包含・OCR metadata・削除フラグ。"""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src.main_twopass import resolve_initial_attitude_rad
from src.product_runtime import (
    PipelineCliError,
    apply_front_profile,
    apply_product_cli,
    apply_sideview_named_input,
    apply_sideview_profile,
    cleanup_sideview_intermediates,
    effective_pipeline,
    resolve_named_front_video,
    resolve_named_run_videos,
    save_ocr_metadata_file,
    validate_pipeline_cli_args,
)
from src.validation.known_z import assert_known_z_covers_window, effective_z_source


def _args(**kwargs):
    base = dict(
        pipeline=None,
        input=None,
        input_dir=None,
        name=None,
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
    base.update(kwargs)
    return Namespace(**base)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_named_prefix_picks_xxx_not_yyy(tmp_path: Path):
    folder = tmp_path / "input"
    for prefix in ("xxx", "yyy"):
        for suf in ("U", "R", "L"):
            _touch(folder / f"{prefix}_{suf}.mp4")
    found = resolve_named_run_videos(folder, "xxx")
    assert set(found) == {"000"}
    assert found["000"].name == "xxx_U.mp4"


def test_named_prefix_first_extension_wins(tmp_path: Path):
    folder = tmp_path / "input"
    _touch(folder / "xxx_U.mp4")
    _touch(folder / "xxx_U.avi")
    _touch(folder / "xxx_R.avi")
    found = resolve_named_run_videos(folder, "xxx")
    assert found["000"].suffix == ".mp4"
    assert set(found) == {"000"}


def test_named_prefix_u_only_is_enough(tmp_path: Path):
    folder = tmp_path / "input"
    _touch(folder / "xxx_U.mp4")
    found = resolve_named_run_videos(folder, "xxx")
    assert found["000"].name == "xxx_U.mp4"


def test_sideview_named_input_clears_missing_l(tmp_path: Path):
    folder = tmp_path / "input"
    _touch(folder / "xxx_U.mp4")
    _touch(folder / "xxx_R.mp4")
    config = Config()
    config.two_direction.run_C.video_path = "stale_L.mp4"
    apply_sideview_named_input(config, str(folder), "xxx")
    assert Path(config.two_direction.run_A.video_path).name == "xxx_U.mp4"
    assert config.two_direction.run_A.run_id == "000"
    assert config.two_direction.run_C.video_path == ""
    assert [r.run_id for r in config.two_direction.runs] == ["000"]


def test_cli_sideview_ignores_json_front_ocr_z_source():
    """JSON が pipeline=front / z_source=OCR でも --pipeline sideview なら known。"""
    cfg = Config.from_dict({
        "pipeline": "front",
        "z_source": "OCR",
        "known_z_metadata_path": "",
        "two_direction": {
            "z_source": "known",
            "known_z_metadata_path": None,
        },
    })
    assert cfg.pipeline == "front"
    apply_product_cli(cfg, _args(pipeline="sideview"))
    assert effective_pipeline(cfg) == "sideview"
    assert cfg.two_direction.z_source == "known"
    assert effective_z_source(cfg) == "known"


def test_front_input_sets_video_path(tmp_path: Path):
    video = tmp_path / "front.mp4"
    _touch(video)
    config = Config()
    config.pipeline = "sideview"
    apply_product_cli(config, _args(pipeline="front", input=video))
    assert effective_pipeline(config) == "front"
    assert config.input.video_path == str(video)


def test_front_named_input_picks_prefix(tmp_path: Path):
    folder = tmp_path / "input"
    _touch(folder / "xxx.mp4")
    _touch(folder / "yyy.mp4")
    _touch(folder / "xxx.avi")
    found = resolve_named_front_video(folder, "xxx")
    assert found.name == "xxx.mp4"
    config = Config()
    apply_product_cli(
        config, _args(pipeline="front", input_dir=folder, name="xxx")
    )
    assert config.input.video_path.endswith("xxx.mp4")


def test_front_named_input_missing_raises(tmp_path: Path):
    folder = tmp_path / "input"
    folder.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_named_front_video(folder, "xxx")


def test_cli_rejects_cross_pipeline_options(tmp_path: Path):
    with pytest.raises(PipelineCliError):
        validate_pipeline_cli_args(
            "front", _args(input=tmp_path / "a.mp4", input_dir=tmp_path, name="xxx")
        )
    with pytest.raises(PipelineCliError):
        validate_pipeline_cli_args("sideview", _args(input=tmp_path / "a.mp4"))
    with pytest.raises(PipelineCliError):
        validate_pipeline_cli_args("sideview", _args(input_dir=tmp_path))
    with pytest.raises(PipelineCliError):
        validate_pipeline_cli_args("front", _args(input_dir=tmp_path))


def test_front_config_parses_initialize_attitude_from_vp():
    cfg = Config.from_dict({"front": {"initialize_attitude_from_vp": False}})
    assert cfg.front.initialize_attitude_from_vp is False
    assert Config().front.initialize_attitude_from_vp is True


def test_initial_attitude_uses_config_when_vp_init_disabled():
    cfg = Config.from_dict({
        "front": {"initialize_attitude_from_vp": False},
        "initial_position": {"theta": 0.0, "phi": 0.0},
    })
    yaw, pitch = resolve_initial_attitude_rad(cfg, [(900.0, 400.0)], camera=None)
    assert yaw == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)


def test_initial_attitude_vp_failure_falls_back_to_config():
    cfg = Config.from_dict({
        "front": {"initialize_attitude_from_vp": True},
        "initial_position": {"theta": 1.5, "phi": -0.5},
    })
    yaw, pitch = resolve_initial_attitude_rad(cfg, [], camera=None)
    assert yaw == pytest.approx(np.radians(1.5))
    assert pitch == pytest.approx(np.radians(-0.5))


def test_front_config_parses_physical_attitude_source():
    cfg = Config.from_dict({"front": {"physical_attitude_source": "dz_only"}})
    assert cfg.front.physical_attitude_source == "dz_only"
    apply_front_profile(cfg)
    assert cfg.front.physical_attitude_source == "dz_only"
    assert (
        cfg.estimation.feature_matching.cylindrical.physical_attitude_source
        == "dz_only"
    )
    cfg2 = Config.from_dict({"front": {"physical_attitude_source": "discard"}})
    apply_front_profile(cfg2)
    assert cfg2.front.physical_attitude_source == "discard"
    assert Config().front.physical_attitude_source == "match"


def test_apply_front_profile_rejects_unknown_attitude_source():
    cfg = Config.from_dict({"front": {"physical_attitude_source": "nope"}})
    with pytest.raises(ValueError, match="physical_attitude_source"):
        apply_front_profile(cfg)


def test_pipeline_profiles_split_camera_and_ocr_increment():
    config = Config.from_dict({
        "camera": {"fov_degrees": 185.0, "fx": 300.0},
        "ocr": {"max_distance_increment_mm": 10.0, "countup_mean_dz_max_mm": 10.0},
        "front": {
            "max_distance_increment_mm": 10.0,
            "countup_mean_dz_max_mm": 10.0,
            "camera": {"fov_degrees": 185.0, "fx": 341.0},
        },
        "two_direction": {
            "max_distance_increment_mm": 60.0,
            "countup_mean_dz_max_mm": 12.0,
            "camera": {"fov_degrees": 60.0, "fx": 900.0, "camera_model": "pinhole"},
        },
    })
    apply_front_profile(config)
    assert config.ocr.max_distance_increment_mm == pytest.approx(10.0)
    assert config.ocr.countup_mean_dz_max_mm == pytest.approx(10.0)
    assert config.camera.fov_degrees == pytest.approx(185.0)
    assert config.camera.fx == pytest.approx(341.0)

    side = Config.from_dict({
        "camera": {"fov_degrees": 185.0, "fx": 300.0},
        "ocr": {"max_distance_increment_mm": 10.0, "countup_mean_dz_max_mm": 10.0},
        "front": {
            "max_distance_increment_mm": 10.0,
            "countup_mean_dz_max_mm": 10.0,
            "camera": {"fov_degrees": 185.0, "fx": 341.0},
        },
        "two_direction": {
            "max_distance_increment_mm": 60.0,
            "countup_mean_dz_max_mm": 12.0,
            "camera": {"fov_degrees": 60.0, "fx": 900.0, "camera_model": "pinhole"},
        },
    })
    apply_sideview_profile(side)
    assert side.ocr.max_distance_increment_mm == pytest.approx(60.0)
    assert side.ocr.countup_mean_dz_max_mm == pytest.approx(12.0)
    assert side.camera.fov_degrees == pytest.approx(60.0)
    assert side.camera.fx == pytest.approx(900.0)
    assert side.camera.camera_model == "pinhole"
    assert side.camera.ocr_roi_ratio == pytest.approx([0.0556, 0.1000, 0.0438, 0.1063])
    assert side.camera.lens_calibration_file in (None, "")
    assert side.front.max_distance_increment_mm == pytest.approx(10.0)


def test_sideview_pinhole_overlay_does_not_keep_fisheye_calib():
    config = Config.from_dict({
        "camera": {
            "fov_degrees": 185.0,
            "camera_model": "fisheye",
            "apply_lens_calibration": False,
            "fx": 343.9,
            "fy": 344.2,
            "cx": 980.3,
            "cy": 552.7,
        },
        "two_direction": {
            "camera": {
                "fov_degrees": 60.0,
                "camera_model": "pinhole",
                "ocr_roi_ratio": [0.0556, 0.1000, 0.0438, 0.1063],
            }
        },
    })
    apply_sideview_profile(config)
    assert config.camera.camera_model == "pinhole"
    assert config.camera.fov_degrees == pytest.approx(60.0)
    assert config.camera.lens_calibration_file in (None, "")
    assert config.camera.fx == pytest.approx(0.0)
    assert config.camera.ocr_roi_ratio == pytest.approx(
        [0.0556, 0.1000, 0.0438, 0.1063]
    )


def test_sideview_profile_keeps_two_direction_ocr_roi():
    config = Config.from_dict({
        "camera": {"ocr_roi_ratio": [0.0, 0.12, 0.0, 0.32]},
        "two_direction": {"max_distance_increment_mm": 60.0},
    })
    apply_sideview_profile(config)
    assert config.camera.ocr_roi_ratio == pytest.approx(
        [0.0556, 0.1000, 0.0438, 0.1063]
    )


def test_apply_front_profile_restores_vp_bounds():
    config = Config()
    config.estimation.use_vanishing_point_constraints = False
    config.estimation.motion.max_yaw = 0.0
    config.estimation.motion.use_vanishing_point_yaw = False
    apply_front_profile(config)
    assert config.estimation.use_vanishing_point_constraints is True
    assert config.estimation.motion.max_yaw == pytest.approx(15.0)
    assert config.estimation.motion.use_vanishing_point_yaw is True
    assert config.estimation.feature_matching.cylindrical.z_axis == "physical"
    assert config.estimation.motion.estimate_droll is False
    assert config.estimation.feature_matching.cylindrical.physical_attitude_source == "match"


def test_sideview_alias_json_merges_into_two_direction():
    cfg = Config.from_dict({
        "pipeline": "sideview",
        "sideview": {
            "run_A": {"video_path": "data/input/xxx_U.mp4", "run_id": "U"},
        },
    })
    assert cfg.two_direction.run_A.video_path.endswith("xxx_U.mp4")
    assert cfg.sideview is cfg.two_direction


def test_known_z_containment_ok_and_fail():
    z = np.arange(10, dtype=float)
    assert_known_z_covers_window(z, 0, 10)
    assert_known_z_covers_window(z, 2, 8)
    with pytest.raises(ValueError, match="開始フレーム"):
        assert_known_z_covers_window(z, 10, None)
    with pytest.raises(ValueError, match="終了フレーム"):
        assert_known_z_covers_window(z, 0, 11)


def test_save_ocr_metadata_roundtrip(tmp_path: Path):
    path = tmp_path / "ocr_metadata.json"
    z = np.array([10.0, 14.5, 19.0])
    saved = save_ocr_metadata_file(path, z, extra={"pipeline": "front"})
    import json
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert data["n"] == 3
    assert data["z_ocr_mm"] == [10.0, 14.5, 19.0]
    assert data["pipeline"] == "front"


def test_delete_temporary_images_flag(tmp_path: Path):
    work = tmp_path / "out"
    mode = work / "mode_A"
    mode.mkdir(parents=True)
    partial = mode / "partial_U.png"
    final = mode / "final.png"
    temp = work / "colormap_temp.png"
    excel = tmp_path / "reports" / "report.xlsx"
    excel.parent.mkdir()
    excel.write_text("keep")
    progress = tmp_path / "progress.json"
    progress.write_text("{}")
    _touch(partial)
    _touch(final)
    _touch(temp)
    dest = work / "colormap.png"
    dest.write_bytes(b"copied")
    config = Config()
    config.two_direction.output_dir = str(work)
    config.two_direction.modes = ["A"]
    config.output.colormap_path = str(dest)
    config.output.temporary_colormap_path = str(temp)
    config.colormap.delete_temporary_images = True
    removed = cleanup_sideview_intermediates(config, ["A"])
    assert not partial.exists()
    assert not temp.exists()
    assert not final.exists()
    assert dest.exists()
    assert excel.exists()
    assert progress.exists()
    assert any(p.name == "partial_U.png" for p in removed)


def test_delete_temporary_images_keeps_when_false(tmp_path: Path):
    work = tmp_path / "out"
    mode = work / "mode_A"
    mode.mkdir(parents=True)
    partial = mode / "partial_U.png"
    _touch(partial)
    config = Config()
    config.two_direction.output_dir = str(work)
    config.two_direction.modes = ["A"]
    config.output.temporary_colormap_path = str(work / "colormap_temp.png")
    config.colormap.delete_temporary_images = False
    cleanup_sideview_intermediates(config, ["A"])
    assert partial.exists()
