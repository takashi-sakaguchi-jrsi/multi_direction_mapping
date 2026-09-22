"""第1段階: 既知zを OCR 読み取り結果の代用とする"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.camera_estimation import CameraEstimator, compute_distance_constraints_only
from src.config import Config, load_config
from src.main_two_direction_validation import build_transformer, run_two_direction_validation
from src.validation.frame_analyzer import FrameAnalyzer
from src.validation.geometry import build_run_reference
from src.validation.known_z import (
    apply_generation_metadata_to_config,
    infer_known_z_metadata_path,
    load_known_z_mm,
    resolve_known_z_mm,
)
from tests.synthetic_pipe import make_sequence

VIDEO_U = Path("data/input/videos/phi250_fisheye_side_U.mp4")
VIDEO_R = Path("data/input/videos/phi250_fisheye_side_R.mp4")
META = Path("data/input/videos/phi250_fisheye_side_metadata.json")


def test_known_z_skips_tesseract_and_fills_ocr_dist(transformer, config):
    config.estimation.use_offset_moving_average = False
    estimator = CameraEstimator(config.estimation, transformer)
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)]
    known = np.array([10.0, 15.0, 20.0, 25.0, 30.0])

    with patch("src.ocr_utils.extract_distance_from_frame") as ocr:
        constraints, z_pos, ocr_dist, success = compute_distance_constraints_only(
            frames=frames,
            estimator=estimator,
            config=config.estimation,
            known_z_mm=known,
        )
        ocr.assert_not_called()

    np.testing.assert_allclose(ocr_dist, known)
    assert np.all(success)
    assert z_pos is not None
    np.testing.assert_allclose(z_pos, known - known[0], atol=1e-5)
    dz_min, dz_max = constraints[2]["dz"]
    assert dz_min <= 5.0 <= dz_max


def test_known_z_length_mismatch_raises(transformer, config):
    estimator = CameraEstimator(config.estimation, transformer)
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
    with pytest.raises(ValueError, match="known_z_mm"):
        compute_distance_constraints_only(
            frames=frames,
            estimator=estimator,
            config=config.estimation,
            known_z_mm=np.array([1.0, 2.0]),
        )


def test_analyzer_records_known_z_as_ocr(transformer, config):
    config.estimation.use_offset_moving_average = False
    estimator = CameraEstimator(config.estimation, transformer)
    analyzer = FrameAnalyzer(config, estimator, mode="A")
    frames = [np.zeros((48, 48, 3), dtype=np.uint8) for _ in range(4)]
    known = np.array([100.0, 108.0, 116.0, 124.0])
    recs, extra = analyzer.analyze_run(
        config.two_direction.run_A, frames=frames, known_z_mm=known
    )
    assert extra["z_source"] == "known"
    np.testing.assert_allclose(extra["ocr_dist"], known)
    assert recs[0].z_ocr == pytest.approx(100.0)
    assert recs[-1].z_ocr == pytest.approx(124.0)


def test_metadata_loader_and_infer(tmp_path):
    video = tmp_path / "demo_fisheye_side_U.mp4"
    video.write_bytes(b"")
    meta = tmp_path / "demo_fisheye_side_metadata.json"
    meta.write_text(
        json.dumps({"z_values_mm": [2.0, 4.0, 6.0], "f_px": 100.0}),
        encoding="utf-8",
    )
    inferred = infer_known_z_metadata_path(str(video))
    assert inferred == meta
    z = load_known_z_mm(meta)
    np.testing.assert_allclose(z, [2.0, 4.0, 6.0])


def test_resolve_known_z_from_config_path(tmp_path, config):
    meta = tmp_path / "z.json"
    meta.write_text(json.dumps({"z_values_mm": [1.0, 2.0]}), encoding="utf-8")
    config.two_direction.z_source = "known"
    config.two_direction.known_z_metadata_path = str(meta)
    z = resolve_known_z_mm(config)
    np.testing.assert_allclose(z, [1.0, 2.0])


def test_source_pixels_per_mm_from_height():
    from src.validation.fisheye_sideview_renderer import pixels_per_mm_from_equirect
    from src.validation.known_z import source_pixels_per_mm_from_metadata

    ppm = pixels_per_mm_from_equirect(2000, 125.0)
    assert ppm == pytest.approx(2000.0 / (2.0 * np.pi * 125.0))
    assert source_pixels_per_mm_from_metadata({"source_pixels_per_mm": ppm}) == pytest.approx(ppm)
    assert source_pixels_per_mm_from_metadata(
        {"colormap_height_px": 2000, "radius_mm": 125.0}
    ) == pytest.approx(ppm)


def test_pipeline_reports_configured_ppm(config, transformer, mapper):
    ref_a = build_run_reference(0.0)
    ref_b = build_run_reference(120.0)
    ori_a = np.array([ref_a["roll_ref_rad"], ref_a["yaw_ref_rad"], ref_a["pitch_ref_rad"]])
    ori_b = np.array([ref_b["roll_ref_rad"], ref_b["yaw_ref_rad"], ref_b["pitch_ref_rad"]])
    frames_a, _ = make_sequence(transformer, mapper, ori_a, n=3, z0=10.0, dz=8.0, width=64, height=48)
    frames_b, _ = make_sequence(transformer, mapper, ori_b, n=3, z0=10.0, dz=8.0, width=64, height=48)
    config.two_direction.modes = ["A"]
    config.two_direction.z_source = "ocr"
    config.two_direction.match_source_pixels_per_mm = False
    config.two_direction.output_dir = "data/output/two_direction/_pytest_ppm"
    config.two_direction.capture.image_width_px = 64
    config.two_direction.capture.image_height_px = 48
    config.two_direction.projection.sample_stride = 4
    config.two_direction.registration.min_inliers = 3
    config.colormap.pixels_per_mm = 2.5
    known = np.array([10.0, 18.0, 26.0])
    result = run_two_direction_validation(
        config, frames_a=frames_a, frames_b=frames_b, modes=["A"], known_z_mm=known
    )
    assert result["pixels_per_mm"] == pytest.approx(2.5)


def test_synthetic_pipeline_with_known_z(config, transformer, mapper):
    ref_a = build_run_reference(0.0)
    ref_b = build_run_reference(120.0)
    ori_a = np.array([ref_a["roll_ref_rad"], ref_a["yaw_ref_rad"], ref_a["pitch_ref_rad"]])
    ori_b = np.array([ref_b["roll_ref_rad"], ref_b["yaw_ref_rad"], ref_b["pitch_ref_rad"]])
    frames_a, _ = make_sequence(transformer, mapper, ori_a, n=4, z0=10.0, dz=8.0, width=96, height=72)
    frames_b, _ = make_sequence(transformer, mapper, ori_b, n=4, z0=10.0, dz=8.0, width=96, height=72)
    config.two_direction.modes = ["A"]
    config.two_direction.z_source = "ocr"
    config.two_direction.output_dir = "data/output/two_direction/_pytest_known_z"
    config.two_direction.capture.image_width_px = 96
    config.two_direction.capture.image_height_px = 72
    config.two_direction.projection.sample_stride = 3
    config.two_direction.registration.min_inliers = 3
    config.estimation.use_offset_moving_average = False
    known = np.array([10.0, 18.0, 26.0, 34.0])

    with patch("src.ocr_utils.extract_distance_from_frame") as ocr:
        result = run_two_direction_validation(
            config, frames_a=frames_a, frames_b=frames_b, modes=["A"], known_z_mm=known
        )
        ocr.assert_not_called()

    assert result["z_source"] == "known"
    assert result["modes"]["A"]["run_A"]["n_frames"] == 4
    assert result["report_path"]


@pytest.mark.skipif(not VIDEO_U.is_file() or not VIDEO_R.is_file() or not META.is_file(),
                    reason="phi250 仮想動画または metadata が無い")
def test_generated_videos_first_stage_known_z(tmp_path):
    cfg = Config.from_json("data/config/two_direction_config.json")
    cfg.two_direction.z_source = "known"
    cfg.two_direction.known_z_metadata_path = str(META)
    cfg.two_direction.modes = ["A"]
    cfg.two_direction.max_frames = 6
    cfg.two_direction.output_dir = str(tmp_path / "two_direction")
    cfg.two_direction.projection.sample_stride = 4
    cfg.estimation.use_offset_moving_average = False
    cfg.estimation.use_vanishing_point_constraints = False

    from src.validation.ocr_simulation import ocr_z_from_metadata

    with patch("src.ocr_utils.extract_distance_from_frame") as ocr:
        result = run_two_direction_validation(cfg, modes=["A"])
        ocr.assert_not_called()

    meta = json.loads(META.read_text(encoding="utf-8"))
    ocr_series = ocr_z_from_metadata(meta)
    assert result["z_source"] == "known"
    assert result["modes"]["A"]["run_A"]["n_frames"] == 6
    assert result["modes"]["A"]["run_A"]["z_ocr_first"] == pytest.approx(float(ocr_series[0]))
    assert result["modes"]["A"]["run_A"]["z_ocr_last"] == pytest.approx(float(ocr_series[5]))
    out = Path(cfg.two_direction.output_dir) / "mode_A"
    assert (out / "partial_A.png").is_file()
    assert (out / "partial_B.png").is_file()
    assert (out / "final.png").is_file()


def test_two_direction_config_disables_production_calib():
    cfg = load_config("data/config/two_direction_config.json")
    assert cfg.camera.lens_calibration_file in (None, "")
    assert cfg.camera._lens_calibration is None
    assert cfg.camera.fx == pytest.approx(341.87536947032544)
    assert cfg.camera.fy == pytest.approx(341.87536947032544)
    assert cfg.camera.fov_degrees == pytest.approx(181.0)
    assert cfg.camera.cx == pytest.approx(960.0)
    assert cfg.camera.cy == pytest.approx(540.0)
    assert cfg.camera.center_offset_x == 0
    assert cfg.camera.center_offset_y == 0

    transformer = build_transformer(cfg)
    assert transformer.calibration is None
    assert transformer.camera.f == pytest.approx(341.87536947032544)
    assert transformer.camera.cx == pytest.approx(960.0)
    assert transformer.camera.cy == pytest.approx(540.0)


def test_apply_generation_metadata_clears_calib(config):
    config.camera.lens_calibration_file = "data/calibration/fisheye_recalibrated_20260217.json"
    config.camera._lens_calibration = object()
    config.camera.center_offset_x = 20
    config.camera.center_offset_y = 12
    apply_generation_metadata_to_config(
        config,
        {
            "f_px": 341.87536947032544,
            "fov_deg": 181.0,
            "width": 1920,
            "height": 1080,
        },
    )
    assert config.camera.lens_calibration_file is None
    assert config.camera._lens_calibration is None
    assert config.camera.fx == pytest.approx(341.87536947032544)
    assert config.camera.cx == pytest.approx(960.0)
    assert config.camera.cy == pytest.approx(540.0)
    assert config.camera.center_offset_x == 0
    assert config.camera.center_offset_y == 0

    transformer = build_transformer(config)
    assert transformer.calibration is None
    assert transformer.camera.f == pytest.approx(341.87536947032544)
    assert transformer.camera.cx == pytest.approx(960.0)
    assert transformer.camera.cy == pytest.approx(540.0)
