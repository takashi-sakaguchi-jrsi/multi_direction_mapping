"""合成フレームによる V1-V6 の通し確認"""

from unittest.mock import patch

import numpy as np

from src.camera_estimation import CameraEstimator
from src.main_two_direction_validation import run_two_direction_validation
from src.validation.geometry import build_run_reference
from tests.synthetic_pipe import make_sequence


def test_synthetic_two_run_pipeline(config, transformer, mapper):
    ref_a = build_run_reference(0.0)
    ref_b = build_run_reference(120.0)
    ori_a = np.array([ref_a["roll_ref_rad"], ref_a["yaw_ref_rad"], ref_a["pitch_ref_rad"]])
    ori_b = np.array([ref_b["roll_ref_rad"], ref_b["yaw_ref_rad"], ref_b["pitch_ref_rad"]])
    frames_a, _ = make_sequence(transformer, mapper, ori_a, n=4, width=96, height=72)
    frames_b, _ = make_sequence(transformer, mapper, ori_b, n=4, width=96, height=72)
    config.two_direction.modes = ["A"]
    config.two_direction.output_dir = "data/output/two_direction/_pytest"
    config.two_direction.capture.image_width_px = 96
    config.two_direction.capture.image_height_px = 72
    config.two_direction.projection.sample_stride = 3
    config.two_direction.registration.min_inliers = 3

    with patch("src.ocr_utils.extract_distance_from_frame", return_value=(100.0, 80)):
        result = run_two_direction_validation(config, frames_a=frames_a, frames_b=frames_b, modes=["A"])

    assert "A" in result["modes"]
    assert result["modes"]["A"]["run_U"]["n_frames"] == 4
    assert result["report_path"]
    from pathlib import Path
    csv_a = Path(result["modes"]["A"]["motion_csv"]["U"])
    csv_b = Path(result["modes"]["A"]["motion_csv"]["R"])
    assert csv_a.is_file()
    assert csv_b.is_file()
    header = csv_a.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header.startswith("frame_num,run_id,mode,status")
    assert csv_a.read_text(encoding="utf-8-sig").count("\n") >= 5


def test_synthetic_three_run_pipeline(config, transformer, mapper):
    ref_a = build_run_reference(0.0)
    ref_b = build_run_reference(120.0)
    ref_c = build_run_reference(240.0)
    ori_a = np.array([ref_a["roll_ref_rad"], ref_a["yaw_ref_rad"], ref_a["pitch_ref_rad"]])
    ori_b = np.array([ref_b["roll_ref_rad"], ref_b["yaw_ref_rad"], ref_b["pitch_ref_rad"]])
    ori_c = np.array([ref_c["roll_ref_rad"], ref_c["yaw_ref_rad"], ref_c["pitch_ref_rad"]])
    frames_a, _ = make_sequence(transformer, mapper, ori_a, n=4, width=96, height=72)
    frames_b, _ = make_sequence(transformer, mapper, ori_b, n=4, width=96, height=72)
    frames_c, _ = make_sequence(transformer, mapper, ori_c, n=4, width=96, height=72)
    config.two_direction.modes = ["A"]
    config.two_direction.output_dir = "data/output/three_direction/_pytest"
    config.two_direction.capture.image_width_px = 96
    config.two_direction.capture.image_height_px = 72
    config.two_direction.projection.sample_stride = 3
    config.two_direction.registration.min_inliers = 3
    config.two_direction.run_C.run_id = "L"
    config.two_direction.run_C.physical_roll_deg = 240.0

    with patch("src.ocr_utils.extract_distance_from_frame", return_value=(100.0, 80)):
        result = run_two_direction_validation(
            config,
            frames_a=frames_a,
            frames_b=frames_b,
            frames_c=frames_c,
            modes=["A"],
        )

    mode = result["modes"]["A"]
    assert result["n_runs"] == 3
    assert mode["n_runs"] == 3
    assert set(mode["run_ids"]) == {"U", "R", "L"}
    assert mode["run_U"]["n_frames"] == 4
    assert mode["run_R"]["n_frames"] == 4
    assert mode["run_L"]["n_frames"] == 4
    assert "L" in mode["motion_csv"]
    from pathlib import Path
    assert Path(mode["motion_csv"]["L"]).is_file()
    assert "U" in mode["coverage"]["run_adoption"]
    assert "R" in mode["coverage"]["run_adoption"]
    assert "L" in mode["coverage"]["run_adoption"]

