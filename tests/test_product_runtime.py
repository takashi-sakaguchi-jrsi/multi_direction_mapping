"""製品版 CLI / Excel / 進捗ステップの単体テスト"""

from argparse import Namespace
from pathlib import Path

import numpy as np

from src.config import Config, load_config
from src.product_runtime import (
    THREE_DIRECTION_STEPS,
    THREE_DIRECTION_STEP_WEIGHTS,
    THREE_DIRECTION_TIMING_SEC_1000,
    apply_input_path,
    apply_output_path,
    apply_product_cli,
    build_direction_progress_plan,
    derive_progress_weights,
    product_end_to_exclusive,
    product_start_to_zero_based,
    write_three_direction_excel,
)
from src.validation.records import FrameAnalysisRecord


def test_product_frame_index_conversion():
    assert product_start_to_zero_based(1) == 0
    assert product_start_to_zero_based(100) == 99
    assert product_end_to_exclusive(500) == 500
    assert product_end_to_exclusive(None) is None


def test_apply_product_cli_start_end(config):
    args = Namespace(
        input=None, output=None, start=10, end=50, debug=False,
        aov=None, pi=300.0, ppm=2.0, use_average_speed=False,
    )
    apply_product_cli(config, args)
    assert config.two_direction.start_frame == 9
    assert config.two_direction.end_frame == 50
    assert config.pipe.diameter_mm == 300.0
    assert config.colormap.pixels_per_mm == 2.0


def test_apply_input_directory(tmp_path, config):
    (tmp_path / "pipe_U.mp4").write_bytes(b"")
    (tmp_path / "pipe_R.mp4").write_bytes(b"")
    (tmp_path / "pipe_L.mp4").write_bytes(b"")
    apply_input_path(config, str(tmp_path))
    assert config.two_direction.run_A.video_path.endswith("pipe_U.mp4")
    assert config.two_direction.run_B.video_path.endswith("pipe_R.mp4")
    assert config.two_direction.run_C.video_path.endswith("pipe_L.mp4")
    assert config.two_direction.run_A.run_id == "U"
    assert config.two_direction.run_A.physical_roll_deg == 0.0
    assert config.two_direction.run_B.physical_roll_deg == 120.0
    assert config.two_direction.run_C.physical_roll_deg == 240.0


def test_apply_input_exact_u_r_l_names(tmp_path, config):
    (tmp_path / "U.mp4").write_bytes(b"")
    (tmp_path / "R.mp4").write_bytes(b"")
    (tmp_path / "L.mp4").write_bytes(b"")
    apply_input_path(config, str(tmp_path))
    assert config.two_direction.run_A.video_path.endswith("U.mp4")
    assert config.two_direction.run_B.video_path.endswith("R.mp4")
    assert config.two_direction.run_C.video_path.endswith("L.mp4")


def test_create_progress_reporter_four_steps(tmp_path, config):
    from src.product_runtime import create_progress_reporter

    config.output.progress_path = str(tmp_path / "progress.json")
    config.two_direction.run_A.video_path = ""
    config.two_direction.run_B.video_path = ""
    config.two_direction.run_C.video_path = ""
    reporter = create_progress_reporter(config, "20260101_120000")
    assert reporter is not None
    assert reporter.process_progress is not None
    assert reporter.process_progress.process_id == "20260101_120000"
    names = [s.name_ja for s in reporter.process_progress.steps]
    assert names[0] == "初期化"
    assert names[-1] == "接合・補正"
    assert any("フレーム解析（" in n for n in names)
    assert any("部分図生成（" in n for n in names)


def test_apply_output_dir_and_png(tmp_path, config):
    apply_output_path(config, str(tmp_path / "out"))
    assert config.two_direction.output_dir.endswith("out")
    assert config.output.colormap_path.endswith("colormap.png")
    apply_output_path(config, str(tmp_path / "map.png"))
    assert config.output.colormap_path.endswith("map.png")


def test_default_config_merges_three_direction():
    cfg = load_config()
    assert cfg.two_direction.enabled is True
    assert cfg.two_direction.z_source == "ocr"
    assert cfg.two_direction.output_dir == "data/output"
    assert cfg.output.progress_path == "progress/progress.json"
    assert cfg.logging.file == "Log/process.log"
    assert "{timestamp}" in cfg.output.report_path
    assert cfg.two_direction.run_A.run_id == "U"
    assert cfg.two_direction.run_C.run_id == "L"
    assert len(THREE_DIRECTION_STEPS) == 4


def test_write_three_direction_excel(tmp_path):
    rec = FrameAnalysisRecord(
        run_id="U",
        frame_num=3,
        timestamp=0.1,
        position=np.array([0.0, 0.0, 12.0]),
        orientation=np.zeros(3),
        reference=np.zeros(5),
        z_ocr=10.0,
        motion_raw=np.zeros(6),
        motion_final=np.zeros(6),
        match_count=8,
        residual_stats={},
        prior_norm=0.0,
        bound_hit={},
        status="OK",
        mode="A",
    )
    path = tmp_path / "report.xlsx"
    out = write_three_direction_excel(path, {"U": [rec]})
    assert Path(out).is_file()


def test_derive_progress_weights_rounds_to_100():
    weights = derive_progress_weights(
        {"init": 2.0, "analyze": 400.0, "accumulate": 80.0, "seam": 50.0}
    )
    assert weights[1] >= 1.0
    assert sum(weights.values()) == 100.0
    assert weights[2] > weights[3] > weights[1]
    assert set(THREE_DIRECTION_STEP_WEIGHTS) == {1, 2, 3, 4}


def test_progress_weights_match_1000_frame_timing():
    derived = derive_progress_weights(THREE_DIRECTION_TIMING_SEC_1000)
    assert derived == THREE_DIRECTION_STEP_WEIGHTS
    assert THREE_DIRECTION_STEP_WEIGHTS == {1: 1.0, 2: 79.0, 3: 16.0, 4: 4.0}
    assert sum(THREE_DIRECTION_STEP_WEIGHTS.values()) == 100.0


def test_build_direction_progress_plan_splits_by_run():
    plan = build_direction_progress_plan(["U", "R", "L"])
    names = [s["name_ja"] for s in plan.steps]
    assert names == [
        "初期化",
        "フレーム解析（1/3）",
        "フレーム解析（2/3）",
        "フレーム解析（3/3）",
        "部分図生成（1/3）",
        "部分図生成（2/3）",
        "部分図生成（3/3）",
        "接合・補正",
    ]
    assert plan.n_runs == 3
    assert plan.analyze_id("R") == 3
    assert plan.accumulate_id("L") == 7
    assert plan.seam_step == 8
    assert abs(sum(plan.weights.values()) - 100.0) < 1e-6
    assert abs(sum(plan.weights[i] for i in plan.analyze_step.values()) - 79.0) < 1e-6
    assert abs(sum(plan.weights[i] for i in plan.accumulate_step.values()) - 16.0) < 1e-6


def test_estimate_remaining_seconds_skips_early_and_scales():
    from src.progress_reporter import estimate_remaining_seconds

    assert estimate_remaining_seconds(1.0, 5.0, 1.0) is None
    assert estimate_remaining_seconds(20.0, 1.0, 1.0) is None
    remaining = estimate_remaining_seconds(20.0, 21.0, 1.0)
    assert remaining is not None
    assert 70.0 < remaining < 90.0
    assert estimate_remaining_seconds(10.0, 100.0, 1.0) == 0.0


def test_progress_json_includes_eta_fields(tmp_path, config):
    import json
    import time

    from src.product_runtime import create_progress_reporter

    config.output.progress_path = str(tmp_path / "progress.json")
    config.two_direction.run_A.video_path = ""
    config.two_direction.run_B.video_path = ""
    config.two_direction.run_C.video_path = ""
    reporter = create_progress_reporter(config, "20260101_120000")
    assert reporter is not None
    reporter.skip_step(1, reason="auto_tune_enabled=false")
    payload = json.loads(Path(reporter.output_path).read_text(encoding="utf-8"))
    assert payload["start_time"]
    assert payload["estimated_remaining_seconds"] is None
    assert payload["estimated_end_time"] is None

    analyze_id = reporter.direction_plan.analyze_id("U")
    reporter.start_step(analyze_id)
    reporter.start_time = time.perf_counter() - 20.0
    reporter.update_step(
        analyze_id, 30.0, details={"run_id": "U", "phase": "motion"}
    )
    payload = json.loads(Path(reporter.output_path).read_text(encoding="utf-8"))
    assert payload["elapsed_time_seconds"] >= 19.0
    assert payload["estimated_remaining_seconds"] is not None
    assert payload["estimated_remaining_seconds"] > 0
    assert payload["estimated_end_time"]
