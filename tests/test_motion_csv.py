"""フレームごとの移動量 CSV 出力"""

import csv

import numpy as np

from src.validation.records import FrameAnalysisRecord
from src.validation.report import FRAME_MOTION_CSV_COLUMNS, write_frame_motion_csv


def _record(**kwargs) -> FrameAnalysisRecord:
    defaults = dict(
        run_id="U",
        frame_num=0,
        timestamp=0.0,
        position=np.array([0.0, 0.0, 10.0]),
        orientation=np.radians([0.0, 180.0, -30.0]),
        reference=np.zeros(5),
        z_ocr=0.0,
        motion_raw=np.zeros(6),
        motion_final=np.zeros(6),
        match_count=0,
        residual_stats={},
        prior_norm=0.0,
        bound_hit={},
        status="INIT",
        mode="A",
    )
    defaults.update(kwargs)
    return FrameAnalysisRecord(**defaults)


def test_write_frame_motion_csv_has_header_and_per_frame_rows(tmp_path):
    records = [
        _record(),
        _record(
            frame_num=1,
            timestamp=1.0 / 30.0,
            position=np.array([0.0, 0.0, 14.5]),
            orientation=np.radians([0.0, 180.1, -29.5]),
            z_ocr=10.0,
            motion_raw=np.array([0.1, -0.2, 4.6, np.radians(0.3), np.radians(0.1), np.radians(0.5)]),
            motion_final=np.array([0.0, 0.0, 4.5, 0.0, np.radians(0.1), np.radians(0.5)]),
            match_count=42,
            residual_stats={"residual_rms": 0.75},
            prior_norm=1.25,
            bound_hit={"x": False, "y": False, "roll": False, "yaw": False, "pitch": True},
            status="OK",
        ),
        _record(
            frame_num=2,
            timestamp=2.0 / 30.0,
            z_ocr=None,
            status="FAILED",
            residual_stats={"error": "no matches"},
        ),
    ]
    path = write_frame_motion_csv(tmp_path / "motion_U.csv", records)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert path.read_text(encoding="utf-8-sig").splitlines()[0] == ",".join(
        FRAME_MOTION_CSV_COLUMNS
    )
    assert len(rows) == 3
    assert rows[0]["frame_num"] == "0"
    assert rows[0]["status"] == "INIT"
    assert rows[0]["z_mm"] == "10.000000"
    assert rows[0]["z_ocr_mm"] == "0.000000"
    assert rows[1]["frame_num"] == "1"
    assert rows[1]["run_id"] == "U"
    assert rows[1]["match_count"] == "42"
    assert rows[1]["dx_final_mm"] == "0.000000"
    assert rows[1]["dz_final_mm"] == "4.500000"
    assert abs(float(rows[1]["dpitch_final_deg"]) - 0.5) < 1e-6
    assert rows[1]["bound_hit"] == "pitch"
    assert rows[1]["residual_rms"] == "0.750000"
    assert rows[2]["z_ocr_mm"] == ""
    assert rows[2]["status"] == "FAILED"


def test_write_frame_motion_csv_empty_still_writes_header(tmp_path):
    path = write_frame_motion_csv(tmp_path / "motion_R.csv", [])
    text = path.read_text(encoding="utf-8-sig")
    assert text.strip() == ",".join(FRAME_MOTION_CSV_COLUMNS)
