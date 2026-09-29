"""製品版（3方向合成）向けの CLI・進捗・Excel 共通化。"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.config import Config
from src.progress_reporter import ProgressReporter
from src.validation.geometry import infer_suffix_from_path, physical_roll_from_suffix

logger = logging.getLogger(__name__)

THREE_DIRECTION_STEPS = [
    {"step_id": 1, "name_ja": "初期化", "name_en": "Initialization"},
    {"step_id": 2, "name_ja": "フレーム解析", "name_en": "Frame Analysis"},
    {"step_id": 3, "name_ja": "部分図生成", "name_en": "Strip Generation"},
    {"step_id": 4, "name_ja": "接合・補正", "name_en": "Seam Join"},
]

# phi250_baseline U/R/L、Mode A、先頭 1000 フレーム、z_source=known（2026-09-29）
THREE_DIRECTION_TIMING_SEC_1000 = {
    "init": 0.029,
    "analyze": 230.036,
    "accumulate": 45.622,
    "seam": 11.739,
}

# 上記実測を整数%（合計100）に丸めた値。初期化は表示用に最低 1%。
# 正面経路の旧配分は 10 / 30 / 55 / 5。
THREE_DIRECTION_STEP_WEIGHTS = {
    1: 1.0,
    2: 79.0,
    3: 16.0,
    4: 4.0,
}

_STEP_TIMING_KEYS = {
    1: "init",
    2: "analyze",
    3: "accumulate",
    4: "seam",
}


def derive_progress_weights(
    timing_sec: Dict[str, float],
    *,
    min_init_pct: float = 1.0,
) -> Dict[int, float]:
    """実測秒から 4 ステップ重み（合計 100）を作る。初期化は最低 min_init_pct%。"""
    seconds = {
        step_id: max(0.0, float(timing_sec.get(key, 0.0)))
        for step_id, key in _STEP_TIMING_KEYS.items()
    }
    total = sum(seconds.values())
    if total <= 0.0:
        return {step_id: float(w) for step_id, w in THREE_DIRECTION_STEP_WEIGHTS.items()}
    raw = {step_id: 100.0 * sec / total for step_id, sec in seconds.items()}
    if raw[1] < min_init_pct:
        deficit = min_init_pct - raw[1]
        donor = max((2, 3, 4), key=lambda i: raw[i])
        raw[donor] = max(1.0, raw[donor] - deficit)
        raw[1] = min_init_pct
        scale = 100.0 / sum(raw.values())
        raw = {i: v * scale for i, v in raw.items()}
    floors = {i: int(raw[i]) for i in (1, 2, 3, 4)}
    remainder = 100 - sum(floors.values())
    order = sorted(
        (1, 2, 3, 4),
        key=lambda i: (raw[i] - floors[i], raw[i]),
        reverse=True,
    )
    for step_id in order:
        if remainder <= 0:
            break
        floors[step_id] += 1
        remainder -= 1
    return {step_id: float(floors[step_id]) for step_id in (1, 2, 3, 4)}


@dataclass
class DirectionProgressPlan:
    """方向数 N に応じた進捗ステップ。フレーム解析・部分図を (i/N) に分割する。"""

    steps: List[Dict[str, str]]
    weights: Dict[int, float]
    analyze_step: Dict[str, int]
    accumulate_step: Dict[str, int]
    seam_step: int
    n_runs: int

    def analyze_id(self, run_id: str) -> int:
        if run_id in self.analyze_step:
            return self.analyze_step[run_id]
        return next(iter(self.analyze_step.values()))

    def accumulate_id(self, run_id: str) -> int:
        if run_id in self.accumulate_step:
            return self.accumulate_step[run_id]
        return next(iter(self.accumulate_step.values()))


def _split_weight(total: float, n: int) -> List[float]:
    if n <= 1:
        return [float(total)]
    each = round(float(total) / n, 4)
    parts = [each] * n
    parts[-1] = round(float(total) - each * (n - 1), 4)
    return parts


def build_direction_progress_plan(run_ids: Sequence[str]) -> DirectionProgressPlan:
    ids = [str(rid) for rid in run_ids if str(rid)]
    if not ids:
        ids = ["U"]
    n = len(ids)
    steps: List[Dict[str, str]] = [
        {"step_id": 1, "name_ja": "初期化", "name_en": "Initialization"},
    ]
    analyze_step: Dict[str, int] = {}
    accumulate_step: Dict[str, int] = {}
    weights: Dict[int, float] = {1: float(THREE_DIRECTION_STEP_WEIGHTS[1])}
    sid = 2
    an_parts = _split_weight(float(THREE_DIRECTION_STEP_WEIGHTS[2]), n)
    for i, rid in enumerate(ids):
        steps.append({
            "step_id": sid,
            "name_ja": f"フレーム解析（{i + 1}/{n}）",
            "name_en": f"Frame Analysis ({i + 1}/{n})",
        })
        analyze_step[rid] = sid
        weights[sid] = an_parts[i]
        sid += 1
    acc_parts = _split_weight(float(THREE_DIRECTION_STEP_WEIGHTS[3]), n)
    for i, rid in enumerate(ids):
        steps.append({
            "step_id": sid,
            "name_ja": f"部分図生成（{i + 1}/{n}）",
            "name_en": f"Strip Generation ({i + 1}/{n})",
        })
        accumulate_step[rid] = sid
        weights[sid] = acc_parts[i]
        sid += 1
    steps.append({
        "step_id": sid,
        "name_ja": "接合・補正",
        "name_en": "Seam Join",
    })
    weights[sid] = float(THREE_DIRECTION_STEP_WEIGHTS[4])
    return DirectionProgressPlan(
        steps=steps,
        weights=weights,
        analyze_step=analyze_step,
        accumulate_step=accumulate_step,
        seam_step=sid,
        n_runs=n,
    )


def progress_run_ids(config: Config) -> List[str]:
    ids: List[str] = []
    fallback = {"A": "U", "B": "R", "C": "L"}
    for tag, run_cfg in config.two_direction.active_run_slots():
        ids.append(run_cfg.run_id or fallback.get(tag, tag))
    return ids

_RUN_TAGS = ("A", "B", "C")
_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


def ensure_product_cwd() -> Path:
    """PyInstaller EXE では exe のあるフォルダを cwd にする（相対パス data/... のため）。"""
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
        os.chdir(root)
        return root
    return Path.cwd()


def product_start_to_zero_based(start: Optional[int]) -> Optional[int]:
    """製品 CLI の開始フレーム（1始まり含む）を two_direction の 0 始まりに変換する。"""
    if start is None:
        return None
    if int(start) <= 0:
        return 0
    return int(start) - 1


def product_end_to_exclusive(end: Optional[int]) -> Optional[int]:
    """製品 CLI の終了フレーム（1始まり含む）を [start, end) の end にする。"""
    if end is None:
        return None
    return int(end)


def apply_product_cli(config: Config, args: Any) -> None:
    """製品 CLI（--input/--output/--start/--end/--debug/--pi/--ppm/--aov）を Config に載せる。"""
    if getattr(args, "input", None):
        apply_input_path(config, str(args.input))
    if getattr(args, "output", None):
        apply_output_path(config, str(args.output))
    if getattr(args, "start", None) is not None:
        start0 = product_start_to_zero_based(int(args.start))
        config.input.start_frame = int(args.start)
        config.two_direction.start_frame = int(start0 or 0)
    if getattr(args, "end", None) is not None:
        config.input.end_frame = args.end
        config.two_direction.end_frame = product_end_to_exclusive(args.end)
    if getattr(args, "debug", False):
        config.debug.enabled = True
        config.logging.level = "DEBUG"
    if getattr(args, "aov", None) is not None:
        config.camera.fov_degrees = float(args.aov)
    if getattr(args, "pi", None) is not None:
        config.pipe.diameter_mm = float(args.pi)
    if getattr(args, "ppm", None) is not None:
        config.colormap.pixels_per_mm = float(args.ppm)
        config.two_direction.match_source_pixels_per_mm = False
    if getattr(args, "use_average_speed", False):
        config.estimation.use_offset_moving_average = False


def apply_input_path(config: Config, raw: str) -> None:
    """--input がファイルなら対応 run、ディレクトリなら U/R/L を割り当てる。"""
    path = Path(raw)
    config.input.video_path = str(path)
    if path.is_dir():
        found = _find_run_videos_in_dir(path)
        for tag, video in found.items():
            _set_run_video(config, tag, video)
        return
    suffix = infer_suffix_from_path(str(path))
    tag = {"U": "A", "R": "B", "L": "C"}.get(suffix or "", "A")
    _set_run_video(config, tag, path)
    if path.parent.is_dir():
        siblings = _find_run_videos_in_dir(path.parent)
        for other_tag, video in siblings.items():
            if other_tag == tag:
                continue
            _set_run_video(config, other_tag, video)


def apply_output_path(config: Config, raw: str) -> None:
    path = Path(raw)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        config.output.colormap_path = str(path)
        config.two_direction.output_dir = str(path.parent)
    else:
        config.two_direction.output_dir = str(path)
        config.output.colormap_path = str(path / "colormap.png")


def _set_run_video(config: Config, tag: str, video: Path) -> None:
    run = getattr(config.two_direction, f"run_{tag}")
    run.video_path = str(video)
    suffix = infer_suffix_from_path(str(video))
    if suffix:
        run.run_id = suffix
        run.physical_roll_deg = physical_roll_from_suffix(suffix)


def _find_run_videos_in_dir(folder: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    mapping = {"U": "A", "R": "B", "L": "C"}
    for suf, tag in mapping.items():
        for ext in _VIDEO_SUFFIXES:
            exact = folder / f"{suf}{ext}"
            if exact.is_file():
                found[tag] = exact
                break
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
    ]
    by_suffix: Dict[str, Path] = {}
    for p in files:
        suf = infer_suffix_from_path(str(p))
        if suf in mapping and suf not in by_suffix:
            by_suffix[suf] = p
    for suf, tag in mapping.items():
        if tag not in found and suf in by_suffix:
            found[tag] = by_suffix[suf]
    return found


def count_video_frames(video_path: str) -> int:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def estimate_total_frames(config: Config) -> int:
    td = config.two_direction
    jobs = td.active_run_slots()
    total = 0
    for _tag, run_cfg in jobs:
        if not run_cfg.video_path:
            continue
        n = count_video_frames(str(run_cfg.video_path))
        start = int(td.start_frame or 0)
        end = td.end_frame if td.end_frame is not None else n
        end = min(int(end), n) if n else int(end or 0)
        if td.max_frames is not None:
            end = min(end, start + int(td.max_frames))
        total += max(0, end - start)
    return max(total, 1)


def create_progress_reporter(config: Config, timestamp: str) -> Optional[ProgressReporter]:
    if not config.output.progress_path:
        return None
    path = Path(
        str(config.output.progress_path).replace("{process_id}", timestamp)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(path, total_frames=estimate_total_frames(config))
    plan = build_direction_progress_plan(progress_run_ids(config))
    reporter.step_weights = dict(plan.weights)
    reporter.initialize_steps(list(plan.steps))
    reporter.direction_plan = plan
    if reporter.process_progress is not None:
        reporter.process_progress.process_id = timestamp
        reporter._write_progress_5steps()
    return reporter


def copy_final_colormap(config: Config, mode: str = "A", timestamp: str = "") -> Optional[Path]:
    src = Path(config.two_direction.output_dir) / f"mode_{mode}" / "final.png"
    dest = Path(str(config.output.colormap_path).replace("{timestamp}", timestamp or "final"))
    if not src.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
        logger.info(f"最終カラーマップをコピー: {dest}")
    return dest


def write_three_direction_excel(
    report_path: Path,
    records_by_run: Dict[str, Sequence[Any]],
) -> Path:
    import pandas as pd

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for run_id, records in records_by_run.items():
        rows = [_record_to_excel_row(run_id, rec) for rec in records]
        frames.append((run_id, pd.DataFrame(rows)))
    if not frames:
        raise ValueError("Excel に書くフレームデータがありません")
    try:
        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            for run_id, df in frames:
                df.to_excel(writer, sheet_name=str(run_id)[:31], index=False)
    except ImportError:
        csv_path = report_path.with_suffix(".csv")
        pd.concat(
            [df.assign(run_id=rid) for rid, df in frames],
            ignore_index=True,
        ).to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Excel 保存できないため CSV: {csv_path}")
        return csv_path
    logger.info(f"3方向レポート: {report_path}")
    return report_path


def _record_to_excel_row(run_id: str, rec: Any) -> Dict[str, Any]:
    pos = np.asarray(getattr(rec, "position", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
    ori = np.asarray(getattr(rec, "orientation", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
    pos = np.pad(pos, (0, max(0, 3 - pos.size)))
    ori = np.pad(ori, (0, max(0, 3 - ori.size)))
    z_ocr = getattr(rec, "z_ocr", None)
    return {
        "方向": run_id,
        "フレーム番号": int(getattr(rec, "frame_num", 0)),
        "X (mm)": float(pos[0]),
        "Y (mm)": float(pos[1]),
        "Z (mm)": float(pos[2]),
        "Roll (度)": float(np.degrees(ori[0])),
        "Yaw (度)": float(np.degrees(ori[1])),
        "Pitch (度)": float(np.degrees(ori[2])),
        "Z_OCR (mm)": float(z_ocr) if z_ocr is not None else None,
        "状態": str(getattr(rec, "status", "")),
        "マッチ数": int(getattr(rec, "match_count", 0) or 0),
    }


def copy_validation_report_to_reports(src: Path, timestamp: str) -> Optional[Path]:
    if not src.is_file():
        return None
    dest = Path("reports") / f"validation_report_{timestamp}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest
