"""製品版（3方向合成）向けの CLI・進捗・Excel 共通化。"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json

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


class PipelineCliError(ValueError):
    """--pipeline と入力オプションの組み合わせが不正。"""


def effective_pipeline(config: Config) -> str:
    return str(getattr(config, "pipeline", "sideview") or "sideview").lower()


def apply_front_profile(config: Config) -> None:
    """正面経路の衝突項目を FrontConfig で上書きする。側視の既定値は触らない。"""
    front = getattr(config, "front", None)
    if front is None:
        return
    config.estimation.use_vanishing_point_constraints = bool(
        front.use_vanishing_point_constraints
    )
    config.estimation.use_ocr_z_constraints = bool(front.use_ocr_z_constraints)
    motion = config.estimation.motion
    motion.max_yaw = float(front.max_yaw)
    motion.max_pitch = float(front.max_pitch)
    motion.use_vanishing_point_yaw = bool(front.use_vanishing_point_yaw)
    motion.use_vanishing_point_pitch = bool(front.use_vanishing_point_pitch)
    if front.auto_tune_enabled is not None:
        config.auto_tune_enabled = bool(front.auto_tune_enabled)
    if bool(getattr(front, "physical_z_matching", True)):
        fm = config.estimation.feature_matching
        fm.use_cylindrical_matching = True
        cyl = fm.cylindrical
        if cyl is None:
            from src.feature_matching_cylindrical import CylindricalMatchingConfig
            cyl = CylindricalMatchingConfig()
            fm.cylindrical = cyl
        cyl.z_axis = "physical"
        cyl.spatial_filter_enabled = False
        cyl.physical_vector_filter_enabled = True
        cyl.physical_max_dyaw_deg = float(
            getattr(front, "physical_max_dyaw_deg", 2.0)
        )
        cyl.physical_max_dpitch_deg = float(
            getattr(front, "physical_max_dpitch_deg", 2.0)
        )
        cyl.physical_max_applied_attitude_deg = float(
            getattr(front, "physical_max_applied_attitude_deg", 1.2)
        )
        cyl.physical_untrusted_attitude_deg = float(
            getattr(front, "physical_untrusted_attitude_deg", 0.8)
        )
        from src.feature_matching_cylindrical import (
            normalize_physical_attitude_source,
        )
        source = normalize_physical_attitude_source(
            getattr(front, "physical_attitude_source", "match")
        )
        front.physical_attitude_source = source
        cyl.physical_attitude_source = source
        filt = config.estimation.feature_filtering
        filt.magnitude_filter_enabled = False
        filt.direction_filter_enabled = False
        filt.vp_circle_filter_enabled = False
        motion.estimate_dx = False
        motion.estimate_droll = False
        motion.hybrid_two_stage_enabled = False
        # 展開は前フレーム姿勢。マッチは dz（と mode により Δyaw/Δpitch）。画素 LS は使わない。
        # 残差 (η, z) 微調整は FrontConfig.residual_paste_warp（既定 true）。姿勢には積まない。
        config.colormap.keep_painted_pixels = bool(
            getattr(front, "keep_painted_colormap", False)
        )
        config.colormap.prefer_peripheral_pixels = bool(
            getattr(front, "prefer_peripheral_colormap", True)
        )
        config.colormap.reject_z_behind_camera = bool(
            getattr(front, "reject_z_behind_camera", True)
        )


def validate_pipeline_cli_args(pipeline: str, args: Any) -> None:
    pipeline = str(pipeline or "sideview").lower()
    has_input = bool(getattr(args, "input", None))
    has_dir = bool(getattr(args, "input_dir", None))
    has_name = getattr(args, "name", None) not in (None, "")
    if has_dir != has_name:
        raise PipelineCliError("--input-dir と --name は同時に指定してください")
    if pipeline == "front":
        if has_input and (has_dir or has_name):
            raise PipelineCliError(
                "正面経路では --input と --input-dir/--name を同時に指定できません"
            )
        return
    if pipeline == "sideview":
        if has_input:
            raise PipelineCliError(
                "側視経路では --input は使えません（--input-dir と --name を使ってください）"
            )
        return
    raise PipelineCliError(f"pipeline は front/sideview のみ: {pipeline}")


def apply_front_input_path(config: Config, raw: str) -> None:
    """--input は正面専用。ファイル1本を input.video_path へ。"""
    path = Path(raw)
    if path.exists() and path.is_dir():
        raise PipelineCliError("--input は正面用の動画ファイルです。フォルダは指定できません")
    config.input.video_path = str(path)


def resolve_named_front_video(input_dir: Path, name: str) -> Path:
    """{dir}/{name}.{ext}。拡張子は .mp4 → .avi → .mov → .mkv の先頭ヒット。"""
    folder = Path(input_dir)
    for ext in _VIDEO_SUFFIXES:
        candidate = folder / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"正面動画が見つかりません: {folder}/{name}.{{mp4,avi,mov,mkv}}"
    )


def apply_front_named_input(config: Config, input_dir: str, name: str) -> None:
    config.input.video_path = str(resolve_named_front_video(Path(input_dir), str(name)))


def resolve_named_run_videos(input_dir: Path, name: str) -> Dict[str, Path]:
    """{dir}/{name}_U/R/L.{ext}。拡張子は .mp4 → .avi → .mov → .mkv の先頭ヒット。"""
    folder = Path(input_dir)
    found: Dict[str, Path] = {}
    mapping = (("U", "A"), ("R", "B"), ("L", "C"))
    for suf, tag in mapping:
        for ext in _VIDEO_SUFFIXES:
            candidate = folder / f"{name}_{suf}{ext}"
            if candidate.is_file():
                found[tag] = candidate
                break
    if "A" not in found or "B" not in found:
        raise FileNotFoundError(
            f"側視動画が見つかりません: {folder}/{name}_U|_R （L は任意）"
        )
    return found


def apply_sideview_named_input(config: Config, input_dir: str, name: str) -> None:
    found = resolve_named_run_videos(Path(input_dir), str(name))
    config.two_direction.run_C.video_path = ""
    for tag, video in found.items():
        _set_run_video(config, tag, video)
    if "C" not in found:
        config.two_direction.run_C.video_path = ""


def apply_pipeline_cli(config: Config, args: Any) -> None:
    """--pipeline と入力オプションを Config に載せる。"""
    if getattr(args, "pipeline", None):
        config.pipeline = str(args.pipeline).lower()
    pipeline = effective_pipeline(config)
    validate_pipeline_cli_args(pipeline, args)
    if pipeline == "front":
        apply_front_profile(config)
        if getattr(args, "input", None):
            apply_front_input_path(config, str(args.input))
        elif getattr(args, "input_dir", None) and getattr(args, "name", None):
            apply_front_named_input(config, str(args.input_dir), str(args.name))
    elif pipeline == "sideview":
        if getattr(args, "input_dir", None) and getattr(args, "name", None):
            apply_sideview_named_input(
                config, str(args.input_dir), str(args.name)
            )


def apply_product_cli(config: Config, args: Any) -> None:
    """製品 CLI（--pipeline/--input/--input-dir/--name/--output/--start/--end）を Config に載せる。"""
    apply_pipeline_cli(config, args)
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
        config.debug.show_feature_points = True
        config.debug.output_debug_images = True
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
    if getattr(args, "z_source", None):
        config.z_source = str(args.z_source)
        config.two_direction.z_source = config.z_source
    if getattr(args, "save_ocr_metadata", False):
        config.save_ocr_metadata = True


def apply_input_path(config: Config, raw: str) -> None:
    """後方互換。正面はファイル1本。側視は --input-dir/--name を使う。"""
    apply_front_input_path(config, raw)


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


FRONT_PROGRESS_STEPS = [
    {"step_id": 1, "name_ja": "パラメータ調整", "name_en": "Parameter Tuning"},
    {"step_id": 2, "name_ja": "フレーム解析", "name_en": "Frame Analysis"},
    {"step_id": 3, "name_ja": "カラーマップ生成", "name_en": "Colormap Generation"},
    {"step_id": 4, "name_ja": "カラーマップ補正", "name_en": "Colormap Correction"},
]


def sideview_work_dir(config: Config) -> Path:
    """側視の中間出力は temporary_colormap_path の親。"""
    return Path(config.output.temporary_colormap_path).expanduser().parent


def apply_sideview_work_dir(config: Config) -> Path:
    work = sideview_work_dir(config)
    work.mkdir(parents=True, exist_ok=True)
    config.two_direction.output_dir = str(work)
    return work


_SIDE_INTERMEDIATE_GLOBS = (
    "partial_*.png",
    "partial_*_corrected.png",
    "*_provenance.npz",
    "motion_*.csv",
    "*.npz",
)


def cleanup_sideview_intermediates(config: Config, modes: Optional[Sequence[str]] = None) -> List[Path]:
    """成功後に部分図・provenance・一時 PNG を消す。Excel / progress / OCR metadata は残す。"""
    if not config.colormap.delete_temporary_images:
        return []
    removed: List[Path] = []
    work = Path(config.two_direction.output_dir)
    modes = list(modes) if modes is not None else list(config.two_direction.modes or ["A"])
    for mode in modes:
        mode_dir = work / f"mode_{mode}"
        if not mode_dir.is_dir():
            continue
        for pattern in _SIDE_INTERMEDIATE_GLOBS:
            for path in mode_dir.glob(pattern):
                if path.name == "final.png":
                    continue
                try:
                    path.unlink()
                    removed.append(path)
                except OSError as exc:
                    logger.warning(f"中間ファイルを削除できません: {path}: {exc}")
        final_src = mode_dir / "final.png"
        dest = Path(str(config.output.colormap_path))
        if final_src.is_file() and dest.is_file():
            try:
                if final_src.resolve() != dest.resolve():
                    final_src.unlink()
                    removed.append(final_src)
            except OSError as exc:
                logger.warning(f"中間 final.png を削除できません: {final_src}: {exc}")
    temp = Path(config.output.temporary_colormap_path)
    if temp.is_file():
        try:
            temp.unlink()
            removed.append(temp)
        except OSError as exc:
            logger.warning(f"一時カラーマップを削除できません: {temp}: {exc}")
    return removed


def ocr_metadata_output_path(config: Config, timestamp: str) -> Path:
    dest = Path(config.output.colormap_path).parent / f"ocr_metadata_{timestamp}.json"
    return dest


def save_ocr_metadata_file(
    path: Path,
    z_mm: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z_list = np.asarray(z_mm, dtype=float).reshape(-1).tolist()
    payload: Dict[str, Any] = {
        "z_ocr_mm": z_list,
        "ocr_z_mm": z_list,
        "n": int(len(z_list)),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"OCR metadata を保存: {path}")
    return path


def create_front_progress_reporter(config: Config, timestamp: str) -> Optional[ProgressReporter]:
    if not config.output.progress_path:
        return None
    video_path = str(config.input.video_path)
    n = count_video_frames(video_path) if video_path else 0
    start = int(config.input.start_frame or 1)
    end = config.input.end_frame if config.input.end_frame is not None else n
    total = max(1, int(end) - int(start) + 1) if n else 1
    path = Path(str(config.output.progress_path).replace("{process_id}", timestamp))
    path.parent.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(path, total_frames=total)
    reporter.initialize_steps(list(FRONT_PROGRESS_STEPS))
    if reporter.process_progress is not None:
        reporter.process_progress.process_id = timestamp
        reporter._write_progress_5steps()
    return reporter

