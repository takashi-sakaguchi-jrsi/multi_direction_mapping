"""検証レポート出力"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

FRAME_MOTION_CSV_COLUMNS = [
    "frame_num",
    "run_id",
    "mode",
    "status",
    "timestamp_s",
    "x_mm",
    "y_mm",
    "z_mm",
    "roll_deg",
    "yaw_deg",
    "pitch_deg",
    "z_ocr_mm",
    "dx_raw_mm",
    "dy_raw_mm",
    "dz_raw_mm",
    "droll_raw_deg",
    "dyaw_raw_deg",
    "dpitch_raw_deg",
    "dx_final_mm",
    "dy_final_mm",
    "dz_final_mm",
    "droll_final_deg",
    "dyaw_final_deg",
    "dpitch_final_deg",
    "match_count",
    "residual_rms",
    "prior_norm",
    "bound_hit",
]


def write_validation_report(
    output_dir: Path,
    payload: Dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation_report.json"
    serializable = _jsonify(payload)
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_frame_motion_csv(path: Path, records: List[Any]) -> Path:
    """フレームごとの移動量・姿勢推定をヘッダ付き CSV に保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FRAME_MOTION_CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(_record_to_motion_row(rec))
    return path


def _record_to_motion_row(rec: Any) -> Dict[str, Any]:
    pos = np.asarray(getattr(rec, "position", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
    ori = np.asarray(getattr(rec, "orientation", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
    raw = np.asarray(getattr(rec, "motion_raw", np.zeros(6)), dtype=float).reshape(-1)
    final = np.asarray(getattr(rec, "motion_final", np.zeros(6)), dtype=float).reshape(-1)
    pos = np.pad(pos, (0, max(0, 3 - pos.size)))
    ori = np.pad(ori, (0, max(0, 3 - ori.size)))
    raw = np.pad(raw, (0, max(0, 6 - raw.size)))
    final = np.pad(final, (0, max(0, 6 - final.size)))
    residual = getattr(rec, "residual_stats", None) or {}
    z_ocr = getattr(rec, "z_ocr", None)
    return {
        "frame_num": int(getattr(rec, "frame_num", 0)),
        "run_id": str(getattr(rec, "run_id", "")),
        "mode": str(getattr(rec, "mode", "")),
        "status": str(getattr(rec, "status", "")),
        "timestamp_s": _fmt_num(getattr(rec, "timestamp", 0.0)),
        "x_mm": _fmt_num(pos[0]),
        "y_mm": _fmt_num(pos[1]),
        "z_mm": _fmt_num(pos[2]),
        "roll_deg": _fmt_num(np.degrees(ori[0])),
        "yaw_deg": _fmt_num(np.degrees(ori[1])),
        "pitch_deg": _fmt_num(np.degrees(ori[2])),
        "z_ocr_mm": _fmt_num(z_ocr) if z_ocr is not None else "",
        "dx_raw_mm": _fmt_num(raw[0]),
        "dy_raw_mm": _fmt_num(raw[1]),
        "dz_raw_mm": _fmt_num(raw[2]),
        "droll_raw_deg": _fmt_num(np.degrees(raw[3])),
        "dyaw_raw_deg": _fmt_num(np.degrees(raw[4])),
        "dpitch_raw_deg": _fmt_num(np.degrees(raw[5])),
        "dx_final_mm": _fmt_num(final[0]),
        "dy_final_mm": _fmt_num(final[1]),
        "dz_final_mm": _fmt_num(final[2]),
        "droll_final_deg": _fmt_num(np.degrees(final[3])),
        "dyaw_final_deg": _fmt_num(np.degrees(final[4])),
        "dpitch_final_deg": _fmt_num(np.degrees(final[5])),
        "match_count": int(getattr(rec, "match_count", 0)),
        "residual_rms": _fmt_num(residual.get("residual_rms", "")),
        "prior_norm": _fmt_num(getattr(rec, "prior_norm", 0.0)),
        "bound_hit": _bound_hit_label(getattr(rec, "bound_hit", None)),
    }


def _fmt_num(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        if not np.isfinite(float(value)):
            return ""
    except (TypeError, ValueError):
        return ""
    return f"{float(value):.6f}"


def _bound_hit_label(bound_hit: Optional[Dict[str, Any]]) -> str:
    if not bound_hit:
        return ""
    return ",".join(str(k) for k, v in bound_hit.items() if v)


def summarize_records(records: List[Any]) -> Dict[str, Any]:
    if not records:
        return {}
    pos = np.stack([r.position for r in records])
    ori = np.stack([r.orientation for r in records])
    ori_deg = np.degrees(ori)
    z_ocr = [r.z_ocr for r in records if r.z_ocr is not None]
    return {
        "n_frames": len(records),
        "match_count_mean": float(np.mean([r.match_count for r in records])),
        "status_counts": _count([r.status for r in records]),
        "position_std_mm": pos.std(axis=0).tolist(),
        "position_max_abs_mm": np.max(np.abs(pos), axis=0).tolist(),
        "orientation_std_deg": ori_deg.std(axis=0).tolist(),
        "orientation_max_abs_deg": np.max(np.abs(ori_deg), axis=0).tolist(),
        "prior_norm_mean": float(np.mean([r.prior_norm for r in records])),
        "bound_hit_frames": int(sum(1 for r in records if any(r.bound_hit.values()))),
        "z_ocr_first": float(z_ocr[0]) if z_ocr else None,
        "z_ocr_last": float(z_ocr[-1]) if z_ocr else None,
    }


def _count(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def _jsonify(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj
