"""生成動画用の微小姿勢振動。

十数フレーム周期の緩やかで中程度の成分と、2〜3 フレームの短く小さい成分を合成する。
dz は平均 z_step のまわり、yaw/pitch は基準姿勢のまわり（積算ドリフトしない）。
OCR は真の累積 z に 10mm 遅れモデルを掛ける。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from src.validation.ocr_simulation import OCR_STEP_MM, ocr_metadata_fields


@dataclass
class PoseJitterConfig:
    slow_period_frames: float = 15.0
    fast_period_frames: float = 2.1
    dz_slow_amp_mm: float = 0.45
    dz_fast_amp_mm: float = 0.22
    dtheta_slow_amp_deg: float = 1.30
    dtheta_fast_amp_deg: float = 0.20
    yaw_slow_period_frames: float = 35.0
    pitch_slow_period_frames: float = 26.0
    yaw_fast_period_frames: float = 4.7
    pitch_fast_period_frames: float = 2.2
    dz_min_mm: float = 1.6
    seed: int = 0


def _two_tone(
    n: int,
    t: np.ndarray,
    amp_slow: float,
    period_slow: float,
    amp_fast: float,
    period_fast: float,
    phase_slow: float,
    phase_fast: float,
) -> np.ndarray:
    slow = amp_slow * np.sin(2.0 * np.pi * t / period_slow + phase_slow)
    fast = amp_fast * np.sin(2.0 * np.pi * t / period_fast + phase_fast)
    return slow + fast


def build_motion_series(
    n_frames: int,
    z_start_mm: float,
    z_step_mm: float,
    z_max_mm: float,
    config: PoseJitterConfig = None,
) -> Dict[str, np.ndarray]:
    """フレーム 0 を z_start、以降は振動付き dz を積算する。"""
    cfg = config or PoseJitterConfig()
    n = max(1, int(n_frames))
    t = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(int(cfg.seed))
    # 位相だけ乱数。振幅・周期は固定して再現性を保つ。
    phases = rng.uniform(0.0, 2.0 * np.pi, size=6)

    dz_off = _two_tone(
        n, t, cfg.dz_slow_amp_mm, cfg.slow_period_frames,
        cfg.dz_fast_amp_mm, cfg.fast_period_frames, phases[0], phases[1],
    )
    dz = np.full(n, float(z_step_mm), dtype=np.float64) + dz_off
    dz[0] = 0.0
    dz[1:] = np.maximum(dz[1:], float(cfg.dz_min_mm))
    z = float(z_start_mm) + np.cumsum(dz)
    keep = z <= float(z_max_mm) + 1e-9
    if not np.any(keep):
        keep[0] = True
    z = z[keep]
    dz = dz[keep]
    n = int(z.size)
    t = np.arange(n, dtype=np.float64)

    yaw_off = _two_tone(
        n, t, cfg.dtheta_slow_amp_deg, cfg.yaw_slow_period_frames,
        cfg.dtheta_fast_amp_deg, cfg.yaw_fast_period_frames, phases[2], phases[3],
    )
    pitch_off = _two_tone(
        n, t, cfg.dtheta_slow_amp_deg, cfg.pitch_slow_period_frames,
        cfg.dtheta_fast_amp_deg, cfg.pitch_fast_period_frames, phases[4], phases[5],
    )
    dyaw = np.zeros(n, dtype=np.float64)
    dpitch = np.zeros(n, dtype=np.float64)
    dyaw[1:] = np.diff(yaw_off)
    dpitch[1:] = np.diff(pitch_off)
    return {
        "z_mm": z,
        "dz_mm": dz,
        "yaw_offset_deg": yaw_off,
        "pitch_offset_deg": pitch_off,
        "dyaw_deg": dyaw,
        "dpitch_deg": dpitch,
    }


def motion_metadata_fields(
    series: Dict[str, np.ndarray],
    z_step_mm: float,
    config: PoseJitterConfig,
    ocr_step_mm: float = OCR_STEP_MM,
) -> dict:
    z = series["z_mm"]
    meta = ocr_metadata_fields(z, z_step_mm=float(z_step_mm), ocr_step_mm=ocr_step_mm)
    meta["ocr_note"] = (
        f"{ocr_step_mm:g}mm 刻み。真の累積 z に対しラインを超えたフレームでカウントアップ。"
        f"走行は公称 {z_step_mm:g}mm/frame に緩急振動を合成。"
    )
    meta["motion_jitter"] = True
    meta["motion_jitter_config"] = asdict(config)
    meta["dz_mm"] = [float(v) for v in series["dz_mm"]]
    meta["dyaw_deg"] = [float(v) for v in series["dyaw_deg"]]
    meta["dpitch_deg"] = [float(v) for v in series["dpitch_deg"]]
    meta["yaw_offset_deg"] = [float(v) for v in series["yaw_offset_deg"]]
    meta["pitch_offset_deg"] = [float(v) for v in series["pitch_offset_deg"]]
    meta["dz_mean_mm"] = float(np.mean(series["dz_mm"][1:])) if series["dz_mm"].size > 1 else 0.0
    return meta
