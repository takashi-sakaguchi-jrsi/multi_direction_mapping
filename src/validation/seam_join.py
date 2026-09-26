"""重なり中央ライン（60° / -60° / 180°）で担当帯を分け、最終展開は θ=0（真上）で切る。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.validation.geometry import eta_to_camera_car_deg, theta_in_camera_car_sector

DEFAULT_UNWRAP_CUT_CAMERA_CAR_DEG = 0.0
"""最終展開図の周期切断。カメラカー 0° = 真上。接合面（60/−60/180）ではない。"""


def unwrap_eta_for_rows(
    n_rows: int,
    theta_min: float = 0.0,
    theta_max: float = 2.0 * np.pi,
) -> np.ndarray:
    """接合キャンバスの行 → プログラム η（roll 前）。

    部分図格子と同じ ``endpoint=False``。最後の行を 2π にすると η=0 と同一角になり、
    周期ラップに1画素の重複／欠落が出る。
    """
    n = max(int(n_rows), 1)
    rows = np.arange(n, dtype=float)
    span = float(theta_max - theta_min)
    return theta_min + rows / n * span


def close_one_pixel_theta_gaps(
    color: np.ndarray,
    filled: np.ndarray,
    source: np.ndarray,
    max_gap: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """θ 周期の短い隙間を隣から埋める。

    接合ワープが周方向を非周期リマップすると、旧切断（η=0／180°）に
    数画素の黒帯が残る。1画素穴だけでなく max_gap 行までの挟まれた穴を閉じる。
    """
    work_c = color.copy()
    work_f = filled.copy()
    work_s = source.copy()
    max_gap = max(int(max_gap), 1)
    for _ in range(max_gap):
        prev_f = np.roll(work_f, 1, axis=0)
        next_f = np.roll(work_f, -1, axis=0)
        acc_p = work_f.copy()
        acc_n = work_f.copy()
        for k in range(max_gap):
            acc_p |= np.roll(work_f, k + 1, axis=0)
            acc_n |= np.roll(work_f, -(k + 1), axis=0)
        hole = (~work_f) & (prev_f | next_f) & acc_p & acc_n
        if not np.any(hole):
            break
        take_prev = hole & prev_f
        take_next = hole & ~take_prev & next_f
        work_c[take_prev] = np.roll(work_c, 1, axis=0)[take_prev]
        work_s[take_prev] = np.roll(work_s, 1, axis=0)[take_prev]
        work_c[take_next] = np.roll(work_c, -1, axis=0)[take_next]
        work_s[take_next] = np.roll(work_s, -1, axis=0)[take_next]
        work_f[hole] = True
    return work_c, work_f, work_s


def unwrap_cut_row(
    n_rows: int,
    cut_camera_car_deg: float = DEFAULT_UNWRAP_CUT_CAMERA_CAR_DEG,
    theta_min: float = 0.0,
    theta_max: float = 2.0 * np.pi,
) -> int:
    """roll 前キャンバスで、切断角に最も近い行。"""
    eta = unwrap_eta_for_rows(n_rows, theta_min, theta_max)
    car = eta_to_camera_car_deg(eta)
    delta = (car - float(cut_camera_car_deg) + 180.0) % 360.0 - 180.0
    return int(np.argmin(np.abs(delta)))


def unwrap_row_camera_car_deg(
    n_rows: int,
    cut_camera_car_deg: float = DEFAULT_UNWRAP_CUT_CAMERA_CAR_DEG,
    theta_min: float = 0.0,
    theta_max: float = 2.0 * np.pi,
) -> np.ndarray:
    """最終展開の各行のカメラカー角[deg]。row0 が切断角。"""
    eta = unwrap_eta_for_rows(n_rows, theta_min, theta_max)
    car = eta_to_camera_car_deg(eta)
    shift = unwrap_cut_row(n_rows, cut_camera_car_deg, theta_min, theta_max)
    return np.roll(car, -shift)


def roll_arrays_to_unwrap_cut(
    arrays: Sequence[np.ndarray],
    cut_camera_car_deg: float = DEFAULT_UNWRAP_CUT_CAMERA_CAR_DEG,
    theta_min: float = 0.0,
    theta_max: float = 2.0 * np.pi,
) -> Tuple[List[np.ndarray], int]:
    """行方向に周期シフトし、上端を切断角にする。"""
    if not arrays:
        return [], 0
    h = int(arrays[0].shape[0])
    shift = unwrap_cut_row(h, cut_camera_car_deg, theta_min, theta_max)
    return [np.roll(a, -shift, axis=0) for a in arrays], shift


def join_at_overlap_centers(
    strips: Sequence[dict],
    pixels_per_mm: float,
    theta_min: float = 0.0,
    theta_max: float = 2.0 * np.pi,
    z_pad_mm: float = 10.0,
    cut_camera_car_deg: float = DEFAULT_UNWRAP_CUT_CAMERA_CAR_DEG,
) -> Dict[str, np.ndarray]:
    """各 strip を担当 120° 帯だけ残して貼る。

    接合面はカメラカー 60° / −60° / 180°。最終画像の上下端（周期切断）は
    真上 θ=0（``cut_camera_car_deg``）で、接合面ではない。

    strip は ``rgb, filled, z_min, z_max, run_id`` を持つ。
    """
    if not strips:
        raise ValueError("接合する部分図がありません")
    ppm = max(float(pixels_per_mm), 1e-9)
    z_lo = min(float(s["z_min"]) for s in strips) - z_pad_mm
    z_hi = max(float(s["z_max"]) for s in strips) + z_pad_mm
    theta_bins = max(s["rgb"].shape[0] for s in strips)
    z_bins = max(8, int(np.ceil((z_hi - z_lo) * ppm)))
    color = np.zeros((theta_bins, z_bins, 3), dtype=np.uint8)
    filled = np.zeros((theta_bins, z_bins), dtype=bool)
    source = np.full((theta_bins, z_bins), -1, dtype=np.int16)
    run_ids: List[str] = []

    eta = unwrap_eta_for_rows(theta_bins, theta_min, theta_max)

    for code, strip in enumerate(strips):
        run_id = str(strip["run_id"])
        run_ids.append(run_id)
        rgb = strip["rgb"]
        mask = strip["filled"]
        h, w = rgb.shape[:2]
        z0 = float(strip["z_min"])
        sector = theta_in_camera_car_sector(eta[:h], run_id)
        if h != theta_bins:
            # 行数が違う場合は η をリサンプルせず、先頭 h 行だけ使う
            pass
        for y in np.where(sector)[0]:
            if y >= h:
                continue
            filled_row = mask[y]
            if not np.any(filled_row):
                continue
            xs = np.where(filled_row)[0]
            z = z0 + xs.astype(float) / ppm
            iz = np.round((z - z_lo) * ppm).astype(int)
            keep = (iz >= 0) & (iz < z_bins)
            if not np.any(keep):
                continue
            iz = iz[keep]
            xs_k = xs[keep]
            # 未充填だけ書く。同セルは先に入った run を維持（帯が排他なので稀）
            empty = ~filled[y, iz]
            if not np.any(empty):
                continue
            iz_e = iz[empty]
            xs_e = xs_k[empty]
            color[y, iz_e] = rgb[y, xs_e]
            filled[y, iz_e] = True
            source[y, iz_e] = code

    color, filled, source = close_one_pixel_theta_gaps(color, filled, source)

    rolled, shift = roll_arrays_to_unwrap_cut(
        [color, filled, source],
        cut_camera_car_deg=cut_camera_car_deg,
        theta_min=theta_min,
        theta_max=theta_max,
    )
    color, filled, source = rolled

    rgb_out = color.copy()
    rgb_out[~filled] = 0
    adoption = {}
    n = max(int(np.sum(filled)), 1)
    for code, name in enumerate(run_ids):
        adoption[name] = float(np.sum(source[filled] == code) / n)
    return {
        "rgb": rgb_out,
        "filled": filled,
        "source_run": source,
        "z_min": z_lo,
        "z_max": z_hi,
        "run_ids": np.array(run_ids, dtype=object),
        "adoption": adoption,
        "cut_camera_car_deg": float(cut_camera_car_deg),
        "unwrap_row_shift": int(shift),
    }
