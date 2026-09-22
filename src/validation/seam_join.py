"""重なり中央ライン（60° / -60° / 180°）でカットして接合する。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.validation.geometry import theta_in_camera_car_sector


def join_at_overlap_centers(
    strips: Sequence[dict],
    pixels_per_mm: float,
    theta_min: float = 0.0,
    theta_max: float = 2.0 * np.pi,
    z_pad_mm: float = 10.0,
) -> Dict[str, np.ndarray]:
    """各 strip を担当 120° 帯だけ残して貼る。

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

    span = theta_max - theta_min
    rows = np.arange(theta_bins, dtype=float)
    eta = theta_min + rows / max(theta_bins - 1, 1) * span

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
    }
