"""既描画との重ね部で新しい帯の (η, z) を合わせる。

マッチ残差ではなく、すでに塗った展開図と新帯の重なりで並進を測る。
姿勢には積まない。正面の貼り付け専用。
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _luma(img: np.ndarray) -> np.ndarray:
    rgb = np.asarray(img)
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    return rgb[..., :3].astype(np.float32).mean(axis=2)


def _wrap_half(shift: float, size: int) -> float:
    if size <= 0:
        return float(shift)
    half = 0.5 * float(size)
    s = float(shift)
    while s > half:
        s -= float(size)
    while s < -half:
        s += float(size)
    return s


def _overlap_windows(
    painted: np.ndarray,
    new_colors: np.ndarray,
    mask: np.ndarray,
    start_col: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """painted と new の重なり輝度。無効は 0。"""
    if painted is None or new_colors is None or mask is None:
        return None
    h, w = new_colors.shape[:2]
    ph, pw = painted.shape[:2]
    if h != ph or w < 2 or h < 8:
        return None
    dest_c0 = max(int(start_col), 0)
    src_c0 = dest_c0 - int(start_col)
    dest_c1 = min(int(start_col) + w, pw)
    src_c1 = src_c0 + (dest_c1 - dest_c0)
    if dest_c1 - dest_c0 < 4 or src_c1 <= src_c0:
        return None
    dest = painted[:, dest_c0:dest_c1]
    src = new_colors[:, src_c0:src_c1]
    src_mask = np.asarray(mask[:, src_c0:src_c1], dtype=bool)
    painted_ok = np.any(dest[..., :3] != 0, axis=-1) if dest.ndim == 3 else dest != 0
    overlap = src_mask & painted_ok
    if int(np.sum(overlap)) < 16:
        return None
    old = np.where(overlap, _luma(dest), 0.0).astype(np.float32)
    new = np.where(overlap, _luma(src), 0.0).astype(np.float32)
    return old, new, overlap


def estimate_overlap_shift_px(
    painted: np.ndarray,
    new_colors: np.ndarray,
    mask: np.ndarray,
    start_col: int,
    *,
    min_pixels: int = 1500,
    min_response: float = 0.04,
    max_drow: float = 80.0,
    max_dcol: float = 45.0,
) -> Optional[Tuple[float, float, float]]:
    """新帯が既描画に対して何 px ずれているか (d_row, d_col, response)。

    OpenCV の符号のまま返す（new = roll(painted, d_row, axis=0) なら d_row>0）。
    貼り直しでは同じ量をサンプリング η, z に足し、模様を既描画へ戻す。
    """
    win = _overlap_windows(painted, new_colors, mask, start_col)
    if win is None:
        return None
    old, new, overlap = win
    n = int(np.sum(overlap))
    if n < int(min_pixels):
        return None
    mean_old = float(old[overlap].mean())
    mean_new = float(new[overlap].mean())
    old_z = np.where(overlap, old - mean_old, 0.0).astype(np.float32)
    new_z = np.where(overlap, new - mean_new, 0.0).astype(np.float32)
    hann = cv2.createHanningWindow((old_z.shape[1], old_z.shape[0]), cv2.CV_32F)
    (d_col, d_row), response = cv2.phaseCorrelate(old_z, new_z, hann)
    if not (np.isfinite(d_col) and np.isfinite(d_row) and np.isfinite(response)):
        return None
    d_row = _wrap_half(d_row, old_z.shape[0])
    d_col = float(d_col)
    if float(response) < float(min_response):
        return None
    if abs(d_row) > float(max_drow) or abs(d_col) > float(max_dcol):
        return None
    return float(d_row), d_col, float(response)


def apply_overlap_shift_to_grids(
    eta_grid: np.ndarray,
    z_grid: np.ndarray,
    d_row: float,
    d_col: float,
    eta_height: int,
    pixels_per_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """サンプリング格子を (d_row, d_col) だけずらす。"""
    d_eta = float(d_row) * (2.0 * np.pi / max(int(eta_height), 1))
    d_z = float(d_col) / max(float(pixels_per_mm), 1e-6)
    return eta_grid + d_eta, z_grid + d_z
