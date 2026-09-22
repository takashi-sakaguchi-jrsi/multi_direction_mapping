"""部分展開図のなだらか補正。

1. OCR 距離に合わせて z 方向を PCHIP 変形
2. 黒境界（エッジライン）の中点が、初期方位から決まるエッジ角中心のまわりに
   残るように θ 方向を変形
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from src.validation.geometry import expected_edge_etas, wrap_angle_rad


@dataclass
class StripCorrectionResult:
    rgb: np.ndarray
    filled: np.ndarray
    z_min: float
    z_max: float
    dtheta_rad: Optional[np.ndarray]
    z_ctrl: Optional[np.ndarray]
    message: str


def correct_strip(
    rgb: np.ndarray,
    filled: np.ndarray,
    z_min: float,
    pixels_per_mm: float,
    z_est_mm: np.ndarray,
    z_ocr_mm: np.ndarray,
    orientation_ref: np.ndarray,
    half_fov_rad: float,
    theta_min: float = 0.0,
    theta_max: float = 2.0 * np.pi,
    control_spacing_mm: float = 50.0,
    warp_theta: bool = True,
) -> StripCorrectionResult:
    """z を OCR に合わせる。warp_theta ならエッジ中心へ一様シフトする。"""
    warped_z, z_min_out, z_max_out, z_ctrl, msg_z = warp_z_to_ocr(
        rgb, filled, z_min, pixels_per_mm, z_est_mm, z_ocr_mm, control_spacing_mm
    )
    if not warp_theta:
        return StripCorrectionResult(
            rgb=warped_z["rgb"],
            filled=warped_z["filled"],
            z_min=z_min_out,
            z_max=z_max_out,
            dtheta_rad=None,
            z_ctrl=z_ctrl,
            message=f"{msg_z}; θ補正スキップ（接合面補正を使う）",
        )
    warped_th, dtheta, msg_th = warp_theta_to_edge_center(
        warped_z["rgb"], warped_z["filled"],
        z_min_out, pixels_per_mm,
        orientation_ref, half_fov_rad,
        theta_min, theta_max, control_spacing_mm,
    )
    return StripCorrectionResult(
        rgb=warped_th["rgb"],
        filled=warped_th["filled"],
        z_min=z_min_out,
        z_max=z_max_out,
        dtheta_rad=dtheta,
        z_ctrl=z_ctrl,
        message=f"{msg_z}; {msg_th}",
    )


def _strict_increasing(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    keep = np.ones(xs.size, dtype=bool)
    keep[1:] = np.diff(xs) > 1e-6
    return xs[keep], ys[keep]


def _eval_pchip_hold(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """区間内は PCHIP、区間外は端点値を保持（傾きの外挿をしない）。"""
    fn = PchipInterpolator(x, y, extrapolate=False)
    yq = np.asarray(fn(xq), dtype=float)
    yq = np.where(xq < x[0], y[0], yq)
    yq = np.where(xq > x[-1], y[-1], yq)
    return yq


def _flatten_end_slopes(
    x: np.ndarray, y: np.ndarray, pad_mm: float
) -> Tuple[np.ndarray, np.ndarray]:
    """端に同じ y の点を足して、開始・終了の微分を 0 に近づける。"""
    pad = max(float(pad_mm), 1.0)
    x0 = float(x[0] - pad)
    xn = float(x[-1] + pad)
    return np.r_[x0, x, xn], np.r_[y[0], y, y[-1]]


def _stable_column_mask(filled: np.ndarray, min_frac: float = 0.85) -> np.ndarray:
    """被覆が薄い開始・終了列を除外する（端の欠けたエッジを制御点にしない）。"""
    heights = np.sum(filled, axis=0).astype(float)
    positive = heights[heights > 0]
    if positive.size == 0:
        return np.zeros(filled.shape[1], dtype=bool)
    med = float(np.median(positive))
    if med < 1.0:
        return heights > 0
    return heights >= (min_frac * med)


def _thin_pairs(
    z_src: np.ndarray,
    z_tgt: np.ndarray,
    spacing_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(z_src)
    zs = z_src[order]
    zt = z_tgt[order]
    keep = [0]
    last = zs[0]
    for i in range(1, zs.size - 1):
        if zs[i] - last >= spacing_mm and abs(zt[i] - zt[keep[-1]]) > 1e-9:
            keep.append(i)
            last = zs[i]
    if keep[-1] != zs.size - 1:
        keep.append(zs.size - 1)
    zs_k = zs[keep]
    zt_k = zt[keep]
    # PCHIP は厳密単調な独立変数が必要
    uniq = np.concatenate([[True], np.diff(zs_k) > 1e-6])
    zs_k, zt_k = zs_k[uniq], zt_k[uniq]
    if zs_k.size < 2:
        return z_src[order][[0, -1]], z_tgt[order][[0, -1]]
    # 目標側も単調に（同一 OCR が続く点は捨てる）
    tgt_ok = np.concatenate([[True], np.diff(zt_k) > 1e-6])
    if int(np.sum(tgt_ok)) >= 2:
        zs_k, zt_k = zs_k[tgt_ok], zt_k[tgt_ok]
    if zs_k.size < 2:
        return z_src[order][[0, -1]], z_tgt[order][[0, -1]]
    return zs_k, zt_k


def _ocr_reference_frames(
    z_est: np.ndarray,
    z_ocr: np.ndarray,
    interval_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """オリジナル補正と同じく、OCR 間隔 L で基準フレームを間引く。"""
    idx = [0]
    current = 0.0
    for i in range(1, int(z_ocr.size)):
        if float(z_ocr[i]) >= current + interval_mm:
            idx.append(i)
            current = float(z_ocr[i])
    if idx[-1] != int(z_ocr.size) - 1:
        idx.append(int(z_ocr.size) - 1)
    zs, zt = np.asarray(z_est[idx], dtype=float), np.asarray(z_ocr[idx], dtype=float)
    if zs.size >= 2 and abs(float(zt[-1]) - float(zt[-2])) <= 1e-6:
        zs = np.r_[zs[:-2], zs[-1]]
        zt = np.r_[zt[:-2], zt[-1]]
    zt, zs = _strict_increasing(zt, zs)
    return zs, zt


def warp_z_to_ocr(
    rgb: np.ndarray,
    filled: np.ndarray,
    z_min: float,
    pixels_per_mm: float,
    z_est_mm: np.ndarray,
    z_ocr_mm: np.ndarray,
    control_spacing_mm: float = 50.0,
) -> Tuple[dict, float, float, Optional[np.ndarray], str]:
    """オリジナル ColormapCorrector と同じ z 補正。

    OCR 軸の 0 ～ za_max だけを PCHIP し、za_max より右（見越し）は
    倍率 1 で元画像を貼る。出力原点は z=0 なので、za_max までの幅は
    U/R で一致する。
    """
    z_est = np.asarray(z_est_mm, dtype=float).reshape(-1)
    z_ocr = np.asarray(z_ocr_mm, dtype=float).reshape(-1)
    n = min(z_est.size, z_ocr.size)
    z_est, z_ocr = z_est[:n], z_ocr[:n]
    valid = np.isfinite(z_est) & np.isfinite(z_ocr)
    z_est, z_ocr = z_est[valid], z_ocr[valid]
    h, w = rgb.shape[:2]
    ppm = max(float(pixels_per_mm), 1e-9)
    src_z_min = float(z_min)
    src_z_max = float(z_min) + w / ppm
    if z_est.size < 2:
        return (
            {"rgb": rgb.copy(), "filled": filled.copy()},
            src_z_min,
            src_z_max,
            None,
            "z補正スキップ（OCR点不足）",
        )

    zs, zt = _ocr_reference_frames(z_est, z_ocr, control_spacing_mm)
    if zs.size < 2:
        return (
            {"rgb": rgb.copy(), "filled": filled.copy()},
            src_z_min,
            src_z_max,
            None,
            "z補正スキップ（単調な制御点が不足）",
        )

    z_max = float(np.max(z_est))
    za_max = float(np.max(z_ocr))
    inverse = PchipInterpolator(zt, zs, extrapolate=True)

    # 補正後幅: za_max まで + 最終カメラ z より右の見越し
    # 元画像 x=0 が z=0 なら w + (za_max - z_max)×ppm に一致する
    tail_mm = max(0.0, src_z_max - z_max)
    z_dst_min = 0.0
    new_w = max(8, int((za_max + tail_mm) * ppm + 0.5))
    z_dst_max = new_w / ppm
    xs = np.arange(new_w, dtype=np.float64)
    mm_dst = z_dst_min + xs / ppm
    mm_src = np.asarray(inverse(mm_dst), dtype=float)
    extra = mm_dst > za_max
    mm_src[extra] = z_max + (mm_dst[extra] - za_max)
    mm_src = np.where(np.isnan(mm_src), 0.0, mm_src)
    map_x = ((mm_src - src_z_min) * ppm).astype(np.float32)
    map_x = np.clip(map_x, 0, w - 1)
    map_x = np.repeat(map_x.reshape(1, -1), h, axis=0)
    map_y = np.repeat(np.arange(h, dtype=np.float32).reshape(-1, 1), new_w, axis=1)
    rgb_o = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    filled_u8 = filled.astype(np.uint8) * 255
    filled_o = cv2.remap(
        filled_u8, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    ) > 0
    rgb_o[~filled_o] = 0
    return (
        {"rgb": rgb_o, "filled": filled_o},
        z_dst_min,
        z_dst_max,
        zs,
        f"z補正 ctrl={zs.size} OCR[0,{za_max:.1f}] tail={tail_mm:.1f}mm",
    )


def _longest_filled_arc(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """円周方向の最長 True 弧。戻り値は [start, end)（end は排他、wrap あり）。"""
    n = int(mask.size)
    if n == 0 or not np.any(mask):
        return None
    if np.all(mask):
        return 0, n
    ext = np.concatenate([mask, mask])
    best_len = 0
    best = (0, 0)
    i = 0
    while i < ext.size:
        if not ext[i]:
            i += 1
            continue
        j = i
        while j < ext.size and ext[j]:
            j += 1
        length = j - i
        if length > best_len:
            best_len = length
            best = (i, j)
        i = j
    start = best[0] % n
    length = min(best_len, n)
    return start, (start + length) % n if length < n else start


def _column_midpoints(
    filled: np.ndarray,
    theta_min: float,
    theta_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = filled.shape[:2]
    span = theta_max - theta_min
    mid = np.full(w, np.nan, dtype=float)
    ok = np.zeros(w, dtype=bool)
    for x in range(w):
        col = filled[:, x]
        arc = _longest_filled_arc(col)
        if arc is None:
            continue
        i0, i1 = arc
        if i0 == i1 and np.all(col):
            mid[x] = 0.5 * (theta_min + theta_max)
            ok[x] = True
            continue
        length = (i1 - i0) % h
        if length == 0:
            length = h
        if length < max(4, h // 20):
            continue
        mid_idx = (i0 + 0.5 * length) % h
        mid[x] = theta_min + mid_idx / max(h - 1, 1) * span
        ok[x] = True
    return mid, ok


def warp_theta_to_edge_center(
    rgb: np.ndarray,
    filled: np.ndarray,
    z_min: float,
    pixels_per_mm: float,
    orientation_ref: np.ndarray,
    half_fov_rad: float,
    theta_min: float = 0.0,
    theta_max: float = 2.0 * np.pi,
    control_spacing_mm: float = 50.0,
) -> Tuple[dict, Optional[np.ndarray], str]:
    h, w = rgb.shape[:2]
    ppm = max(float(pixels_per_mm), 1e-9)
    center, _lo, _hi = expected_edge_etas(
        float(orientation_ref[0]),
        float(orientation_ref[1]),
        float(orientation_ref[2]),
        float(half_fov_rad),
    )
    mid, ok = _column_midpoints(filled, theta_min, theta_max)
    ok = ok & _stable_column_mask(filled)
    xs = np.where(ok)[0]
    if xs.size < 4:
        return {"rgb": rgb.copy(), "filled": filled.copy()}, None, "θ補正スキップ（エッジ不足）"

    z_ok = z_min + xs.astype(float) / ppm
    delta = np.array([wrap_angle_rad(center - m) for m in mid[ok]], dtype=float)
    keep = [0]
    last = z_ok[0]
    for i in range(1, z_ok.size - 1):
        if z_ok[i] - last >= control_spacing_mm:
            keep.append(i)
            last = z_ok[i]
    if keep[-1] != z_ok.size - 1:
        keep.append(z_ok.size - 1)
    zs = z_ok[keep]
    ds = delta[keep]
    uniq = np.concatenate([[True], np.diff(zs) > 1e-6])
    zs, ds = zs[uniq], ds[uniq]
    if zs.size < 2:
        return {"rgb": rgb.copy(), "filled": filled.copy()}, None, "θ補正スキップ（制御点不足）"

    # 端の外れ値を内点の中央値へ寄せ、開始点で急なせん断が起きないようにする
    interior = ds[max(0, zs.size // 4) : max(1, 3 * zs.size // 4)]
    d_ref = float(np.median(interior if interior.size else ds))
    spread = float(np.median(np.abs(ds - d_ref))) + 1e-6
    ds = ds.copy()
    ds[0] = d_ref if abs(ds[0] - d_ref) > 2.5 * spread else ds[0]
    ds[-1] = d_ref if abs(ds[-1] - d_ref) > 2.5 * spread else ds[-1]
    zs, ds = _flatten_end_slopes(zs, ds, pad_mm=max(control_spacing_mm, 20.0))
    z_cols = z_min + np.arange(w, dtype=float) / ppm
    dtheta = _eval_pchip_hold(zs, ds, z_cols)
    span = theta_max - theta_min
    d_rows = dtheta / span * (h - 1)
    map_x = np.repeat(np.arange(w, dtype=np.float32).reshape(1, -1), h, axis=0)
    rows = np.arange(h, dtype=np.float32).reshape(-1, 1)
    map_y = (rows - d_rows.reshape(1, -1)) % h
    rgb_o = cv2.remap(rgb, map_x, map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    filled_u8 = filled.astype(np.uint8) * 255
    filled_o = cv2.remap(
        filled_u8, map_x, map_y.astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    ) > 0
    rgb_o[~filled_o] = 0
    return (
        {"rgb": rgb_o, "filled": filled_o},
        dtheta,
        f"θ補正 ctrl={zs.size} center={np.degrees(center):.1f}deg",
    )
