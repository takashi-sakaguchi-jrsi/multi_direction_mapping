"""展開後の既描画（画像1）と現フレーム帯（画像2）の特徴点重なり誤差。

魚眼上の既存マッチは使わない。展開画像から ORB を取り直し、IQR と
RANSAC で例外を除く。補正は貼り位置ではなく絶対座標 (z, roll) を直し、
その姿勢で取り直して貼る。移動量推定の droll=0 はフレーム間の roll 増分を
出さないだけで、補正後の絶対 roll は次フレームでも維持される。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class StripOverlapResult:
    n_raw: int
    n_kept: int
    n_inlier: int
    med_drow_px: float
    med_dcol_px: float
    rms_px: float
    med_abs_px: float
    affine: Optional[np.ndarray]
    pts_painted: np.ndarray
    pts_strip: np.ndarray
    inlier_mask: np.ndarray
    src_c0: int
    dest_c0: int
    ok: bool


def _luma_u8(img: np.ndarray) -> np.ndarray:
    rgb = np.asarray(img)
    if rgb.ndim == 2:
        g = rgb
    else:
        g = rgb[..., :3].astype(np.float32).mean(axis=2)
    return np.clip(g, 0, 255).astype(np.uint8)


def _overlap_slices(
    painted: np.ndarray,
    strip: np.ndarray,
    mask: np.ndarray,
    start_col: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    h, w = strip.shape[:2]
    ph, pw = painted.shape[:2]
    if h != ph or w < 8 or h < 32:
        return None
    dest_c0 = max(int(start_col), 0)
    src_c0 = dest_c0 - int(start_col)
    dest_c1 = min(int(start_col) + w, pw)
    src_c1 = src_c0 + (dest_c1 - dest_c0)
    if dest_c1 - dest_c0 < 8 or src_c1 <= src_c0:
        return None
    dest = painted[:, dest_c0:dest_c1]
    src = strip[:, src_c0:src_c1]
    src_mask = np.asarray(mask[:, src_c0:src_c1], dtype=bool)
    painted_ok = np.any(dest[..., :3] != 0, axis=-1) if dest.ndim == 3 else dest != 0
    overlap = src_mask & painted_ok
    if int(np.sum(overlap)) < 64:
        return None
    return dest, src, overlap, src_c0, dest_c0


def _iqr_keep(du: np.ndarray, dv: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    keep = np.ones(len(du), dtype=bool)
    if len(du) < 8:
        return keep
    for d in (du, dv):
        q1, q3 = np.percentile(d, [25.0, 75.0])
        iqr = float(q3 - q1)
        if iqr < 1e-6:
            continue
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        keep &= (d >= lo) & (d <= hi)
    return keep


def match_unwrapped_strips(
    painted: np.ndarray,
    strip: np.ndarray,
    mask: np.ndarray,
    start_col: int,
    *,
    max_features: int = 2000,
    ratio: float = 0.75,
    ransac_px: float = 3.0,
    min_inliers: int = 8,
) -> StripOverlapResult:
    """画像1=既描画、画像2=現帯。戻り点は帯パッチ座標 (x=列, y=行)。"""
    empty = StripOverlapResult(
        n_raw=0, n_kept=0, n_inlier=0,
        med_drow_px=0.0, med_dcol_px=0.0, rms_px=0.0, med_abs_px=0.0,
        affine=None,
        pts_painted=np.zeros((0, 2), dtype=float),
        pts_strip=np.zeros((0, 2), dtype=float),
        inlier_mask=np.zeros(0, dtype=bool),
        src_c0=0, dest_c0=0, ok=False,
    )
    slices = _overlap_slices(painted, strip, mask, start_col)
    if slices is None:
        return empty
    dest, src, overlap, src_c0, dest_c0 = slices
    g1 = _luma_u8(dest)
    g2 = _luma_u8(src)
    m = overlap.astype(np.uint8) * 255
    orb = cv2.ORB_create(nfeatures=int(max_features))
    k1, d1 = orb.detectAndCompute(g1, m)
    k2, d2 = orb.detectAndCompute(g2, m)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return empty
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(d1, d2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < float(ratio) * b.distance:
            good.append(a)
    if len(good) < 4:
        empty.n_raw = len(good)
        return empty
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    n_raw = len(p1)
    du, dv = p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1]
    keep = _iqr_keep(du, dv)
    p1, p2 = p1[keep], p2[keep]
    n_kept = len(p1)
    if n_kept < 4:
        empty.n_raw = n_raw
        empty.n_kept = n_kept
        return empty
    _, inliers = cv2.estimateAffinePartial2D(
        p2.reshape(-1, 1, 2),
        p1.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_px),
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if inliers is None:
        inlier_mask = np.ones(n_kept, dtype=bool)
    else:
        inlier_mask = inliers.reshape(-1).astype(bool)
        if int(np.sum(inlier_mask)) < int(min_inliers):
            inlier_mask = np.ones(n_kept, dtype=bool)
    p1i, p2i = p1[inlier_mask], p2[inlier_mask]
    dcol = p2i[:, 0] - p1i[:, 0]
    drow = p2i[:, 1] - p1i[:, 1]
    mag = np.hypot(dcol, drow)
    med_dcol = float(np.median(dcol))
    med_drow = float(np.median(drow))
    # 帯パッチ座標へ
    off = np.array([float(src_c0), 0.0], dtype=float)
    pts_painted = p1 + off
    pts_strip = p2 + off
    # src→dst 並進。warpAffine 既定はこれを逆に使う。回転は入れない。
    affine = np.array(
        [[1.0, 0.0, -med_dcol], [0.0, 1.0, -med_drow]],
        dtype=np.float64,
    )
    return StripOverlapResult(
        n_raw=n_raw,
        n_kept=n_kept,
        n_inlier=int(np.sum(inlier_mask)),
        med_drow_px=med_drow,
        med_dcol_px=med_dcol,
        rms_px=float(np.sqrt(np.mean(mag ** 2))),
        med_abs_px=float(np.median(mag)),
        affine=affine,
        pts_painted=pts_painted.astype(float),
        pts_strip=pts_strip.astype(float),
        inlier_mask=inlier_mask,
        src_c0=int(src_c0),
        dest_c0=int(dest_c0),
        ok=int(np.sum(inlier_mask)) >= int(min_inliers),
    )


def leftover_to_pose_delta(
    med_drow_px: float,
    med_dcol_px: float,
    eta_resolution: int,
    pixels_per_mm: float,
) -> Tuple[float, float]:
    """帯-既描画の中央値並進を (dz_mm, droll_rad) へ。

    展開の η 並進は円筒軸回りの roll。z 並進はカメラ z。
    帯が既描画より +η/+z なら、現在姿勢がその分だけ進んでいる。
    適用は両方とも引く（z0 - dz, roll0 - droll）。実画像はカメラ +roll で
    帯の行が + になる。
    """
    d_roll = float(med_drow_px) * (2.0 * np.pi / max(int(eta_resolution), 1))
    d_z = float(med_dcol_px) / max(float(pixels_per_mm), 1e-6)
    return d_z, d_roll


N_ETA_BINS = 8
MIN_BIN_POINTS = 3
MIN_ETA_BINS = 3
SEAM_FRAC = 0.4
BIN_WORSE_PX = 1.0
DCOL_BIN_AGREE_PX = 3.0


def _inlier_deltas(
    result: StripOverlapResult,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not result.ok or result.pts_strip.size == 0:
        return None
    mask = np.asarray(result.inlier_mask, dtype=bool).reshape(-1)
    if mask.size != len(result.pts_strip):
        return None
    pts_s = np.asarray(result.pts_strip, dtype=float)[mask]
    pts_p = np.asarray(result.pts_painted, dtype=float)[mask]
    if len(pts_s) < 4:
        return None
    dcol = pts_s[:, 0] - pts_p[:, 0]
    drow = pts_s[:, 1] - pts_p[:, 1]
    return pts_s, drow, dcol


def eta_bin_med_abs(
    result: StripOverlapResult,
    eta_height: int,
    n_bins: int = N_ETA_BINS,
    min_bin_points: int = MIN_BIN_POINTS,
) -> np.ndarray:
    """インライアの |d| 中央値を η ビンごと。点不足は nan。"""
    out = np.full(int(n_bins), np.nan, dtype=float)
    got = _inlier_deltas(result)
    if got is None:
        return out
    pts_s, drow, dcol = got
    h = max(int(eta_height), 1)
    mag = np.hypot(dcol, drow)
    bins = np.floor(pts_s[:, 1] * float(n_bins) / float(h)).astype(int)
    bins = np.clip(bins, 0, int(n_bins) - 1)
    for b in range(int(n_bins)):
        sel = bins == b
        if int(np.sum(sel)) < int(min_bin_points):
            continue
        out[b] = float(np.median(mag[sel]))
    return out


def _eta_bin_medians(
    drow: np.ndarray,
    dcol: np.ndarray,
    rows: np.ndarray,
    eta_height: int,
    n_bins: int,
    min_bin_points: int,
) -> Tuple[list, list]:
    h = max(int(eta_height), 1)
    bins = np.floor(rows * float(n_bins) / float(h)).astype(int)
    bins = np.clip(bins, 0, int(n_bins) - 1)
    med_r: list = []
    med_c: list = []
    for b in range(int(n_bins)):
        sel = bins == b
        if int(np.sum(sel)) < int(min_bin_points):
            continue
        med_r.append(float(np.median(drow[sel])))
        med_c.append(float(np.median(dcol[sel])))
    return med_r, med_c


def pose_dc_from_inliers(
    result: StripOverlapResult,
    eta_height: int,
    *,
    seam_frac: float = SEAM_FRAC,
    n_bins: int = N_ETA_BINS,
    min_bin_points: int = MIN_BIN_POINTS,
    min_bins: int = MIN_ETA_BINS,
) -> Optional[Tuple[float, float, int]]:
    """継ぎ目側（高 z）と η ビンの DC を (drow, dcol, n_bins_used) で返す。

    継ぎ目だけではビンが足りないときは全インライアでやり直す。
    それでも 3 ビン未満なら、dcol が揃っている 2 ビンから z だけ出す。
    """
    got = _inlier_deltas(result)
    if got is None:
        return None
    pts_s, drow, dcol = got
    attempts = []
    x_cut = float(np.quantile(pts_s[:, 0], max(0.0, 1.0 - float(seam_frac))))
    seam = pts_s[:, 0] >= x_cut
    if int(np.sum(seam)) >= int(min_bin_points) * 2:
        attempts.append((drow[seam], dcol[seam], pts_s[seam, 1]))
    attempts.append((drow, dcol, pts_s[:, 1]))
    last_c: list = []
    for dr, dc, rows in attempts:
        med_r, med_c = _eta_bin_medians(
            dr, dc, rows, eta_height, n_bins, min_bin_points
        )
        last_c = med_c
        if len(med_r) >= int(min_bins):
            return float(np.median(med_r)), float(np.median(med_c)), len(med_r)
    if len(last_c) >= 2 and float(np.ptp(np.asarray(last_c, dtype=float))) <= DCOL_BIN_AGREE_PX:
        return 0.0, float(np.median(last_c)), len(last_c)
    return None


def allow_bin_worse_for_pose(allow_flag: bool, d_roll: float) -> bool:
    """z のみは常にビン悪化を許容。z+roll は設定フラグに従う。"""
    return bool(allow_flag) or abs(float(d_roll)) < 1e-12


def pose_correction_verdict(
    before: StripOverlapResult,
    after: StripOverlapResult,
    eta_height: int,
    *,
    n_bins: int = N_ETA_BINS,
    min_bin_points: int = MIN_BIN_POINTS,
    worse_px: float = BIN_WORSE_PX,
    allow_bin_worse: bool = False,
) -> Tuple[bool, str]:
    """取り直しの合否と、不合格ならどの判定で落ちたかを返す。

    allow_bin_worse は z のみ、または設定で z+roll もビン悪化を通すときに真。
    """
    if not after.ok:
        return False, (
            "取り直しマッチ失敗 "
            f"n={after.n_raw}->{after.n_kept}->{after.n_inlier}"
        )
    if not np.isfinite(after.rms_px):
        return False, "取り直しRMSが非数"
    if after.rms_px >= before.rms_px:
        return False, (
            "全体RMS非改善 "
            f"{before.rms_px:.2f}->{after.rms_px:.2f}px "
            f"n={before.n_inlier}->{after.n_inlier}"
        )
    b = eta_bin_med_abs(before, eta_height, n_bins, min_bin_points)
    a = eta_bin_med_abs(after, eta_height, n_bins, min_bin_points)
    both = np.isfinite(b) & np.isfinite(a)
    worse = both & (a > b + float(worse_px))
    if np.any(worse):
        parts = [
            f"η{i}:{b[i]:.2f}->{a[i]:.2f}"
            for i in np.where(worse)[0]
        ]
        detail = (
            "ηビン悪化 "
            + ",".join(parts)
            + f" rms {before.rms_px:.2f}->{after.rms_px:.2f}px "
            f"n={before.n_inlier}->{after.n_inlier}"
        )
        if not allow_bin_worse:
            return False, detail
        return True, "改善(ηビン悪化許容) " + detail
    return True, (
        f"改善 rms {before.rms_px:.2f}->{after.rms_px:.2f}px "
        f"n={before.n_inlier}->{after.n_inlier}"
    )


def pose_correction_helps(
    before: StripOverlapResult,
    after: StripOverlapResult,
    eta_height: int,
    *,
    n_bins: int = N_ETA_BINS,
    min_bin_points: int = MIN_BIN_POINTS,
    worse_px: float = BIN_WORSE_PX,
    allow_bin_worse: bool = False,
) -> bool:
    """取り直し後、全体 RMS が下がり、η ビンのどれも悪化していないときだけ真。

    allow_bin_worse なら η ビンは見ない。z のみは呼び出し側が常に真を渡す。
    (z, roll) がこれを満たさず z だけが満たす場合は、呼び出し側で z のみ採用する。
    """
    ok, _ = pose_correction_verdict(
        before,
        after,
        eta_height,
        n_bins=n_bins,
        min_bin_points=min_bin_points,
        worse_px=worse_px,
        allow_bin_worse=allow_bin_worse,
    )
    return ok


def shift_strip_eta(
    colors: np.ndarray,
    mask: np.ndarray,
    sample_radius: Optional[np.ndarray],
    drow: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """帯の内容を -drow 行だけ回す。η は周期。z は配列内では動かさない。"""
    h, w = colors.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(xs, ys)
    map_y = np.mod(map_y + np.float32(drow), float(h)).astype(np.float32)
    warped = cv2.remap(
        colors, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask_u8 = mask.astype(np.uint8) * 255
    mask_w = cv2.remap(
        mask_u8, map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 127
    radius_w = sample_radius
    if sample_radius is not None:
        radius_w = cv2.remap(
            sample_radius.astype(np.float32), map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return warped, mask_w, radius_w


def warp_strip_affine(
    colors: np.ndarray,
    mask: np.ndarray,
    sample_radius: Optional[np.ndarray],
    affine: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """帯全体を (η, z) 並進。η は周期、z は帯の外を切る。特徴が無い画素も同じ変換。"""
    h, w = colors.shape[:2]
    m = np.asarray(affine, dtype=np.float64).reshape(2, 3)
    tx, ty = float(m[0, 2]), float(m[1, 2])
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(xs, ys)
    map_x = map_x - np.float32(tx)
    map_y = np.mod(map_y - np.float32(ty), float(h)).astype(np.float32)
    warped = cv2.remap(
        colors, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask_u8 = mask.astype(np.uint8) * 255
    mask_w = cv2.remap(
        mask_u8, map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 127
    radius_w = sample_radius
    if sample_radius is not None:
        radius_w = cv2.remap(
            sample_radius.astype(np.float32), map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return warped, mask_w, radius_w


def draw_strip_error_vectors(
    painted: np.ndarray,
    strip: np.ndarray,
    mask: np.ndarray,
    start_col: int,
    result: StripOverlapResult,
    *,
    max_arrows: int = 400,
) -> Optional[np.ndarray]:
    """既描画の帯に、画像1→画像2 の誤差ベクトルを重ねる（BGR）。"""
    slices = _overlap_slices(painted, strip, mask, start_col)
    if slices is None or not result.ok:
        return None
    dest, src, overlap, src_c0, dest_c0 = slices
    vis = dest[..., :3].copy()
    if vis.dtype != np.uint8:
        vis = np.clip(vis, 0, 255).astype(np.uint8)
    # RGB → BGR
    vis = vis[:, :, ::-1].copy()
    src_bgr = src[..., :3][:, :, ::-1]
    blend = vis.astype(np.float32)
    src_f = src_bgr.astype(np.float32)
    ov = overlap[..., None]
    # 重なりは補正前の画像2を100%。矢尻が画像2の特徴点に乗る。
    vis = np.clip(blend * (1.0 - ov) + src_f * ov, 0, 255).astype(np.uint8)
    pts_p = result.pts_painted[result.inlier_mask] - np.array([src_c0, 0.0])
    pts_s = result.pts_strip[result.inlier_mask] - np.array([src_c0, 0.0])
    n = min(len(pts_p), int(max_arrows))
    h, w = vis.shape[:2]
    for i in range(n):
        x0, y0 = pts_p[i]
        x1, y1 = pts_s[i]
        if not (np.isfinite([x0, y0, x1, y1]).all()):
            continue
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))
        mag = float(np.hypot(x1 - x0, y1 - y0))
        if mag < 2:
            color = (0, 220, 0)
        elif mag < 12:
            color = (0, 220, 220)
        else:
            color = (0, 0, 220)
        cv2.arrowedLine(vis, p0, p1, color, 1, cv2.LINE_AA, tipLength=0.25)
        if 0 <= p1[0] < w and 0 <= p1[1] < h:
            cv2.circle(vis, p1, 2, color, -1)
    label = (
        f"inlier {result.n_inlier}/{result.n_raw}  "
        f"med {result.med_abs_px:.1f}px  rms {result.rms_px:.1f}px"
    )
    cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis
