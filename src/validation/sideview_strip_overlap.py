"""側視の同一 Run 帯重なり。既描画と現帯の ORB leftover から絶対 (z, pitch) を直す。

正面の front_strip_overlap は import しない。leftover の行はプログラム pitch、
列は z。帯が既描画より +η/+z なら、現在姿勢をその分だけ引く。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

N_ETA_BINS = 6
MIN_BIN_POINTS = 2
MIN_ETA_BINS = 3
SEAM_FRAC = 0.4
BIN_WORSE_PX = 1.0
DCOL_BIN_AGREE_PX = 3.0
MIN_LEFTOVER_PX = 0.75


@dataclass
class StripOverlapResult:
    n_raw: int
    n_kept: int
    n_inlier: int
    med_drow_px: float
    med_dcol_px: float
    rms_px: float
    med_abs_px: float
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
    """画像1=既描画、画像2=現帯。点は帯パッチ座標 (x=列=z, y=行=η)。"""
    empty = StripOverlapResult(
        n_raw=0, n_kept=0, n_inlier=0,
        med_drow_px=0.0, med_dcol_px=0.0, rms_px=0.0, med_abs_px=0.0,
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
    off = np.array([float(src_c0), 0.0], dtype=float)
    return StripOverlapResult(
        n_raw=n_raw,
        n_kept=n_kept,
        n_inlier=int(np.sum(inlier_mask)),
        med_drow_px=float(np.median(drow)),
        med_dcol_px=float(np.median(dcol)),
        rms_px=float(np.sqrt(np.mean(mag ** 2))),
        med_abs_px=float(np.median(mag)),
        pts_painted=(p1 + off).astype(float),
        pts_strip=(p2 + off).astype(float),
        inlier_mask=inlier_mask,
        src_c0=int(src_c0),
        dest_c0=int(dest_c0),
        ok=int(np.sum(inlier_mask)) >= int(min_inliers),
    )


def painted_row_span(mask: np.ndarray) -> Tuple[int, int]:
    """マスクが立つ行の [row0, row1)。空なら全高。"""
    m = np.asarray(mask)
    if m.ndim >= 2:
        occupied = np.any(m != 0, axis=tuple(range(1, m.ndim)))
    else:
        occupied = m != 0
    rows = np.where(occupied)[0]
    if rows.size == 0:
        return 0, int(m.shape[0]) if m.ndim >= 1 else 1
    return int(rows[0]), int(rows[-1] + 1)


def _eta_bin_index(
    rows: np.ndarray,
    row0: int,
    row1: int,
    n_bins: int,
) -> np.ndarray:
    """塗れている η 範囲だけを n_bins 等分する。全キャンバス高では割らない。"""
    span = max(float(int(row1) - int(row0)), 1.0)
    bins = np.floor(
        (np.asarray(rows, dtype=float) - float(row0)) * float(n_bins) / span
    ).astype(int)
    return np.clip(bins, 0, max(int(n_bins) - 1, 0))


def leftover_to_pose_delta(
    med_drow_px: float,
    med_dcol_px: float,
    eta_resolution: int,
    pixels_per_mm: float,
) -> Tuple[float, float]:
    """帯-既描画の中央値並進を (dz_mm, dpitch_rad) へ。

    drow>0（η 下向きが +）なら帯が +η → pitch = pitch0 − d_pitch
    dcol>0 なら帯が +z → z = z0 − dz
    """
    d_pitch = float(med_drow_px) * (2.0 * np.pi / max(int(eta_resolution), 1))
    d_z = float(med_dcol_px) / max(float(pixels_per_mm), 1e-6)
    return d_z, d_pitch


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
    row0: Optional[int] = None,
    row1: Optional[int] = None,
) -> np.ndarray:
    out = np.full(int(n_bins), np.nan, dtype=float)
    got = _inlier_deltas(result)
    if got is None:
        return out
    pts_s, drow, dcol = got
    r0 = 0 if row0 is None else int(row0)
    r1 = max(int(eta_height), 1) if row1 is None else int(row1)
    mag = np.hypot(dcol, drow)
    bins = _eta_bin_index(pts_s[:, 1], r0, r1, n_bins)
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
    row0: int,
    row1: int,
    n_bins: int,
    min_bin_points: int,
) -> Tuple[list, list]:
    bins = _eta_bin_index(rows, row0, row1, n_bins)
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
    row0: Optional[int] = None,
    row1: Optional[int] = None,
    seam_frac: float = SEAM_FRAC,
    n_bins: int = N_ETA_BINS,
    min_bin_points: int = MIN_BIN_POINTS,
    min_bins: int = MIN_ETA_BINS,
) -> Optional[Tuple[float, float, int]]:
    got = _inlier_deltas(result)
    if got is None:
        return None
    pts_s, drow, dcol = got
    r0 = 0 if row0 is None else int(row0)
    r1 = max(int(eta_height), 1) if row1 is None else int(row1)
    attempts = []
    x_cut = float(np.quantile(pts_s[:, 0], max(0.0, 1.0 - float(seam_frac))))
    seam = pts_s[:, 0] >= x_cut
    if int(np.sum(seam)) >= int(min_bin_points) * 2:
        attempts.append((drow[seam], dcol[seam], pts_s[seam, 1]))
    attempts.append((drow, dcol, pts_s[:, 1]))
    last_c: list = []
    for dr, dc, rows in attempts:
        med_r, med_c = _eta_bin_medians(
            dr, dc, rows, r0, r1, n_bins, min_bin_points
        )
        last_c = med_c
        if len(med_r) >= int(min_bins):
            return float(np.median(med_r)), float(np.median(med_c)), len(med_r)
    if len(last_c) >= 2 and float(np.ptp(np.asarray(last_c, dtype=float))) <= DCOL_BIN_AGREE_PX:
        return 0.0, float(np.median(last_c)), len(last_c)
    return None


def allow_bin_worse_for_pose(allow_flag: bool, d_pitch: float) -> bool:
    return bool(allow_flag) or abs(float(d_pitch)) < 1e-12


def pose_correction_verdict(
    before: StripOverlapResult,
    after: StripOverlapResult,
    eta_height: int,
    *,
    n_bins: int = N_ETA_BINS,
    min_bin_points: int = MIN_BIN_POINTS,
    worse_px: float = BIN_WORSE_PX,
    allow_bin_worse: bool = False,
    row0: Optional[int] = None,
    row1: Optional[int] = None,
) -> Tuple[bool, str]:
    if not after.ok:
        return False, (
            "取り直しマッチ失敗 "
            f"n={after.n_raw}->{after.n_kept}->{after.n_inlier}"
        )
    if not np.isfinite(after.rms_px) or after.rms_px >= before.rms_px:
        return False, (
            "全体RMS非改善 "
            f"{before.rms_px:.2f}->{after.rms_px:.2f}px "
            f"n={before.n_inlier}->{after.n_inlier}"
        )
    b = eta_bin_med_abs(
        before, eta_height, n_bins, min_bin_points, row0=row0, row1=row1
    )
    a = eta_bin_med_abs(
        after, eta_height, n_bins, min_bin_points, row0=row0, row1=row1
    )
    both = np.isfinite(b) & np.isfinite(a)
    worse = both & (a > b + float(worse_px))
    if np.any(worse):
        parts = [f"η{i}:{b[i]:.2f}->{a[i]:.2f}" for i in np.where(worse)[0]]
        detail = (
            "ηビン悪化 "
            + ",".join(parts)
            + f" rms {before.rms_px:.2f}->{after.rms_px:.2f}px"
        )
        if not allow_bin_worse:
            return False, detail
        return True, "改善(ηビン悪化許容) " + detail
    return True, (
        f"改善 rms {before.rms_px:.2f}->{after.rms_px:.2f}px "
        f"n={before.n_inlier}->{after.n_inlier}"
    )


def _start_col(acc, z_min_mm: float) -> int:
    ppm = max(float(acc.buf.pixels_per_mm), 1e-9)
    return int(np.floor((float(z_min_mm) - float(acc.buf.z_min)) * ppm))


def _try_abs_pose(
    mapper,
    acc,
    rec,
    frame,
    run_id: str,
    before: StripOverlapResult,
    z0: float,
    pitch0: float,
    d_z: float,
    d_pitch: float,
    max_dz: float,
    alpha: float,
    ppm: float,
    allow_bin_worse: bool,
    row0: int,
    row1: int,
    n_bins: int,
    min_bin_points: int,
) -> Optional[Tuple[dict, str, float, float]]:
    pos = np.asarray(rec.position, dtype=float).copy()
    ori = np.asarray(rec.orientation, dtype=float).copy()
    pos[2] = z0 - d_z
    ori[2] = pitch0 - d_pitch
    strip2 = mapper.sample_strip(
        frame, pos, ori,
        theta_bins=acc.buf.theta_bins,
        pixels_per_mm=ppm,
        max_dz=max_dz,
        alpha=alpha,
        theta_min=acc.theta_min,
        theta_max=acc.theta_max,
        run_id=run_id,
        frame_num=rec.frame_num,
    )
    start2 = _start_col(acc, float(strip2["z_min_strip"]))
    if start2 < 0:
        return None
    check = match_unwrapped_strips(
        acc.buf.color, strip2["colors_2d"], strip2["mask_2d"], start2
    )
    ok, why = pose_correction_verdict(
        before, check, acc.buf.theta_bins,
        n_bins=n_bins,
        min_bin_points=min_bin_points,
        row0=row0,
        row1=row1,
        allow_bin_worse=allow_bin_worse_for_pose(allow_bin_worse, d_pitch),
    )
    note = (
        f"dz={-d_z:+.2f}mm dpitch={float(np.degrees(-d_pitch)):+.3f}° {why}"
    )
    if not ok:
        logger.debug("側視帯重なり補正候補: %s", note)
        return None
    return strip2, note, float(pos[2]), float(ori[2])


def inherit_running_pose(
    z_ana: float,
    pitch_ana: float,
    z_ana_prev: Optional[float],
    pitch_ana_prev: Optional[float],
    z_run: Optional[float],
    pitch_run: Optional[float],
) -> Tuple[float, float]:
    """帯重なりで直した絶対 (z, pitch) に、解析のフレーム間増分を足して次の初期値にする。

    正面の camera_state と同じ。マッチの droll=0 / 側視 dz_only では周方向増分が 0 なので、
    直した pitch が次フレームの初期 pitch になる。deroll はしない。
    """
    if (
        z_run is None
        or pitch_run is None
        or z_ana_prev is None
        or pitch_ana_prev is None
    ):
        return float(z_ana), float(pitch_ana)
    return (
        float(z_run) + (float(z_ana) - float(z_ana_prev)),
        float(pitch_run) + (float(pitch_ana) - float(pitch_ana_prev)),
    )


def maybe_correct_strip(
    mapper,
    acc,
    strip: Dict,
    rec,
    frame,
    run_id: str,
    config,
    max_dz: float,
    alpha: float,
    ppm: float,
) -> Dict:
    """測定オンなら leftover を測り、補正オンかつ改善すれば取り直した帯を返す。

    失敗時は補正前の帯を返す（測定ログだけ残す）。
    """
    td = config.two_direction
    if not bool(getattr(td, "strip_overlap_match", False)):
        return strip
    start = _start_col(acc, float(strip["z_min_strip"]))
    result = match_unwrapped_strips(
        acc.buf.color, strip["colors_2d"], strip["mask_2d"], start
    )
    logger.info(
        "側視帯重なり: frame=%s n=%s->%s->%s "
        "med_dη=%.2fpx med_dz=%.2fpx rms=%.2fpx ok=%s",
        rec.frame_num, result.n_raw, result.n_kept, result.n_inlier,
        result.med_drow_px, result.med_dcol_px, result.rms_px, result.ok,
    )
    if not bool(getattr(td, "strip_overlap_correct", False)) or not result.ok:
        return strip
    n_bins = int(getattr(td, "residual_theta_bins", N_ETA_BINS) or N_ETA_BINS)
    min_pts = int(getattr(td, "residual_min_bin_points", MIN_BIN_POINTS) or MIN_BIN_POINTS)
    row0, row1 = painted_row_span(strip["mask_2d"])
    dc = pose_dc_from_inliers(
        result, acc.buf.theta_bins,
        row0=row0, row1=row1, n_bins=n_bins, min_bin_points=min_pts,
    )
    if dc is None:
        logger.info("側視帯重なり補正: ηビン不足のためスキップ frame=%s", rec.frame_num)
        return strip
    dc_drow, dc_dcol, n_bins_used = dc
    if abs(dc_drow) < MIN_LEFTOVER_PX and abs(dc_dcol) < MIN_LEFTOVER_PX:
        return strip
    if abs(dc_drow) > 16.0 or abs(dc_dcol) > 16.0:
        logger.info(
            "側視帯重なり補正: leftover 上限超え drow=%.2f dcol=%.2f frame=%s",
            dc_drow, dc_dcol, rec.frame_num,
        )
        return strip
    d_z, d_pitch = leftover_to_pose_delta(
        dc_drow, dc_dcol, acc.buf.theta_bins, ppm
    )
    z0 = float(rec.position[2])
    pitch0 = float(rec.orientation[2])
    apply_z = abs(dc_dcol) >= MIN_LEFTOVER_PX
    apply_pitch = abs(dc_drow) >= MIN_LEFTOVER_PX
    allow_bin = bool(getattr(td, "strip_overlap_allow_bin_worse", False))
    best = None
    tag = "z+pitch"
    dz_try = d_z if apply_z else 0.0
    dp_try = d_pitch if apply_pitch else 0.0
    if apply_z or apply_pitch:
        best = _try_abs_pose(
            mapper, acc, rec, frame, run_id, result,
            z0, pitch0, dz_try, dp_try, max_dz, alpha, ppm, allow_bin,
            row0, row1, n_bins, min_pts,
        )
        if apply_z and not apply_pitch:
            tag = "z"
        elif apply_pitch and not apply_z:
            tag = "pitch"
    if best is None and apply_z and apply_pitch:
        best = _try_abs_pose(
            mapper, acc, rec, frame, run_id, result,
            z0, pitch0, d_z, 0.0, max_dz, alpha, ppm, allow_bin,
            row0, row1, n_bins, min_pts,
        )
        tag = "z"
    if best is None:
        logger.info("側視帯重なり補正: 取り直し不採用 frame=%s", rec.frame_num)
        return strip
    strip2, note, z_new, pitch_new = best
    rec.position = np.asarray(rec.position, dtype=float)
    rec.orientation = np.asarray(rec.orientation, dtype=float)
    rec.position[2] = z_new
    rec.orientation[2] = pitch_new
    logger.info(
        "側視帯重なり補正: mode=%s dz=%+.2fmm dpitch=%+.3f° dc_nbin=%s %s frame=%s",
        tag, z_new - z0, np.degrees(pitch_new - pitch0), n_bins_used, note, rec.frame_num,
    )
    return strip2
