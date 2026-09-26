"""接合面の 2 次元残差を半分ずつ戻す補正。

OCR 距離補正のあと、各接合線で (Δz, Δθ) を測り、端は固定しない。
カメラ i, j に u_i = −m/2, u_j = +m/2 を与え、120° 帯の内部は
θ 方向に線形補間する。相手の無い接合は u=0（OCR のまま）。
3 方向では U が R 側と L 側の両方を境界条件に取る。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.validation.geometry import (
    OVERLAP_CENTER_CAMERA_CAR_DEG,
    SECTOR_CAMERA_CAR_DEG,
    camera_car_deg_to_eta,
    eta_to_camera_car_deg,
)
from src.validation.strip_correction import (
    _eval_pchip_hold,
    _flatten_end_slopes,
    remap_with_periodic_theta,
)


SEAM_PAIRS: Tuple[Tuple[str, str], ...] = (("U", "R"), ("U", "L"), ("R", "L"))
# 各 run の担当帯 (lo, hi) に対応する相手。lo が先。
RUN_BOUNDARY_PARTNERS = {
    "U": ("L", "R"),
    "R": ("U", "L"),
    "L": ("R", "U"),
}


def normalize_run_id(run_id: str) -> str:
    key = str(run_id or "").strip().upper()
    if key == "A":
        return "U"
    if key == "B":
        return "R"
    if key == "C":
        return "L"
    return key


def pair_key(a: str, b: str) -> Optional[Tuple[str, str]]:
    ua, ub = normalize_run_id(a), normalize_run_id(b)
    for key in SEAM_PAIRS:
        if set(key) == {ua, ub}:
            return key
    return None


def sector_blend_weight(car_deg, run_id: str) -> np.ndarray:
    """担当帯 lo→hi で 0→1。帯の外は 0 または 1 にクリップ。"""
    key = normalize_run_id(run_id)
    lo, _hi = SECTOR_CAMERA_CAR_DEG[key]
    delta = (np.asarray(car_deg, dtype=float) - lo + 180.0) % 360.0 - 180.0
    return np.clip(delta / 120.0, 0.0, 1.0)


@dataclass
class SeamMismatch:
    pair: Tuple[str, str]
    z_mm: np.ndarray
    mz_mm: np.ndarray
    mth_rad: np.ndarray
    n_raw: int
    n_used: int
    message: str = ""


@dataclass
class SeamWarpResult:
    strips: List[dict]
    mismatches: List[SeamMismatch] = field(default_factory=list)
    success: bool = False
    message: str = ""

    def to_report(self) -> dict:
        seams = {}
        for m in self.mismatches:
            name = f"{m.pair[0]}_{m.pair[1]}"
            seams[name] = {
                "n_raw": int(m.n_raw),
                "n_used": int(m.n_used),
                "mz_median_mm": float(np.median(m.mz_mm)) if m.mz_mm.size else 0.0,
                "mth_median_deg": float(np.degrees(np.median(m.mth_rad))) if m.mth_rad.size else 0.0,
                "message": m.message,
            }
        return {
            "success": bool(self.success),
            "message": self.message,
            "seams": seams,
        }


def warp_strips_to_seams(
    strips: Sequence[dict],
    pixels_per_mm: float,
    theta_min: float = 0.0,
    theta_max: float = 2.0 * np.pi,
    config=None,
) -> SeamWarpResult:
    """OCR 後の部分図リストを接合面基準で変形する。"""
    if not strips:
        return SeamWarpResult(strips=[], message="部分図がありません")
    cfg = config
    enabled = True if cfg is None else bool(getattr(cfg, "enabled", True))
    if not enabled:
        return SeamWarpResult(
            strips=[_copy_strip(s) for s in strips],
            message="接合補正スキップ（disabled）",
        )

    prepared = [_copy_strip(s) for s in strips]
    for s in prepared:
        s["run_id"] = normalize_run_id(s["run_id"])
    present = {s["run_id"] for s in prepared}
    ppm = max(float(pixels_per_mm), 1e-9)
    commons = _paste_common(prepared, ppm)
    mismatches: List[SeamMismatch] = []
    for a_id, b_id in SEAM_PAIRS:
        if a_id not in present or b_id not in present:
            continue
        sa = commons[a_id]
        sb = commons[b_id]
        mm = _match_seam_pair(
            sa, sb, a_id, b_id, ppm, theta_min, theta_max, cfg
        )
        mismatches.append(mm)

    if not mismatches or all(m.n_used < 2 for m in mismatches):
        return SeamWarpResult(
            strips=prepared,
            mismatches=mismatches,
            success=False,
            message="接合補正スキップ（マッチ不足）",
        )

    by_id = {s["run_id"]: s for s in prepared}
    warped: List[dict] = []
    for run_id, strip in by_id.items():
        warped.append(
            _remap_run(strip, run_id, mismatches, ppm, theta_min, theta_max, cfg)
        )
    # 元の順を保つ
    order = [normalize_run_id(s["run_id"]) for s in strips]
    by_w = {s["run_id"]: s for s in warped}
    out = [by_w[k] for k in order if k in by_w]
    n_ok = sum(1 for m in mismatches if m.n_used >= 2)
    return SeamWarpResult(
        strips=out,
        mismatches=mismatches,
        success=n_ok > 0,
        message=f"接合補正 seams={n_ok}/{len(mismatches)}",
    )


def _copy_strip(s: dict) -> dict:
    rgb = s["rgb"]
    filled = s["filled"]
    if int(getattr(rgb, "size", 0)) >= 8_000_000:
        rgb_out = rgb
        filled_out = filled
    else:
        rgb_out = np.ascontiguousarray(rgb.copy())
        filled_out = np.ascontiguousarray(filled.copy())
    return {
        "rgb": rgb_out,
        "filled": filled_out,
        "z_min": float(s["z_min"]),
        "z_max": float(s.get("z_max", s["z_min"])),
        "run_id": s["run_id"],
    }


def _cfg_val(cfg, name: str, default):
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def _paste_common(strips: Sequence[dict], ppm: float) -> Dict[str, dict]:
    shapes = [tuple(s["rgb"].shape[:2]) for s in strips]
    z_mins = [float(s["z_min"]) for s in strips]
    if (
        shapes
        and all(sh == shapes[0] for sh in shapes)
        and all(abs(z - z_mins[0]) <= 1e-3 for z in z_mins)
    ):
        return {
            s["run_id"]: {
                "rgb": s["rgb"],
                "filled": s["filled"],
                "z_min": float(s["z_min"]),
                "z_max": float(s.get("z_max", s["z_min"] + s["rgb"].shape[1] / ppm)),
            }
            for s in strips
        }
    z_lo = min(float(s["z_min"]) for s in strips)
    z_hi = max(
        float(s.get("z_max", s["z_min"] + s["rgb"].shape[1] / ppm))
        for s in strips
    )
    h = max(int(s["rgb"].shape[0]) for s in strips)
    w = max(8, int(np.ceil((z_hi - z_lo) * ppm)))
    out: Dict[str, dict] = {}
    for s in strips:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        filled = np.zeros((h, w), dtype=bool)
        x0 = int(round((float(s["z_min"]) - z_lo) * ppm))
        hh, ww = s["rgb"].shape[:2]
        x1 = min(w, x0 + ww)
        y1 = min(h, hh)
        if x1 > max(x0, 0) and y1 > 0:
            xs = max(x0, 0)
            rgb[:y1, xs:x1] = s["rgb"][:y1, xs - x0 : x1 - x0]
            filled[:y1, xs:x1] = s["filled"][:y1, xs - x0 : x1 - x0]
        out[s["run_id"]] = {
            "rgb": rgb,
            "filled": filled,
            "z_min": z_lo,
            "z_max": z_lo + w / ppm,
        }
    return out


def _seam_row(h: int, car_deg: float, theta_min: float, theta_max: float) -> float:
    eta = float(np.asarray(camera_car_deg_to_eta(car_deg)).reshape(-1)[0])
    span = float(theta_max - theta_min)
    t = ((eta - theta_min) / max(span, 1e-12)) % 1.0
    return t * max(h - 1, 1)


def _match_seam_pair(
    sa: dict,
    sb: dict,
    a_id: str,
    b_id: str,
    ppm: float,
    theta_min: float,
    theta_max: float,
    cfg,
) -> SeamMismatch:
    pair = pair_key(a_id, b_id)
    assert pair is not None
    car = OVERLAP_CENTER_CAMERA_CAR_DEG[pair]
    gray_a = sa["rgb"]
    gray_b = sb["rgb"]
    h, w = gray_a.shape[:2]
    row = _seam_row(h, car, theta_min, theta_max)
    half_deg = float(_cfg_val(cfg, "half_band_deg", 8.0))
    half_rows = max(4.0, half_deg / 360.0 * max(h - 1, 1))
    zs: List[float] = []
    mz: List[float] = []
    mth: List[float] = []

    orb_pts = _orb_band_matches(
        gray_a, gray_b, sa["filled"], sb["filled"], row, half_rows
    )
    span = float(theta_max - theta_min)
    z0 = float(sa["z_min"])
    if orb_pts is not None:
        pa, pb = orb_pts
        zs.extend((0.5 * (pa[:, 0] + pb[:, 0]) / ppm + z0).tolist())
        mz.extend(((pa[:, 0] - pb[:, 0]) / ppm).tolist())
        dy = _wrap_row_delta(pa[:, 1] - pb[:, 1], h)
        mth.extend((dy / max(h - 1, 1) * span).tolist())

    ncc = _ncc_band_matches(
        gray_a, gray_b, sa["filled"], sb["filled"],
        row, half_rows, ppm, z0, span, h, cfg,
    )
    zs.extend(ncc[0])
    mz.extend(ncc[1])
    mth.extend(ncc[2])

    z_arr = np.asarray(zs, dtype=float)
    mz_arr = np.asarray(mz, dtype=float)
    mth_arr = np.asarray(mth, dtype=float)
    n_raw = int(z_arr.size)
    keep = np.isfinite(z_arr) & np.isfinite(mz_arr) & np.isfinite(mth_arr)
    max_dz = float(_cfg_val(cfg, "max_dz_mm", 10.0))
    max_dth = np.radians(float(_cfg_val(cfg, "max_dtheta_deg", 4.0)))
    keep &= np.abs(mz_arr) <= max_dz
    keep &= np.abs(mth_arr) <= max_dth
    z_arr, mz_arr, mth_arr = z_arr[keep], mz_arr[keep], mth_arr[keep]
    keep = _iqr_keep(mz_arr) & _iqr_keep(mth_arr)
    z_arr, mz_arr, mth_arr = z_arr[keep], mz_arr[keep], mth_arr[keep]

    min_n = int(_cfg_val(cfg, "min_matches", 6))
    if z_arr.size < max(2, min_n // 2):
        return SeamMismatch(
            pair=pair, z_mm=np.zeros(0), mz_mm=np.zeros(0), mth_rad=np.zeros(0),
            n_raw=n_raw, n_used=int(z_arr.size),
            message="マッチ不足",
        )

    spacing = float(_cfg_val(cfg, "control_spacing_mm", 30.0))
    zc, mzc, mtc = _flatten_pair(z_arr, mz_arr, mth_arr, spacing)
    mzc = np.clip(mzc, -max_dz, max_dz)
    mtc = np.clip(mtc, -max_dth, max_dth)
    return SeamMismatch(
        pair=pair,
        z_mm=zc,
        mz_mm=mzc,
        mth_rad=mtc,
        n_raw=n_raw,
        n_used=int(z_arr.size),
        message="ok",
    )


def _flatten_pair(
    z: np.ndarray, mz: np.ndarray, mth: np.ndarray, spacing: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zc, mzc, mtc = _bin_medians(z, mz, mth, spacing)
    pad = max(float(spacing), 20.0)
    if zc.size < 1:
        return zc, mzc, mtc
    if zc.size == 1:
        return zc, mzc, mtc
    z_pad, mz_pad = _flatten_end_slopes(zc, mzc, pad)
    _z2, mt_pad = _flatten_end_slopes(zc, mtc, pad)
    return z_pad, mz_pad, mt_pad


def _bin_medians(
    z: np.ndarray, mz: np.ndarray, mth: np.ndarray, spacing: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(z)
    z, mz, mth = z[order], mz[order], mth[order]
    if z.size == 1:
        return z.copy(), mz.copy(), mth.copy()
    spacing = max(float(spacing), 1e-3)
    z0, z1 = float(z[0]), float(z[-1])
    edges = np.arange(z0, z1 + spacing * 0.5, spacing)
    if edges.size < 2:
        return (
            np.array([np.median(z)], dtype=float),
            np.array([np.median(mz)], dtype=float),
            np.array([np.median(mth)], dtype=float),
        )
    zs, a, b = [], [], []
    for i in range(edges.size - 1):
        sel = (z >= edges[i]) & (z < edges[i + 1] + (1e-9 if i == edges.size - 2 else 0.0))
        if i == edges.size - 2:
            sel = z >= edges[i]
        if not np.any(sel):
            continue
        zs.append(float(np.median(z[sel])))
        a.append(float(np.median(mz[sel])))
        b.append(float(np.median(mth[sel])))
    if len(zs) < 1:
        return (
            np.array([np.median(z)], dtype=float),
            np.array([np.median(mz)], dtype=float),
            np.array([np.median(mth)], dtype=float),
        )
    zc = np.asarray(zs, dtype=float)
    uniq = np.concatenate([[True], np.diff(zc) > 1e-6]) if zc.size > 1 else np.array([True])
    return zc[uniq], np.asarray(a, dtype=float)[uniq], np.asarray(b, dtype=float)[uniq]


def _iqr_keep(vals: np.ndarray, mult: float = 1.5, min_pts: int = 4) -> np.ndarray:
    if vals.size < min_pts:
        return np.ones(vals.size, dtype=bool)
    q1, q3 = np.percentile(vals, [25.0, 75.0])
    iqr = q3 - q1
    if iqr < 1e-9:
        return np.ones(vals.size, dtype=bool)
    lo, hi = q1 - mult * iqr, q3 + mult * iqr
    return (vals >= lo) & (vals <= hi)


def _to_gray(
    img: np.ndarray,
    y0: int = 0,
    y1: Optional[int] = None,
    x0: int = 0,
    x1: Optional[int] = None,
) -> np.ndarray:
    """切り出し帯だけを灰度化する。全画面 (h, w) の変換はしない。"""
    if y1 is None:
        y1 = int(img.shape[0])
    if x1 is None:
        x1 = int(img.shape[1])
    band = img[y0:y1, x0:x1]
    if band.ndim == 2:
        return band
    if not band.flags["C_CONTIGUOUS"]:
        band = np.ascontiguousarray(band)
    return cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)


def _wrap_row_delta(dy: np.ndarray, h: int) -> np.ndarray:
    """θ 周期の行差。η=0 ラップをまたぐ 180° 接合用。"""
    hh = max(int(h), 1)
    return np.mod(np.asarray(dy, dtype=float) + 0.5 * hh, hh) - 0.5 * hh


def _periodic_band_index(h: int, row: float, half_rows: float, extra: int = 0):
    """接合帯を θ 周期で切り出す行インデックス。180°（η=0）でも L 側下端を含む。"""
    h = max(int(h), 1)
    half = max(int(np.ceil(half_rows)), 4) + max(int(extra), 0)
    center = int(np.round(row)) % h
    rel = np.arange(-half, half + 1, dtype=int)
    idx = np.mod(center + rel, h)
    return idx, rel, center


def _band_slices(h: int, row: float, half_rows: float) -> Tuple[int, int]:
    y0 = max(0, int(np.floor(row - half_rows)))
    y1 = min(h, int(np.ceil(row + half_rows)) + 1)
    if y1 - y0 < 4:
        mid = int(round(row))
        y0 = max(0, mid - 2)
        y1 = min(h, mid + 3)
    return y0, y1


def _orb_band_matches(
    gray_a, gray_b, filled_a, filled_b, row: float, half_rows: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    h, w = gray_a.shape[:2]
    idx, rel, center = _periodic_band_index(h, row, half_rows)
    band_a = np.take(gray_a, idx, axis=0)
    band_b = np.take(gray_b, idx, axis=0)
    mask_full_a = filled_a[idx]
    mask_full_b = filled_b[idx]
    y_base = float(center + rel[0])
    orb = cv2.ORB_create(nfeatures=800)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pts_a: List[np.ndarray] = []
    pts_b: List[np.ndarray] = []
    chunk = 2048
    overlap = 256
    x0 = 0
    while x0 < w:
        x1 = min(w, x0 + chunk)
        ga = _to_gray(band_a, 0, band_a.shape[0], x0, x1)
        gb = _to_gray(band_b, 0, band_b.shape[0], x0, x1)
        mask_a = mask_full_a[:, x0:x1].astype(np.uint8) * 255
        mask_b = mask_full_b[:, x0:x1].astype(np.uint8) * 255
        kp1, des1 = orb.detectAndCompute(ga, mask_a)
        kp2, des2 = orb.detectAndCompute(gb, mask_b)
        if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
            knn = matcher.knnMatch(des1, des2, k=2)
            good = []
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) >= 3:
                pa = np.float32([kp1[m.queryIdx].pt for m in good])
                pb = np.float32([kp2[m.trainIdx].pt for m in good])
                pa[:, 0] += x0
                pb[:, 0] += x0
                pa[:, 1] = np.mod(y_base + pa[:, 1], h)
                pb[:, 1] = np.mod(y_base + pb[:, 1], h)
                pts_a.append(pa)
                pts_b.append(pb)
        nxt = x0 + chunk - overlap
        if nxt <= x0:
            break
        x0 = nxt
    if not pts_a:
        return None
    return np.vstack(pts_a), np.vstack(pts_b)


def _ncc_band_matches(
    gray_a, gray_b, filled_a, filled_b,
    row: float, half_rows: float, ppm: float, z0: float, span: float, h: int, cfg,
) -> Tuple[List[float], List[float], List[float]]:
    hh, w = gray_a.shape[:2]
    win = max(12, int(round(float(_cfg_val(cfg, "ncc_window_mm", 40.0)) * ppm)))
    step = max(4, int(round(float(_cfg_val(cfg, "ncc_step_mm", 15.0)) * ppm)))
    max_dz = float(_cfg_val(cfg, "max_dz_mm", 10.0))
    max_dth = np.radians(float(_cfg_val(cfg, "max_dtheta_deg", 4.0)))
    sx = max(2, int(np.ceil(max_dz * ppm)))
    sy = max(1, int(np.ceil(max_dth / max(span, 1e-12) * max(h - 1, 1))))
    zs, mz, mth = [], [], []
    idx_t, rel_t, _center = _periodic_band_index(hh, row, half_rows)
    idx_s, rel_s, _c2 = _periodic_band_index(hh, row, half_rows, extra=sy)
    off = int(rel_t[0] - rel_s[0])
    templ_h = int(idx_t.size)
    if templ_h < 4 or w < win + 2 * sx or off < 0:
        return zs, mz, mth
    band_a = np.take(gray_a, idx_t, axis=0)
    band_b = np.take(gray_b, idx_s, axis=0)
    fill_a = filled_a[idx_t]
    fill_b = filled_b[idx_s]
    for x in range(sx, w - win - sx, step):
        ta = _to_gray(band_a, 0, templ_h, x, x + win).astype(np.float32)
        ma = fill_a[:, x : x + win]
        if float(np.mean(ma)) < 0.6 or float(ta.std()) < 4.0:
            continue
        xs0 = x - sx
        xs1 = x + win + sx
        search = _to_gray(band_b, 0, band_b.shape[0], xs0, xs1).astype(np.float32)
        mb = fill_b[:, xs0:xs1]
        if float(np.mean(mb)) < 0.5:
            continue
        if search.shape[0] < ta.shape[0] or search.shape[1] < ta.shape[1]:
            continue
        res = cv2.matchTemplate(search, ta, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(res)
        if peak < 0.35:
            continue
        px, py = loc
        dx = float((xs0 + px) - x)
        dy = float(py - off)
        zs.append(z0 + (x + 0.5 * win) / ppm)
        mz.append(-dx / ppm)
        mth.append(-dy / max(h - 1, 1) * span)
    return zs, mz, mth


def eval_mismatch(mm: SeamMismatch, zq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    zq = np.asarray(zq, dtype=float)
    if mm.z_mm.size == 0:
        return np.zeros_like(zq), np.zeros_like(zq)
    if mm.z_mm.size == 1:
        return np.full_like(zq, float(mm.mz_mm[0])), np.full_like(zq, float(mm.mth_rad[0]))
    return (
        _eval_pchip_hold(mm.z_mm, mm.mz_mm, zq),
        _eval_pchip_hold(mm.z_mm, mm.mth_rad, zq),
    )


def boundary_displacement(
    run_id: str,
    z_mm: np.ndarray,
    mismatches: Sequence[SeamMismatch],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """run の lo/hi 接合での u=(δz[mm], δθ[rad])。相手無しは 0。"""
    key = normalize_run_id(run_id)
    lo_p, hi_p = RUN_BOUNDARY_PARTNERS[key]
    z_mm = np.asarray(z_mm, dtype=float)
    u_lo_z, u_lo_th = _u_against(key, lo_p, z_mm, mismatches)
    u_hi_z, u_hi_th = _u_against(key, hi_p, z_mm, mismatches)
    return u_lo_z, u_lo_th, u_hi_z, u_hi_th


def _u_against(
    run_id: str,
    partner: str,
    z_mm: np.ndarray,
    mismatches: Sequence[SeamMismatch],
) -> Tuple[np.ndarray, np.ndarray]:
    pk = pair_key(run_id, partner)
    if pk is None:
        return np.zeros_like(z_mm), np.zeros_like(z_mm)
    found = None
    for m in mismatches:
        if m.pair == pk and m.z_mm.size:
            found = m
            break
    if found is None:
        return np.zeros_like(z_mm), np.zeros_like(z_mm)
    mz, mth = eval_mismatch(found, z_mm)
    # m = p_first - p_second。first は −m/2、second は +m/2
    if normalize_run_id(run_id) == pk[0]:
        return -0.5 * mz, -0.5 * mth
    return 0.5 * mz, 0.5 * mth


def _remap_run(
    strip: dict,
    run_id: str,
    mismatches: Sequence[SeamMismatch],
    ppm: float,
    theta_min: float,
    theta_max: float,
    cfg,
) -> dict:
    rgb = strip["rgb"]
    filled = strip["filled"]
    h, w = rgb.shape[:2]
    z_min = float(strip["z_min"])
    xs = np.arange(w, dtype=float)
    ys = np.arange(h, dtype=float)
    z_cols = z_min + xs / ppm
    u_lo_z, u_lo_th, u_hi_z, u_hi_th = boundary_displacement(run_id, z_cols, mismatches)
    span = float(theta_max - theta_min)
    eta = theta_min + ys / max(h, 1) * span
    car = eta_to_camera_car_deg(eta)
    wt = np.asarray(sector_blend_weight(car, run_id), dtype=float).reshape(-1, 1)
    th_scale = h / max(span, 1e-12)
    if rgb.ndim == 3:
        rgb_o = np.zeros((h, w, rgb.shape[2]), dtype=rgb.dtype)
    else:
        rgb_o = np.zeros((h, w), dtype=rgb.dtype)
    filled_o = np.zeros((h, w), dtype=bool)
    chunk = 512
    for x0 in range(0, w, chunk):
        x1 = min(w, x0 + chunk)
        uz = (1.0 - wt) * u_lo_z[x0:x1].reshape(1, -1) + wt * u_hi_z[x0:x1].reshape(1, -1)
        uth = (1.0 - wt) * u_lo_th[x0:x1].reshape(1, -1) + wt * u_hi_th[x0:x1].reshape(1, -1)
        map_x = (xs[x0:x1].reshape(1, -1) - uz * ppm).astype(np.float32)
        map_y = (ys.reshape(-1, 1) - uth * th_scale).astype(np.float32)
        chunk_rgb, chunk_filled = remap_with_periodic_theta(rgb, filled, map_x, map_y)
        rgb_o[:, x0:x1] = chunk_rgb
        filled_o[:, x0:x1] = chunk_filled
        del uz, uth, map_x, map_y, chunk_rgb, chunk_filled
    return {
        "rgb": rgb_o,
        "filled": filled_o,
        "z_min": z_min,
        "z_max": z_min + w / ppm,
        "run_id": run_id,
    }
