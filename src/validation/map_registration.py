"""2Run 仮展開図の重複領域登録と共同 z 補正"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


@dataclass
class RegistrationResult:
    success: bool
    dz: float
    dtheta: float
    inliers: int
    raw_matches: int
    inlier_ratio: float
    residual_before: float
    residual_after: float
    z_ctrl: Optional[np.ndarray]
    delta_a: Optional[np.ndarray]
    delta_b: Optional[np.ndarray]
    message: str = ""

    def apply_z(self, z: np.ndarray, run: str) -> np.ndarray:
        if self.z_ctrl is None:
            shift = self.dz if run.upper().endswith("B") or run == "B" else 0.0
            return z + shift
        delta = self.delta_b if run.upper().endswith("B") or run in ("B", "R", "L") else self.delta_a
        if delta is None:
            return z
        interp = PchipInterpolator(self.z_ctrl, delta, extrapolate=True)
        return z + interp(z)

    def apply_theta(self, theta: np.ndarray, run: str) -> np.ndarray:
        if run.upper().endswith("B") or run in ("B", "R", "L"):
            return (theta + self.dtheta) % (2.0 * np.pi)
        return theta


class MapRegistrar:
    def __init__(self, config):
        self.cfg = config.two_direction.registration

    def register(
        self,
        map_a: np.ndarray,
        map_b: np.ndarray,
        filled_a: np.ndarray,
        filled_b: np.ndarray,
        z_min: float,
        pixels_per_mm: float,
        theta_min: float,
        theta_max: float,
        z_ocr_a: Optional[np.ndarray] = None,
        z_ocr_b: Optional[np.ndarray] = None,
        z_est_a: Optional[np.ndarray] = None,
        z_est_b: Optional[np.ndarray] = None,
    ) -> RegistrationResult:
        gray_a = _to_gray(map_a)
        gray_b = _to_gray(map_b)
        if filled_a.shape != filled_b.shape:
            return RegistrationResult(
                success=False, dz=0.0, dtheta=0.0, inliers=0, raw_matches=0,
                inlier_ratio=0.0, residual_before=np.nan, residual_after=np.nan,
                z_ctrl=None, delta_a=None, delta_b=None,
                message=f"部分図サイズ不一致 {filled_a.shape} vs {filled_b.shape}",
            )
        overlap = filled_a & filled_b
        if np.sum(overlap) < self.cfg.min_inliers:
            # 重複が薄い場合でも特徴点で試す
            pass
        pts_a, pts_b = _match_maps(gray_a, gray_b, filled_a, filled_b)
        raw = 0 if pts_a is None else len(pts_a)
        if pts_a is None or raw < max(3, self.cfg.min_inliers // 2):
            return RegistrationResult(
                success=False, dz=0.0, dtheta=0.0, inliers=0, raw_matches=raw,
                inlier_ratio=0.0, residual_before=np.nan, residual_after=np.nan,
                z_ctrl=None, delta_a=None, delta_b=None,
                message="RANSAC inlier不足",
            )

        za, ta = _pixel_to_z_theta(pts_a, z_min, pixels_per_mm, theta_min, theta_max, map_a.shape)
        zb, tb = _pixel_to_z_theta(pts_b, z_min, pixels_per_mm, theta_min, theta_max, map_b.shape)

        dz, dtheta, inlier_mask, res_before = self._ransac_global(za, ta, zb, tb)
        inliers = int(np.sum(inlier_mask))
        if inliers < self.cfg.min_inliers:
            return RegistrationResult(
                success=False, dz=float(dz), dtheta=float(dtheta), inliers=inliers,
                raw_matches=raw, inlier_ratio=inliers / max(raw, 1),
                residual_before=float(res_before), residual_after=float(res_before),
                z_ctrl=None, delta_a=None, delta_b=None,
                message="rigid のみ（inlier不足のため非線形補正スキップ）",
            )

        za_i, zb_i = za[inlier_mask], zb[inlier_mask]
        z_ctrl, da, db, res_after = self._joint_z_correction(
            za_i, zb_i, dz, z_ocr_a, z_ocr_b, z_est_a, z_est_b
        )
        return RegistrationResult(
            success=True,
            dz=float(dz),
            dtheta=float(dtheta),
            inliers=inliers,
            raw_matches=raw,
            inlier_ratio=inliers / max(raw, 1),
            residual_before=float(res_before),
            residual_after=float(res_after),
            z_ctrl=z_ctrl,
            delta_a=da,
            delta_b=db,
            message="ok",
        )

    def _ransac_global(
        self,
        za: np.ndarray,
        ta: np.ndarray,
        zb: np.ndarray,
        tb: np.ndarray,
    ) -> Tuple[float, float, np.ndarray, float]:
        n = len(za)
        rng = np.random.default_rng(0)
        best_in = np.zeros(n, dtype=bool)
        best_dz = 0.0
        best_dth = 0.0
        thr = self.cfg.ransac_inlier_threshold_mm
        thr_th = np.radians(self.cfg.theta_search_margin_deg)
        for _ in range(self.cfg.ransac_iterations):
            i = int(rng.integers(0, n))
            dz = za[i] - zb[i]
            dth = _wrap_pi(ta[i] - tb[i])
            err_z = np.abs((zb + dz) - za)
            err_th = np.abs(_wrap_pi((tb + dth) - ta)) * (self.cfg.z_search_margin_mm / max(thr_th, 1e-6))
            inl = (err_z < thr) & (err_th < thr)
            if np.sum(inl) > np.sum(best_in):
                best_in = inl
                best_dz, best_dth = dz, dth
        if np.any(best_in):
            best_dz = float(np.median(za[best_in] - zb[best_in]))
            best_dth = float(np.median(_wrap_pi(ta[best_in] - tb[best_in])))
        res = float(np.mean(np.abs((zb + best_dz) - za))) if n else np.nan
        return best_dz, best_dth, best_in, res

    def _joint_z_correction(
        self,
        za: np.ndarray,
        zb: np.ndarray,
        dz_rigid: float,
        z_ocr_a,
        z_ocr_b,
        z_est_a,
        z_est_b,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        z_lo = float(min(za.min(), zb.min()) - 1.0)
        z_hi = float(max(za.max(), zb.max()) + 1.0)
        spacing = max(20.0, self.cfg.control_point_spacing_mm)
        n_ctrl = max(3, int(np.ceil((z_hi - z_lo) / spacing)) + 1)
        z_ctrl = np.linspace(z_lo, z_hi, n_ctrl)
        x0 = np.concatenate([np.zeros(n_ctrl), np.full(n_ctrl, dz_rigid)])

        def residuals(x):
            da = x[:n_ctrl]
            db = x[n_ctrl:]
            fa = PchipInterpolator(z_ctrl, da, extrapolate=True)
            fb = PchipInterpolator(z_ctrl, db, extrapolate=True)
            e_match = (za + fa(za)) - (zb + fb(zb))
            e_smooth = np.diff(da, n=2).tolist() + np.diff(db, n=2).tolist()
            e_ocr = []
            if z_ocr_a is not None and z_est_a is not None and len(z_ocr_a) == len(z_est_a):
                e_ocr.extend(list(self.cfg.lambda_ocr * ((z_est_a + fa(z_est_a)) - z_ocr_a)))
            if z_ocr_b is not None and z_est_b is not None and len(z_ocr_b) == len(z_est_b):
                e_ocr.extend(list(self.cfg.lambda_ocr * ((z_est_b + fb(z_est_b)) - z_ocr_b)))
            return np.concatenate([
                e_match,
                self.cfg.lambda_smooth * np.array(e_smooth, dtype=float),
                np.array(e_ocr, dtype=float) if e_ocr else np.zeros(1),
            ])

        result = least_squares(residuals, x0, loss="soft_l1")
        da = np.clip(result.x[:n_ctrl], -self.cfg.max_correction_mm, self.cfg.max_correction_mm)
        db = np.clip(result.x[n_ctrl:], -self.cfg.max_correction_mm, self.cfg.max_correction_mm)
        if np.any(np.diff(z_ctrl + da) <= 0) or np.any(np.diff(z_ctrl + db) <= 0):
            da = np.zeros_like(da)
            db = np.full_like(db, dz_rigid)
        fa = PchipInterpolator(z_ctrl, da, extrapolate=True)
        fb = PchipInterpolator(z_ctrl, db, extrapolate=True)
        res_after = float(np.mean(np.abs((za + fa(za)) - (zb + fb(zb)))))
        return z_ctrl, da, db, res_after


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _match_maps(gray_a, gray_b, filled_a, filled_b):
    orb = cv2.ORB_create(nfeatures=2000)
    mask_a = filled_a.astype(np.uint8) * 255
    mask_b = filled_b.astype(np.uint8) * 255
    kp1, des1 = orb.detectAndCompute(gray_a, mask_a)
    kp2, des2 = orb.detectAndCompute(gray_b, mask_b)
    if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) >= 3:
            pts_a = np.float32([kp1[m.queryIdx].pt for m in good])
            pts_b = np.float32([kp2[m.trainIdx].pt for m in good])
            return pts_a, pts_b
    return _match_maps_phase(gray_a, gray_b, filled_a, filled_b)


def _match_maps_phase(gray_a, gray_b, filled_a, filled_b):
    """ORBが効かない平滑画像向け。位相相関で並進を推定し格子対応点を作る。"""
    overlap = filled_a & filled_b
    if int(np.sum(overlap)) < 16:
        return None, None
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    a = np.where(overlap, a, 0.0)
    b = np.where(overlap, b, 0.0)
    (dx, dy), _resp = cv2.phaseCorrelate(a, b)
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return None, None
    h, w = gray_a.shape[:2]
    step = 8
    yy, xx = np.mgrid[4:h - 4:step, 4:w - 4:step]
    pts_a = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    pts_b = pts_a + np.array([dx, dy], dtype=np.float32)
    ia = np.round(pts_a).astype(int)
    ib = np.round(pts_b).astype(int)
    keep = (
        (ia[:, 0] >= 0) & (ia[:, 0] < w) & (ia[:, 1] >= 0) & (ia[:, 1] < h)
        & (ib[:, 0] >= 0) & (ib[:, 0] < w) & (ib[:, 1] >= 0) & (ib[:, 1] < h)
    )
    if not np.any(keep):
        return None, None
    ia, ib = ia[keep], ib[keep]
    pts_a, pts_b = pts_a[keep], pts_b[keep]
    filled = filled_a[ia[:, 1], ia[:, 0]] & filled_b[ib[:, 1], ib[:, 0]]
    if int(np.sum(filled)) < 3:
        return None, None
    return pts_a[filled], pts_b[filled]


def _pixel_to_z_theta(pts, z_min, ppm, theta_min, theta_max, shape):
    # pts: (x, y) = (z-col, theta-row)
    x, y = pts[:, 0], pts[:, 1]
    h, w = shape[:2]
    z = z_min + x / max(ppm, 1e-6)
    theta = theta_min + y / max(h - 1, 1) * (theta_max - theta_min)
    return z, theta


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi
