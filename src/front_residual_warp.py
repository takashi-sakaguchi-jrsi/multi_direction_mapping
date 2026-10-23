"""直視タイル貼り付け時の残差 (η, z) 微調整。

3DoF（dz, Δyaw, Δpitch）で説明したあとの特徴点残差を粗いビンの中央値にし、
新しい帯のサンプリンググリッドだけずらす。カメラ姿勢には積まない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# オリジナル展開: η = matcher_θ + π/2
COLORMAP_ETA_OFFSET_RAD = 0.5 * np.pi


def _wrap_deg(delta) -> np.ndarray:
    return (np.asarray(delta, dtype=float) + 180.0) % 360.0 - 180.0


def _wrap_rad(delta) -> np.ndarray:
    return (np.asarray(delta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _wrap_rad_0_2pi(theta) -> np.ndarray:
    return np.asarray(theta, dtype=float) % (2.0 * np.pi)


def point_residuals(
    prev_cyl: np.ndarray,
    curr_cyl: np.ndarray,
    dz: float,
    dyaw: float,
    dpitch: float,
    pipe_radius_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """3DoF 予測を引いた点ごと残差。

    Returns:
        theta_rad: 前フレームの管 θ [rad]
        z_mm: 前フレームの管 Z [mm]
        r_theta_rad: Δθ 残差 [rad]
        r_z_mm: ΔZ 残差 [mm]
    """
    prev_cyl = np.asarray(prev_cyl, dtype=float)
    curr_cyl = np.asarray(curr_cyl, dtype=float)
    if prev_cyl.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, empty
    theta = np.deg2rad(prev_cyl[:, 0])
    z = np.maximum(prev_cyl[:, 1], 1e-3)
    radius = max(float(pipe_radius_mm), 1e-3)
    c = np.cos(theta)
    s = np.sin(theta)
    kZ = radius + (z * z) / radius
    kth = z / radius
    dZ_obs = curr_cyl[:, 1] - prev_cyl[:, 1]
    dth_obs = np.deg2rad(_wrap_deg(curr_cyl[:, 0] - prev_cyl[:, 0]))
    dZ_pred = -float(dz) - kZ * (float(dyaw) * c + float(dpitch) * s)
    dth_pred = kth * (float(dpitch) * c - float(dyaw) * s)
    r_z = dZ_obs - dZ_pred
    r_th = _wrap_rad(dth_obs - dth_pred)
    return theta, z, r_th, r_z


def _smooth_bin_grid(grid: np.ndarray) -> np.ndarray:
    """θ 周期・Z は端複製の 3 点平滑。1 回だけ。"""
    g = np.asarray(grid, dtype=float)
    if g.size == 0:
        return g
    nz, nt = g.shape
    if nt >= 2:
        padded = np.pad(g, ((0, 0), (1, 1)), mode="wrap")
        g = 0.25 * padded[:, 0:-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
    if nz >= 2:
        padded_z = np.pad(g, ((1, 1), (0, 0)), mode="edge")
        g = 0.25 * padded_z[0:-2] + 0.5 * padded_z[1:-1] + 0.25 * padded_z[2:]
    return g


@dataclass
class ResidualWarpField:
    """ビン中央の (dη, dz)。空ビンは 0。"""

    theta_centers_rad: np.ndarray
    z_centers_mm: np.ndarray
    d_eta_rad: np.ndarray
    d_z_mm: np.ndarray
    eta_offset_rad: float = COLORMAP_ETA_OFFSET_RAD

    def deltas_on_grid(
        self,
        eta_grid: np.ndarray,
        z_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """展開図グリッド上の (dη [rad], dz [mm])。範囲外は 0。"""
        eta = np.asarray(eta_grid, dtype=float)
        z = np.asarray(z_grid, dtype=float)
        d_eta = np.zeros(eta.shape, dtype=float)
        d_z = np.zeros(z.shape, dtype=float)
        th = self.theta_centers_rad
        zz = self.z_centers_mm
        if th.size < 1 or zz.size < 1:
            return d_eta, d_z
        if not np.any(np.abs(self.d_eta_rad) > 1e-12) and not np.any(
            np.abs(self.d_z_mm) > 1e-12
        ):
            return d_eta, d_z
        theta_q = _wrap_rad_0_2pi(eta - float(self.eta_offset_rad))
        th_ext, d_eta_ext, d_z_ext = self._periodic_theta_tables()
        if zz.size == 1:
            zz = np.array([zz[0] - 0.5, zz[0] + 0.5], dtype=float)
            d_eta_ext = np.vstack([d_eta_ext, d_eta_ext])
            d_z_ext = np.vstack([d_z_ext, d_z_ext])
        kwargs = dict(bounds_error=False, fill_value=0.0)
        interp_eta = RegularGridInterpolator(
            (zz, th_ext), d_eta_ext, **kwargs
        )
        interp_z = RegularGridInterpolator((zz, th_ext), d_z_ext, **kwargs)
        pts = np.column_stack([z.ravel(), theta_q.ravel()])
        d_eta = interp_eta(pts).reshape(eta.shape)
        d_z = interp_z(pts).reshape(z.shape)
        return d_eta, d_z

    def _periodic_theta_tables(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        th = np.asarray(self.theta_centers_rad, dtype=float)
        d_eta = np.asarray(self.d_eta_rad, dtype=float)
        d_z = np.asarray(self.d_z_mm, dtype=float)
        two_pi = 2.0 * np.pi
        if th.size == 1:
            th_ext = np.array([th[0] - two_pi, th[0], th[0] + two_pi], dtype=float)
            d_eta_ext = np.concatenate([d_eta, d_eta, d_eta], axis=1)
            d_z_ext = np.concatenate([d_z, d_z, d_z], axis=1)
            return th_ext, d_eta_ext, d_z_ext
        th_ext = np.concatenate([th[-1:] - two_pi, th, th[:1] + two_pi])
        d_eta_ext = np.concatenate([d_eta[:, -1:], d_eta, d_eta[:, :1]], axis=1)
        d_z_ext = np.concatenate([d_z[:, -1:], d_z, d_z[:, :1]], axis=1)
        return th_ext, d_eta_ext, d_z_ext


def apply_residual_to_grids(
    eta_grid: np.ndarray,
    z_grid: np.ndarray,
    field: Optional[ResidualWarpField],
) -> Tuple[np.ndarray, np.ndarray]:
    """サンプリング用 (η, z)。field が無ければ入力のまま。"""
    if field is None:
        return eta_grid, z_grid
    d_eta, d_z = field.deltas_on_grid(eta_grid, z_grid)
    return eta_grid + d_eta, z_grid + d_z


def build_residual_warp_field(
    prev_cyl: np.ndarray,
    curr_cyl: np.ndarray,
    *,
    dz: float,
    dyaw: float,
    dpitch: float,
    pipe_radius_mm: float,
    theta_bins: int = 12,
    z_bins: int = 4,
    max_dtheta_deg: float = 2.0,
    max_dz_mm: float = 8.0,
    min_bin_points: int = 2,
    eta_offset_rad: float = COLORMAP_ETA_OFFSET_RAD,
) -> Optional[ResidualWarpField]:
    """クリップ済み点のビン中央値から残差フィールドを作る。不足なら None。"""
    if prev_cyl is None or curr_cyl is None:
        return None
    prev_cyl = np.asarray(prev_cyl, dtype=float)
    curr_cyl = np.asarray(curr_cyl, dtype=float)
    if prev_cyl.ndim != 2 or curr_cyl.ndim != 2 or len(prev_cyl) < 2:
        return None
    n_th = max(int(theta_bins), 1)
    n_z = max(int(z_bins), 1)
    min_n = max(int(min_bin_points), 1)
    theta, z, r_th, r_z = point_residuals(
        prev_cyl, curr_cyl, dz, dyaw, dpitch, pipe_radius_mm
    )
    keep = (
        np.isfinite(theta)
        & np.isfinite(z)
        & np.isfinite(r_th)
        & np.isfinite(r_z)
        & (np.abs(np.degrees(r_th)) <= abs(float(max_dtheta_deg)))
        & (np.abs(r_z) <= abs(float(max_dz_mm)))
    )
    theta, z, r_th, r_z = theta[keep], z[keep], r_th[keep], r_z[keep]
    if theta.size < min_n:
        return None
    theta_deg = np.rad2deg(_wrap_rad_0_2pi(theta))
    th_edges = np.linspace(0.0, 360.0, n_th + 1)
    z0 = float(np.min(z))
    z1 = float(np.max(z))
    if z1 - z0 < 1.0:
        z_edges = np.array([z0 - 0.5, z1 + 0.5], dtype=float)
        n_z_use = 1
    else:
        z_edges = np.linspace(z0, z1, n_z + 1)
        n_z_use = n_z
    d_eta = np.zeros((n_z_use, n_th), dtype=float)
    d_z = np.zeros((n_z_use, n_th), dtype=float)
    th_idx = np.clip(np.digitize(theta_deg, th_edges) - 1, 0, n_th - 1)
    z_idx = np.clip(np.digitize(z, z_edges) - 1, 0, n_z_use - 1)
    for iz in range(n_z_use):
        for it in range(n_th):
            sel = (z_idx == iz) & (th_idx == it)
            if int(np.sum(sel)) < min_n:
                continue
            d_eta[iz, it] = float(np.median(r_th[sel]))
            d_z[iz, it] = float(np.median(r_z[sel]))
    if not np.any(np.abs(d_eta) > 1e-12) and not np.any(np.abs(d_z) > 1e-12):
        return None
    d_eta = _smooth_bin_grid(d_eta)
    d_z = _smooth_bin_grid(d_z)
    th_centers = np.deg2rad(0.5 * (th_edges[:-1] + th_edges[1:]))
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    return ResidualWarpField(
        theta_centers_rad=th_centers,
        z_centers_mm=z_centers,
        d_eta_rad=d_eta,
        d_z_mm=d_z,
        eta_offset_rad=float(eta_offset_rad),
    )
