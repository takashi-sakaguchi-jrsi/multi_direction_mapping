"""展開図格子 → フレーム（逆投影）＋双線形補間。ColorMapGenerator と同じ手法。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.validation.geometry import pose_R_c2w, world_to_eta
from src.validation.records import FrameProjectionSummary


class SideviewProjectionMapper:
    """ColorMapGenerator と同じ逆投影で帯を復元する。

    展開図の各 (η, z) を円筒面上の点にし、world_to_pixel でフレーム座標へ戻し、
    ``cv2.remap(..., INTER_LINEAR)`` で周囲画素を双線形補間する。
    真横撮影では画像中心が最近壁なので、正面用の内側ドーナツ切欠きは使わない。
    """

    def __init__(self, config, transformer):
        self.config = config
        self.transformer = transformer
        self.pipe_radius = float(transformer.pipe_radius)
        self.allow_signed_z = config.two_direction.projection.allow_signed_relative_z
        self.sample_stride = max(1, int(config.two_direction.projection.sample_stride))
        cap = config.two_direction.capture
        self.outer_ratio = cap.usable_outer_radius_ratio
        self.roi_rect = cap.roi_rect

    def default_roi_rect(self, width: int, height: int) -> List[int]:
        if self.roi_rect is not None:
            return list(self.roi_rect)
        margin_x = int(width * 0.08)
        margin_y = int(height * 0.08)
        return [margin_x, margin_y, width - margin_x, height - margin_y]

    def max_radius_px(self, width: int, height: int) -> float:
        return min(width, height) / 2.0 * self.outer_ratio

    def generate_eta_z_grid(
        self,
        z_range: Tuple[float, float],
        colormap_shape: Tuple[int, int],
        theta_min: float = 0.0,
        theta_max: float = 2.0 * np.pi,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """η-z 格子。行は world_to_eta（row0 = η=0）、列は z。"""
        z_min, z_max = z_range
        height, width = colormap_shape
        z_vals = np.linspace(z_min, z_max, num=max(int(width), 1))
        eta_vals = np.linspace(theta_min, theta_max, num=max(int(height), 1), endpoint=False)
        return np.meshgrid(eta_vals, z_vals, indexing="ij")

    def eta_z_to_world(self, eta_grid: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
        """world_to_eta の逆。η=0 は −Y。"""
        x = self.pipe_radius * np.sin(eta_grid)
        y = -self.pipe_radius * np.cos(eta_grid)
        return np.stack((x, y, z_grid), axis=-1)

    def calculate_dynamic_z_range(
        self,
        frame_shape: Tuple[int, int],
        camera_position: np.ndarray,
        orientation: np.ndarray,
        max_dz: float,
        alpha: float,
    ) -> Tuple[float, float]:
        """カメラ z を相対 0 とし、帯は [0, max_dz * alpha] mm。"""
        del frame_shape, orientation
        z_cam = float(np.asarray(camera_position, dtype=float).reshape(-1)[2])
        span = max(0.0, float(max_dz) * float(alpha))
        if span <= 0.0:
            span = max(float(max_dz), 1e-6)
        return z_cam, z_cam + span

    def apply_z_length_limit(self, z_min: float, z_max: float) -> Tuple[float, float]:
        cmap = self.config.colormap
        if not cmap.enable_z_length_limit:
            return z_min, z_max
        ppm = max(float(cmap.pixels_per_mm), 1e-9)
        if cmap.max_z_length_per_frame is not None:
            max_z_px = float(cmap.max_z_length_per_frame) * ppm
        else:
            max_z_px = float(cmap.max_z_pixels_per_frame)
        z_len_px = (z_max - z_min) * ppm
        if z_len_px <= max_z_px:
            return z_min, z_max
        return z_min, z_min + max_z_px / ppm

    def sample_colors_from_frame(
        self,
        frame: np.ndarray,
        eta_grid: np.ndarray,
        z_grid: np.ndarray,
        camera_position: np.ndarray,
        orientation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """展開図格子の各点をフレームへ戻し、双線形補間で色を取る。"""
        height, width = eta_grid.shape
        fh, fw = frame.shape[:2]
        world = self.eta_z_to_world(eta_grid, z_grid).reshape(-1, 3)
        cam_pos = np.asarray(camera_position, dtype=float).reshape(3)
        roll, yaw, pitch = [float(v) for v in orientation[:3]]
        pixels = self.transformer.world_to_pixel(
            world, cam_pos, roll, yaw, pitch, apply_distortion=True
        )
        map_x = pixels[:, 0].astype(np.float32)
        map_y = pixels[:, 1].astype(np.float32)
        map_x_2d = np.nan_to_num(
            map_x.reshape(height, width), nan=-1.0, posinf=-1.0, neginf=-1.0
        )
        map_y_2d = np.nan_to_num(
            map_y.reshape(height, width), nan=-1.0, posinf=-1.0, neginf=-1.0
        )
        sampled = cv2.remap(
            frame,
            map_x_2d,
            map_y_2d,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        if sampled.ndim == 2:
            colors = cv2.cvtColor(sampled, cv2.COLOR_GRAY2RGB)
        else:
            colors = cv2.cvtColor(sampled, cv2.COLOR_BGR2RGB)

        camera = self.transformer.camera
        r_sq = (map_x - camera.cx) ** 2 + (map_y - camera.cy) ** 2
        max_r = self.max_radius_px(fw, fh)
        R = pose_R_c2w(roll, yaw, pitch)
        pts_cam = (world - cam_pos.reshape(1, 3)) @ R.T
        in_front = pts_cam[:, 2] > 1e-6
        in_disk = r_sq <= max_r ** 2
        in_img = (
            (map_x >= 0.0) & (map_x <= fw - 1.0)
            & (map_y >= 0.0) & (map_y <= fh - 1.0)
        )
        valid = in_front & in_disk & in_img & np.isfinite(map_x) & np.isfinite(map_y)
        if not self.allow_signed_z:
            valid = valid & (world[:, 2] >= cam_pos[2])
        return colors, valid.reshape(height, width)

    def sample_strip(
        self,
        frame: np.ndarray,
        camera_position: np.ndarray,
        orientation: np.ndarray,
        theta_bins: int,
        pixels_per_mm: float,
        max_dz: float,
        alpha: float,
        theta_min: float = 0.0,
        theta_max: float = 2.0 * np.pi,
        run_id: str = "",
        frame_num: int = 0,
    ) -> Dict[str, np.ndarray]:
        """1フレーム分の帯を逆投影する。z はカメラ位置から max_dz*alpha mm。"""
        z_min, z_max = self.calculate_dynamic_z_range(
            frame.shape[:2], camera_position, orientation, max_dz, alpha
        )
        z_min, z_max = self.apply_z_length_limit(z_min, z_max)
        ppm = max(float(pixels_per_mm), 1e-9)
        width = max(1, int(np.ceil((z_max - z_min) * ppm)))
        height = max(int(theta_bins), 1)
        eta_grid, z_grid = self.generate_eta_z_grid(
            (z_min, z_max), (height, width), theta_min, theta_max
        )
        colors, mask = self.sample_colors_from_frame(
            frame, eta_grid, z_grid, camera_position, orientation
        )
        return {
            "colors_2d": colors,
            "mask_2d": mask,
            "z_min_strip": np.float64(z_min),
            "z_max_strip": np.float64(z_max),
            "eta_grid": eta_grid,
            "z_grid": z_grid,
            "run_id": run_id,
            "frame_num": int(frame_num),
        }

    def project_frame(
        self,
        frame: np.ndarray,
        camera_position: np.ndarray,
        orientation: np.ndarray,
        run_id: str = "",
        frame_num: int = 0,
        roi_rect: Optional[List[int]] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """ROI 内画素を円筒へ順投影する（幾何テスト用）。蓄積には使わない。"""
        h, w = frame.shape[:2]
        if roi_rect is None:
            roi_rect = self.default_roi_rect(w, h)
        x0, y0, x1, y1 = [int(v) for v in roi_rect]
        stride = self.sample_stride
        us = np.arange(x0, x1, stride, dtype=np.float64)
        vs = np.arange(y0, y1, stride, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)
        pts = np.column_stack((uu.ravel(), vv.ravel()))

        camera = self.transformer.camera
        cx, cy, f = camera.cx, camera.cy, camera.f
        dx = pts[:, 0] - cx
        dy = -(pts[:, 1] - cy)
        r = np.hypot(dx, dy) + 1e-12
        gamma = r / f
        max_r = self.max_radius_px(w, h)
        in_circle = r <= max_r
        if valid_mask is not None:
            ui = np.clip(pts[:, 0].astype(int), 0, w - 1)
            vi = np.clip(pts[:, 1].astype(int), 0, h - 1)
            in_circle = in_circle & (valid_mask[vi, ui] > 0)

        p_wall, t_hit, hit_ok = self._forward_wall_from_pixels(
            pts, camera_position, orientation
        )
        d_wall = np.linalg.norm(p_wall - camera_position.reshape(1, 3), axis=1)
        z = p_wall[:, 2]
        theta = world_to_eta(p_wall[:, 0], p_wall[:, 1])
        z_cam = float(camera_position[2])
        if not self.allow_signed_z:
            hit_ok = hit_ok & (z >= z_cam)

        valid = in_circle & hit_ok & np.isfinite(d_wall) & (d_wall > 1e-6)

        ui = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        vi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        bgr = frame[vi, ui]
        colors = bgr[:, ::-1].copy()

        return {
            "u": pts[:, 0],
            "v": pts[:, 1],
            "z": z,
            "theta": theta,
            "d_wall": d_wall,
            "gamma": gamma,
            "valid": valid,
            "colors": colors,
            "t_hit": t_hit,
            "p_wall": p_wall,
            "roi_rect": np.array(roi_rect, dtype=np.int32),
            "run_id": run_id,
            "frame_num": frame_num,
        }

    def summarize(self, proj: Dict[str, np.ndarray], run_id: str, frame_num: int) -> FrameProjectionSummary:
        if "mask_2d" in proj:
            mask = proj["mask_2d"]
            if not np.any(mask):
                return FrameProjectionSummary(
                    run_id=run_id, frame_num=frame_num,
                    z_min=np.nan, z_max=np.nan, theta_min=np.nan, theta_max=np.nan,
                    valid_ratio=0.0, d_wall_mean=np.nan, d_wall_min=np.nan, gamma_mean=np.nan,
                )
            z = proj["z_grid"][mask]
            th = proj["eta_grid"][mask]
            return FrameProjectionSummary(
                run_id=run_id,
                frame_num=frame_num,
                z_min=float(z.min()),
                z_max=float(z.max()),
                theta_min=float(th.min()),
                theta_max=float(th.max()),
                valid_ratio=float(np.mean(mask)),
                d_wall_mean=np.nan,
                d_wall_min=np.nan,
                gamma_mean=np.nan,
            )
        valid = proj["valid"]
        if not np.any(valid):
            return FrameProjectionSummary(
                run_id=run_id, frame_num=frame_num,
                z_min=np.nan, z_max=np.nan, theta_min=np.nan, theta_max=np.nan,
                valid_ratio=0.0, d_wall_mean=np.nan, d_wall_min=np.nan, gamma_mean=np.nan,
            )
        z = proj["z"][valid]
        th = proj["theta"][valid]
        dw = proj["d_wall"][valid]
        return FrameProjectionSummary(
            run_id=run_id,
            frame_num=frame_num,
            z_min=float(z.min()),
            z_max=float(z.max()),
            theta_min=float(th.min()),
            theta_max=float(th.max()),
            valid_ratio=float(np.mean(valid)),
            d_wall_mean=float(dw.mean()),
            d_wall_min=float(dw.min()),
            gamma_mean=float(proj["gamma"][valid].mean()),
        )

    def _forward_wall_from_pixels(
        self,
        pts: np.ndarray,
        camera_position: np.ndarray,
        orientation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        camera = self.transformer.camera
        yaw_cam, pitch_cam = camera.pixel_to_angles_cam(pts)
        roll, yaw, pitch = orientation
        vx = np.cos(pitch_cam) * np.sin(yaw_cam)
        vy = np.sin(pitch_cam)
        vz = np.cos(pitch_cam) * np.cos(yaw_cam)
        v_cam = np.stack((vx, vy, vz), axis=-1)
        R = pose_R_c2w(roll, yaw, pitch)
        v_w = v_cam @ R
        direction = v_w / (np.linalg.norm(v_w, axis=1, keepdims=True) + 1e-12)
        origin = np.repeat(np.asarray(camera_position, dtype=float).reshape(1, 3), len(pts), axis=0)
        p_wall, t_hit, hit_ok = self._intersect_forward_wall(origin, direction)
        return p_wall, t_hit, hit_ok

    def _intersect_forward_wall(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        radius = self.pipe_radius
        x_o, y_o = origin[:, 0], origin[:, 1]
        dx, dy, dz = direction[:, 0], direction[:, 1], direction[:, 2]
        a = dx ** 2 + dy ** 2
        b = 2.0 * (x_o * dx + y_o * dy)
        c = x_o ** 2 + y_o ** 2 - radius ** 2
        disc = b ** 2 - 4.0 * a * c
        ok = (a > 1e-12) & (disc >= 0.0)
        sqrt_d = np.sqrt(np.clip(disc, 0.0, None))
        t1 = (-b - sqrt_d) / (2.0 * a + 1e-15)
        t2 = (-b + sqrt_d) / (2.0 * a + 1e-15)
        t_pos = np.full(len(origin), np.nan)
        for t_cand in (t1, t2):
            better = ok & (t_cand > 1e-6) & (np.isnan(t_pos) | (t_cand < t_pos))
            t_pos = np.where(better, t_cand, t_pos)
        hit = np.isfinite(t_pos)
        p = origin + t_pos[:, None] * direction
        p[~hit] = np.nan
        return p, t_pos, hit
