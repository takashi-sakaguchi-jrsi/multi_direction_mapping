"""CameraCarDemo 円筒レンダラの魚眼（等距離射影）差し替え。

ピンホールの tan(FOV/2) をやめ、r = fθ でレイを出す。
姿勢は Demo のカメラカー定義（roll, pitch）、またはプログラム Euler（roll, yaw, pitch）。
距離 overlay は持たない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.camera_utils import compute_fisheye_focal_length
from src.validation.geometry import pose_R_c2w


DEFAULT_FOV_DEG = 181.0
DEFAULT_PITCH_DEG = 90.0
PIPE_DIAMETER_MM = 250.0
DEFAULT_RADIUS_MM = PIPE_DIAMETER_MM / 2.0
DEFAULT_COLORMAP_RELATIVE = Path("original_colormap") / "phi250tenkaizu.png"
DEFAULT_N_FRAMES = 150
DEFAULT_Z_START_MM = 10.0
DEFAULT_Z_STEP_MM = 4.5
N_FRAMES_HINT_MIN = 100
N_FRAMES_HINT_MAX = 300


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_colormap_path() -> Path:
    return project_root() / DEFAULT_COLORMAP_RELATIVE


def pixels_per_mm_from_equirect(height_px: int, radius_mm: float) -> float:
    """展開図の円周方向高さから pix/mm を求める（Z 方向も同じ密度）。"""
    circ = 2.0 * np.pi * float(radius_mm)
    if circ <= 0.0 or height_px <= 0:
        raise ValueError(f"不正な展開図サイズ: height={height_px}, radius={radius_mm}")
    return float(height_px) / circ


def demo_wall_angle_deg(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CameraCarDemo の周方向角[deg]。画像上端が真上 (+Y)。"""
    return (np.degrees(np.arctan2(y, x)) + 270.0) % 360.0


def make_demo_layout_colormap(
    radius_mm: float = 125.0,
    z_max_mm: float = 400.0,
    pixels_per_mm: float = 1.0,
) -> np.ndarray:
    """周方向が Demo 規約の合成展開図 (H=円周, W=Z)。"""
    height = max(8, int(round(2.0 * np.pi * radius_mm * pixels_per_mm)))
    width = max(8, int(round(z_max_mm * pixels_per_mm)))
    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    angle = yy / height * 360.0
    z = xx / pixels_per_mm
    r = (np.sin(np.radians(angle) * 3.0) * 0.5 + 0.5) * 255.0
    g = (np.cos(z / 40.0) * 0.5 + 0.5) * 255.0
    b = ((angle / 360.0 + z / max(z_max_mm, 1.0)) % 1.0) * 255.0
    return np.stack([b, g, r], axis=-1).astype(np.uint8)


class FisheyeSideviewRenderer:
    """展開図テクスチャ + 魚眼レイ + Demo の roll/pitch 回転。"""

    def __init__(
        self,
        colormap: Union[str, Path, np.ndarray],
        output_size: Tuple[int, int],
        radius_mm: float,
        fov_deg: float = DEFAULT_FOV_DEG,
        apply_distance_shading: bool = False,
        decay: float = 0.002,
        min_alpha: float = 0.0001,
        keep: float = 0.0,
    ):
        if isinstance(colormap, np.ndarray):
            img = colormap
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("colormap は BGR 画像である必要があります")
            self.equirect_img = img.copy()
            self.colormap_path = None
        else:
            path = Path(colormap)
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"展開図を読めません: {path}")
            self.equirect_img = img
            self.colormap_path = str(path)

        self.output_width, self.output_height = int(output_size[0]), int(output_size[1])
        self.radius = float(radius_mm)
        self.apply_distance_shading = bool(apply_distance_shading)
        self.decay = float(decay)
        self.min_alpha = float(min_alpha)
        self.keep = min(float(keep), self.radius)
        self.eq_h, self.eq_w = self.equirect_img.shape[:2]
        self.vert_deg_per_px = 360.0 / self.eq_h
        self.horiz_mm_per_px = (2.0 * np.pi * self.radius) / self.eq_h
        self.z_extent_mm = self.eq_w * self.horiz_mm_per_px
        self.fov_deg = float(fov_deg)
        self.f = float(
            compute_fisheye_focal_length(
                self.fov_deg, (self.output_height, self.output_width)
            )
        )
        self.cx = self.output_width / 2.0
        self.cy = self.output_height / 2.0
        self.r_max = min(self.output_width, self.output_height) / 2.0
        self._view_rays, self._r_px = self._build_fisheye_rays()

    def _build_fisheye_rays(self) -> Tuple[np.ndarray, np.ndarray]:
        i, j = np.meshgrid(
            np.arange(self.output_width, dtype=np.float64),
            np.arange(self.output_height, dtype=np.float64),
        )
        cam_x = i - self.cx
        cam_y = -(j - self.cy)
        r_px = np.hypot(cam_x, cam_y)
        rho = np.maximum(r_px, 1e-12)
        gamma = r_px / self.f
        vx = np.sin(gamma) * (cam_x / rho)
        vy = np.sin(gamma) * (cam_y / rho)
        vz = np.cos(gamma)
        on_axis = r_px < 1e-12
        vx = np.where(on_axis, 0.0, vx)
        vy = np.where(on_axis, 0.0, vy)
        vz = np.where(on_axis, 1.0, vz)
        rays = np.stack([vx, vy, vz], axis=-1)
        rays = rays / (np.linalg.norm(rays, axis=-1, keepdims=True) + 1e-12)
        return rays, r_px

    @staticmethod
    def demo_rotation(roll_deg: float, pitch_deg: float) -> np.ndarray:
        """カメラカーの roll/pitch から Demo と同じ回転行列を作る。

        Demo は右キーで roll を減らす（右ロールで底面が右に見える）。
        カメラカーの右ロール正（R=+120）は、Demo へ渡す前に符号を反転する。
        pitch は Demo どおり符号反転して R = R_roll @ R_pitch。
        """
        roll = np.radians(-float(roll_deg))
        pitch = -np.radians(float(pitch_deg))
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        r_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
        return r_roll @ r_pitch

    def _rotate_rays(self, roll_deg: float, pitch_deg: float) -> np.ndarray:
        rotation = self.demo_rotation(roll_deg, pitch_deg)
        return self._apply_rotation(rotation)

    def _apply_rotation(self, rotation: np.ndarray) -> np.ndarray:
        flat = self._view_rays.reshape(-1, 3).T
        rotated = np.asarray(rotation, dtype=np.float64) @ flat
        return rotated.T.reshape(self.output_height, self.output_width, 3)

    def _intersect(
        self,
        cam_position: Sequence[float],
        roll_deg: float,
        pitch_deg: float,
        rotation: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        x0, y0, z0 = [float(v) for v in cam_position]
        rays = self._apply_rotation(rotation) if rotation is not None else self._rotate_rays(roll_deg, pitch_deg)
        dx, dy, dz = rays[..., 0], rays[..., 1], rays[..., 2]
        a = dx * dx + dy * dy
        b = 2.0 * (x0 * dx + y0 * dy)
        c = x0 * x0 + y0 * y0 - self.radius * self.radius
        disc = b * b - 4.0 * a * c
        in_fov = self._r_px <= (self.r_max + 1e-6)
        ok = (disc >= 0.0) & (a > 1e-12) & in_fov
        sqrt_d = np.sqrt(np.clip(disc, 0.0, None))
        t1 = (-b - sqrt_d) / (2.0 * a + 1e-15)
        t2 = (-b + sqrt_d) / (2.0 * a + 1e-15)
        t = np.full_like(a, np.nan)
        for t_cand in (t1, t2):
            better = ok & (t_cand > 1e-6) & (np.isnan(t) | (t_cand < t))
            t = np.where(better, t_cand, t)
        valid = np.isfinite(t)
        x = x0 + t * dx
        y = y0 + t * dy
        z = z0 + t * dz
        d_wall = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
        return {
            "valid": valid,
            "X": x,
            "Y": y,
            "Z": z,
            "t": t,
            "d_wall": d_wall,
        }

    def _remap_texture(self, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
        """周方向は周期、Z 方向は端で黒。巨大展開図は NumPy 補間。

        周方向の 0/360 継ぎ目を INTER_LINEAR が黒境界と混ぜないよう、
        上下 1 行を周期パディングしてからサンプリングする。
        """
        src = self.equirect_img
        h, w = src.shape[:2]
        map_y = np.mod(map_y.astype(np.float32), float(h))
        map_x = map_x.astype(np.float32)
        if h < 32766 and w < 32767:
            padded = np.concatenate([src[-1:], src, src[:1]], axis=0)
            return cv2.remap(
                padded,
                map_x,
                map_y + 1.0,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        x = map_x
        y = map_y
        inside = (x >= 0.0) & (x <= (w - 1))
        x = np.clip(x, 0.0, float(w - 1))
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32) % h
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = (y0 + 1) % h
        wx = (x - x0.astype(np.float32))[..., None]
        wy = (y - np.floor(y))[..., None]
        c00 = src[y0, x0].astype(np.float32)
        c01 = src[y0, x1].astype(np.float32)
        c10 = src[y1, x0].astype(np.float32)
        c11 = src[y1, x1].astype(np.float32)
        blended = (
            c00 * (1.0 - wy) * (1.0 - wx)
            + c01 * (1.0 - wy) * wx
            + c10 * wy * (1.0 - wx)
            + c11 * wy * wx
        )
        frame = np.zeros(map_x.shape + (3,), dtype=np.uint8)
        frame[inside] = np.clip(blended[inside], 0, 255).astype(np.uint8)
        return frame

    def _sample_texture(self, hits: Dict[str, np.ndarray]) -> np.ndarray:
        valid = hits["valid"]
        angle = demo_wall_angle_deg(hits["X"], hits["Y"])
        map_x = np.zeros(valid.shape, dtype=np.float32)
        map_y = np.zeros(valid.shape, dtype=np.float32)
        map_x[valid] = (hits["Z"][valid] / self.horiz_mm_per_px).astype(np.float32)
        map_y[valid] = (angle[valid] / self.vert_deg_per_px).astype(np.float32)
        frame = self._remap_texture(map_x, map_y)
        frame = frame.copy()
        frame[~valid] = 0
        if self.apply_distance_shading:
            alpha = np.ones(valid.shape, dtype=np.float32)
            d = hits["d_wall"].astype(np.float32)
            alpha[valid] = np.exp(
                -self.decay * np.maximum(0.0, d[valid] - self.keep)
            )
            alpha = np.clip(alpha, self.min_alpha, 1.0)
            frame = (frame.astype(np.float32) * alpha[..., None]).astype(np.uint8)
            frame[~valid] = 0
        return frame

    def render_view(
        self,
        cam_position: Sequence[float],
        cam_orientation: Sequence[float],
        fov_deg: Optional[float] = None,
        return_hits: bool = False,
        program_rpy_rad: Optional[Sequence[float]] = None,
    ):
        if fov_deg is not None and abs(float(fov_deg) - self.fov_deg) > 1e-6:
            raise ValueError("FOV は初期化時の値のみ対応しています。レンダラを作り直してください。")
        rotation = None
        if program_rpy_rad is not None:
            roll, yaw, pitch = [float(v) for v in program_rpy_rad]
            # Demo は R @ v_cam。プログラムは v_cam @ R_c2w = R_c2w.T @ v_cam
            rotation = pose_R_c2w(roll, yaw, pitch).T
            hits = self._intersect(cam_position, 0.0, 0.0, rotation=rotation)
        else:
            roll, pitch = float(cam_orientation[0]), float(cam_orientation[1])
            hits = self._intersect(cam_position, roll, pitch)
        frame = self._sample_texture(hits)
        if return_hits:
            return frame, hits
        return frame

    def render_sequence(
        self,
        z_values: Sequence[float],
        roll_deg: float,
        pitch_deg: float = DEFAULT_PITCH_DEG,
        cam_xy: Tuple[float, float] = (0.0, 0.0),
        program_rpy_rad: Optional[Sequence[Sequence[float]]] = None,
    ) -> Tuple[List[np.ndarray], List[float]]:
        frames: List[np.ndarray] = []
        zs: List[float] = []
        for i, z in enumerate(z_values):
            pos = (float(cam_xy[0]), float(cam_xy[1]), float(z))
            rpy = None
            if program_rpy_rad is not None:
                rpy = program_rpy_rad[i]
            frames.append(self.render_view(pos, (roll_deg, pitch_deg), program_rpy_rad=rpy))
            zs.append(float(z))
        return frames, zs

    def reconstruct_from_hits(
        self,
        frames: Sequence[np.ndarray],
        positions: Sequence[Sequence[float]],
        orientations: Sequence[Sequence[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """既知姿勢で展開図へ戻す（Phase1 / OCR なし）。周方向は Demo 規約。"""
        canvas = np.zeros_like(self.equirect_img)
        filled = np.zeros((self.eq_h, self.eq_w), dtype=bool)
        dmin = np.full((self.eq_h, self.eq_w), np.inf, dtype=np.float32)
        for frame, pos, ori in zip(frames, positions, orientations):
            ori = tuple(float(v) for v in ori)
            if len(ori) >= 3:
                _, hits = self.render_view(pos, ori[:2], return_hits=True, program_rpy_rad=ori[:3])
            else:
                _, hits = self.render_view(pos, ori, return_hits=True)
            valid = hits["valid"]
            if not np.any(valid):
                continue
            x = np.where(valid, hits["X"], 0.0)
            y = np.where(valid, hits["Y"], 0.0)
            z = np.where(valid, hits["Z"], 0.0)
            angle = demo_wall_angle_deg(x, y)
            eq_x = np.rint(z / self.horiz_mm_per_px).astype(np.int32)
            eq_y = np.rint(angle / self.vert_deg_per_px).astype(np.int32) % self.eq_h
            yy = frame.shape[0]
            xx = frame.shape[1]
            jj, ii = np.meshgrid(np.arange(yy), np.arange(xx), indexing="ij")
            sel = (
                valid
                & (eq_x >= 0) & (eq_x < self.eq_w)
                & (eq_y >= 0) & (eq_y < self.eq_h)
            )
            if not np.any(sel):
                continue
            ix = eq_x[sel]
            iy = eq_y[sel]
            dw = hits["d_wall"][sel].astype(np.float32)
            colors = frame[jj[sel], ii[sel]]
            better = dw < dmin[iy, ix]
            unset = ~filled[iy, ix]
            take = better | unset
            if not np.any(take):
                continue
            iy, ix = iy[take], ix[take]
            canvas[iy, ix] = colors[take]
            dmin[iy, ix] = dw[take]
            filled[iy, ix] = True
        canvas[~filled] = 0
        return canvas, filled


def z_values_for_segment(
    renderer: FisheyeSideviewRenderer,
    z_start_mm: float,
    n_frames: int,
    z_step_mm: float,
    z_margin_mm: float = 10.0,
) -> np.ndarray:
    """展開図の一部だけを掃引する。全長は使わない。

    z_start_mm を変えると、別の特徴が出ている区間で試せる。
    """
    extent = float(renderer.z_extent_mm)
    z_min = float(max(0.0, z_margin_mm))
    z_max = float(max(z_min + abs(float(z_step_mm)), extent - float(z_margin_mm)))
    z_max = min(z_max, extent)
    if z_max <= z_min:
        z_min, z_max = 0.0, extent
    step = float(z_step_mm)
    if step <= 0.0:
        raise ValueError(f"z_step_mm は正である必要があります: {z_step_mm}")
    n = max(1, int(n_frames))
    z0 = float(np.clip(z_start_mm, z_min, z_max))
    zs = z0 + np.arange(n, dtype=np.float64) * step
    zs = zs[zs <= z_max + 1e-9]
    if zs.size == 0:
        zs = np.array([z0], dtype=np.float64)
    return zs


def z_values_for_map(
    renderer: FisheyeSideviewRenderer,
    n_frames: int,
    z_margin_mm: float = 20.0,
    z_start_mm: float = DEFAULT_Z_START_MM,
    z_step_mm: float = DEFAULT_Z_STEP_MM,
) -> np.ndarray:
    """互換: 区間掃引。全長 linspace はしない。"""
    return z_values_for_segment(
        renderer,
        z_start_mm=z_start_mm,
        n_frames=n_frames,
        z_step_mm=z_step_mm,
        z_margin_mm=z_margin_mm,
    )


def crop_colormap_z_window(
    image: np.ndarray,
    renderer: FisheyeSideviewRenderer,
    z0_mm: float,
    z1_mm: float,
    pad_mm: float = 250.0,
) -> np.ndarray:
    """目視用に、生成した距離区間付近だけ切り出す。"""
    x0 = int(np.floor(max(0.0, z0_mm - pad_mm) / renderer.horiz_mm_per_px))
    x1 = int(np.ceil(min(renderer.z_extent_mm, z1_mm + pad_mm) / renderer.horiz_mm_per_px))
    x1 = min(renderer.eq_w, max(x0 + 1, x1))
    return image[:, x0:x1]


def default_colormap_candidates() -> List[Path]:
    root = project_root()
    return [
        root / DEFAULT_COLORMAP_RELATIVE,
        root / "original_colormap",
        root / "data" / "input" / "colormaps",
        root.parent / "CameraCarDemo" / "color_map",
        Path("/home/tsakaguchi/CameraCarDemo/color_map"),
    ]


def resolve_colormap_path(explicit: Optional[Union[str, Path]] = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            cand = project_root() / path
            if cand.is_file():
                return cand
        if path.is_file():
            return path
        raise FileNotFoundError(f"展開図が見つかりません: {path}")
    preferred = default_colormap_path()
    if preferred.is_file():
        return preferred
    names = ("phi250tenkaizu.png", "vu250_tenkaizu.png")
    for folder in default_colormap_candidates():
        if folder.is_file() and folder.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return folder
        if not folder.is_dir():
            continue
        for name in names:
            cand = folder / name
            if cand.is_file():
                return cand
        pngs = sorted(folder.glob("*.png"))
        if pngs:
            return pngs[0]
    searched = ", ".join(str(p) for p in default_colormap_candidates())
    raise FileNotFoundError(
        "展開図 PNG が見つかりません。--colormap で指定してください。"
        f" 探索先: {searched}"
    )


def write_mp4(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("フレームが空です")
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"動画を開けません: {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
