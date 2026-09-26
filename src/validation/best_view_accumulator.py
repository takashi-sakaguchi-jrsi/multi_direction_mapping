"""座標位置への帯の上書き蓄積（ColorMapGenerator.update_colormap 相当）"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class AccumulatorBuffers:
    color: np.ndarray
    filled: np.ndarray
    source_run: Optional[np.ndarray]
    z_min: float
    theta_min: float
    pixels_per_mm: float
    theta_bins: int
    d_wall: Optional[np.ndarray] = None
    source_frame: Optional[np.ndarray] = None
    source_u: Optional[np.ndarray] = None
    source_v: Optional[np.ndarray] = None
    source_gamma: Optional[np.ndarray] = None


class BestViewAccumulator:
    """同一 (η, z) 画素は後から来た帯で上書きする。d_wall では選ばない。"""

    def __init__(
        self,
        config,
        z_min: float,
        z_max: float,
        theta_min: float = 0.0,
        theta_max: float = 2.0 * np.pi,
        theta_bins: Optional[int] = None,
    ):
        self.config = config
        self.tie = config.two_direction.best_view.distance_tie_tolerance_mm
        ppm = config.colormap.pixels_per_mm
        radius = config.pipe.diameter_mm / 2.0
        if theta_bins is None:
            theta_span = theta_max - theta_min
            circ = radius * theta_span
            theta_bins = max(8, int(np.ceil(circ * ppm)))
        z_bins = max(8, int(np.ceil((z_max - z_min) * ppm)))
        shape = (theta_bins, z_bins)
        self.buf = AccumulatorBuffers(
            color=np.zeros(shape + (3,), dtype=np.uint8),
            filled=np.zeros(shape, dtype=bool),
            source_run=None,
            z_min=float(z_min),
            theta_min=float(theta_min),
            pixels_per_mm=float(ppm),
            theta_bins=int(theta_bins),
        )
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.z_max = float(z_max)
        self._run_ids: Dict[str, int] = {}
        self._grow_block = 2048

    def _grow_to(self, min_cols: int) -> None:
        """右端へゼロ列を足して幅を伸ばす。1px ずつではなく 2048 列単位。"""
        min_cols = int(min_cols)
        canvas_w = int(self.buf.color.shape[1])
        if canvas_w >= min_cols:
            return
        extra = min_cols - canvas_w
        grow_cols = ((extra + self._grow_block - 1) // self._grow_block) * self._grow_block
        h = int(self.buf.color.shape[0])
        self.buf.color = np.concatenate(
            [self.buf.color, np.zeros((h, grow_cols, 3), dtype=self.buf.color.dtype)],
            axis=1,
        )
        self.buf.filled = np.concatenate(
            [self.buf.filled, np.zeros((h, grow_cols), dtype=self.buf.filled.dtype)],
            axis=1,
        )
        if self.buf.source_run is not None:
            self.buf.source_run = np.concatenate(
                [self.buf.source_run, np.zeros((h, grow_cols), dtype=self.buf.source_run.dtype)],
                axis=1,
            )
        for name, fill in (
            ("d_wall", np.inf),
            ("source_frame", -1),
            ("source_u", np.nan),
            ("source_v", np.nan),
            ("source_gamma", np.nan),
        ):
            arr = getattr(self.buf, name)
            if arr is None:
                continue
            pad = np.full((h, grow_cols), fill, dtype=arr.dtype)
            setattr(self.buf, name, np.concatenate([arr, pad], axis=1))
        self.z_max = float(self.buf.z_min) + (
            self.buf.color.shape[1] / max(float(self.buf.pixels_per_mm), 1e-9)
        )

    def _ensure_point_bufs(self) -> None:
        """点投影用の大きな配列は初回だけ確保する。"""
        shape = self.buf.filled.shape
        if self.buf.d_wall is None:
            self.buf.d_wall = np.full(shape, np.inf, dtype=np.float32)
        if self.buf.source_frame is None:
            self.buf.source_frame = np.full(shape, -1, dtype=np.int32)
        if self.buf.source_u is None:
            self.buf.source_u = np.full(shape, np.nan, dtype=np.float32)
        if self.buf.source_v is None:
            self.buf.source_v = np.full(shape, np.nan, dtype=np.float32)
        if self.buf.source_gamma is None:
            self.buf.source_gamma = np.full(shape, np.nan, dtype=np.float32)

    def _run_code(self, run_id: str) -> int:
        if run_id not in self._run_ids:
            self._run_ids[run_id] = len(self._run_ids)
        return self._run_ids[run_id]

    def _index(self, z: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        iz = np.round((z - self.buf.z_min) * self.buf.pixels_per_mm).astype(int)
        span = self.theta_max - self.theta_min
        it = np.round((theta - self.theta_min) / span * (self.buf.theta_bins - 1)).astype(int)
        valid = (
            (iz >= 0) & (iz < self.buf.color.shape[1])
            & (it >= 0) & (it < self.buf.color.shape[0])
        )
        return it, iz, valid

    def add_strip(
        self,
        colors: np.ndarray,
        mask: np.ndarray,
        z_min_mm: float,
        run_id: str,
        frame_num: int = 0,
    ) -> None:
        """帯を start_col = floor((z_min - canvas_z_min) * ppm) へ上書きする。"""
        if colors.ndim != 3 or mask.ndim != 2:
            raise ValueError("add_strip は (H,W,3) と (H,W) の帯が必要です")
        h, w = mask.shape
        if h != self.buf.color.shape[0]:
            raise ValueError(
                f"η 行数がキャンバスと不一致: strip={h}, canvas={self.buf.color.shape[0]}"
            )
        start = int(np.floor((float(z_min_mm) - self.buf.z_min) * self.buf.pixels_per_mm))
        src0, dst0 = 0, start
        src1, dst1 = w, start + w
        if dst0 < 0:
            src0 = -dst0
            dst0 = 0
        canvas_w = self.buf.color.shape[1]
        if dst1 > canvas_w:
            self._grow_to(dst1)
            canvas_w = self.buf.color.shape[1]
        if src0 >= src1 or dst0 >= dst1:
            return
        patch = colors[:, src0:src1]
        m = mask[:, src0:src1]
        if not np.any(m):
            return
        self.buf.color[:, dst0:dst1][m] = patch[m]
        self.buf.filled[:, dst0:dst1][m] = True
        if self.buf.source_run is not None:
            code = self._run_code(run_id)
            self.buf.source_run[:, dst0:dst1][m] = code
        if self.buf.source_frame is not None:
            self.buf.source_frame[:, dst0:dst1][m] = int(frame_num)

    def add_projection(self, proj: Dict[str, np.ndarray], run_id: str) -> None:
        """座標上書き。2D 帯があれば add_strip、なければ点列を後勝ちで書く。"""
        if "colors_2d" in proj and "mask_2d" in proj:
            self.add_strip(
                proj["colors_2d"],
                proj["mask_2d"],
                float(proj["z_min_strip"]),
                run_id,
                int(proj.get("frame_num", 0)),
            )
            return
        sel = proj["valid"]
        if not np.any(sel):
            return
        z = proj["z"][sel]
        theta = proj["theta"][sel]
        colors = proj["colors"][sel]
        u = proj["u"][sel]
        v = proj["v"][sel]
        gamma = proj["gamma"][sel]
        d_wall = proj["d_wall"][sel] if "d_wall" in proj else np.full(z.shape, np.nan)
        frame_num = int(proj.get("frame_num", 0))
        it, iz, in_rng = self._index(z, theta)
        if not np.any(in_rng):
            return
        self._ensure_point_bufs()
        it, iz = it[in_rng], iz[in_rng]
        colors = colors[in_rng]
        u, v, gamma = u[in_rng], v[in_rng], gamma[in_rng]
        d_wall = d_wall[in_rng]
        run_code = self._run_code(run_id)
        # 後から来た点で上書き（同一呼び出し内は配列末尾が残るよう逆順代入）
        self.buf.color[it, iz] = colors
        self.buf.filled[it, iz] = True
        if self.buf.source_run is not None:
            self.buf.source_run[it, iz] = run_code
        self.buf.source_frame[it, iz] = frame_num
        self.buf.source_u[it, iz] = u
        self.buf.source_v[it, iz] = v
        self.buf.source_gamma[it, iz] = gamma
        self.buf.d_wall[it, iz] = d_wall

    def unfilled_ratio(self) -> float:
        return float(1.0 - np.mean(self.buf.filled))

    def run_adoption_ratio(self) -> Dict[str, float]:
        if self.buf.source_run is None:
            return {}
        filled = self.buf.filled
        n = max(int(np.sum(filled)), 1)
        out = {}
        inv = {v: k for k, v in self._run_ids.items()}
        for code, name in inv.items():
            out[name] = float(np.sum(self.buf.source_run[filled] == code) / n)
        return out

    def colormap_rgb(self) -> np.ndarray:
        if self.buf.color.size >= 8_000_000:
            self.buf.color[~self.buf.filled] = 0
            return self.buf.color
        img = self.buf.color.copy()
        img[~self.buf.filled] = 0
        return img

    def provenance(self) -> Dict[str, np.ndarray]:
        self._ensure_point_bufs()
        dw = self.buf.d_wall.copy()
        dw[~self.buf.filled] = np.nan
        return {
            "color": self.buf.color,
            "d_wall": dw,
            "filled": self.buf.filled,
            "source_run": self.buf.source_run,
            "source_frame": self.buf.source_frame,
            "source_u": self.buf.source_u,
            "source_v": self.buf.source_v,
            "source_gamma": self.buf.source_gamma,
            "run_id_map": np.array(list(self._run_ids.items()), dtype=object),
        }
