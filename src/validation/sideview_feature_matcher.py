"""壁面正対用の特徴点マッチング

レンズ中央の低歪矩形で抽出し、Lowe 比などの品質フィルタ、
逆走除外、ベクトル長と方向角の IQR 外れ値除外を行う。
8×8 空間間引きはしない。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


class SideviewFeatureMatcher:
    """レンズ中央矩形 + 品質フィルタ + 逆走除外"""

    def __init__(self, config):
        side = config.two_direction.feature_matching_sideview
        fm = config.estimation.feature_matching
        self.max_features = side.max_features
        self.ratio_test_threshold = side.ratio_test_threshold
        self.max_displacement = side.max_pixel_displacement_sideview
        self.min_match_count = side.min_match_count
        self.center_rect_half_ratio = float(
            getattr(side, "center_rect_half_ratio", 0.4)
        )
        self.reject_reverse_travel = bool(
            getattr(side, "reject_reverse_travel", True)
        )
        self.reverse_flow_tolerance_px = float(
            getattr(side, "reverse_flow_tolerance_px", 1.0)
        )
        self.outlier_iqr_multiplier = float(
            getattr(side, "outlier_iqr_multiplier", 1.5)
        )
        self.outlier_min_points = int(getattr(side, "outlier_min_points", 4))
        self.orb = cv2.ORB_create(
            nfeatures=max(self.max_features * 2, fm.max_features * 2),
            scaleFactor=fm.scale_factor,
            nlevels=fm.n_levels,
            edgeThreshold=fm.edge_threshold,
            firstLevel=fm.first_level,
            WTA_K=fm.wta_k,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=fm.patch_size,
            fastThreshold=fm.fast_threshold,
        )

    def center_rect(
        self,
        shape: Tuple[int, int],
        center: Optional[Tuple[float, float]] = None,
    ) -> List[int]:
        """レンズ中心まわりの正方形。一辺は min(w,h) * half_ratio。"""
        h, w = int(shape[0]), int(shape[1])
        if center is None:
            cx, cy = w / 2.0, h / 2.0
        else:
            cx, cy = float(center[0]), float(center[1])
        half = max(4.0, self.center_rect_half_ratio * (min(w, h) / 2.0))
        x0 = int(np.floor(cx - half))
        y0 = int(np.floor(cy - half))
        x1 = int(np.ceil(cx + half))
        y1 = int(np.ceil(cy + half))
        return [
            max(0, x0), max(0, y0),
            min(w, x1), min(h, y1),
        ]

    def build_roi_mask(
        self,
        shape: Tuple[int, int],
        roi_rect: Optional[List[int]],
        valid_mask: Optional[np.ndarray],
        center: Optional[Tuple[float, float]] = None,
        max_radius_px: Optional[float] = None,
    ) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if roi_rect is None:
            roi_rect = self.center_rect(shape, center)
        x0, y0, x1, y1 = [int(v) for v in roi_rect]
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
        if max_radius_px is not None and center is not None:
            yy, xx = np.ogrid[:h, :w]
            rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
            mask[rr > max_radius_px] = 0
        if valid_mask is not None:
            mask = np.where(valid_mask > 0, mask, 0).astype(np.uint8)
        return mask

    def filter_motion_vectors(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """逆走除外のあと、ベクトル長と方向角の IQR 外れ値を落とす。"""
        if len(pts1) == 0:
            return pts1, pts2
        d = pts2 - pts1
        keep = np.linalg.norm(d, axis=1) < self.max_displacement
        if self.reject_reverse_travel:
            keep = keep & (d[:, 1] <= self.reverse_flow_tolerance_px)
        pts1, pts2 = pts1[keep], pts2[keep]
        if len(pts1) < self.outlier_min_points:
            return pts1, pts2
        mag_keep = self._iqr_mask(np.linalg.norm(pts2 - pts1, axis=1), floor=0.5)
        if int(np.sum(mag_keep)) >= 3:
            pts1, pts2 = pts1[mag_keep], pts2[mag_keep]
        if len(pts1) < self.outlier_min_points:
            return pts1, pts2
        ang_keep = self._direction_iqr_mask(pts2 - pts1)
        if int(np.sum(ang_keep)) >= 3:
            pts1, pts2 = pts1[ang_keep], pts2[ang_keep]
        return pts1, pts2

    def _iqr_mask(self, values: np.ndarray, floor: float = 0.0) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        n = values.size
        if n < self.outlier_min_points:
            return np.ones(n, dtype=bool)
        q1, q3 = np.percentile(values, [25.0, 75.0])
        iqr = max(float(q3 - q1), float(floor))
        k = self.outlier_iqr_multiplier
        lo, hi = q1 - k * iqr, q3 + k * iqr
        return (values >= lo) & (values <= hi)

    def _direction_iqr_mask(self, delta: np.ndarray) -> np.ndarray:
        """円周まわりの方向角を中央値基準で折り返し、IQR する。"""
        ang = np.arctan2(delta[:, 1], delta[:, 0])
        mean = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
        wrapped = (ang - mean + np.pi) % (2.0 * np.pi) - np.pi
        return self._iqr_mask(wrapped, floor=np.radians(1.0))

    def detect_and_match(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        roi_rect: Optional[List[int]] = None,
        valid_mask: Optional[np.ndarray] = None,
        center: Optional[Tuple[float, float]] = None,
        max_radius_px: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if curr_frame.ndim == 3 else curr_frame
        mask1 = self.build_roi_mask(prev_gray.shape, roi_rect, valid_mask, center, max_radius_px)
        mask2 = self.build_roi_mask(curr_gray.shape, roi_rect, valid_mask, center, max_radius_px)

        kp1, des1 = self.orb.detectAndCompute(prev_gray, mask1)
        kp2, des2 = self.orb.detectAndCompute(curr_gray, mask2)
        if kp1 is None or kp2 is None or des1 is None or des2 is None:
            return None, None
        if len(kp1) < 4 or len(kp2) < 4:
            return None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test_threshold * n.distance:
                good.append(m)
        if len(good) == 0:
            return None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        pts1, pts2 = self.filter_motion_vectors(pts1, pts2)
        if len(pts1) < 3:
            return None, None
        return pts1, pts2
