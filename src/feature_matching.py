"""特徴点マッチングモジュール

ORB特徴点を使用してフレーム間の対応点を検出し、
カメラの移動量推定に必要な情報を提供します。

主な機能:
- ORB特徴点の検出
- Brute-Force Matcher + crossCheckによるマッチング
- ドーナツ状フィルタリング（有効領域の制限）
- Lowe's比率テスト
- 距離制限によるアウトライア除去
- 空間分散フィルタリング（θ-φ空間でのバランス抽出）

使用例:
    >>> from config import Config
    >>> config = Config.from_defaults()
    >>> matcher = FeatureMatcher(config)
    >>> prev_points, curr_points = matcher.detect_and_match(prev_frame, curr_frame, camera_params)
"""

import logging
from typing import Tuple, List, Optional

import cv2
import numpy as np


# ============================================================================
# カスタム例外クラス
# ============================================================================

class FeatureMatchingError(Exception):
    """特徴点マッチング関連のベース例外"""
    pass


class InsufficientFeaturesError(FeatureMatchingError):
    """特徴点数が不足している"""
    pass


class MatchingFailedError(FeatureMatchingError):
    """マッチング処理に失敗"""
    pass


# ============================================================================
# 特徴点マッチングクラス
# ============================================================================

class FeatureMatcher:
    """特徴点マッチング処理クラス
    
    ORB特徴点を使用してフレーム間の対応点を検出します。
    
    Attributes:
        config: 設定情報（ORBパラメータ、マッチングパラメータ等）
        logger: ロガー
        orb: ORB特徴点検出器
    """
    
    def __init__(self, config):
        """初期化
        
        Args:
            config: 設定情報（Config.estimation.feature_matching）
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ORB特徴点検出器の初期化
        self.orb = cv2.ORB_create(
            nfeatures=self.config.max_features * 2,  # マッチング前の候補数を多めに
            scaleFactor=self.config.scale_factor,
            nlevels=self.config.n_levels,
            edgeThreshold=self.config.edge_threshold,
            firstLevel=self.config.first_level,
            WTA_K=self.config.wta_k,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=self.config.patch_size,
            fastThreshold=self.config.fast_threshold
        )
        
        self.logger.info(f"FeatureMatcher initialized: max_features={self.config.max_features}")
    
    def detect_and_match(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        camera_params: dict
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """特徴点検出とマッチング
        
        Args:
            prev_frame: 前フレーム画像 (H, W, 3) BGR
            curr_frame: 現フレーム画像 (H, W, 3) BGR
            camera_params: カメラパラメータ（center, radius等）
        
        Returns:
            prev_points: 前フレームのマッチ点座標 (N, 2)、失敗時はNone
            curr_points: 現フレームのマッチ点座標 (N, 2)、失敗時はNone
        """
        # グレースケール変換
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # 特徴点検出
        kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(curr_gray, None)
        
        if kp1 is None or kp2 is None or des1 is None or des2 is None:
            self.logger.warning("特徴点検出に失敗")
            return None, None
        
        self.logger.debug(f"検出特徴点数: prev={len(kp1)}, curr={len(kp2)}")
        
        # ドーナツ状フィルタリング
        center = camera_params['center']
        radius = camera_params['radius']
        
        # Legacy実装では crip_margin_px と crip_size を使用
        # config経由で取得
        radius_min = self.config.radius_min_ratio * radius
        radius_max = self.config.radius_max_ratio * radius
        
        kp1, des1 = self._filter_keypoints_by_radius(kp1, des1, center, radius_min, radius_max)
        kp2, des2 = self._filter_keypoints_by_radius(kp2, des2, center, radius_min, radius_max)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            self.logger.warning(f"フィルタ後の特徴点数が不足: prev={len(kp1) if kp1 else 0}, curr={len(kp2) if kp2 else 0}")
            return None, None
        
        self.logger.debug(f"フィルタ後特徴点数: prev={len(kp1)}, curr={len(kp2)}")
        
        # マッチング
        matches = self._match_features(des1, des2)
        
        if len(matches) == 0:
            self.logger.warning("マッチング結果が空")
            return None, None
        
        # 座標抽出
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # 距離制限フィルタ
        pts1, pts2 = self._filter_by_distance(pts1, pts2)
        
        if len(pts1) < 10:
            self.logger.warning(f"距離フィルタ後の点数が不足: {len(pts1)}")
            return None, None
        
        self.logger.info(f"マッチング成功: {len(pts1)}点")
        
        return pts1, pts2
    
    def _filter_keypoints_by_radius(
        self,
        kp: List[cv2.KeyPoint],
        des: np.ndarray,
        center: Tuple[float, float],
        radius_min: float,
        radius_max: float
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """特徴点をドーナツ状に制限
        
        Args:
            kp: 特徴点リスト
            des: 記述子
            center: レンズ中心座標 (cx, cy)
            radius_min: 最小半径
            radius_max: 最大半径
        
        Returns:
            filtered_kp: フィルタ後の特徴点リスト
            filtered_des: フィルタ後の記述子
        """
        c_x, c_y = center
        filtered_kp = []
        filtered_des = []
        
        for i, point in enumerate(kp):
            x, y = point.pt
            dist = np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)
            
            if radius_min <= dist < radius_max:
                filtered_kp.append(point)
                filtered_des.append(des[i])
        
        if filtered_des:
            filtered_des = np.array(filtered_des)
        else:
            filtered_des = None
        
        return filtered_kp, filtered_des
    
    def _match_features(
        self,
        des1: np.ndarray,
        des2: np.ndarray
    ) -> List[cv2.DMatch]:
        """特徴点マッチング（Lowe's比率テスト適用）
        
        Args:
            des1: 前フレームの記述子
            des2: 現フレームの記述子
        
        Returns:
            good_matches: 良好なマッチのリスト
        """
        # Brute-Force Matcher（HAMMING距離）
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # k=2 で2つの最も近いマッチを取得（比率テストに使用）
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) < 2:
                continue
            
            m, n = match_pair
            
            # Lowe's Ratio Test
            if m.distance < self.config.ratio_test_threshold * n.distance:
                good_matches.append(m)
        
        self.logger.debug(f"Lowe's比率テスト後のマッチ数: {len(good_matches)}")
        
        return good_matches
    
    def _filter_by_distance(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """距離制限によるフィルタリング
        
        フレーム間で特徴点が大きく移動している場合（誤マッチ）を除外します。
        
        Args:
            pts1: 前フレームの点座標 (N, 2)
            pts2: 現フレームの点座標 (N, 2)
        
        Returns:
            filtered_pts1: フィルタ後の前フレーム点 (M, 2)
            filtered_pts2: フィルタ後の現フレーム点 (M, 2)
        """
        # ユークリッド距離を計算
        distances = np.linalg.norm(pts1 - pts2, axis=1)
        
        # しきい値以下のものを選択
        valid_indices = distances < self.config.max_distance_threshold
        
        filtered_pts1 = pts1[valid_indices]
        filtered_pts2 = pts2[valid_indices]
        
        self.logger.debug(
            f"距離フィルタ: {len(pts1)} -> {len(filtered_pts1)} "
            f"(threshold={self.config.max_distance_threshold})"
        )
        
        return filtered_pts1, filtered_pts2
    
    def select_spatially_balanced_points(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        max_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """空間分散フィルタリング（θ-φ空間でのバランス抽出）
        
        視線角度に基づいてグリッド分割し、各グリッドから均等に点を選出します。
        これにより、画像の特定領域に点が偏ることを防ぎます。
        
        Args:
            pts1: 前フレームの点座標 (N, 2)
            pts2: 現フレームの点座標 (N, 2)
            theta: 各特徴点の水平方向視線角度（ラジアン） (N,)
            phi: 各特徴点の垂直方向視線角度（ラジアン） (N,)
            max_points: 選出する最大点数
        
        Returns:
            selected_pts1: 選出された前フレーム点 (M, 2)
            selected_pts2: 選出された現フレーム点 (M, 2)
        """
        if len(theta) <= max_points:
            return pts1, pts2
        
        # 自動スケーリング：極端な外れ値の影響を除外
        theta_min, theta_max = np.percentile(theta, [1, 99])
        phi_min, phi_max = np.percentile(phi, [1, 99])
        
        num_theta_bins = self.config.spatial_grid_theta_bins
        num_phi_bins = self.config.spatial_grid_phi_bins
        
        theta_edges = np.linspace(theta_min, theta_max, num_theta_bins + 1)
        phi_edges = np.linspace(phi_min, phi_max, num_phi_bins + 1)
        
        theta_bins = np.digitize(theta, theta_edges) - 1
        phi_bins = np.digitize(phi, phi_edges) - 1
        
        # 優先度（中心からの角度距離）
        priority = theta**2 + phi**2
        
        # グリッドごとにグループ化
        grid = {}
        for i, (tb, pb) in enumerate(zip(theta_bins, phi_bins)):
            if 0 <= tb < num_theta_bins and 0 <= pb < num_phi_bins:
                key = (tb, pb)
                grid.setdefault(key, []).append((priority[i], i))
        
        # 各グリッド内で優先度順にソート
        for key in grid:
            grid[key].sort(reverse=True)
        
        # ラウンドロビン方式で各グリッドから選出
        selected_indices = []
        while len(selected_indices) < max_points:
            any_added = False
            for key in sorted(grid.keys()):
                if grid[key]:
                    _, idx = grid[key].pop(0)
                    selected_indices.append(idx)
                    any_added = True
                    if len(selected_indices) == max_points:
                        break
            if not any_added:
                break
        
        selected_indices = np.array(selected_indices)
        
        self.logger.debug(
            f"空間分散フィルタ: {len(pts1)} -> {len(selected_indices)} "
            f"(grid={num_theta_bins}x{num_phi_bins})"
        )
        
        return pts1[selected_indices], pts2[selected_indices]


# ============================================================================
# ユーティリティ関数（Legacy互換）
# ============================================================================

def match_features(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    center: Tuple[float, float],
    radius_min: float,
    radius_max: float,
    max_distance_threshold: float = 200.0,
    ratio_test_threshold: float = 0.75,
    max_features: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """特徴点マッチング（Legacy互換関数）
    
    Legacy実装の match_features() 関数と互換性を保つためのラッパー関数です。
    
    Args:
        prev_gray: 前フレーム（グレースケール）
        curr_gray: 現フレーム（グレースケール）
        center: レンズ中心座標 (cx, cy)
        radius_min: 最小半径
        radius_max: 最大半径
        max_distance_threshold: 特徴点間の最大許容距離（ピクセル）
        ratio_test_threshold: Lowe's比率テストの閾値
        max_features: 最大特徴点数
    
    Returns:
        pts1: 前フレームのマッチ点座標 (N, 2)、失敗時はNone
        pts2: 現フレームのマッチ点座標 (N, 2)、失敗時はNone
    """
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    
    # ドーナツ状フィルタリング
    c_x, c_y = center
    filtered_kp1 = []
    filtered_des1 = []
    
    for i, point in enumerate(kp1):
        x, y = point.pt
        dist = np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)
        
        if radius_min <= dist < radius_max:
            filtered_kp1.append(point)
            filtered_des1.append(des1[i])
    
    filtered_kp2 = []
    filtered_des2 = []
    
    for i, point in enumerate(kp2):
        x, y = point.pt
        dist = np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)
        
        if radius_min <= dist < radius_max:
            filtered_kp2.append(point)
            filtered_des2.append(des2[i])
    
    if filtered_des1:
        des1 = np.array(filtered_des1)
    else:
        des1 = None
    
    if filtered_des2:
        des2 = np.array(filtered_des2)
    else:
        des2 = None
    
    kp1 = filtered_kp1
    kp2 = filtered_kp2
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, None
    
    # Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # k=2 で2つの最も近いマッチを取得
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        # Lowe's Ratio Test
        if m.distance < ratio_test_threshold * n.distance:
            good_matches.append(m)
    
    if good_matches:
        # 座標抽出
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # 距離制限
        distances = np.linalg.norm(pts1 - pts2, axis=1)
        valid_indices = distances < max_distance_threshold
        
        pts1 = pts1[valid_indices]
        pts2 = pts2[valid_indices]
        
        filtered_matches = np.array(good_matches)[valid_indices]
        
        # 類似度の高い順に max_features 個を選択
        if max_features is None or len(filtered_matches) < max_features:
            max_features = len(filtered_matches)
        
        sorted_indices = np.argsort([m.distance for m in filtered_matches])[:max_features]
        pts1 = pts1[sorted_indices]
        pts2 = pts2[sorted_indices]
    else:
        pts1 = np.empty((0, 2), dtype=np.float32)
        pts2 = np.empty((0, 2), dtype=np.float32)
    
    return pts1, pts2


def select_spatially_balanced_indices(
    theta: np.ndarray,
    phi: np.ndarray,
    max_points: int,
    num_theta_bins: int = 8,
    num_phi_bins: int = 6
) -> np.ndarray:
    """空間分散フィルタリング（Legacy互換関数）
    
    theta, phi の角度分布から自動スケーリングされたグリッドにより、
    空間的に分散された特徴点のインデックスを抽出します。
    
    Args:
        theta: 各特徴点の水平方向視線角度（ラジアン） (N,)
        phi: 各特徴点の垂直方向視線角度（ラジアン） (N,)
        max_points: 選出する最大点数
        num_theta_bins: θ方向のグリッド分割数
        num_phi_bins: φ方向のグリッド分割数
    
    Returns:
        selected_indices: 選出された特徴点のインデックス配列
    """
    # 自動スケーリング：極端な外れ値の影響を除外
    theta_min, theta_max = np.percentile(theta, [1, 99])
    phi_min, phi_max = np.percentile(phi, [1, 99])
    
    theta_edges = np.linspace(theta_min, theta_max, num_theta_bins + 1)
    phi_edges = np.linspace(phi_min, phi_max, num_phi_bins + 1)
    
    theta_bins = np.digitize(theta, theta_edges) - 1
    phi_bins = np.digitize(phi, phi_edges) - 1
    priority = theta**2 + phi**2
    grid = {}
    for i, (tb, pb) in enumerate(zip(theta_bins, phi_bins)):
        if 0 <= tb < num_theta_bins and 0 <= pb < num_phi_bins:
            key = (tb, pb)
            grid.setdefault(key, []).append((priority[i], i))
    for key in grid:
        grid[key].sort(reverse=True)
    selected_indices = []
    while len(selected_indices) < max_points:
        any_added = False
        for key in sorted(grid.keys()):
            if grid[key]:
                _, idx = grid[key].pop(0)
                selected_indices.append(idx)
                any_added = True
                if len(selected_indices) == max_points:
                    break
        if not any_added:
            break
    return np.array(selected_indices)
