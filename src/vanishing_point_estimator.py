"""消失点推定モジュール

特徴点マッチング結果から、RANSACを用いて消失点を推定します。

Author: Camera Car Simulator Project
Created: 2025-10-16
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import logging
import numpy as np


@dataclass
class VanishingPointConfig:
    """消失点推定設定（Phase 1専用）

    このクラスは消失点推定処理（Phase 1）で使用されるパラメータのみを
    定義します。カメラ姿勢推定（Phase 2）で使用される角度制約パラメータ
    （max_abs_yaw_degrees, max_abs_pitch_degrees）は、EstimationConfigに
    定義されています。

    Attributes:
        enable_feature_based_vp: 特徴点ベース消失点推定を有効化
        frame_skip_for_vp: 消失点推定用のフレームスキップ間隔（1=スキップなし、2=1フレームおき）
        vp_ransac_iterations: RANSAC反復回数
        vp_ransac_min_samples: RANSAC最小サンプル数（2直線で交点計算）
        vp_inlier_threshold: インライア判定閾値（ピクセル）
        vp_min_inlier_ratio: 最小インライア比率（0〜1）
        vp_min_inlier_count: 最小インライア数
        min_movement_threshold: 最小移動閾値（ピクセル、小さすぎる移動を除外）
        max_movement_threshold: 最大移動閾値（ピクセル、大きすぎる移動を除外）
        enable_feature_vp_in_phase1: Phase1での特徴点ベースVP推定を有効化
        ocr_distance_interval: OCR距離の累積間隔（mm、基準フレーム選択用）
        min_reference_frames: 最小基準フレーム数（不足時は暗部重心VPのみ）
        vp_valid_region_radius: 消失点存在可能円の半径（ピクセル）
            移動ベクトルの後方延長線がこの円の内側を通らない場合、
            ミスマッチとして除外する。0で無効化。
            中心座標: レンズ中心 (center_x + cx, center_y + cy)

    Notes:
        frame_skip_for_vp:
        - 1: 全フレームで推定実行（デフォルト）
        - 2: 1フレームおきに推定実行（フレーム0, 2, 4, ...）
        - 背景: 30fpsの動画では毎フレーム処理するとdz（フレーム間移動距離）が小さすぎる（7-13mm = 0.6-1.0px）
        - 1フレームおきに処理することで、dzが16-17mm（1.2-1.4px）となり、推定精度が向上
        - 期待成功率: 3.3% → 60-80%
    """

    enable_feature_based_vp: bool = False  # Task 4.2.6決定: 暗部中心法をデフォルトに設定
    frame_skip_for_vp: int = 1
    vp_ransac_iterations: int = 1000
    vp_ransac_min_samples: int = 2
    vp_inlier_threshold: float = 5.0
    vp_min_inlier_ratio: float = 0.5
    vp_min_inlier_count: int = 20
    min_movement_threshold: float = 1.0
    max_movement_threshold: float = 100.0

    # Phase1: 特徴点ベースVP推定設定（累積10mm間隔）
    enable_feature_vp_in_phase1: bool = True
    ocr_distance_interval: float = 10.0
    min_reference_frames: int = 3

    # 角度制約付きペア選択（Task B.1, 改善版: 60°-120°）
    vp_min_angle_diff: float = 60.0  # 最小角度差（度）- より直交に近いペアを選択
    vp_max_angle_diff: float = 120.0  # 最大角度差（度）- より直交に近いペアを選択

    # 外れ値除去設定
    iqr_multiplier: float = 1.5

    # アルゴリズム選択
    vp_estimation_method: str = "intersection_centroid"  # "ransac" or "intersection_centroid"

    # 交点集合法設定
    outlier_removal_method: str = "iqr"  # "iqr", "mad", or "none"
    mad_threshold: float = 3.0           # MAD法の閾値
    min_intersection_count: int = 10     # 最小交点数
    min_inlier_count: int = 5            # 外れ値除去後の最小点数

    # 消失点存在可能円フィルタ（ミスマッチ除外用）
    vp_valid_region_radius: float = 300.0  # 消失点存在可能円の半径（ピクセル）、0で無効化

    # TASK-17: max_abs_yaw_degrees, max_abs_pitch_degrees は EstimationConfig に移動
    # （Phase 2カメラ姿勢推定で使用されるため、Phase 1専用のこのクラスからは削除）

    # 消失点変動量閾値（BUG-012追加）
    vp_magnitude_threshold: float = 50.0  # 消失点変動量閾値(px)

    # 🆕 BUG-013修正: 局所異常判定ウィンドウサイズ
    vanishing_point_outlier_window_size: int = 50
    """局所異常判定のウィンドウサイズ（フレーム数）

    各フレームを中心に ±window_size/2 の範囲で統計値を計算し、
    異常判定を行う。これにより処理範囲に依存しない一貫性のある判定を実現。

    推奨値:
    - 50: デフォルト（約1.7秒分、30fps想定）
    - 30: 短い動画や急激な変化が多い場合
    - 100: 長い動画で緩やかな変化が主体の場合

    Note:
        BUG-013修正により、グローバル統計ベース判定からスライディングウィンドウ方式の
        局所統計ベース判定に変更。処理範囲（--start, --end）によって同じフレームの
        判定結果が変わる問題を解決。
    """



@dataclass
class VanishingPointEstimationResult:
    """消失点推定結果

    Attributes:
        success: 推定成功フラグ
        vp_x: 消失点x座標（ピクセル）
        vp_y: 消失点y座標（ピクセル）
        inliers: インライアマスク (N,) bool配列
        inlier_count: インライア数
        inlier_ratio: インライア比率
        residual_mean: 残差平均（ピクセル）
        residual_std: 残差標準偏差（ピクセル）
        failure_reason: 失敗理由（失敗時のみ）
    """

    success: bool
    vp_x: Optional[float] = None
    vp_y: Optional[float] = None
    inliers: Optional[np.ndarray] = None
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    residual_mean: Optional[float] = None
    residual_std: Optional[float] = None
    failure_reason: Optional[str] = None


class FeatureBasedVPEstimator:
    """特徴点ベース消失点推定器

    フレーム座標系の特徴点マッチング結果から、
    RANSACを用いて消失点を推定します。

    処理フロー:
    1. フレームスキップ判定（frame_skip_for_vp > 1の場合）
    2. 移動ベクトルフィルタリング（小さすぎる/大きすぎる移動を除外）
    3. 移動ベクトルから直線パラメータを計算
    4. RANSAC直線交点推定
    5. 最小二乗法で最終推定
    6. インライア検証

    Attributes:
        config: 消失点推定設定
        logger: ロガー
        frame_counter: フレームカウンタ（スキップ判定用）
    """

    def __init__(self, config: VanishingPointConfig):
        """初期化

        Args:
            config: 消失点推定設定
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.frame_counter = 0

        skip_info = f", frame_skip={config.frame_skip_for_vp}" if config.frame_skip_for_vp > 1 else ""
        self.logger.info(
            f"FeatureBasedVPEstimator initialized: "
            f"iterations={config.vp_ransac_iterations}, "
            f"threshold={config.vp_inlier_threshold}px, "
            f"min_inliers={config.vp_min_inlier_count}"
            f"{skip_info}"
        )

    def estimate(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        image_center: Optional[Tuple[float, float]] = None
    ) -> VanishingPointEstimationResult:
        """消失点推定のメインメソッド

        Args:
            prev_points: 前フレームの特徴点 (N, 2) [u, v]
            curr_points: 現フレームの特徴点 (N, 2) [u, v]
            image_center: 画像中心座標 (x, y)、Noneの場合はデフォルト(640, 360)

        Returns:
            VanishingPointEstimationResult: 推定結果
        """
        # デフォルト値設定
        if image_center is None:
            image_center = (640.0, 360.0)  # 1280x720の中心
        
        # フレームカウンタを増加
        self.frame_counter += 1

        # フレームスキップ判定
        if self.config.frame_skip_for_vp > 1:
            if self.frame_counter % self.config.frame_skip_for_vp != 0:
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason=f"Skipped frame (frame_skip={self.config.frame_skip_for_vp})"
                )

        # アルゴリズム選択
        if self.config.vp_estimation_method == "intersection_centroid":
            return self.estimate_vp_intersection_centroid(prev_points, curr_points, image_center)
        elif self.config.vp_estimation_method == "ransac":
            return self.estimate_vp_ransac(prev_points, curr_points)
        else:
            self.logger.error(f"Unknown VP estimation method: {self.config.vp_estimation_method}")
            return VanishingPointEstimationResult(
                success=False,
                failure_reason=f"Unknown method: {self.config.vp_estimation_method}"
            )



    def estimate_vp_ransac(
        self, prev_points: np.ndarray, curr_points: np.ndarray
    ) -> VanishingPointEstimationResult:
        """RANSAC法でVPを推定（従来の実装）

        Args:
            prev_points: 前フレームの特徴点 (N, 2)
            curr_points: 現フレームの特徴点 (N, 2)

        Returns:
            VanishingPointEstimationResult
        """
        try:
            # 入力検証
            if prev_points is None or curr_points is None:
                return VanishingPointEstimationResult(
                    success=False, failure_reason="Input points are None"
                )

            if len(prev_points) != len(curr_points):
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason=f"Point count mismatch: {len(prev_points)} vs {len(curr_points)}",
                )

            if len(prev_points) < self.config.vp_min_inlier_count:
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason=f"Insufficient points: {len(prev_points)} < {self.config.vp_min_inlier_count}",
                )

            # Step 1: 移動ベクトル計算とフィルタリング
            movement_vectors = curr_points - prev_points
            movement_magnitudes = np.linalg.norm(movement_vectors, axis=1)

            valid_mask = (
                movement_magnitudes >= self.config.min_movement_threshold
            ) & (movement_magnitudes <= self.config.max_movement_threshold)

            if np.sum(valid_mask) < self.config.vp_min_inlier_count:
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason=f"Insufficient valid movements: {np.sum(valid_mask)} < {self.config.vp_min_inlier_count}",
                )

            prev_points_filtered = prev_points[valid_mask]
            curr_points_filtered = curr_points[valid_mask]

            self.logger.debug(
                f"Movement filtering: {len(prev_points)} → {len(prev_points_filtered)} points"
            )

            # Step 2: 直線パラメータ計算
            lines = self._matches_to_lines(prev_points_filtered, curr_points_filtered)

            # Step 3: RANSAC直線交点推定
            vp, inliers_filtered = self._ransac_vanishing_point(lines)

            if vp is None:
                return VanishingPointEstimationResult(
                    success=False, failure_reason="RANSAC failed to find vanishing point"
                )

            # Step 4: インライア検証
            inlier_count = np.sum(inliers_filtered)
            inlier_ratio = inlier_count / len(lines)

            if inlier_count < self.config.vp_min_inlier_count:
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason=f"Insufficient inliers: {inlier_count} < {self.config.vp_min_inlier_count}",
                )

            if inlier_ratio < self.config.vp_min_inlier_ratio:
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason=f"Low inlier ratio: {inlier_ratio:.2%} < {self.config.vp_min_inlier_ratio:.2%}",
                )

            # Step 5: 残差計算
            inlier_lines = [line for i, line in enumerate(lines) if inliers_filtered[i]]
            residuals = np.array(
                [self._point_to_line_distance(vp, line) for line in inlier_lines]
            )
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)

            # Step 6: 元のインデックスに対応するinliersマスクを作成
            inliers_original = np.zeros(len(prev_points), dtype=bool)
            inliers_original[valid_mask] = inliers_filtered

            self.logger.info(
                f"VP estimation (ransac) succeeded: vp=({vp[0]:.1f}, {vp[1]:.1f}), "
                f"inliers={inlier_count}/{len(prev_points)} ({inlier_ratio:.2%}), "
                f"residual={residual_mean:.2f}±{residual_std:.2f}px"
            )

            return VanishingPointEstimationResult(
                success=True,
                vp_x=float(vp[0]),
                vp_y=float(vp[1]),
                inliers=inliers_original,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                residual_mean=residual_mean,
                residual_std=residual_std,
            )

        except Exception as e:
            self.logger.error(f"VP estimation (ransac) error: {e}", exc_info=True)
            return VanishingPointEstimationResult(
                success=False, failure_reason=f"Exception: {str(e)}"
            )

    def estimate_vp_intersection_centroid(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        image_center: Tuple[float, float] = (640.0, 360.0)
    ) -> VanishingPointEstimationResult:
        """交点集合の重心法でVPを推定

        Args:
            prev_points: 前フレームの特徴点 (N, 2)
            curr_points: 現フレームの特徴点 (N, 2)
            image_center: 画像中心座標 (x, y)

        Returns:
            VanishingPointEstimationResult
        """
        try:
            # 入力検証
            if prev_points is None or curr_points is None:
                return VanishingPointEstimationResult(
                    success=False, failure_reason="Input points are None"
                )

            if len(prev_points) != len(curr_points):
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason=f"Point count mismatch: {len(prev_points)} vs {len(curr_points)}",
                )

            # Step 1: 移動ベクトルフィルタリング（既存と同じ）
            movement_vectors = curr_points - prev_points
            movement_magnitudes = np.linalg.norm(movement_vectors, axis=1)

            valid_mask = (
                movement_magnitudes >= self.config.min_movement_threshold
            ) & (movement_magnitudes <= self.config.max_movement_threshold)

            if np.sum(valid_mask) < 3:
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason=f"Insufficient valid movements: {np.sum(valid_mask)}",
                )

            prev_points_filtered = prev_points[valid_mask]
            curr_points_filtered = curr_points[valid_mask]

            # Step 1.5: ミスマッチフィルタ
            if self.config.vp_valid_region_radius > 0 and image_center is not None:
                mismatch_mask = self._filter_mismatched_movements_vp(
                    prev_points_filtered,
                    curr_points_filtered,
                    image_center
                )
                
                if np.sum(mismatch_mask) < 3:
                    return VanishingPointEstimationResult(
                        success=False,
                        failure_reason=f"Insufficient points after mismatch filter: {np.sum(mismatch_mask)}",
                    )
                
                prev_points_filtered = prev_points_filtered[mismatch_mask]
                curr_points_filtered = curr_points_filtered[mismatch_mask]

            # Step 2: 直線パラメータと角度を計算
            lines, angles = self._matches_to_lines_with_angles(
                prev_points_filtered, curr_points_filtered
            )

            if len(lines) < 3:
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason=f"Insufficient lines: {len(lines)}"
                )

            # Step 3: 角度制約付きペア選択
            pairs = self._select_angle_constrained_pairs(lines, angles)

            if len(pairs) < 3:
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason=f"Insufficient valid pairs: {len(pairs)}"
                )

            # Step 4: 全ペアの交点計算
            intersections = self._compute_all_intersections(lines, pairs, prev_points_filtered, curr_points_filtered)

            if len(intersections) < self.config.min_intersection_count:
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason=f"Insufficient intersections: {len(intersections)}"
                )

            # Step 5: 外れ値除去
            if self.config.outlier_removal_method == "iqr":
                inliers = self._remove_outliers_iqr(intersections)
            elif self.config.outlier_removal_method == "mad":
                inliers = self._remove_outliers_mad(
                    intersections, self.config.mad_threshold
                )
            else:
                inliers = intersections  # 外れ値除去なし

            if len(inliers) < self.config.min_inlier_count:
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason=f"Insufficient inliers after outlier removal: {len(inliers)}"
                )

            inlier_ratio = len(inliers) / len(intersections)

            # Step 6: 重心計算
            vp = self._compute_centroid(inliers)

            if vp is None:
                return VanishingPointEstimationResult(
                    success=False,
                    failure_reason="Failed to compute centroid"
                )

            # Step 7: 残差計算（評価用）
            residuals = np.linalg.norm(inliers - vp, axis=1)
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)

            self.logger.info(
                f"VP estimation (intersection_centroid) succeeded: "
                f"vp=({vp[0]:.1f}, {vp[1]:.1f}), "
                f"inliers={len(inliers)}/{len(intersections)} ({inlier_ratio:.2%}), "
                f"residual={residual_mean:.2f}±{residual_std:.2f}px"
            )

            return VanishingPointEstimationResult(
                success=True,
                vp_x=float(vp[0]),
                vp_y=float(vp[1]),
                inliers=None,  # 交点のinliers（直線のinliersではない）
                inlier_count=len(inliers),
                inlier_ratio=inlier_ratio,
                residual_mean=residual_mean,
                residual_std=residual_std,
            )

        except Exception as e:
            self.logger.error(f"VP estimation (intersection_centroid) error: {e}", exc_info=True)
            return VanishingPointEstimationResult(
                success=False, failure_reason=f"Exception: {str(e)}"
            )

    def _compute_centroid(self, inliers: np.ndarray) -> Optional[np.ndarray]:
        """交点群の重心を計算

        Args:
            inliers: 外れ値除去後の交点配列 (K, 2)

        Returns:
            vp: 消失点座標 (2,) or None
        """
        if len(inliers) == 0:
            return None

        # 単純平均
        vp = np.mean(inliers, axis=0)

        return vp


    def _matches_to_lines(
        self, prev_points: np.ndarray, curr_points: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """特徴点ペアから直線パラメータを計算

        Args:
            prev_points: 前フレームの特徴点 (N, 2)
            curr_points: 現フレームの特徴点 (N, 2)

        Returns:
            lines: 直線パラメータリスト [(a, b, c), ...]
                   直線の式: ax + by + c = 0
        """
        lines = []

        for (u1, v1), (u2, v2) in zip(prev_points, curr_points):
            # 移動ベクトル
            du = u2 - u1
            dv = v2 - v1

            # 直線の式: ax + by + c = 0
            # 点(u1, v1)を通り、方向ベクトル(du, dv)の直線
            a = dv
            b = -du
            c = -(a * u1 + b * v1)

            # 正規化
            norm = np.sqrt(a**2 + b**2)
            if norm > 1e-10:  # ゼロ除算回避
                a /= norm
                b /= norm
                c /= norm
                lines.append((a, b, c))

        return lines


    def _matches_to_lines_with_angles(
        self, prev_points: np.ndarray, curr_points: np.ndarray
    ) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
        """特徴点ペアから直線パラメータと角度を計算

        Args:
            prev_points: 前フレームの特徴点 (N, 2)
            curr_points: 現フレームの特徴点 (N, 2)

        Returns:
            lines: 直線パラメータリスト [(a, b, c), ...]
                   直線の式: ax + by + c = 0
            angles: 各直線の角度（度） [θ1, θ2, ..., θN]
        """
        lines = []
        angles = []

        for (u1, v1), (u2, v2) in zip(prev_points, curr_points):
            # 移動ベクトル
            du = u2 - u1
            dv = v2 - v1

            # 角度を計算（-180°～180°）
            angle = np.arctan2(dv, du) * 180.0 / np.pi
            # 0°～360°に正規化
            if angle < 0:
                angle += 360.0

            # 直線パラメータ（既存と同じ）
            a = dv
            b = -du
            c = -(a * u1 + b * v1)

            # 正規化
            norm = np.sqrt(a**2 + b**2)
            if norm > 1e-10:
                a /= norm
                b /= norm
                c /= norm
                lines.append((a, b, c))
                angles.append(angle)

        return lines, np.array(angles)

    def _select_angle_constrained_pairs(
        self,
        lines: List[Tuple[float, float, float]],
        angles: np.ndarray
    ) -> List[Tuple[int, int]]:
        """角度制約を満たす直線ペアを全て列挙

        Args:
            lines: 直線リスト
            angles: 各直線の角度（度）

        Returns:
            pairs: [(i, j), ...] インデックスペアリスト
        """
        N = len(lines)
        pairs = []

        min_angle_diff = self.config.vp_min_angle_diff
        max_angle_diff = self.config.vp_max_angle_diff

        for i in range(N):
            for j in range(i + 1, N):
                # 角度差を計算（0°～180°）
                angle_diff = abs(angles[i] - angles[j])
                if angle_diff > 180.0:
                    angle_diff = 360.0 - angle_diff

                # 角度制約をチェック
                if min_angle_diff <= angle_diff <= max_angle_diff:
                    pairs.append((i, j))

        self.logger.debug(
            f"Selected {len(pairs)} pairs from {len(lines)} lines "
            f"(angle constraint: {min_angle_diff}°-{max_angle_diff}°)"
        )

        return pairs

    def _ransac_vanishing_point(
        self, lines: List[Tuple[float, float, float]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """RANSAC直線交点推定

        Args:
            lines: 直線パラメータリスト [(a, b, c), ...]

        Returns:
            vp: 消失点座標 (2,) or None
            inliers: インライアマスク (N,) or None
        """
        N = len(lines)

        if N < self.config.vp_ransac_min_samples:
            self.logger.warning(f"Insufficient lines for RANSAC: {N}")
            return None, None

        best_vp = None
        best_inliers = None
        best_inlier_count = 0

        rng = np.random.RandomState(42)

        for iteration in range(self.config.vp_ransac_iterations):
            # Step 1: ランダムに2直線を選択
            indices = rng.choice(N, self.config.vp_ransac_min_samples, replace=False)
            line1 = lines[indices[0]]
            line2 = lines[indices[1]]

            # Step 2: 交点を計算
            vp_candidate = self._line_intersection(line1, line2)

            if vp_candidate is None:
                continue

            # Step 3: 全直線との距離を計算
            distances = np.array(
                [self._point_to_line_distance(vp_candidate, line) for line in lines]
            )

            # Step 4: インライア判定
            inliers = distances < self.config.vp_inlier_threshold
            inlier_count = np.sum(inliers)

            # Step 5: 最良モデル更新
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_vp = vp_candidate
                best_inliers = inliers

        if best_vp is None:
            return None, None

        # Step 6: インライアで最終推定（最小二乗法）
        inlier_lines = [line for i, line in enumerate(lines) if best_inliers[i]]
        vp_final = self._least_squares_vp(inlier_lines)

        if vp_final is None:
            vp_final = best_vp

        self.logger.debug(
            f"RANSAC completed: {best_inlier_count}/{N} inliers "
            f"({best_inlier_count/N:.2%})"
        )

        return vp_final, best_inliers


    def _is_intersection_behind_movement(
        self,
        intersection: np.ndarray,
        prev_point: np.ndarray,
        curr_point: np.ndarray
    ) -> bool:
        """交点が移動ベクトルのマイナス側（後方、消失点方向）にあるか判定
        
        Args:
            intersection: 交点座標 (2,)
            prev_point: 前フレーム特徴点 (2,)
            curr_point: 現フレーム特徴点 (2,)
        
        Returns:
            bool: マイナス側（消失点方向）ならTrue
        """
        # 移動ベクトル: curr_point - prev_point
        movement = curr_point - prev_point
        
        # 交点へのベクトル: intersection - curr_point
        to_intersection = intersection - curr_point
        
        # 内積が負 → 交点は移動ベクトルの逆方向（消失点方向）
        dot_product = np.dot(movement, to_intersection)
        
        return dot_product < 0

    @staticmethod
    def _line_intersection(
        line1: Tuple[float, float, float], line2: Tuple[float, float, float]
    ) -> Optional[np.ndarray]:
        """2直線の交点を計算

        Args:
            line1: 直線1 (a1, b1, c1)
            line2: 直線2 (a2, b2, c2)

        Returns:
            交点 (x, y) or None（平行な場合）
        """
        a1, b1, c1 = line1
        a2, b2, c2 = line2

        # 行列式
        det = a1 * b2 - a2 * b1

        if abs(det) < 1e-10:  # 平行
            return None

        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det

        return np.array([x, y])

    @staticmethod
    def _point_to_line_distance(
        point: np.ndarray, line: Tuple[float, float, float]
    ) -> float:
        """点から直線までの距離

        Args:
            point: 点 (x, y)
            line: 直線 (a, b, c)

        Returns:
            距離
        """
        a, b, c = line
        x, y = point

        # 正規化されているので |ax + by + c|
        return abs(a * x + b * y + c)

    def _least_squares_vp(
        self, lines: List[Tuple[float, float, float]]
    ) -> Optional[np.ndarray]:
        """最小二乗法で消失点を推定

        Args:
            lines: 直線パラメータリスト [(a, b, c), ...]

        Returns:
            消失点 (x, y) or None
        """
        if len(lines) < 2:
            return None

        A_mat = []
        b_vec = []

        for a, b, c in lines:
            # ax + by + c = 0 を ax + by = -c の形に
            A_mat.append([a, b])
            b_vec.append(-c)

        A_mat = np.array(A_mat)  # (N, 2)
        b_vec = np.array(b_vec)  # (N,)

        try:
            # 最小二乗解: (A^T A)^-1 A^T b
            vp, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            return vp
        except np.linalg.LinAlgError:
            return None

    def _compute_all_intersections(
        self,
        lines: List[Tuple[float, float, float]],
        pairs: List[Tuple[int, int]],
        prev_points: np.ndarray,
        curr_points: np.ndarray
    ) -> np.ndarray:
        """全ペアの交点を計算（マイナス側延長線のみ）

        Args:
            lines: 直線リスト
            pairs: ペアインデックスリスト
            prev_points: 前フレーム特徴点 (N, 2)
            curr_points: 現フレーム特徴点 (N, 2)

        Returns:
            intersections: 交点座標配列 (M, 2)
        """
        intersections = []

        for i, j in pairs:
            line1 = lines[i]
            line2 = lines[j]

            vp = self._line_intersection(line1, line2)

            if vp is not None:
                # 画像範囲の拡張範囲内の交点のみ保持
                if self._is_valid_vp_position(vp):
                    # 両方の移動ベクトルのマイナス側にある交点のみ保持
                    is_behind_i = self._is_intersection_behind_movement(
                        vp, prev_points[i], curr_points[i]
                    )
                    is_behind_j = self._is_intersection_behind_movement(
                        vp, prev_points[j], curr_points[j]
                    )
                    
                    if is_behind_i and is_behind_j:
                        intersections.append(vp)

        self.logger.debug(f"Computed {len(intersections)} intersections (minus-side only)")

        return np.array(intersections) if intersections else np.array([]).reshape(0, 2)

    def _is_valid_vp_position(self, vp: np.ndarray) -> bool:
        """VP位置が妥当な範囲内かをチェック

        Args:
            vp: VP座標 (2,)

        Returns:
            valid: 妥当ならTrue
        """
        # 画像サイズの2倍の範囲まで許容
        # （VPは画像外にあることも多い）
        # Note: image_width, image_heightはconfigから取得する必要があるため、
        # 代わりに大きな範囲（例: -5000～5000）を使用
        x_min = -5000.0
        x_max = 5000.0
        y_min = -5000.0
        y_max = 5000.0

        return (x_min <= vp[0] <= x_max) and (y_min <= vp[1] <= y_max)


    def _filter_mismatched_movements_vp(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        image_center: Tuple[float, float]
    ) -> np.ndarray:
        """VP推定用のミスマッチフィルタ
        
        Args:
            prev_points: 前フレーム特徴点 (N, 2)
            curr_points: 現フレーム特徴点 (N, 2)
            image_center: 画像中心座標 (x, y)
        
        Returns:
            valid_mask: 有効な移動ベクトルのマスク (N,) bool
        """
        vp_valid_radius = self.config.vp_valid_region_radius
        cx_vp, cy_vp = image_center
        
        valid_mask = []
        
        for (u1, v1), (u2, v2) in zip(prev_points, curr_points):
            dx = u2 - u1
            dy = v2 - v1
            
            # 移動ベクトルの後方延長線が円の内側を通過するかチェック
            a = dy
            b = -dx
            c = dx * v2 - dy * u2
            
            norm = np.sqrt(a**2 + b**2)
            if norm < 1e-10:
                valid_mask.append(False)
                continue
            
            distance = abs(a * cx_vp + b * cy_vp + c) / norm
            
            # 【追加】方向判定
            # 移動ベクトル (dx, dy) と 現在点→円中心 (cx_vp - u2, cy_vp - v2) の内積
            to_vp_center = (cx_vp - u2, cy_vp - v2)
            dot_product = dx * to_vp_center[0] + dy * to_vp_center[1]
            
            # 円中心が移動ベクトルの逆方向（後方延長側）にあり、かつ距離が閾値以内の場合のみ有効
            is_minus_side = (dot_product < 0)
            is_within_radius = (distance < vp_valid_radius)
            
            valid_mask.append(is_minus_side and is_within_radius)
        
        result_mask = np.array(valid_mask)
        
        filtered_count = len(result_mask) - np.sum(result_mask)
        if filtered_count > 0:
            self.logger.debug(
                f"VP mismatch filter (direction check): {len(result_mask)} → {np.sum(result_mask)} "
                f"({filtered_count} removed: radius+direction)"
            )
        
        return result_mask

    def _remove_outliers_iqr(self, intersections: np.ndarray) -> np.ndarray:
        """IQR法で外れ値を除去

        Args:
            intersections: 交点座標配列 (M, 2)

        Returns:
            inliers: 外れ値除去後の交点配列 (K, 2)
        """
        if len(intersections) == 0:
            return intersections

        # X座標とY座標それぞれでIQR計算
        x_coords = intersections[:, 0]
        y_coords = intersections[:, 1]

        # X座標のIQR
        q1_x = np.percentile(x_coords, 25)
        q3_x = np.percentile(x_coords, 75)
        iqr_x = q3_x - q1_x
        multiplier = self.config.iqr_multiplier  # デフォルト1.5
        lower_x = q1_x - multiplier * iqr_x
        upper_x = q3_x + multiplier * iqr_x

        # Y座標のIQR
        q1_y = np.percentile(y_coords, 25)
        q3_y = np.percentile(y_coords, 75)
        iqr_y = q3_y - q1_y
        lower_y = q1_y - multiplier * iqr_y
        upper_y = q3_y + multiplier * iqr_y

        # 両方の範囲内の点のみを保持
        mask = (
            (x_coords >= lower_x) & (x_coords <= upper_x) &
            (y_coords >= lower_y) & (y_coords <= upper_y)
        )

        inliers = intersections[mask]

        self.logger.debug(
            f"IQR outlier removal: {len(intersections)} → {len(inliers)} "
            f"({len(inliers)/len(intersections):.2%})"
        )

        return inliers

    def _remove_outliers_mad(
        self, intersections: np.ndarray, threshold: float = 3.0
    ) -> np.ndarray:
        """MAD法で外れ値を除去

        Args:
            intersections: 交点座標配列 (M, 2)
            threshold: MADの何倍を外れ値とするか

        Returns:
            inliers: 外れ値除去後の交点配列 (K, 2)
        """
        if len(intersections) == 0:
            return intersections

        # 中央値
        median = np.median(intersections, axis=0)

        # 各点から中央値までの距離
        distances = np.linalg.norm(intersections - median, axis=1)

        # MAD（中央絶対偏差）
        mad = np.median(distances)

        if mad < 1e-10:  # MADが0に近い場合
            return intersections

        # 修正Z-score
        modified_z_scores = 0.6745 * distances / mad

        # 閾値以下の点のみを保持
        mask = modified_z_scores < threshold
        inliers = intersections[mask]

        self.logger.debug(
            f"MAD outlier removal: {len(intersections)} → {len(inliers)} "
            f"({len(inliers)/len(intersections):.2%})"
        )

        return inliers
