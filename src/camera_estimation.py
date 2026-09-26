"""カメラ位置・姿勢推定モジュール

フレーム間の特徴点マッチングから移動量を推定し、
暗部重心からカメラの姿勢方向を推定します。

主な機能:
- 暗部重心検出による消失点推定
- カメラ姿勢推定（θ, φ）
- フレーム間移動量推定（6DoF）
- カメラ状態管理
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.config import Config, EstimationConfig, PoseConfig
from src.coordinate_transform import (
    CoordinateTransformer,
    CameraModel,
    R_c2w,
    CylinderIntersectionError
)


# ============================================================================
# 定数定義
# ============================================================================

# 姿勢角の初期値が絶対制限範囲外の場合に使用する微小角度範囲（radians）
ANGLE_EPSILON = 0.01  # 約0.57度

# 位置の初期値がbounds境界にある場合に使用するオフセット（mm）
# soft_l1損失関数が境界で正しく機能するよう、境界から少し内側に初期値を設定
POSITION_EPSILON = 0.01  # 0.01mm


# ============================================================================
# カスタム例外クラス
# ============================================================================

class CameraEstimationError(Exception):
    """カメラ推定関連のベース例外"""
    pass


class DarkRegionNotFoundError(CameraEstimationError):
    """暗部領域が見つからない
    
    Attributes:
        fallback_point: エラー時のフォールバック座標 (x, y)
            Noneの場合はフォールバックなし
    """
    def __init__(self, message: str, fallback_point: Optional[Tuple[float, float]] = None):
        """初期化
        
        Args:
            message: エラーメッセージ
            fallback_point: フォールバック座標 (x, y)
        """
        super().__init__(message)
        self.fallback_point = fallback_point


class MotionEstimationError(CameraEstimationError):
    """移動量推定エラー"""
    pass


# ============================================================================
# CameraEstimator クラス
# ============================================================================

class CameraEstimator:
    """カメラ位置・姿勢推定クラス
    
    フレーム間の特徴点マッチングから移動量を推定し、
    暗部重心からカメラの姿勢方向を推定します。
    
    Attributes:
        config: 推定設定
        transformer: 座標変換器
        logger: ロガー
    """
    
    def __init__(
        self,
        config: EstimationConfig,
        transformer: CoordinateTransformer,
        pixels_per_mm: float = 1.0
    ):
        """初期化

        Args:
            config: 推定設定
            transformer: 座標変換器（pixel <-> world変換に使用）
            pixels_per_mm: ピクセルからmmへの変換係数（停止フレーム検出に使用）
        """
        self.config = config
        self.transformer = transformer
        self.logger = logging.getLogger(__name__)

        # 円筒座標マッチャー初期化
        if (config.feature_matching.use_cylindrical_matching and
            config.feature_matching.cylindrical is not None):
            try:
                from src.feature_matching_cylindrical import CylindricalFeatureMatcher

                self.cylindrical_matcher = CylindricalFeatureMatcher(
                    transformer=transformer,
                    config=config.feature_matching.cylindrical,
                    feature_filtering_config=config.feature_filtering,
                    static_frame_detection_config=config.static_frame_detection,
                    pixels_per_mm=pixels_per_mm
                )
                self.logger.info("円筒座標マッチャー初期化完了")
            except ImportError as e:
                self.logger.warning(
                    f"CylindricalFeatureMatcher のインポート失敗: {e}"
                )
                self.cylindrical_matcher = None
            except Exception as e:
                self.logger.error(
                    f"円筒座標マッチャー初期化エラー: {e}"
                )
                self.cylindrical_matcher = None
        else:
            self.cylindrical_matcher = None
            if config.feature_matching.use_cylindrical_matching:
                self.logger.info("従来のORBマッチング使用（円筒座標設定未指定）")
            else:
                self.logger.debug("従来のORBマッチング使用")


        # 特徴点ベース消失点推定器初期化
        if (config.vanishing_point is not None and 
            hasattr(config.vanishing_point, 'enable_feature_based_vp') and
            config.vanishing_point.enable_feature_based_vp):
            try:
                from src.vanishing_point_estimator import FeatureBasedVPEstimator
                
                self.vp_estimator = FeatureBasedVPEstimator(
                    config=config.vanishing_point
                )
                self.logger.info("特徴点ベース消失点推定器初期化完了")
            except ImportError as e:
                self.logger.warning(
                    f"FeatureBasedVPEstimator のインポート失敗: {e}"
                )
                self.vp_estimator = None
            except Exception as e:
                self.logger.error(
                    f"特徴点ベース消失点推定器初期化エラー: {e}"
                )
                self.vp_estimator = None
        else:
            self.vp_estimator = None
            if config.vanishing_point is None:
                self.logger.debug("消失点推定設定なし（暗部重心法のみ使用）")
            else:
                self.logger.debug("特徴点ベース消失点推定無効（暗部重心法のみ使用）")
        # 統計情報の初期化
        self.stats = {
            'x_constrained_count': 0,
            'y_constrained_count': 0,
            'yaw_constrained_count': 0,
            'pitch_constrained_count': 0,
            'roll_constrained_count': 0,
        }
    
    def detect_dark_region_centroid(
        self,
        frame: np.ndarray,
        threshold: int = 30,
        radius_limit: Optional[float] = None,
        target_area: Optional[int] = None,
        threshold_range: Tuple[int, int] = (1, 100)
    ) -> Tuple[float, float]:
        """暗部重心検出（適応的閾値選択対応）

        Args:
            frame: 入力フレーム画像（BGR）
            threshold: 暗部判定の閾値（0-255）
                target_areaが指定された場合は無視される
            radius_limit: 暗部検出対象の半径制限（ピクセル）
            target_area: 目標暗部ピクセル数（指定時は適応的閾値選択を使用）
                Noneの場合は固定閾値を使用（従来の動作）
            threshold_range: 適応的閾値探索範囲 (min, max)

        Returns:
            centroid_x: 重心x座標（ピクセル）
            centroid_y: 重心y座標（ピクセル）

        Raises:
            DarkRegionNotFoundError: 暗部が検出できない場合

        処理の流れ:
            - target_area指定時（適応的閾値選択）:
                1. threshold_range内で閾値を変化させながら輪郭面積を確認
                2. 面積がtarget_area以上となる最小の閾値を採用
                3. 最大輪郭の重心を計算
            - target_area未指定時（固定閾値）:
                1. グレースケール化
                2. 二値化（固定閾値処理）
                3. 最大輪郭の重心を計算
            - 暗部重心検出後、消失点周辺の明るさチェックを実施
                （出口光パターンの検出）

        Note:
            適応的閾値選択はレガシープログラムのauto_adjust_threshold_for_dark_region()
            と同じアルゴリズム。暗部面積を一定に保つことで、照明条件の変化に
            頑健な消失点検出を実現する。
        """
        # グレースケール化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        camera = self.transformer.camera

        # 半径制限マスク作成
        mask_radius = None
        if radius_limit is not None:
            Y, X = np.ogrid[:h, :w]
            dist_sq = (X - camera.cx)**2 + (Y - camera.cy)**2
            mask_radius = (dist_sq <= radius_limit**2).astype(np.uint8)

        # 適応的閾値選択
        if target_area is not None:
            best_threshold = None
            best_mask = None
            best_contour = None

            for thresh in range(threshold_range[0], threshold_range[1] + 1):
                # 二値化
                _, dark_mask = cv2.threshold(gray, thresh, 1, cv2.THRESH_BINARY_INV)

                # 半径制限適用
                if mask_radius is not None:
                    valid_mask = dark_mask & mask_radius
                else:
                    valid_mask = dark_mask

                # 輪郭検出
                contours, _ = cv2.findContours(
                    valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if not contours:
                    continue

                # 最大輪郭
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                # 目標面積以上なら採用（最小の閾値を優先）
                if area >= target_area:
                    best_threshold = thresh
                    best_mask = valid_mask
                    best_contour = largest
                    break

            if best_threshold is None:
                raise DarkRegionNotFoundError(
                    f"目標面積{target_area}ピクセル以上の暗部が見つかりません",
                    fallback_point=(camera.cx, camera.cy)
                )

            self.logger.debug(
                f"適応的閾値選択: threshold={best_threshold}, "
                f"area={cv2.contourArea(best_contour):.0f}px"
            )

            # 重心計算
            M = cv2.moments(best_contour)
            if M["m00"] == 0:
                raise DarkRegionNotFoundError(
                    "暗部重心の面積がゼロです",
                    fallback_point=(camera.cx, camera.cy)
                )

            centroid_x = M["m10"] / M["m00"]
            centroid_y = M["m01"] / M["m00"]

        # 固定閾値（従来の動作）
        else:
            # 暗部抽出（黒が1）
            _, dark_mask = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY_INV)

            # 半径制限がある場合
            if mask_radius is not None:
                dark_mask = dark_mask & mask_radius

            # 輪郭検出
            contours, _ = cv2.findContours(
                dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                raise DarkRegionNotFoundError(
                "暗部輪郭が見つかりません",
                fallback_point=(camera.cx, camera.cy)
            )

            # 最大輪郭
            largest = max(contours, key=cv2.contourArea)

            # 重心計算
            M = cv2.moments(largest)
            if M["m00"] == 0:
                raise DarkRegionNotFoundError(
                "暗部重心の面積がゼロです",
                fallback_point=(camera.cx, camera.cy)
            )

            centroid_x = M["m10"] / M["m00"]
            centroid_y = M["m01"] / M["m00"]

        # 消失点周辺の明るさチェック
        pose_config = self.config.pose
        if pose_config.brightness_check_enabled:
            # 消失点（暗部重心）を中心とした円形マスク作成
            check_radius = pose_config.brightness_check_radius
            Y_check, X_check = np.ogrid[:h, :w]
            dist_sq_check = (X_check - centroid_x)**2 + (Y_check - centroid_y)**2
            center_mask = (dist_sq_check <= check_radius**2)

            # 消失点周辺の明るいピクセル数カウント
            center_bright_pixels = np.sum(gray[center_mask] > pose_config.brightness_threshold)

            # TASK-31: 中心/周辺比較が有効な場合、周辺領域もカウント
            should_raise_error = center_bright_pixels >= pose_config.bright_pixel_count_threshold

            if should_raise_error and pose_config.brightness_center_comparison_enabled:
                # 周辺領域マスク作成（消失点周辺外 〜 dark_region_radius_limit_ratio以内）
                outer_radius = int(w * pose_config.dark_region_radius_limit_ratio)
                outer_mask = (dist_sq_check <= outer_radius**2) & ~center_mask

                # 周辺領域の明るいピクセル数カウント
                outer_bright_pixels = np.sum(gray[outer_mask] > pose_config.brightness_threshold)


                # 各領域の総ピクセル数を計算
                center_total_pixels = np.sum(center_mask)
                outer_total_pixels = np.sum(outer_mask)

                # ゼロ除算対策
                if outer_total_pixels == 0:
                    # 周辺領域がない場合は中心のみで判定
                    should_raise_error = center_bright_pixels >= pose_config.bright_pixel_count_threshold
                    self.logger.debug(
                        f"TASK-31 周辺領域なし: 中心明るさピクセル={center_bright_pixels}, "
                        f"閾値={pose_config.bright_pixel_count_threshold}, 検出={should_raise_error}"
                    )
                else:
                    # ピクセル密度を計算
                    center_density = center_bright_pixels / center_total_pixels
                    outer_density = outer_bright_pixels / outer_total_pixels

                    # 判定1: 中心領域の密度が周辺領域より高い (TASK-31既存ロジック)
                    density_comparison_result = center_density > outer_density

                    # 判定2: 中心密度が絶対閾値を超過（管路出口光の漸増検出）
                    density_threshold = pose_config.brightness_center_density_threshold
                    absolute_threshold_result = center_density > density_threshold

                    should_raise_error = density_comparison_result or absolute_threshold_result

                    self.logger.debug(
                        f"TASK-31 中心/周辺密度比較: "
                        f"中心密度={center_density:.4f} ({center_bright_pixels}/{center_total_pixels}), "
                        f"周辺密度={outer_density:.4f} ({outer_bright_pixels}/{outer_total_pixels}), "
                        f"絶対閾値={density_threshold:.4f}, "
                        f"密度比較={density_comparison_result}, 絶対値検出={absolute_threshold_result}, "
                        f"検出={should_raise_error}"
                    )
            else:
                self.logger.debug(
                    f"消失点周辺の明るさチェック: "
                    f"半径={check_radius}px, 明るいピクセル数={center_bright_pixels}, "
                    f"閾値={pose_config.bright_pixel_count_threshold} "
                    f"(明度>{pose_config.brightness_threshold})"
                )

            if should_raise_error:
                raise DarkRegionNotFoundError(
                    f"消失点周辺に明るい部分が存在します "
                    f"(明度>{pose_config.brightness_threshold}のピクセル数={center_bright_pixels} "
                    f">= 閾値={pose_config.bright_pixel_count_threshold})",
                    fallback_point=(camera.cx, camera.cy)
                )

        return centroid_x, centroid_y

    def estimate_camera_pose(
        self,
        frame: np.ndarray,
        roll: float,
        threshold: int = 30,
        radius_limit: Optional[float] = None,
        target_area: Optional[int] = None,
        threshold_range: Tuple[int, int] = (1, 100)
    ) -> Tuple[Optional[float], Optional[float]]:
        """カメラ姿勢推定（暗部重心から）

        Args:
            frame: 入力フレーム（BGR）
            roll: カメラのロール角（ラジアン）
            threshold: 暗部判定閾値（0-255）
            radius_limit: 暗部検出の半径制限（ピクセル）
            target_area: 目標暗部ピクセル数（適応的閾値選択用）
            threshold_range: 適応的閾値探索範囲 (min, max)

        Returns:
            theta: 左右方向角度（ラジアン）、検出失敗時はNone
            phi: 上下方向角度（ラジアン）、検出失敗時はNone

        処理の流れ:
            1. 暗部重心を検出
            2. coordinate_transform.convert_pixel_to_pose()で姿勢角に変換
        """
        try:
            # 暗部重心検出
            gx, gy = self.detect_dark_region_centroid(
                frame, threshold, radius_limit, target_area, threshold_range
            )
        except DarkRegionNotFoundError as e:
            self.logger.warning(f"暗部検出失敗: {e}")
            return None, None

        # 歪み補正: 歪んだ重心座標を理想座標に変換
        if self.transformer.calibration is not None:
            dist_coeffs = self.transformer._get_dist_coeffs(
                self.transformer.calibration
            )
            if not np.allclose(dist_coeffs, 0.0):
                pts = np.array([[gx, gy]])
                pts_undist = self.transformer.camera.undistort_points(
                    pts, dist_coeffs
                )
                gx, gy = pts_undist[0, 0], pts_undist[0, 1]

        # 消失点座標から姿勢角に変換（coordinate_transformの共通関数を使用）
        from src.coordinate_transform import convert_pixel_to_pose
        yaw, pitch = convert_pixel_to_pose(gx, gy, self.transformer.camera, roll)

        return yaw, pitch
    
    def estimate_motion(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        camera_state: Dict[str, Any],
        camera_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """フレーム間移動量推定
        
        Args:
            prev_points: 前フレームのマッチ点（ピクセル座標、N×2）
            curr_points: 現フレームのマッチ点（ピクセル座標、N×2）
            camera_state: 現在のカメラ状態
                - position: [x, y, z]
                - orientation: [roll, yaw, pitch]
            camera_params: カメラパラメータ
                - f: 焦点距離
                - center: レンズ中心
                - model: 'fisheye' or 'pinhole'
        
        Returns:
            motion: 移動量 {'dx', 'dy', 'dz', 'dtheta', 'dphi', 'droll'}
        
        Raises:
            MotionEstimationError: 推定失敗時
        
        処理の流れ:
            1. ピクセル座標→ワールド座標の角度に変換
            2. 最小二乗法で6DoF移動量を推定
            3. 移動制約を適用
        """
        # 最低必要点数: 6DOF最小二乗法に必要な最低点数
        MIN_ALGORITHMIC_POINTS = 3
        n_points = len(prev_points)
        if n_points < MIN_ALGORITHMIC_POINTS:
            raise MotionEstimationError(
                f"マッチ点数不足: {n_points} < {MIN_ALGORITHMIC_POINTS} "
                f"(最小二乗法に必要な最低点数)"
            )
        if n_points < self.config.feature_matching.min_match_count:
            self.logger.warning(
                f"マッチ点数が推奨値未満: {n_points} < "
                f"{self.config.feature_matching.min_match_count} (続行)"
            )

        # カメラ状態を展開
        x0, y0, z0 = camera_state['position']
        roll_0, yaw_0, phi_0 = camera_state['orientation']
        # 姿勢推定から得たyaw, pitchを使用（存在する場合）
        
        f = camera_params['f']
        center = camera_params['center']
        model = camera_params['model']
        
        # 前フレームの特徴点: ワールド座標系の角度に変換
        theta1, phi1 = self._pixel_to_angles_world(
            prev_points, f, center, roll_0, yaw_0, phi_0, model
        )
        
        # 現フレームの特徴点: カメラ座標系の角度
        theta2, phi2 = self._pixel_to_angles_cam(curr_points, f, center, model)
        
        # 最小二乗法による推定
        def residuals_6dof(params):
            """残差関数
            
            Args:
                params: [dx, dy, dz, droll]
            
            Returns:
                errors: 残差ベクトル
            """
            dx, dy, dz, droll, dtheta, dphi = params  # 6次元に拡張
            N = len(theta1)
            
            # カメラ位置（元と変化後）
            offset0 = (
                np.full(N, x0),
                np.full(N, y0),
                np.zeros(N)
            )
            offset1 = (
                np.full(N, x0 + dx),
                np.full(N, y0 + dy),
                np.full(N, dz)
            )
            
            roll2 = roll_0 + droll
            yaw_2 = yaw_0 + dtheta    # 変化量を加算
            phi_2 = phi_0 + dphi      # 変化量を加算
            
            # 姿勢のブレによるロール回転の後でヨー・ピッチを反映
            theta2_w, phi2_w = self._cam_to_world_angles(
                theta2, phi2, roll2, yaw_2, phi_2
            )
            
            # 前フレームの特徴点の円筒座標
            try:
                p0 = self._compute_cylinder_intersection_vect(
                    offset0, theta1, phi1
                )
            except CylinderIntersectionError:
                return np.full(N, 1e6)  # 大きなペナルティ
            
            # 現フレームの特徴点の円筒座標
            try:
                p1 = self._compute_cylinder_intersection_vect(
                    offset1, theta2_w, phi2_w
                )
            except CylinderIntersectionError:
                return np.full(N, 1e6)  # 大きなペナルティ
            
            # 有効なインデックス
            valid = (~np.isnan(p0[:, 0])) & (~np.isnan(p1[:, 0]))
            
            if not np.any(valid):
                return np.full(N, 1e6)  # 大きなペナルティ
            
            errors = np.linalg.norm(p0[valid] - p1[valid], axis=1)
            
            return errors
        
        # 初期値と制約（6次元）
        # 姿勢推定がない場合、dyaw=0, dphi=0を仮定
        max_dx = self.config.motion.max_dx
        max_dy = self.config.motion.max_dy
        max_dz = self.config.motion.max_dz
        max_dr = np.radians(self.config.motion.max_droll)  # ロール制限
        max_dtheta = np.radians(self.config.motion.max_dtheta)
        max_dphi = np.radians(self.config.motion.max_dphi)

        initial_guess = [0, 0, self.config.motion.max_dz / 2, 0, 0, 0]  # 6次元

        bounds = (
            [-max_dx, -max_dy, 0.0, -max_dr, -max_dtheta, -max_dphi],
            [max_dx, max_dy, max_dz, max_dr, max_dtheta, max_dphi]
        )
        
        try:
            result = least_squares(
                residuals_6dof,
                initial_guess,
                loss='soft_l1',
                bounds=bounds
            )
        except Exception as e:
            raise MotionEstimationError(f"最小二乗法失敗: {e}")

        dx, dy, dz, droll, dtheta, dphi = result.x

        # 移動制約の適用（追加の位置制約がある場合）
        # TODO: pos_areaによる制約を追加
        
        motion = {
            'dx': dx,
            'dy': dy,
            'dz': dz,
            'droll': droll,
            'dtheta': 0.0,  # ヨー変化量（現在は暗部推定から取得）
            'dphi': 0.0     # ピッチ変化量（現在は暗部推定から取得）
        }
        
        self.logger.debug(
            f"移動量推定完了: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, "
            f"droll={np.degrees(droll):.4f}°"
        )
        
        return motion

    def estimate_motion_from_frames(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        camera_state: Dict[str, Any],
        camera_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """フレーム画像から移動量を推定（新規メソッド）
        
        Args:
            prev_frame: 前フレーム画像（BGR）
            curr_frame: 現フレーム画像（BGR）
            camera_state: カメラ状態
                - position: [x, y, z]
                - orientation: [roll, yaw, pitch]
            camera_params: カメラパラメータ
                - f: 焦点距離
                - center: レンズ中心
                - model: 'fisheye' or 'pinhole'
                - pipe_diameter: 管路直径（mm）
        
        Returns:
            motion: 移動量辞書 {'dx', 'dy', 'dz', 'dtheta', 'dphi', 'droll'}
        
        Raises:
            MotionEstimationError: 推定失敗時
        
        処理の流れ:
            1. 特徴点マッチング（円筒座標 or 従来のORB）
            2. 既存のestimate_motion()を呼び出し
        
        Note:
            このメソッドはmain_twopass.pyから呼び出される。
            use_cylindrical_matchingフラグに応じて、円筒座標マッチングか
            従来のORBマッチングを自動選択する。
        """
        # 特徴点マッチング
        if self.cylindrical_matcher is not None:
            # 円筒座標マッチング
            try:
                prev_points, curr_points, status = self.cylindrical_matcher.extract_and_match(
                    prev_frame, curr_frame, camera_params
                )
                self.logger.debug(
                    f"円筒座標マッチング: {len(prev_points)}点"
                )

                # 静止状態・失敗チェック
                if status == "STATIC_FRAME":
                    raise MotionEstimationError("静止状態フレームのため移動量推定をスキップ")
                elif status == "FAILED" or prev_points is None:
                    raise MotionEstimationError("特徴点抽出失敗")

                # ミスマッチフィルタ（消失点存在可能円フィルタ）
                valid_mask = self._filter_mismatched_movements(prev_points, curr_points)
                if np.sum(valid_mask) < 3:
                    self.logger.warning(f"Insufficient points after mismatch filter: {np.sum(valid_mask)}")
                    # 特徴点不足の場合、フォールバック処理
                    # 既存のestimate_motion()でエラーハンドリングされる
                    pass
                else:
                    prev_points = prev_points[valid_mask]
                    curr_points = curr_points[valid_mask]
                    self.logger.debug(
                        f"ミスマッチフィルタ適用後: {len(prev_points)}点"
                    )
            except Exception as e:
                self.logger.warning(
                    f"円筒座標マッチング失敗: {e}、従来手法にフォールバック"
                )
                # フォールバック: 従来のORBマッチング
                prev_points, curr_points = self._extract_orb_features_legacy(
                    prev_frame, curr_frame
                )
        else:
            # 従来のORBマッチング（フレーム座標系）
            prev_points, curr_points = self._extract_orb_features_legacy(
                prev_frame, curr_frame
            )
        
        # 特徴点ベース消失点推定（有効な場合のみ）
        vp_result = None
        if self.vp_estimator is not None and prev_points is not None and curr_points is not None:
            try:
                vp_result = self.vp_estimator.estimate(prev_points, curr_points)
                if vp_result.success:
                    self.logger.info(
                        f"特徴点ベース消失点推定成功: "
                        f"vp=({vp_result.vp_x:.1f}, {vp_result.vp_y:.1f}), "
                        f"inliers={vp_result.inlier_count} ({vp_result.inlier_ratio:.2%})"
                    )
                else:
                    self.logger.debug(
                        f"特徴点ベース消失点推定失敗: {vp_result.failure_reason}"
                    )
            except Exception as e:
                self.logger.warning(
                    f"特徴点ベース消失点推定でエラー: {e}", exc_info=True
                )
        
        # 既存のestimate_motion()を呼び出し
        motion = self.estimate_motion(
            prev_points, curr_points, camera_state, camera_params
        )
        
        # 特徴点ベース消失点推定結果を追加
        if vp_result is not None:
            motion['vp_feature_based_success'] = vp_result.success
            if vp_result.success:
                motion['vp_feature_based_x'] = vp_result.vp_x
                motion['vp_feature_based_y'] = vp_result.vp_y
                motion['vp_feature_based_inlier_count'] = vp_result.inlier_count
                motion['vp_feature_based_inlier_ratio'] = vp_result.inlier_ratio
                motion['vp_feature_based_residual_mean'] = vp_result.residual_mean
                motion['vp_feature_based_residual_std'] = vp_result.residual_std
            else:
                motion['vp_feature_based_x'] = None
                motion['vp_feature_based_y'] = None
                motion['vp_feature_based_failure_reason'] = vp_result.failure_reason
        else:
            motion['vp_feature_based_success'] = False
            motion['vp_feature_based_x'] = None
            motion['vp_feature_based_y'] = None
        
        return motion

    def _extract_orb_features_legacy(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """従来のORB特徴点抽出（フレーム座標系）
        
        互換性のため残す。円筒座標マッチングが失敗した場合の
        フォールバックとしても使用される。
        
        Args:
            prev_frame: 前フレーム画像（BGR）
            curr_frame: 現フレーム画像（BGR）
        
        Returns:
            prev_points: 前フレームの特徴点（N×2）
            curr_points: 現フレームの特徴点（N×2）
        
        Note:
            ドーナツ状フィルタは適用しない（簡易版）。
            必要に応じて後で拡張する。
        """
        orb = cv2.ORB_create(
            nfeatures=self.config.feature_matching.max_features
        )
        
        # グレースケール化
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # 特徴点抽出
        kp1, desc1 = orb.detectAndCompute(prev_gray, None)
        kp2, desc2 = orb.detectAndCompute(curr_gray, None)
        
        if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
            raise MotionEstimationError(
                f"ORB特徴点抽出失敗: prev={len(kp1) if kp1 else 0}点, "
                f"curr={len(kp2) if kp2 else 0}点"
            )
        
        # マッチング
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < self.config.feature_matching.min_match_count:
            raise MotionEstimationError(
                f"マッチ点数不足: {len(matches)} < "
                f"{self.config.feature_matching.min_match_count}"
            )
        
        # 座標抽出
        prev_points = np.array(
            [kp1[m.queryIdx].pt for m in matches],
            dtype=np.float32
        )
        curr_points = np.array(
            [kp2[m.trainIdx].pt for m in matches],
            dtype=np.float32
        )
        
        self.logger.debug(
            f"従来のORBマッチング: {len(matches)}点"
        )
        
        return prev_points, curr_points

    def _filter_mismatched_movements(
        self,
        prev_frame_points: np.ndarray,
        curr_frame_points: np.ndarray,
        image_size: Tuple[int, int] = None,
    ) -> np.ndarray:
        """消失点存在可能円を通らない移動ベクトル（ミスマッチ）を除外

        処理フロー:
            特徴点マッチング（円筒座標系）
            ↓
            フレーム座標系変換
            ↓
            ミスマッチフィルタ ← この関数
            ↓
            移動回転量推定（最小二乗法）

        Args:
            prev_frame_points: 前フレームの特徴点（フレーム座標系） (N, 2)
            curr_frame_points: 現フレームの特徴点（フレーム座標系） (N, 2)
            image_size: (height, width) 画像サイズ、Noneの場合はself.current_frameから取得

        Returns:
            valid_mask: 有効な移動ベクトルのマスク (N,) bool
        """
        # 消失点存在可能円のパラメータ
        vp_valid_radius = self.config.vanishing_point.vp_valid_region_radius

        # 無効化チェック
        if vp_valid_radius <= 0:
            return np.ones(len(prev_frame_points), dtype=bool)

        # レンズ中心座標（画像中心 + principal point offset）
        if image_size is not None:
            height, width = image_size
        elif hasattr(self, 'current_frame') and self.current_frame is not None:
            height, width = self.current_frame.shape[:2]
        else:
            # Fallback: transformerのcameraから画像サイズ取得
            height = self.transformer.camera.image_height
            width = self.transformer.camera.image_width

        center_x = width / 2.0
        center_y = height / 2.0

        # カメラ内部パラメータ（principal point offset、レンズ中心からの画像中心のオフセット）
        # 通常は0だが、カメラキャリブレーション済みの場合は非ゼロの可能性
        cx = 0.0
        cy = 0.0

        vp_center = (center_x + cx, center_y + cy)
        
        # 各移動ベクトルをチェック
        valid_mask = []
        cx_vp, cy_vp = vp_center
        
        for (u1, v1), (u2, v2) in zip(prev_frame_points, curr_frame_points):
            dx = u2 - u1
            dy = v2 - v1
            
            # 移動ベクトルの後方延長線が円の内側を通過するかチェック
            # 直線: 点(u2, v2)を通り、方向ベクトル-(dx, dy)
            # 直線の式: dy*x - dx*y + (dx*v2 - dy*u2) = 0
            a = dy
            b = -dx
            c = dx * v2 - dy * u2
            
            norm = np.sqrt(a**2 + b**2)
            if norm < 1e-10:
                valid_mask.append(False)  # ゼロベクトル
                continue
            
            # 円中心から直線までの距離
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
        
        # ログ出力
        filtered_count = len(result_mask) - np.sum(result_mask)
        if filtered_count > 0:
            self.logger.debug(
                f"Mismatch filter (direction check): {len(result_mask)} → {np.sum(result_mask)} "
                f"({filtered_count} removed: radius+direction, {filtered_count/len(result_mask)*100:.1f}%)"
            )
        
        return result_mask




    
    def update_camera_state(
        self,
        camera_state: Dict[str, Any],
        motion: Dict[str, float]
    ) -> Dict[str, Any]:
        """カメラ状態更新
        
        Args:
            camera_state: 現在のカメラ状態
            motion: フレーム間移動量
        
        Returns:
            updated_state: 更新後のカメラ状態
        """
        import copy
        updated_state = copy.deepcopy(camera_state)
        
        # 位置更新
        updated_state['position'][0] += motion['dx']
        updated_state['position'][1] += motion['dy']
        updated_state['position'][2] += motion['dz']
        
        # 姿勢更新
        updated_state['orientation'][0] += motion['droll']
        updated_state['orientation'][1] += motion['dtheta']
        updated_state['orientation'][2] += motion['dphi']
        
        return updated_state
    
    # ========================================================================
    # プライベートヘルパーメソッド
    # ========================================================================

    def estimate_dz_from_cylindrical(
        self,
        prev_points_cyl: np.ndarray,
        curr_points_cyl: np.ndarray,
        dz_bounds: Optional[Tuple[float, float]] = None,
        pipe_radius_mm: Optional[float] = None
    ) -> Dict[str, Any]:
        """円筒座標での2変数最適化によるdz推定（ハイブリッド2段階推定の第1段階）

        dx=dy=0を仮定し、円筒座標(theta_deg, z_mm)空間で
        dz（前進量）とdroll_composite（合成回転量）を同時最適化する。

        z_mmは円筒グリッドのパラメータであり物理的な管軸方向距離ではない。
        物理距離 d = R² / z_mm（逆比例）の関係にあるため、
        z_mm空間では一定シフトモデルが成立しない。

        pipe_radius_mmが指定された場合、z_mmを物理距離dに変換してから
        残差を計算する。物理距離空間ではカメラ前進時に全点が一様に
        -dz_actualシフトするため、線形モデルが正確に成立する。

        Args:
            prev_points_cyl: 前フレームの円筒座標 (N,2) [theta_deg, z_mm]
            curr_points_cyl: 現フレームの円筒座標 (N,2) [theta_deg, z_mm]
            dz_bounds: dz探索範囲 (dz_min, dz_max)。Noneの場合はconfig値を使用
            pipe_radius_mm: 管の半径(mm)。指定時にz_mmを物理距離に変換

        Returns:
            結果辞書:
                - dz: 推定前進量(mm)
                - droll_composite: 合成回転量(度、参考値)
                - residual_mean: 残差平均
                - residual_std: 残差標準偏差
                - n_points: 使用した特徴点数
                - success: 推定成功フラグ
        """
        MIN_POINTS = 2  # 2変数(dz, droll)最適化は2点で解ける
        n_points = len(prev_points_cyl)

        if n_points < MIN_POINTS:
            self.logger.warning(
                f"第1段階: 点数不足 {n_points} < {MIN_POINTS}"
            )
            return {'dz': 0.0, 'droll_composite': 0.0,
                    'residual_mean': float('inf'), 'residual_std': float('inf'),
                    'n_points': n_points, 'success': False}

        # 成分抽出
        prev_theta = prev_points_cyl[:, 0]
        curr_theta = curr_points_cyl[:, 0]
        prev_z = prev_points_cyl[:, 1]
        curr_z = curr_points_cyl[:, 1]

        # pipe_radius_mmが未指定の場合、transformerから取得を試みる
        R = pipe_radius_mm
        if R is None and hasattr(self, 'transformer') and self.transformer is not None:
            R = getattr(self.transformer, 'pipe_radius', None)

        if R is not None and R > 0:
            # z_mmを物理距離に変換: d = R² / z_mm
            # z_mm ≈ 0 の点は遠方で数値的に不安定なため除外
            Z_MIN_THRESHOLD = 1.0  # mm
            valid_mask = (prev_z > Z_MIN_THRESHOLD) & (curr_z > Z_MIN_THRESHOLD)
            n_valid = int(np.sum(valid_mask))

            if n_valid < MIN_POINTS:
                self.logger.warning(
                    f"第1段階: z_mm変換後の有効点数不足 {n_valid} < {MIN_POINTS}"
                )
                return {'dz': 0.0, 'droll_composite': 0.0,
                        'residual_mean': float('inf'), 'residual_std': float('inf'),
                        'n_points': n_valid, 'success': False}

            prev_theta = prev_theta[valid_mask]
            curr_theta = curr_theta[valid_mask]
            prev_d = R * R / prev_z[valid_mask]
            curr_d = R * R / curr_z[valid_mask]

            # z_grid上で同一ピクセルの点（z方向変位なし）を除外
            # ORB整数座標の量子化でz_grid_diff=0となる点はdz推定に寄与しない
            z_has_motion = (prev_z[valid_mask] != curr_z[valid_mask])
            n_with_motion = int(np.sum(z_has_motion))

            if n_with_motion >= MIN_POINTS:
                # z方向に変位のある点のみ使用
                prev_theta = prev_theta[z_has_motion]
                curr_theta = curr_theta[z_has_motion]
                prev_d = prev_d[z_has_motion]
                curr_d = curr_d[z_has_motion]
                n_points = n_with_motion
                self.logger.debug(
                    f"第1段階: z変位あり点={n_with_motion}/{n_valid}, "
                    f"prev_d=[{prev_d.min():.1f}, {prev_d.max():.1f}]mm"
                )
            else:
                # z変位のある点が不足 → 全有効点を使用（θ情報も活用）
                n_points = n_valid
                self.logger.debug(
                    f"第1段階: z変位あり点不足({n_with_motion}<{MIN_POINTS}), "
                    f"全{n_valid}点使用, "
                    f"prev_d=[{prev_d.min():.1f}, {prev_d.max():.1f}]mm"
                )

            # 物理距離空間での初期値: カメラ前進時 prev_d > curr_d → dz > 0
            dz_init = float(np.median(prev_d - curr_d))
            droll_init = float(np.median(prev_theta - curr_theta))
        else:
            # フォールバック: z_mm空間のまま（後方互換性）
            dz_init = float(np.median(prev_z - curr_z))
            droll_init = float(np.median(prev_theta - curr_theta))
            prev_d = prev_z
            curr_d = curr_z

        # bounds設定
        if dz_bounds is not None:
            dz_min, dz_max = dz_bounds
        else:
            dz_min = self.config.motion.hybrid_stage1_dz_min
            dz_max_cfg = self.config.motion.hybrid_stage1_dz_max
            dz_max = dz_max_cfg if dz_max_cfg is not None else self.config.motion.max_dz

        # 初期値をbounds内にクランプ
        dz_init = max(dz_min, min(dz_init, dz_max))

        bounds_lower = [dz_min, -360.0]
        bounds_upper = [dz_max, 360.0]

        # 2変数残差関数（物理距離空間）
        # カメラ前進dz時: curr_d = prev_d - dz → res_d = curr_d - (prev_d - dz) = 0
        def residuals_cyl_2var(params):
            dz, droll_comp = params
            res_d = curr_d - prev_d + dz
            res_theta = curr_theta - prev_theta + droll_comp
            return np.concatenate([res_d, res_theta])

        loss = self.config.motion.hybrid_stage1_loss

        try:
            result = least_squares(
                residuals_cyl_2var,
                [dz_init, droll_init],
                loss=loss,
                bounds=(bounds_lower, bounds_upper)
            )
            dz_est, droll_comp_est = result.x
            residuals = result.fun
            res_mean = float(np.mean(np.abs(residuals)))
            res_std = float(np.std(residuals))

            self.logger.info(
                f"第1段階完了: dz={dz_est:.3f}mm "
                f"(init={dz_init:.3f}), "
                f"droll_comp={droll_comp_est:.4f}° "
                f"(init={droll_init:.4f}), "
                f"残差mean={res_mean:.4f}, std={res_std:.4f}, "
                f"N={n_points}"
            )

            return {
                'dz': dz_est,
                'droll_composite': droll_comp_est,
                'residual_mean': res_mean,
                'residual_std': res_std,
                'n_points': n_points,
                'success': True
            }

        except Exception as e:
            self.logger.warning(f"第1段階最適化失敗: {e}")
            return {'dz': 0.0, 'droll_composite': 0.0,
                    'residual_mean': float('inf'), 'residual_std': float('inf'),
                    'n_points': n_points, 'success': False}

    def estimate_motion_with_constraints(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        camera_state: Dict[str, Any],
        camera_params: Dict[str, Any],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        fix_angles: bool = True,
        dz_hint: Optional[float] = None,
        fixed_dz: Optional[float] = None,
        fixed_droll: Optional[float] = None
    ) -> Dict[str, float]:
        """探索範囲制約付きフレーム間移動量推定

        Args:
            prev_points: 前フレームのマッチ点（ピクセル座標、N×2）
            curr_points: 現フレームのマッチ点（ピクセル座標、N×2）
            camera_state: 現在のカメラ状態
                - position: [x, y, z]
                - orientation: [roll, yaw, pitch]
                - yaw_estimated: 消失点から推定したyaw（オプション）
                - phi_estimated: 消失点から推定したpitch（オプション）
            camera_params: カメラパラメータ
            constraints: 探索範囲制約（省略時は既存の広い範囲を使用）
                - 'dz': (dz_min, dz_max)
                - 'dyaw': (dyaw_min, dyaw_max)  # fix_angles=Falseの場合のみ有効
                - 'dpitch': (dpitch_min, dpitch_max)  # fix_angles=Falseの場合のみ有効
                - 'dx': (dx_min, dx_max)  # オプション
                - 'dy': (dy_min, dy_max)  # オプション
                - 'droll': (droll_min, droll_max)  # オプション
            fix_angles: Trueの場合、dyaw/dpitchを消失点推定値に固定（4次元最適化）
                        Falseの場合、dyaw/dpitchも最小二乗法で推定（6次元最適化）
            fixed_dz: 位相相関から推定したdz（mm）。指定時はdzを固定値として扱う。
            fixed_droll: 位相相関から推定したdroll（ラジアン）。指定時はdrollを固定値として扱う。
                fixed_dzとfixed_drollが両方指定された場合、2DOF最適化（dx, dyのみ）を実行。

        Returns:
            motion: 移動量 {'dx', 'dy', 'dz', 'dtheta', 'dphi', 'droll'}
                ★特徴点ベース消失点推定の結果も含む:
                    - 'vp_feature_based_success': 推定成功フラグ
                    - 'vp_feature_based_x': VP x座標（px、成功時のみ）
                    - 'vp_feature_based_y': VP y座標（px、成功時のみ）
                    - 'vp_feature_based_inlier_count': インライア数
                    - 'vp_feature_based_inlier_ratio': インライア比率
                    - 'vp_feature_based_residual_mean': 残差平均
                    - 'vp_feature_based_residual_std': 残差標準偏差

        Raises:
            MotionEstimationError: 推定失敗時

        処理の流れ:
            1. 特徴点ベース消失点推定（有効な場合のみ）
            2. 制約が指定されている場合はboundsを上書き
            3. fix_angles=Trueの場合は4次元最適化（dyaw, dpitch固定）
               fix_angles=Falseの場合は6次元最適化（全パラメータ推定）
            4. 制約範囲内での推定を保証

        Note:
            タスク1.1, 1.2で計算した範囲を使用することで、
            探索空間を大幅に削減し、処理速度と精度を向上させる。
            fix_angles=Trueは方法2（消失点固定）、Falseは方法1（最小二乗推定）に対応。
        """
        # 最低必要点数: ハイブリッドモード(fixed_dz指定)は3DOF→3点、非ハイブリッドは4DOF以上→4点
        if fixed_dz is not None:
            MIN_ALGORITHMIC_POINTS = 3  # 3DOFモード(dx, dy, droll)
        else:
            MIN_ALGORITHMIC_POINTS = 4  # 4DOF以上
        n_points = len(prev_points)
        if n_points < MIN_ALGORITHMIC_POINTS:
            raise MotionEstimationError(
                f"マッチ点数不足: {n_points} < {MIN_ALGORITHMIC_POINTS} "
                f"(最小二乗法に必要な最低点数)"
            )
        if n_points < self.config.feature_matching.min_match_count:
            self.logger.warning(
                f"マッチ点数が推奨値未満: {n_points} < "
                f"{self.config.feature_matching.min_match_count} (続行)"
            )

        # ★特徴点ベース消失点推定（有効な場合のみ）
        vp_result = None
        if self.vp_estimator is not None and prev_points is not None and curr_points is not None:
            try:
                vp_result = self.vp_estimator.estimate(prev_points, curr_points)
                if vp_result.success:
                    self.logger.info(
                        f"特徴点ベース消失点推定成功: "
                        f"vp=({vp_result.vp_x:.1f}, {vp_result.vp_y:.1f}), "
                        f"inliers={vp_result.inlier_count} ({vp_result.inlier_ratio:.2%})"
                    )
                else:
                    self.logger.debug(
                        f"特徴点ベース消失点推定失敗: {vp_result.failure_reason}"
                    )
            except Exception as e:
                self.logger.warning(
                    f"特徴点ベース消失点推定でエラー: {e}", exc_info=True
                )
        
        # カメラ状態を展開
        x0, y0, z0 = camera_state['position']
        roll_0, yaw_0, phi_0 = camera_state['orientation']
        
        # 姿勢推定から得たyaw, pitchを使用（存在する場合）
        yaw_2 = camera_state.get('yaw_estimated', yaw_0)
        phi_2 = camera_state.get('phi_estimated', phi_0)
        
        f = camera_params['f']
        center = camera_params['center']
        model = camera_params['model']
        
        # 前フレームの特徴点: ワールド座標系の角度に変換
        theta1, phi1 = self._pixel_to_angles_world(
            prev_points, f, center, roll_0, yaw_0, phi_0, model
        )
        
        # 現フレームの特徴点: カメラ座標系の角度
        theta2, phi2 = self._pixel_to_angles_cam(curr_points, f, center, model)
        
        # 最小二乗法による推定
        def residuals_6dof(params):
            """残差関数
            
            Args:
                params: [dx, dy, dz, droll]
            
            Returns:
                errors: 残差ベクトル
            """
            dx, dy, dz, droll, dtheta, dphi = params  # 6次元に拡張
            N = len(theta1)
            
            # カメラ位置（元と変化後）
            offset0 = (
                np.full(N, x0),
                np.full(N, y0),
                np.zeros(N)
            )
            offset1 = (
                np.full(N, x0 + dx),
                np.full(N, y0 + dy),
                np.full(N, dz)
            )
            
            roll2 = roll_0 + droll
            yaw_2 = yaw_0 + dtheta    # 変化量を加算
            phi_2 = phi_0 + dphi      # 変化量を加算
            # 姿勢のブレによるロール回転の後でヨー・ピッチを反映
            theta2_w, phi2_w = self._cam_to_world_angles(
                theta2, phi2, roll2, yaw_2, phi_2
            )
            
            # 前フレームの特徴点の円筒座標
            try:
                p0 = self._compute_cylinder_intersection_vect(
                    offset0, theta1, phi1
                )
            except CylinderIntersectionError:
                return np.full(N, 1e6)  # 大きなペナルティ
            
            # 現フレームの特徴点の円筒座標
            try:
                p1 = self._compute_cylinder_intersection_vect(
                    offset1, theta2_w, phi2_w
                )
            except CylinderIntersectionError:
                return np.full(N, 1e6)  # 大きなペナルティ
            
            # 有効なインデックス
            valid = (~np.isnan(p0[:, 0])) & (~np.isnan(p1[:, 0]))
            
            if not np.any(valid):
                return np.full(N, 1e6)  # 大きなペナルティ
            
            errors = np.linalg.norm(p0[valid] - p1[valid], axis=1)

            return errors

        def residuals_4dof(params):
            """残差関数（4次元: dx, dy, dz, droll）

            fix_angles=Trueの場合に使用。dtheta, dphiは固定値として扱う。

            Args:
                params: [dx, dy, dz, droll]

            Returns:
                errors: 残差ベクトル
            """
            dx, dy, dz, droll = params  # 4次元
            dtheta = dtheta_ini  # 固定値
            dphi = dphi_ini      # 固定値
            N = len(theta1)

            # カメラ位置（元と変化後）
            offset0 = (
                np.full(N, x0),
                np.full(N, y0),
                np.zeros(N)
            )
            offset1 = (
                np.full(N, x0 + dx),
                np.full(N, y0 + dy),
                np.full(N, dz)
            )

            roll2 = roll_0 + droll
            yaw_2 = yaw_0 + dtheta    # 固定値を加算
            phi_2 = phi_0 + dphi      # 固定値を加算
            # 姿勢のブレによるロール回転の後でヨー・ピッチを反映
            theta2_w, phi2_w = self._cam_to_world_angles(
                theta2, phi2, roll2, yaw_2, phi_2
            )

            # 前フレームの特徴点の円筒座標
            try:
                p0 = self._compute_cylinder_intersection_vect(
                    offset0, theta1, phi1
                )
            except CylinderIntersectionError:
                return np.full(N, 1e6)  # 大きなペナルティ

            # 現フレームの特徴点の円筒座標
            try:
                p1 = self._compute_cylinder_intersection_vect(
                    offset1, theta2_w, phi2_w
                )
            except CylinderIntersectionError:
                return np.full(N, 1e6)  # 大きなペナルティ

            # 有効なインデックス
            valid = (~np.isnan(p0[:, 0])) & (~np.isnan(p1[:, 0]))

            if not np.any(valid):
                return np.full(N, 1e6)  # 大きなペナルティ

            errors = np.linalg.norm(p0[valid] - p1[valid], axis=1)

            return errors

        def residuals_2dof(params):
            """残差関数（2次元: dx, dy）

            fixed_dz/fixed_drollが指定された場合に使用。
            dz, drollは位相相関推定値に固定、dtheta, dphiも消失点推定値に固定。

            Args:
                params: [dx, dy]

            Returns:
                errors: 残差ベクトル
            """
            dx, dy = params  # 2次元
            dz = fixed_dz       # 位相相関固定値
            droll = fixed_droll  # 位相相関固定値
            dtheta = dtheta_ini  # 消失点固定値
            dphi = dphi_ini      # 消失点固定値
            N = len(theta1)

            # カメラ位置（元と変化後）
            offset0 = (
                np.full(N, x0),
                np.full(N, y0),
                np.zeros(N)
            )
            offset1 = (
                np.full(N, x0 + dx),
                np.full(N, y0 + dy),
                np.full(N, dz)
            )

            roll2 = roll_0 + droll
            yaw_2 = yaw_0 + dtheta
            phi_2 = phi_0 + dphi
            theta2_w, phi2_w = self._cam_to_world_angles(
                theta2, phi2, roll2, yaw_2, phi_2
            )

            try:
                p0 = self._compute_cylinder_intersection_vect(
                    offset0, theta1, phi1
                )
            except CylinderIntersectionError:
                return np.full(N, 1e6)

            try:
                p1 = self._compute_cylinder_intersection_vect(
                    offset1, theta2_w, phi2_w
                )
            except CylinderIntersectionError:
                return np.full(N, 1e6)

            valid = (~np.isnan(p0[:, 0])) & (~np.isnan(p1[:, 0]))

            if not np.any(valid):
                return np.full(N, 1e6)

            errors = np.linalg.norm(p0[valid] - p1[valid], axis=1)

            return errors

        def residuals_3dof(params):
            """残差関数（3次元: dx, dy, droll）— dz固定、角度固定

            ハイブリッド2段階推定の第2段階で使用。
            dzは第1段階で確定済み、dtheta/dphiは消失点推定値に固定。

            Args:
                params: [dx, dy, droll]

            Returns:
                errors: 残差ベクトル
            """
            dx, dy, droll = params  # 3次元
            dz = fixed_dz       # 第1段階確定値
            dtheta = dtheta_ini  # 消失点固定値
            dphi = dphi_ini      # 消失点固定値
            N = len(theta1)

            # カメラ位置（元と変化後）
            offset0 = (
                np.full(N, x0),
                np.full(N, y0),
                np.zeros(N)
            )
            offset1 = (
                np.full(N, x0 + dx),
                np.full(N, y0 + dy),
                np.full(N, dz)
            )

            roll2 = roll_0 + droll
            yaw_2 = yaw_0 + dtheta
            phi_2 = phi_0 + dphi
            theta2_w, phi2_w = self._cam_to_world_angles(
                theta2, phi2, roll2, yaw_2, phi_2
            )

            try:
                p0 = self._compute_cylinder_intersection_vect(
                    offset0, theta1, phi1
                )
            except CylinderIntersectionError:
                return np.full(N, 1e6)

            try:
                p1 = self._compute_cylinder_intersection_vect(
                    offset1, theta2_w, phi2_w
                )
            except CylinderIntersectionError:
                return np.full(N, 1e6)

            valid = (~np.isnan(p0[:, 0])) & (~np.isnan(p1[:, 0]))

            if not np.any(valid):
                return np.full(N, 1e6)

            errors = np.linalg.norm(p0[valid] - p1[valid], axis=1)

            return errors

        # ハイブリッド3DOFモード判定（第1段階でdz確定、drollは未確定）
        use_hybrid_3dof = (
            self.config.motion.hybrid_two_stage_enabled
            and fixed_dz is not None
            and fixed_droll is None
        )

        # 2DOFモード判定（位相相関によるdz/droll固定）
        use_2dof = (fixed_dz is not None and fixed_droll is not None
                    and not use_hybrid_3dof)

        # デフォルトの制約
        max_dx = self.config.motion.max_dx
        max_dy = self.config.motion.max_dy
        max_dz = self.config.motion.max_dz
        max_dr = np.radians(self.config.motion.max_droll)  # ロール制限


        # dyaw, dpitchの初期値を計算
        yaw_0 = camera_state['orientation'][1]
        phi_0 = camera_state['orientation'][2]

        yaw_est = camera_state.get('yaw_estimated', yaw_0)
        phi_est = camera_state.get('phi_estimated', phi_0)

        dtheta_ini = yaw_est - yaw_0
        dphi_ini = phi_est - phi_0

        # ========== dphi/dtheta安全上限: クランプ + 6DOFフォールバック ========== #
        # 障害物乗り越え等による正当な角度変化（~1.3°）を許容するため、
        # ソフト閾値（CLAMP_DEG）まではVP値をクランプして4DOF維持。
        # ソフト閾値を大幅に超える場合（FALLBACK_DEG）のみ6DOFに切り替える。
        #
        # 旧実装（閾値1.0°→0リセット→6DOF）は、max_dphi=0.2°の探索空間で
        # 実際の変化量（~1.3°）を捕捉できず悪循環を引き起こしていた。
        CLAMP_DEG = self.config.motion.vp_angle_clamp_degrees
        FALLBACK_DEG = self.config.motion.vp_angle_fallback_degrees
        clamp_rad = np.radians(CLAMP_DEG)
        fallback_rad = np.radians(FALLBACK_DEG)
        force_6dof = False

        dphi_deg = np.degrees(dphi_ini)
        dtheta_deg = np.degrees(dtheta_ini)

        if abs(dphi_ini) > fallback_rad or abs(dtheta_ini) > fallback_rad:
            # 極端な乖離 → VP異常と判断し6DOFフォールバック
            # ただしクランプ値を初期値として使用（0リセットしない）
            self.logger.warning(
                f"VP角度乖離が極端なため6DOFフォールバック: "
                f"dphi={dphi_deg:.2f}°, dtheta={dtheta_deg:.2f}° "
                f"(閾値±{FALLBACK_DEG}°)"
            )
            dtheta_ini = np.clip(dtheta_ini, -clamp_rad, clamp_rad)
            dphi_ini = np.clip(dphi_ini, -clamp_rad, clamp_rad)
            force_6dof = True
            fix_angles = False
        elif abs(dphi_ini) > clamp_rad or abs(dtheta_ini) > clamp_rad:
            # クランプ閾値超過 → VP方向を信頼しつつ大きさを制限、4DOF維持
            dphi_clamped = np.clip(dphi_ini, -clamp_rad, clamp_rad)
            dtheta_clamped = np.clip(dtheta_ini, -clamp_rad, clamp_rad)
            self.logger.info(
                f"VP角度変化をクランプ: "
                f"dphi={dphi_deg:.2f}°→{np.degrees(dphi_clamped):.2f}°, "
                f"dtheta={dtheta_deg:.2f}°→{np.degrees(dtheta_clamped):.2f}° "
                f"(上限±{CLAMP_DEG}°)"
            )
            dtheta_ini = dtheta_clamped
            dphi_ini = dphi_clamped
        # ========== dphi/dtheta安全上限ここまで ========== #

        # ========== ★ TASK-17修正: カメラ姿勢推定角度の制約 ========== #
        # Phase 2のカメラ姿勢推定で使用される角度制約パラメータは
        # estimation.max_abs_yaw_degrees, estimation.max_abs_pitch_degrees
        # （従来の vanishing_point.max_abs_* から移動）

        # Yaw角の制約
        if self.config.max_abs_yaw_degrees is not None and self.config.max_abs_yaw_degrees > 0:
            max_abs_yaw_rad = np.radians(self.config.max_abs_yaw_degrees)
            if abs(yaw_est) > max_abs_yaw_rad:
                yaw_est_original = yaw_est
                yaw_est = np.clip(yaw_est, -max_abs_yaw_rad, max_abs_yaw_rad)
                dtheta_ini = yaw_est - yaw_0

                self.logger.warning(
                    f"カメラ姿勢Yaw角を制約: "
                    f"{np.degrees(yaw_est_original):.2f}° → {np.degrees(yaw_est):.2f}° "
                    f"(max_abs_yaw={self.config.max_abs_yaw_degrees}°)"
                )

                # 統計情報を更新
                if not hasattr(self, 'stats'):
                    self.stats = {}
                self.stats['vp_yaw_constrained_count'] = self.stats.get('vp_yaw_constrained_count', 0) + 1

        # Pitch角の制約
        if self.config.max_abs_pitch_degrees is not None and self.config.max_abs_pitch_degrees > 0:
            max_abs_pitch_rad = np.radians(self.config.max_abs_pitch_degrees)
            if abs(phi_est) > max_abs_pitch_rad:
                phi_est_original = phi_est
                phi_est = np.clip(phi_est, -max_abs_pitch_rad, max_abs_pitch_rad)
                dphi_ini = phi_est - phi_0

                self.logger.warning(
                    f"カメラ姿勢Pitch角を制約: "
                    f"{np.degrees(phi_est_original):.2f}° → {np.degrees(phi_est):.2f}° "
                    f"(max_abs_pitch={self.config.max_abs_pitch_degrees}°)"
                )

                # 統計情報を更新
                if not hasattr(self, 'stats'):
                    self.stats = {}
                self.stats['vp_pitch_constrained_count'] = self.stats.get('vp_pitch_constrained_count', 0) + 1
        # ========== TASK-17修正ここまで ========== #

        # fix_anglesに応じてboundsを設定（force_6dofの場合は角度を自由推定）
        # 2DOFモード: 位相相関でdz/drollが固定された場合
        if use_2dof and fix_angles and not force_6dof:
            self.logger.info(
                f"2DOFモード（位相相関固定）: "
                f"dz={fixed_dz:.3f}mm, droll={np.degrees(fixed_droll):.4f}°, "
                f"dtheta={np.degrees(dtheta_ini):.4f}°, dphi={np.degrees(dphi_ini):.4f}°"
            )

            # 2次元のbounds: [dx, dy]
            bounds_lower = [-max_dx, -max_dy]
            bounds_upper = [max_dx, max_dy]

            # 制約が指定されている場合は上書き
            if constraints is not None:
                if 'dx' in constraints:
                    bounds_lower[0], bounds_upper[0] = constraints['dx']
                if 'dy' in constraints:
                    bounds_lower[1], bounds_upper[1] = constraints['dy']

            bounds = (bounds_lower, bounds_upper)

        elif fix_angles and not force_6dof:
            # 方法2: dyaw, dpitchを消失点に固定（4次元最適化）
            self.logger.debug(
                f"方法2（固定）: dtheta={np.degrees(dtheta_ini):.4f}°, "
                f"dphi={np.degrees(dphi_ini):.4f}°"
            )

            # 4次元のbounds: [dx, dy, dz, droll]
            bounds_lower = [-max_dx, -max_dy, 0.0, -max_dr]
            bounds_upper = [max_dx, max_dy, max_dz, max_dr]

            # 制約が指定されている場合は上書き
            if constraints is not None:
                if 'dx' in constraints:
                    bounds_lower[0], bounds_upper[0] = constraints['dx']
                if 'dy' in constraints:
                    bounds_lower[1], bounds_upper[1] = constraints['dy']
                if 'dz' in constraints:
                    bounds_lower[2], bounds_upper[2] = constraints['dz']
                if 'droll' in constraints:
                    bounds_lower[3], bounds_upper[3] = constraints['droll']

                self.logger.debug(
                    f"制約付き推定: dz=[{bounds_lower[2]:.2f}, {bounds_upper[2]:.2f}]mm, "
                    f"droll=[{np.degrees(bounds_lower[3]):.2f}, {np.degrees(bounds_upper[3]):.2f}]°"
                )

            bounds = (bounds_lower, bounds_upper)

        else:
            # 方法1: 最小二乗法で推定（消失点を初期値とする、6次元最適化）
            max_dtheta = np.radians(self.config.motion.max_dtheta)
            max_dphi = np.radians(self.config.motion.max_dphi)

            # dtheta, dphiのbounds設定
            # 初期値が絶対制限範囲内か範囲外かで変動域を切り替える
            if -max_dtheta <= dtheta_ini <= max_dtheta:
                # 初期値が範囲内 → 全範囲を使用（constraintsとの交差を取る）
                # ただし6DOFフォールバック時はVP由来のconstraintsが信頼できないため無視
                if not force_6dof and constraints and 'dyaw' in constraints:
                    abs_lower, abs_upper = constraints['dyaw']
                    dtheta_lower = max(abs_lower, -max_dtheta)
                    dtheta_upper = min(abs_upper, max_dtheta)
                else:
                    dtheta_lower = -max_dtheta
                    dtheta_upper = max_dtheta
            else:
                # 初期値が範囲外 → εベースの微小範囲を使用
                dtheta_lower = dtheta_ini - ANGLE_EPSILON
                dtheta_upper = dtheta_ini + ANGLE_EPSILON

            if -max_dphi <= dphi_ini <= max_dphi:
                # 初期値が範囲内 → 全範囲を使用（constraintsとの交差を取る）
                # ただし6DOFフォールバック時はVP由来のconstraintsが信頼できないため無視
                if not force_6dof and constraints and 'dpitch' in constraints:
                    abs_lower, abs_upper = constraints['dpitch']
                    dphi_lower = max(abs_lower, -max_dphi)
                    dphi_upper = min(abs_upper, max_dphi)
                else:
                    dphi_lower = -max_dphi
                    dphi_upper = max_dphi
            else:
                # 初期値が範囲外 → εベースの微小範囲を使用
                dphi_lower = dphi_ini - ANGLE_EPSILON
                dphi_upper = dphi_ini + ANGLE_EPSILON

            self.logger.debug(
                f"方法1（推定）: dtheta初期値={np.degrees(dtheta_ini):.4f}°, "
                f"範囲=[{np.degrees(dtheta_lower):.2f}, {np.degrees(dtheta_upper):.2f}]°, "
                f"dphi初期値={np.degrees(dphi_ini):.4f}°, "
                f"範囲=[{np.degrees(dphi_lower):.2f}, {np.degrees(dphi_upper):.2f}]°"
            )

            # 6次元のbounds: [dx, dy, dz, droll, dtheta, dphi]
            bounds_lower = [-max_dx, -max_dy, 0.0, -max_dr, dtheta_lower, dphi_lower]
            bounds_upper = [max_dx, max_dy, max_dz, max_dr, dtheta_upper, dphi_upper]

            # 制約が指定されている場合は上書き（dx, dy, dz, droll）
            if constraints is not None:
                if 'dx' in constraints:
                    bounds_lower[0], bounds_upper[0] = constraints['dx']
                if 'dy' in constraints:
                    bounds_lower[1], bounds_upper[1] = constraints['dy']
                if 'dz' in constraints:
                    bounds_lower[2], bounds_upper[2] = constraints['dz']
                if 'droll' in constraints:
                    bounds_lower[3], bounds_upper[3] = constraints['droll']

                self.logger.debug(
                    f"制約付き推定: dz=[{bounds_lower[2]:.2f}, {bounds_upper[2]:.2f}]mm, "
                    f"droll=[{np.degrees(bounds_lower[3]):.2f}, {np.degrees(bounds_upper[3]):.2f}]°"
                )

            bounds = (bounds_lower, bounds_upper)

        # initial_guessを設定
        # ★重要: boundsが動的に変更された後は、initial_guessを境界から少し内側に設定
        # soft_l1損失関数は境界で正しく機能しない場合があるため
        def safe_initial_value(lower: float, upper: float, epsilon: float = POSITION_EPSILON) -> float:
            """boundsの中央値を返すが、範囲が狭い場合は境界から epsilon 離れた位置を返す

            Args:
                lower: bounds下限
                upper: bounds上限
                epsilon: 境界からのオフセット（デフォルト: POSITION_EPSILON）

            Returns:
                安全な初期値
            """
            mid = (lower + upper) / 2
            range_width = upper - lower

            # 範囲が十分広い場合は中央値
            if range_width > 2 * epsilon:
                return mid

            # 範囲が狭い場合は、下限 + epsilon と 上限 - epsilon の間の中央値
            safe_lower = lower + epsilon
            safe_upper = upper - epsilon

            # さらに狭い場合（upper - lower < 2*epsilon）は、可能な範囲で中央を取る
            if safe_lower >= safe_upper:
                return mid  # フォールバック: 中央値

            return (safe_lower + safe_upper) / 2

        # ========== 2DOFモードの場合はdz探索をスキップ ==========
        if use_2dof and fix_angles and not force_6dof:
            # 2DOFモード: dx, dyのみ最適化
            initial_guess = [
                safe_initial_value(bounds_lower[0], bounds_upper[0]),  # dx
                safe_initial_value(bounds_lower[1], bounds_upper[1]),  # dy
            ]

            self.logger.debug(
                f"2DOF初期推定値: dx={initial_guess[0]:.2f}mm, dy={initial_guess[1]:.2f}mm"
            )

            try:
                result = least_squares(
                    residuals_2dof,
                    initial_guess,
                    loss='soft_l1',
                    bounds=bounds
                )
            except Exception as e:
                raise MotionEstimationError(f"2DOF最小二乗法失敗: {e}")

            dx, dy = result.x
            dz = fixed_dz
            droll = fixed_droll
            dtheta = dtheta_ini
            dphi = dphi_ini

            motion = {
                'dx': dx,
                'dy': dy,
                'dz': dz,
                'droll': droll,
                'dtheta': dtheta,
                'dphi': dphi,
                'phase_correlation_used': True,
            }

            # ★特徴点ベース消失点推定結果を追加
            if vp_result is not None:
                motion['vp_feature_based_success'] = vp_result.success
                if vp_result.success:
                    motion['vp_feature_based_x'] = vp_result.vp_x
                    motion['vp_feature_based_y'] = vp_result.vp_y
                    motion['vp_feature_based_inlier_count'] = vp_result.inlier_count
                    motion['vp_feature_based_inlier_ratio'] = vp_result.inlier_ratio
                    motion['vp_feature_based_residual_mean'] = vp_result.residual_mean
                    motion['vp_feature_based_residual_std'] = vp_result.residual_std
                else:
                    motion['vp_feature_based_x'] = None
                    motion['vp_feature_based_y'] = None
                    motion['vp_feature_based_failure_reason'] = vp_result.failure_reason
            else:
                motion['vp_feature_based_success'] = False
                motion['vp_feature_based_x'] = None
                motion['vp_feature_based_y'] = None

            self.logger.debug(
                f"2DOF移動量推定完了: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, "
                f"droll={np.degrees(droll):.4f}°, dtheta={np.degrees(dtheta):.4f}°, dphi={np.degrees(dphi):.4f}°"
            )

            return motion

        # ========== ハイブリッド3DOFモード ==========
        if use_hybrid_3dof and fix_angles and not force_6dof:
            self.logger.info(
                f"3DOFモード（ハイブリッド第2段階）: "
                f"dz={fixed_dz:.3f}mm固定, "
                f"dtheta={np.degrees(dtheta_ini):.4f}°, dphi={np.degrees(dphi_ini):.4f}°"
            )

            # 3次元のbounds: [dx, dy, droll]
            bounds_lower_3dof = [-max_dx, -max_dy, -max_dr]
            bounds_upper_3dof = [max_dx, max_dy, max_dr]

            # 制約が指定されている場合は上書き
            if constraints is not None:
                if 'dx' in constraints:
                    bounds_lower_3dof[0], bounds_upper_3dof[0] = constraints['dx']
                if 'dy' in constraints:
                    bounds_lower_3dof[1], bounds_upper_3dof[1] = constraints['dy']
                if 'droll' in constraints:
                    bounds_lower_3dof[2], bounds_upper_3dof[2] = constraints['droll']

            bounds_3dof = (bounds_lower_3dof, bounds_upper_3dof)

            initial_guess_3dof = [
                safe_initial_value(bounds_lower_3dof[0], bounds_upper_3dof[0]),
                safe_initial_value(bounds_lower_3dof[1], bounds_upper_3dof[1]),
                safe_initial_value(bounds_lower_3dof[2], bounds_upper_3dof[2], ANGLE_EPSILON),
            ]

            self.logger.debug(
                f"3DOF初期推定値: dx={initial_guess_3dof[0]:.2f}mm, "
                f"dy={initial_guess_3dof[1]:.2f}mm, "
                f"droll={np.degrees(initial_guess_3dof[2]):.4f}°"
            )

            try:
                result = least_squares(
                    residuals_3dof,
                    initial_guess_3dof,
                    loss='soft_l1',
                    bounds=bounds_3dof
                )
            except Exception as e:
                raise MotionEstimationError(f"3DOF最小二乗法失敗: {e}")

            dx, dy, droll = result.x
            dz = fixed_dz
            dtheta = dtheta_ini
            dphi = dphi_ini

            motion = {
                'dx': dx,
                'dy': dy,
                'dz': dz,
                'droll': droll,
                'dtheta': dtheta,
                'dphi': dphi,
                'hybrid_two_stage_used': True,
            }

            # 特徴点ベース消失点推定結果を追加
            if vp_result is not None:
                motion['vp_feature_based_success'] = vp_result.success
                if vp_result.success:
                    motion['vp_feature_based_x'] = vp_result.vp_x
                    motion['vp_feature_based_y'] = vp_result.vp_y
                    motion['vp_feature_based_inlier_count'] = vp_result.inlier_count
                    motion['vp_feature_based_inlier_ratio'] = vp_result.inlier_ratio
                    motion['vp_feature_based_residual_mean'] = vp_result.residual_mean
                    motion['vp_feature_based_residual_std'] = vp_result.residual_std
                else:
                    motion['vp_feature_based_x'] = None
                    motion['vp_feature_based_y'] = None
                    motion['vp_feature_based_failure_reason'] = vp_result.failure_reason
            else:
                motion['vp_feature_based_success'] = False
                motion['vp_feature_based_x'] = None
                motion['vp_feature_based_y'] = None

            self.logger.debug(
                f"3DOF移動量推定完了: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, "
                f"droll={np.degrees(droll):.4f}°, dtheta={np.degrees(dtheta):.4f}°, "
                f"dphi={np.degrees(dphi):.4f}°"
            )

            return motion
        # ========== ハイブリッド3DOFモードここまで ==========

        # ========== 粗密2段階dz探索 ==========
        # Step 1: dz候補点で生の残差二乗和を評価し、最良のdz初期値を決定
        # soft_l1は大残差をダウンウェイトするため、dz=0の局所解に吸い込まれやすい。
        # 生の残差二乗和で評価すれば壁面側の特徴点情報が失われず、正しいdz近傍を選択できる。
        N_COARSE_POINTS = 5
        dz_lo = bounds_lower[2]
        dz_hi = bounds_upper[2]
        dz_range = dz_hi - dz_lo

        residual_fn = residuals_4dof if fix_angles else residuals_6dof

        if dz_range > POSITION_EPSILON * 2:
            # 他パラメータは中央値に固定してdzだけ変えて評価
            dx_mid = safe_initial_value(bounds_lower[0], bounds_upper[0])
            dy_mid = safe_initial_value(bounds_lower[1], bounds_upper[1])
            droll_mid = safe_initial_value(bounds_lower[3], bounds_upper[3], ANGLE_EPSILON)

            coarse_dz_candidates = [
                dz_lo + dz_range * (i + 0.5) / N_COARSE_POINTS
                for i in range(N_COARSE_POINTS)
            ]

            best_dz_coarse = coarse_dz_candidates[N_COARSE_POINTS // 2]  # フォールバック: 中央
            best_cost = float('inf')

            for dz_c in coarse_dz_candidates:
                if fix_angles:
                    probe = [dx_mid, dy_mid, dz_c, droll_mid]
                else:
                    probe = [dx_mid, dy_mid, dz_c, droll_mid, dtheta_ini, dphi_ini]
                res = residual_fn(probe)
                cost = float(np.sum(res ** 2))
                if cost < best_cost:
                    best_cost = cost
                    best_dz_coarse = dz_c

            dz_initial = best_dz_coarse

            self.logger.debug(
                f"粗密探索Step1: dz候補{N_COARSE_POINTS}点 "
                f"[{dz_lo:.2f}..{dz_hi:.2f}]mm → 最良dz={dz_initial:.3f}mm "
                f"(残差二乗和={best_cost:.2f})"
            )
        else:
            dz_initial = safe_initial_value(dz_lo, dz_hi)

        # dz_hintがある場合、粗密探索結果とhintを比較して良い方を採用
        if dz_hint is not None:
            dz_hint_clamped = max(dz_lo + POSITION_EPSILON,
                                 min(dz_hint, dz_hi - POSITION_EPSILON))
            if fix_angles:
                probe_hint = [dx_mid if dz_range > POSITION_EPSILON * 2 else 0.0,
                              dy_mid if dz_range > POSITION_EPSILON * 2 else 0.0,
                              dz_hint_clamped,
                              droll_mid if dz_range > POSITION_EPSILON * 2 else 0.0]
            else:
                probe_hint = [dx_mid if dz_range > POSITION_EPSILON * 2 else 0.0,
                              dy_mid if dz_range > POSITION_EPSILON * 2 else 0.0,
                              dz_hint_clamped,
                              droll_mid if dz_range > POSITION_EPSILON * 2 else 0.0,
                              dtheta_ini, dphi_ini]
            hint_cost = float(np.sum(residual_fn(probe_hint) ** 2))

            if hint_cost < best_cost if dz_range > POSITION_EPSILON * 2 else True:
                self.logger.debug(
                    f"dz_hint採用: hint={dz_hint:.3f}→clamped={dz_hint_clamped:.3f}mm "
                    f"(hint残差={hint_cost:.2f} < coarse残差={best_cost:.2f})"
                    if dz_range > POSITION_EPSILON * 2 else
                    f"dz_hint採用: hint={dz_hint:.3f}→{dz_hint_clamped:.3f}mm"
                )
                dz_initial = dz_hint_clamped
            else:
                self.logger.debug(
                    f"粗密探索結果を維持: coarse={dz_initial:.3f}mm(残差={best_cost:.2f}) "
                    f"vs hint={dz_hint_clamped:.3f}mm(残差={hint_cost:.2f})"
                )
        # ========== 粗密2段階dz探索ここまで ==========

        if fix_angles:
            # 4次元の初期値: dx, dy, dz, drollは制約範囲の中央値（境界から安全な距離）
            initial_guess = [
                safe_initial_value(bounds_lower[0], bounds_upper[0]),  # dx
                safe_initial_value(bounds_lower[1], bounds_upper[1]),  # dy
                dz_initial,  # dz（粗密探索で決定）
                safe_initial_value(bounds_lower[3], bounds_upper[3], ANGLE_EPSILON),  # droll
            ]
        else:
            # 6次元の初期値
            initial_guess = [
                safe_initial_value(bounds_lower[0], bounds_upper[0]),  # dx
                safe_initial_value(bounds_lower[1], bounds_upper[1]),  # dy
                dz_initial,  # dz（粗密探索で決定）
                safe_initial_value(bounds_lower[3], bounds_upper[3], ANGLE_EPSILON),  # droll
                dtheta_ini,  # dyaw（消失点から計算）
                dphi_ini     # dpitch（消失点から計算）
            ]

        self.logger.debug(
            f"初期推定値: dx={initial_guess[0]:.2f}mm, dy={initial_guess[1]:.2f}mm, "
            f"dz={initial_guess[2]:.2f}mm, droll={np.degrees(initial_guess[3]):.4f}deg"
        )

        # bounds設定の詳細ログ（DEBUGレベル）
        if fix_angles:
            self.logger.debug(
                f"bounds設定（4次元）: "
                f"dx=[{bounds_lower[0]:.4f}, {bounds_upper[0]:.4f}]mm, "
                f"dy=[{bounds_lower[1]:.4f}, {bounds_upper[1]:.4f}]mm, "
                f"dz=[{bounds_lower[2]:.4f}, {bounds_upper[2]:.4f}]mm, "
                f"droll=[{np.degrees(bounds_lower[3]):.4f}, {np.degrees(bounds_upper[3]):.4f}]deg"
            )
        else:
            self.logger.debug(
                f"bounds設定（6次元）: "
                f"dx=[{bounds_lower[0]:.4f}, {bounds_upper[0]:.4f}]mm, "
                f"dy=[{bounds_lower[1]:.4f}, {bounds_upper[1]:.4f}]mm, "
                f"dz=[{bounds_lower[2]:.4f}, {bounds_upper[2]:.4f}]mm, "
                f"droll=[{np.degrees(bounds_lower[3]):.4f}, {np.degrees(bounds_upper[3]):.4f}]deg, "
                f"dyaw=[{np.degrees(bounds_lower[4]):.4f}, {np.degrees(bounds_upper[4]):.4f}]deg, "
                f"dpitch=[{np.degrees(bounds_lower[5]):.4f}, {np.degrees(bounds_upper[5]):.4f}]deg"
            )

        # Step 2: 粗密探索で決めた初期値からleast_squaresで精密最適化
        try:
            if fix_angles:
                # 4次元最適化
                result = least_squares(
                    residuals_4dof,
                    initial_guess,
                    loss='soft_l1',
                    bounds=bounds
                )
            else:
                # 6次元最適化
                result = least_squares(
                    residuals_6dof,
                    initial_guess,
                    loss='soft_l1',
                    bounds=bounds
                )
        except Exception as e:
            raise MotionEstimationError(f"最小二乗法失敗: {e}")

        # 結果を6次元辞書に変換
        if fix_angles:
            # 4次元結果 + 固定値2つ
            dx, dy, dz, droll = result.x
            dtheta = dtheta_ini  # 固定値
            dphi = dphi_ini      # 固定値
        else:
            # 6次元結果そのまま
            dx, dy, dz, droll, dtheta, dphi = result.x

        motion = {
            'dx': dx,
            'dy': dy,
            'dz': dz,
            'droll': droll,
            'dtheta': dtheta,
            'dphi': dphi
        }

        # ★特徴点ベース消失点推定結果を追加
        if vp_result is not None:
            motion['vp_feature_based_success'] = vp_result.success
            if vp_result.success:
                motion['vp_feature_based_x'] = vp_result.vp_x
                motion['vp_feature_based_y'] = vp_result.vp_y
                motion['vp_feature_based_inlier_count'] = vp_result.inlier_count
                motion['vp_feature_based_inlier_ratio'] = vp_result.inlier_ratio
                motion['vp_feature_based_residual_mean'] = vp_result.residual_mean
                motion['vp_feature_based_residual_std'] = vp_result.residual_std
            else:
                motion['vp_feature_based_x'] = None
                motion['vp_feature_based_y'] = None
                motion['vp_feature_based_failure_reason'] = vp_result.failure_reason
        else:
            motion['vp_feature_based_success'] = False
            motion['vp_feature_based_x'] = None
            motion['vp_feature_based_y'] = None

        self.logger.debug(
            f"移動量推定完了: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, "
            f"droll={np.degrees(droll):.4f}°, dtheta={np.degrees(dtheta):.4f}°, dphi={np.degrees(dphi):.4f}°"
        )

        return motion


    def _constrain_camera_pose(
        self,
        motion: Dict[str, float],
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        frame_num: int,
        camera_state: Optional[Dict[str, Any]] = None,
        run_reference: Optional[Dict[str, float]] = None,
        hard_bounds: Optional[Any] = None,
    ) -> Dict[str, float]:
        """カメラ姿勢と位置を制約範囲内に強制的に収める
        
        Args:
            motion: 推定された移動量辞書 {'dx', 'dy', 'dz', 'dtheta', 'dphi', 'droll'}
            current_position: 現在の位置 [x, y, z] (mm)
            current_orientation: 現在の姿勢 [roll, yaw, pitch] (ラジアン)
            frame_num: フレーム番号(ログ出力用)
            camera_state: カメラ状態辞書(オプション)
                - yaw_estimated: 消失点から推定したyaw角(ラジアン、use_vanishing_point_yaw=true時に使用)
                - phi_estimated: 消失点から推定したpitch角(ラジアン、use_vanishing_point_pitch=true時に使用)
        
        Returns:
            制約範囲内に補正された移動量辞書
        
        Note:
            補正が発生した場合は警告ログを出力し、統計情報に記録する。
            
            制約方法:
            - X位置: [-max_x, +max_x] の範囲にクリップ
            - Y位置: [-max_y, +max_y] の範囲にクリップ
            
            - Yaw角(ヨー角):
              - モードA (fix_angles_to_vanishing_point=true):
                  max_abs_yaw_degrees が設定されている場合は [-max_abs_yaw, +max_abs_yaw]
                  未設定の場合は [-max_yaw, +max_yaw]
              - モードB (use_vanishing_point_yaw=false): 絶対値範囲 [-max_yaw, +max_yaw]
              - モードC (use_vanishing_point_yaw=true): 消失点推定値±max_yaw の相対範囲
            
            - Pitch角(ピッチ角):
              - モードA (fix_angles_to_vanishing_point=true):
                  max_abs_pitch_degrees が設定されている場合は [-max_abs_pitch, +max_abs_pitch]
                  未設定の場合は [-max_pitch, +max_pitch]
              - モードB (use_vanishing_point_pitch=false): 絶対値範囲 [-max_pitch, +max_pitch]
              - モードC (use_vanishing_point_pitch=true): 消失点推定値±max_pitch の相対範囲
            
            - Roll角: [-max_roll, +max_roll] の範囲にクリップ(固定)
            
            max_x, max_y, max_yaw, max_pitch, max_rollは self.config.motion から取得
        """
        if run_reference is not None:
            return self._constrain_camera_pose_to_reference(
                motion, current_position, current_orientation, frame_num,
                run_reference, hard_bounds,
            )
        # 移動量をコピー(元のデータを変更しない)
        constrained_motion = motion.copy()
        
        # X/Y位置の制約
        x_current, y_current, z_current = current_position
        
        # 更新後のX/Y位置を計算
        x_new = x_current + motion['dx']
        y_new = y_current + motion['dy']
        
        # X位置の制約
        max_x_limit = self.config.motion.max_x
        if abs(x_new) > max_x_limit:
            original_x = x_new
            x_constrained = np.clip(x_new, -max_x_limit, max_x_limit)
            
            # dx を補正
            constrained_motion['dx'] = x_constrained - x_current
            
            self.logger.warning(
                f"フレーム{frame_num}: X位置を制約範囲内に補正: "
                f"{original_x:.2f}mm → {x_constrained:.2f}mm"
            )
            self.stats['x_constrained_count'] += 1
        
        # Y位置の制約
        max_y_limit = self.config.motion.max_y
        if abs(y_new) > max_y_limit:
            original_y = y_new
            y_constrained = np.clip(y_new, -max_y_limit, max_y_limit)
            
            # dy を補正
            constrained_motion['dy'] = y_constrained - y_current
            
            self.logger.warning(
                f"フレーム{frame_num}: Y位置を制約範囲内に補正: "
                f"{original_y:.2f}mm → {y_constrained:.2f}mm"
            )
            self.stats['y_constrained_count'] += 1
        
        # orientation = (roll, yaw, pitch) の順序
        roll_current, yaw_current, pitch_current = current_orientation
        
        # 更新後の姿勢を計算
        roll_new = roll_current + motion['droll']
        yaw_new = yaw_current + motion['dtheta']
        pitch_new = pitch_current + motion['dphi']
        
        # Yaw角の制約
        max_yaw_rad = np.radians(self.config.motion.max_yaw)
        
        if self.config.motion.fix_angles_to_vanishing_point:
            # ========== ★ 新規追加: モードA（角度固定時の絶対値制約） ========== #
            # 消失点推定角度自体を制約（Phase1で既に制約済みだが最終防衛線として）
            vp_config = self.config.vanishing_point
            if self.config.max_abs_yaw_degrees is not None and self.config.max_abs_yaw_degrees > 0:
                max_abs_yaw_rad = np.radians(self.config.max_abs_yaw_degrees)
                yaw_min = -max_abs_yaw_rad
                yaw_max = max_abs_yaw_rad
            else:
                # max_abs_yaw_degreesが未設定の場合はmax_yawを使用
                yaw_min = -max_yaw_rad
                yaw_max = max_yaw_rad
            # ========== 新規追加ここまで ========== #
        elif self.config.motion.use_vanishing_point_yaw and camera_state is not None:
            # モードC: 消失点推定値からの相対範囲
            yaw_base = camera_state.get('yaw_estimated', yaw_current)
            yaw_min = yaw_base - max_yaw_rad
            yaw_max = yaw_base + max_yaw_rad
        else:
            # モードB: 絶対値範囲(従来通り)
            yaw_min = -max_yaw_rad
            yaw_max = max_yaw_rad
        
        if yaw_new < yaw_min or yaw_new > yaw_max:
            original_yaw_deg = np.degrees(yaw_new)
            yaw_constrained = np.clip(yaw_new, yaw_min, yaw_max)
            constrained_yaw_deg = np.degrees(yaw_constrained)
            
            # dyaw を補正
            constrained_motion['dtheta'] = yaw_constrained - yaw_current
            
            self.logger.warning(
                f"フレーム{frame_num}: Yaw角を制約範囲内に補正: "
                f"{original_yaw_deg:.2f}° → {constrained_yaw_deg:.2f}° "
                f"(範囲: [{np.degrees(yaw_min):.2f}°, {np.degrees(yaw_max):.2f}°])"
            )
            # 統計情報を更新(存在しない場合は初期化)
            if not hasattr(self, 'stats'):
                self.stats = {}
            self.stats['yaw_constrained_count'] = self.stats.get('yaw_constrained_count', 0) + 1
        
        # Pitch角の制約
        max_pitch_rad = np.radians(self.config.motion.max_pitch)
        
        if self.config.motion.fix_angles_to_vanishing_point:
            # ========== ★ 新規追加: モードA（角度固定時の絶対値制約） ========== #
            # 消失点推定角度自体を制約（Phase1で既に制約済みだが最終防衛線として）
            vp_config = self.config.vanishing_point
            if self.config.max_abs_pitch_degrees is not None and self.config.max_abs_pitch_degrees > 0:
                max_abs_pitch_rad = np.radians(self.config.max_abs_pitch_degrees)
                pitch_min = -max_abs_pitch_rad
                pitch_max = max_abs_pitch_rad
            else:
                # max_abs_pitch_degreesが未設定の場合はmax_pitchを使用
                pitch_min = -max_pitch_rad
                pitch_max = max_pitch_rad
            # ========== 新規追加ここまで ========== #
        elif self.config.motion.use_vanishing_point_pitch and camera_state is not None:
            # モードC: 消失点推定値からの相対範囲
            phi_base = camera_state.get('phi_estimated', pitch_current)
            pitch_min = phi_base - max_pitch_rad
            pitch_max = phi_base + max_pitch_rad
        else:
            # モードB: 絶対値範囲(従来通り)
            pitch_min = -max_pitch_rad
            pitch_max = max_pitch_rad
        
        if pitch_new < pitch_min or pitch_new > pitch_max:
            original_pitch_deg = np.degrees(pitch_new)
            pitch_constrained = np.clip(pitch_new, pitch_min, pitch_max)
            constrained_pitch_deg = np.degrees(pitch_constrained)
            
            # dpitch を補正
            constrained_motion['dphi'] = pitch_constrained - pitch_current
            
            self.logger.warning(
                f"フレーム{frame_num}: Pitch角を制約範囲内に補正: "
                f"{original_pitch_deg:.2f}° → {constrained_pitch_deg:.2f}° "
                f"(範囲: [{np.degrees(pitch_min):.2f}°, {np.degrees(pitch_max):.2f}°])"
            )
            # 統計情報を更新
            if not hasattr(self, 'stats'):
                self.stats = {}
            self.stats['pitch_constrained_count'] = self.stats.get('pitch_constrained_count', 0) + 1
        
        # Roll角の制約
        max_roll_rad = np.radians(self.config.motion.max_roll)
        if abs(roll_new) > max_roll_rad:
            original_roll_deg = np.degrees(roll_new)
            roll_constrained = np.clip(roll_new, -max_roll_rad, max_roll_rad)
            constrained_roll_deg = np.degrees(roll_constrained)
            
            # droll を補正
            constrained_motion['droll'] = roll_constrained - roll_current
            
            self.logger.warning(
                f"フレーム{frame_num}: Roll角を制約範囲内に補正: "
                f"{original_roll_deg:.2f}° → {constrained_roll_deg:.2f}°"
            )
            # 統計情報を更新
            if not hasattr(self, 'stats'):
                self.stats = {}
            self.stats['roll_constrained_count'] = self.stats.get('roll_constrained_count', 0) + 1
        
        return constrained_motion



    def estimate_motion_flexible(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        camera_state: Dict[str, Any],
        camera_params: Dict[str, Any],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        run_reference: Optional[Dict[str, float]] = None,
        center_prior: Optional[Any] = None,
        estimation_mode: Optional[str] = None,
        hard_bounds: Optional[Any] = None,
    ) -> Dict[str, float]:
        """柔軟なパラメータ推定（段階的テスト用）

        設定に基づいて推定するパラメータを動的に決定します。
        yaw, pitchも推定変数に含めることができます。

        Args:
            prev_points: 前フレームのマッチ点（ピクセル座標、N×2）
            curr_points: 現フレームのマッチ点（ピクセル座標、N×2）
            camera_state: 現在のカメラ状態
                - position: [x, y, z]
                - orientation: [roll, yaw, pitch]
            camera_params: カメラパラメータ
            constraints: 探索範囲制約（省略時は設定から決定）

        Returns:
            motion: 移動量 {'dx', 'dy', 'dz', 'dtheta', 'dphi', 'droll'}

        Raises:
            MotionEstimationError: 推定失敗時

        Note:
            config.motion.estimate_*フラグで推定するパラメータを制御します。
            - estimate_dx: dx, dyを推定するか
            - estimate_dz: dzを推定するか
            - estimate_droll: drollを推定するか
            - estimate_dyaw: dyawを推定するか
            - estimate_dpitch: dpitchを推定するか
            - use_vanishing_point_yaw/pitch: 消失点値を基準値として使用するか
        """
        # 最低必要点数: least_squaresのパラメータ数以上の残差が必要
        # 4DOF=4点、6DOF=6点が理論最小だが、3点あれば実用上動作する
        MIN_ALGORITHMIC_POINTS = 3
        n_points = len(prev_points)
        if n_points < MIN_ALGORITHMIC_POINTS:
            raise MotionEstimationError(
                f"マッチ点数不足: {n_points} < {MIN_ALGORITHMIC_POINTS} "
                f"(最小二乗法に必要な最低点数)"
            )
        if run_reference is not None:
            return self._estimate_motion_with_reference(
                prev_points,
                curr_points,
                camera_state,
                camera_params,
                constraints,
                run_reference,
                center_prior,
                estimation_mode,
                hard_bounds,
            )
        if n_points < self.config.feature_matching.min_match_count:
            self.logger.warning(
                f"マッチ点数が推奨値未満: {n_points} < "
                f"{self.config.feature_matching.min_match_count} (続行)"
            )

        # カメラ状態を展開
        x0, y0, z0 = camera_state['position']
        roll_0, yaw_0, phi_0 = camera_state['orientation']

        # 姿勢の基準値を決定
        if self.config.motion.use_vanishing_point_yaw:
            yaw_base = camera_state.get('yaw_estimated', yaw_0)
        else:
            yaw_base = np.radians(self.config.motion.fixed_yaw)

        if self.config.motion.use_vanishing_point_pitch:
            phi_base = camera_state.get('phi_estimated', phi_0)
        else:
            phi_base = np.radians(self.config.motion.fixed_pitch)

        roll_base = np.radians(self.config.motion.fixed_roll)

        f = camera_params['f']
        center = camera_params['center']
        model = camera_params['model']

        # 推定するパラメータのリストを構築
        param_names = []
        bounds_lower = []
        bounds_upper = []
        initial_guess = []

        # dx, dy
        if self.config.motion.estimate_dx:
            param_names.extend(['dx', 'dy'])

            if constraints and 'dx' in constraints and 'dy' in constraints:
                # 制約がある場合はそれを使用（Phase1で計算済み）
                bounds_lower.extend([constraints['dx'][0], constraints['dy'][0]])
                bounds_upper.extend([constraints['dx'][1], constraints['dy'][1]])
            else:
                # 制約なしの場合（通常は発生しない）
                max_dx = self.config.motion.max_dx
                max_dy = self.config.motion.max_dy
                bounds_lower.extend([-max_dx, -max_dy])
                bounds_upper.extend([max_dx, max_dy])
            initial_guess.extend([0.0, 0.0])

        # dz
        if self.config.motion.estimate_dz:
            param_names.append('dz')
            max_dz = self.config.motion.max_dz
            if constraints and 'dz' in constraints:
                bounds_lower.append(constraints['dz'][0])
                bounds_upper.append(constraints['dz'][1])
                # 制約範囲の中央値を初期推定値として使用
                initial_guess.append((constraints['dz'][0] + constraints['dz'][1]) / 2)
            else:
                bounds_lower.append(0.0)
                bounds_upper.append(max_dz)
                initial_guess.append(max_dz / 2)

        # droll
        if self.config.motion.estimate_droll:
            param_names.append('droll')
            max_dr = np.radians(self.config.motion.max_droll)
            if constraints and 'droll' in constraints:
                bounds_lower.append(constraints['droll'][0])
                bounds_upper.append(constraints['droll'][1])
            else:
                bounds_lower.append(-max_dr)
                bounds_upper.append(max_dr)
            initial_guess.append(0.0)

        # dyaw
        if self.config.motion.estimate_dyaw:
            param_names.append('dyaw')
            max_dyaw = np.radians(self.config.motion.max_dtheta)
            if constraints and 'dyaw' in constraints:
                bounds_lower.append(constraints['dyaw'][0])
                bounds_upper.append(constraints['dyaw'][1])
                # 制約範囲の中央値を初期推定値として使用
                initial_guess.append((constraints['dyaw'][0] + constraints['dyaw'][1]) / 2)
            else:
                bounds_lower.append(-max_dyaw)
                bounds_upper.append(max_dyaw)
                initial_guess.append(0.0)

        # dpitch
        if self.config.motion.estimate_dpitch:
            param_names.append('dpitch')
            max_dpitch = np.radians(self.config.motion.max_dphi)
            if constraints and 'dpitch' in constraints:
                bounds_lower.append(constraints['dpitch'][0])
                bounds_upper.append(constraints['dpitch'][1])
                # 制約範囲の中央値を初期推定値として使用
                initial_guess.append((constraints['dpitch'][0] + constraints['dpitch'][1]) / 2)
            else:
                bounds_lower.append(-max_dpitch)
                bounds_upper.append(max_dpitch)
                initial_guess.append(0.0)

        if len(param_names) == 0:
            raise MotionEstimationError("推定するパラメータが1つも設定されていません")

        self.logger.debug(f"推定パラメータ: {param_names}")

        # 前フレームの特徴点: ワールド座標系の角度に変換
        theta1, phi1 = self._pixel_to_angles_world(
            prev_points, f, center, roll_0, yaw_0, phi_0, model
        )

        # 現フレームの特徴点: カメラ座標系の角度
        theta2, phi2 = self._pixel_to_angles_cam(curr_points, f, center, model)

        # 残差関数
        def residuals(params):
            """残差関数

            Args:
                params: 推定パラメータのリスト

            Returns:
                errors: 残差ベクトル
            """
            # パラメータを辞書に変換
            param_dict = dict(zip(param_names, params))

            # 各パラメータの値を取得（推定しない場合は0または基準値）
            dx = param_dict.get('dx', 0.0)
            dy = param_dict.get('dy', 0.0)
            dz = param_dict.get('dz', 0.0)
            droll = param_dict.get('droll', 0.0) if self.config.motion.estimate_droll else roll_base - roll_0
            dyaw = param_dict.get('dyaw', 0.0)
            dpitch = param_dict.get('dpitch', 0.0)

            N = len(theta1)

            # カメラ位置（元と変化後）
            offset0 = (
                np.full(N, x0),
                np.full(N, y0),
                np.zeros(N)
            )
            offset1 = (
                np.full(N, x0 + dx),
                np.full(N, y0 + dy),
                np.full(N, dz)
            )

            # 姿勢を計算
            roll2 = roll_0 + droll
            yaw2 = yaw_base + dyaw
            pitch2 = phi_base + dpitch

            # 姿勢のブレによるロール回転の後でヨー・ピッチを反映
            theta2_w, phi2_w = self._cam_to_world_angles(
                theta2, phi2, roll2, yaw2, pitch2
            )

            # 前フレームの特徴点の円筒座標
            try:
                p0 = self._compute_cylinder_intersection_vect(
                    offset0, theta1, phi1
                )
            except CylinderIntersectionError:
                return np.full(N, 1e6)  # 大きなペナルティ

            # 現フレームの特徴点の円筒座標
            try:
                p1 = self._compute_cylinder_intersection_vect(
                    offset1, theta2_w, phi2_w
                )
            except CylinderIntersectionError:
                return np.full(N, 1e6)  # 大きなペナルティ

            # 有効なインデックス
            valid = (~np.isnan(p0[:, 0])) & (~np.isnan(p1[:, 0]))

            if not np.any(valid):
                return np.full(N, 1e6)  # 大きなペナルティ

            errors = np.linalg.norm(p0[valid] - p1[valid], axis=1)

            return errors

        bounds = (bounds_lower, bounds_upper)

        self.logger.debug(
            f"初期推定値: {dict(zip(param_names, initial_guess))}"
        )
        self.logger.debug(
            f"探索範囲: lower={dict(zip(param_names, bounds_lower))}, "
            f"upper={dict(zip(param_names, bounds_upper))}"
        )

        try:
            result = least_squares(
                residuals,
                initial_guess,
                loss='soft_l1',
                bounds=bounds
            )
        except Exception as e:
            raise MotionEstimationError(f"最小二乗法失敗: {e}")

        # 結果を辞書に変換
        result_dict = dict(zip(param_names, result.x))

        # 全パラメータの移動量を構築
        motion = {
            'dx': result_dict.get('dx', 0.0),
            'dy': result_dict.get('dy', 0.0),
            'dz': result_dict.get('dz', 0.0),
            'droll': result_dict.get('droll', 0.0) if self.config.motion.estimate_droll else roll_base - roll_0,
            'dtheta': result_dict.get('dyaw', 0.0),
            'dphi': result_dict.get('dpitch', 0.0)
        }

        self.logger.debug(
            f"移動量推定完了: {motion}"
        )

        return motion

    def _pixel_to_angles_world(
        self,
        pts_px: np.ndarray,
        f: float,
        center: Tuple[float, float],
        roll: float,
        yaw: float,
        pitch: float,
        model: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ピクセル座標→ワールド座標系の角度
        
        Args:
            pts_px: ピクセル座標（N×2）
            f: 焦点距離
            center: レンズ中心
            roll: ロール角
            yaw: ヨー角
            pitch: ピッチ角
            model: カメラモデル（'fisheye' or 'pinhole'）
        
        Returns:
            theta_world: ワールド座標系ヨー角
            phi_world: ワールド座標系ピッチ角
        """
        # カメラ座標系の角度
        theta_cam, phi_cam = self._pixel_to_angles_cam(pts_px, f, center, model)
        
        # ワールド座標系に変換
        theta_world, phi_world = self.transformer.cam_to_world_angles(
            theta_cam, phi_cam, roll, yaw, pitch
        )
        
        return theta_world, phi_world
    
    def _pixel_to_angles_cam(
        self,
        pts_px: np.ndarray,
        f: float,
        center: Tuple[float, float],
        model: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ピクセル座標→カメラ座標系の角度
        
        Args:
            pts_px: ピクセル座標（N×2）
            f: 焦点距離
            center: レンズ中心
            model: カメラモデル
        
        Returns:
            yaw_cam: カメラ座標系ヨー角
            pitch_cam: カメラ座標系ピッチ角
        """
        from src.coordinate_transform import FisheyeCamera, PinholeCamera
        
        if model == 'fisheye':
            # 魚眼カメラの角度計算
            dx = pts_px[:, 0] - center[0]
            dy = -(pts_px[:, 1] - center[1])  # カメラ座標系では上が正
            
            r = np.hypot(dx, dy) + 1e-12
            gamma = r / f
            phi_fish = np.arctan2(dy, dx)
            
            vx = np.sin(gamma) * np.cos(phi_fish)
            vy = np.sin(gamma) * np.sin(phi_fish)
            vz = np.cos(gamma)
            
            yaw_cam = np.arctan2(vx, vz)
            pitch_cam = np.arctan2(vy, np.hypot(vx, vz))
        
        elif model == 'pinhole':
            # ピンホールカメラの角度計算
            dx = pts_px[:, 0] - center[0]
            dy = center[1] - pts_px[:, 1]  # カメラ座標系では上が正
            
            x_cam = dx
            y_cam = dy
            z_cam = f
            
            yaw_cam = np.arctan2(x_cam, z_cam)
            pitch_cam = np.arctan2(y_cam, np.sqrt(z_cam**2 + x_cam**2))
        
        else:
            raise CameraEstimationError(f"未対応のカメラモデル: {model}")
        
        return yaw_cam, pitch_cam
    
    def _cam_to_world_angles(
        self,
        yaw_cam: np.ndarray,
        pitch_cam: np.ndarray,
        roll: float,
        yaw: float,
        pitch: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """カメラ座標系の角度→ワールド座標系の角度
        
        Args:
            yaw_cam: カメラ座標系ヨー角
            pitch_cam: カメラ座標系ピッチ角
            roll: カメラのロール角
            yaw: カメラのヨー角
            pitch: カメラのピッチ角
        
        Returns:
            yaw_world: ワールド座標系ヨー角
            pitch_world: ワールド座標系ピッチ角
        """
        # カメラ座標系の方向ベクトル
        vx = np.cos(pitch_cam) * np.sin(yaw_cam)
        vy = np.sin(pitch_cam)
        vz = np.cos(pitch_cam) * np.cos(yaw_cam)
        v_cam = np.stack((vx, vy, vz), axis=-1)
        
        # ワールド座標系に変換
        R = R_c2w(roll, pitch, yaw)
        v_w = v_cam @ R
        vx_w, vy_w, vz_w = np.moveaxis(v_w, -1, 0)
        
        # ワールド座標系の角度
        yaw_world = np.arctan2(vx_w, vz_w)
        pitch_world = np.arctan2(vy_w, np.hypot(vx_w, vz_w))
        
        return yaw_world, pitch_world
    
    def _compute_cylinder_intersection_vect(
        self,
        offset: Tuple[np.ndarray, np.ndarray, np.ndarray],
        theta: np.ndarray,
        phi: np.ndarray
    ) -> np.ndarray:
        """レイと円筒の交点を計算
        
        Args:
            offset: カメラ位置（x, y, z）のタプル
            theta: 水平方向の視線角度（ラジアン）
            phi: 垂直方向の視線角度（ラジアン）
        
        Returns:
            intersection: 交点座標（N×3）
        
        Raises:
            CylinderIntersectionError: 交点計算失敗
        """
        x_f, y_f, z_f = offset
        
        # 視線ベクトル
        dx_v = np.cos(phi) * np.sin(theta)  # 右側が+
        dy_v = np.sin(phi)
        dz_v = np.cos(phi) * np.cos(theta)
        
        # 二次方程式の係数
        A = dx_v**2 + dy_v**2
        B = 2 * (x_f * dx_v + y_f * dy_v)
        C = x_f**2 + y_f**2 - self.transformer.pipe_radius**2
        
        # レイが円筒軸にほぼ平行な場合
        A_threshold = 1e-20
        if np.any(np.abs(A) < A_threshold):
            n_invalid = np.sum(np.abs(A) < A_threshold)
            raise CylinderIntersectionError(
                f"レイが円筒軸にほぼ平行（A≈0）：{n_invalid}/{len(A)} 点"
            )
        
        # 判別式
        D = B**2 - 4 * A * C
        
        # 交点が存在しない場合
        if np.any(D < 0):
            n_invalid = np.sum(D < 0)
            raise CylinderIntersectionError(
                f"判別式が負：交点なし（{n_invalid}/{len(D)} 点）"
            )
        
        # 解を計算（遠い方の交点を選択）
        sqrt_D = np.sqrt(D)
        t = (-B + sqrt_D) / (2 * A)
        
        # カメラ前方の交点のみ有効（t > 0）
        if np.any(t <= 0):
            n_invalid = np.sum(t <= 0)
            raise CylinderIntersectionError(
                f"交点がレイの原点より後ろ側（t≤0）：{n_invalid}/{len(t)} 点"
            )
        
        # 交点座標
        x = x_f + t * dx_v
        y = y_f + t * dy_v
        z = z_f + t * dz_v
        
        return np.stack((x, y, z), axis=-1)

    # ========================================
    # Task 4.2.4.1: Phase1実装 - 累積10mm間隔VP推定
    # ========================================

    def validate_ocr_distance(
        self,
        frame_idx: int,
        ocr_distance: Optional[float],
        prev_ocr_distance: Optional[float],
        prev_ocr_success: bool,
        base_ocr_distance: Optional[float],
        frame_gap: int,
        max_increment: float
    ) -> Optional[float]:
        """OCR距離の妥当性を検証（物理制約ベース）

        呼び出し側でbase_ocr_distanceにprev_ocr_distance（直前の成功OCR値）を
        渡し、frame_gapにその成功フレームからの差を渡すことで、
        per-frame incrementベースの検証を行う。

        Args:
            frame_idx: 現在のフレーム番号
            ocr_distance: 現在フレームのOCR読取り結果(mm)
            prev_ocr_distance: 前フレームのOCR距離(mm)（逆転チェック用）
            prev_ocr_success: 前フレームのOCR読取り成功フラグ
            base_ocr_distance: 参照距離(mm)（prev_ocr_distanceまたはbase）
            frame_gap: 参照フレームからのフレーム差
            max_increment: 1フレームあたりの最大移動量(mm)

        Returns:
            検証済みOCR距離(mm)。破棄の場合はNone

        検証ルール:
            1. ocr_distance が None の場合はそのまま返す（読取り失敗）
            2. 参照距離なし（最初のフレーム）→ 検証なしで合格
            3. 逆転検出（前フレーム成功時のみ）: ocr_distance < prev_ocr_distance → 破棄
            4. 範囲チェック: ocr_distance ∈ [ref, ref + max_increment * gap]
        """
        # 読取り失敗の場合はそのまま返す
        if ocr_distance is None:
            return None

        # 最初のフレーム（基準なし）→ 検証なしで合格
        if base_ocr_distance is None:
            return ocr_distance

        # 逆転チェック（前フレーム成功時のみ）
        if prev_ocr_success and prev_ocr_distance is not None:
            if ocr_distance < prev_ocr_distance:
                self.logger.warning(
                    f"OCR距離の逆転を検出: frame={frame_idx}, "
                    f"value={ocr_distance:.1f}mm < prev={prev_ocr_distance:.1f}mm、破棄"
                )
                return None

        # 基準フレームからの範囲チェック
        min_distance = base_ocr_distance
        max_distance = base_ocr_distance + max_increment * frame_gap

        if not (min_distance <= ocr_distance <= max_distance):
            self.logger.warning(
                f"OCR距離検証失敗: frame={frame_idx}, "
                f"value={ocr_distance:.1f}mm, "
                f"expected=[{min_distance:.1f}, {max_distance:.1f}]mm (gap={frame_gap}), "
                f"reason=範囲外"
            )
            return None

        # 検証合格
        return ocr_distance

    def _collect_reference_frames(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        ocr_roi_ratio: Tuple[float, float, float, float],
        ocr_tesseract_config: str,
        ocr_preprocessing_enabled: bool,
        max_distance_increment_mm: float = 10.0
    ) -> List[Dict]:
        """基準フレームを収集（OCR距離検証統合版）

        OCR距離が累積10mm増加するごとに基準フレームを記録

        Args:
            video_path: 動画ファイルパス
            start_frame: 開始フレーム
            end_frame: 終了フレーム
            ocr_roi_ratio: OCR読み取りROI領域 (x1, y1, x2, y2)
            ocr_tesseract_config: Tesseract設定文字列
            ocr_preprocessing_enabled: OCR前処理の有効化フラグ
            max_distance_increment_mm: 1フレームあたりの最大距離増分(mm)（デフォルト: 10.0）

        Returns:
            reference_frames: 基準フレームリスト
                [{
                    'frame_idx': int,
                    'ocr_distance': float,
                    'dark_center': (float, float)
                }, ...]
        """
        from src.ocr_utils import extract_distance_from_frame

        reference_frames = []
        prev_ocr_distance = None  # 前フレームのOCR距離
        prev_ocr_success = False   # 前フレームのOCR読取り成功フラグ
        prev_ocr_frame_idx = None  # 直前のOCR成功フレームインデックス
        base_ocr_distance = None   # 直前の基準フレームのOCR距離
        base_ocr_frame_idx = start_frame  # 直前の基準フレームのインデックス

        # BUG-011統計情報用
        ocr_raw_success_count = 0      # OCR読取り成功数（検証前）
        ocr_validated_success_count = 0  # OCR読取り成功数（検証後）
        total_frames = end_frame - start_frame + 1

        cap = cv2.VideoCapture(video_path)
        
        for frame_idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 暗部重心VP取得
            try:
                dark_center = self.detect_dark_region_centroid(
                    frame,
                    threshold=self.config.pose.dark_threshold,
                    radius_limit=min(
                        self.transformer.camera.image_height,
                        self.transformer.camera.image_width
                    ) // 2 * self.config.pose.dark_region_radius_limit_ratio if hasattr(self.config.pose, 'dark_region_radius_limit_ratio') else None,
                    target_area=self.config.pose.target_dark_area if self.config.pose.use_adaptive_threshold else None,
                    threshold_range=self.config.pose.threshold_search_range if self.config.pose.use_adaptive_threshold else (1, 100)
                )
            except DarkRegionNotFoundError:
                camera = self.transformer.camera
                dark_center = (camera.cx, camera.cy)
                self.logger.warning(
                    f"フレーム{frame_idx}: 暗部重心VP検出失敗、"
                    f"レンズ中心を使用 ({camera.cx:.1f}, {camera.cy:.1f})"
                )
            
            # OCR距離読み取り
            try:
                from src.ocr_utils import OCRConfig as OCRConfigUtils
                _ocr_config = OCRConfigUtils(
                    psm_mode=OCRConfigUtils._extract_psm_mode(ocr_tesseract_config),
                    preprocessing_enabled=ocr_preprocessing_enabled,
                )
                ocr_distance_raw, _ = extract_distance_from_frame(
                    frame,
                    ocr_roi_ratio,
                    config=_ocr_config
                )
                if ocr_distance_raw is not None:
                    ocr_raw_success_count += 1
            except Exception as e:
                self.logger.warning(f"フレーム{frame_idx}: OCR読み取り失敗 - {e}")
                ocr_distance_raw = None

            # BUG-011修正: OCR距離の妥当性検証（prev_ocr_distanceベースを優先）
            if prev_ocr_distance is not None and prev_ocr_frame_idx is not None:
                gap_for_validate = frame_idx - prev_ocr_frame_idx
                ref_distance = prev_ocr_distance
            elif base_ocr_distance is not None:
                gap_for_validate = frame_idx - base_ocr_frame_idx
                ref_distance = base_ocr_distance
            else:
                gap_for_validate = 1
                ref_distance = None
            ocr_distance = self.validate_ocr_distance(
                frame_idx=frame_idx,
                ocr_distance=ocr_distance_raw,
                prev_ocr_distance=prev_ocr_distance,
                prev_ocr_success=prev_ocr_success,
                base_ocr_distance=ref_distance,
                frame_gap=gap_for_validate,
                max_increment=max_distance_increment_mm
            )

            # OCR距離が検証合格した場合、統計カウント
            if ocr_distance is not None:
                ocr_validated_success_count += 1

            # OCRインクリメント検出と基準フレーム判定
            if prev_ocr_distance is not None and ocr_distance is not None:
                if ocr_distance > prev_ocr_distance:  # インクリメント発生

                    if base_ocr_distance is None:
                        # 最初の基準フレーム（Rule 3）
                        reference_frames.append({
                            'frame_idx': frame_idx,
                            'ocr_distance': ocr_distance,
                            'dark_center': dark_center
                        })
                        base_ocr_distance = ocr_distance
                        base_ocr_frame_idx = frame_idx
                        self.logger.info(
                            f"基準フレーム記録（初回）: frame={frame_idx}, "
                            f"OCR={ocr_distance:.1f}mm, "
                            f"dark_center=({dark_center[0]:.1f}, {dark_center[1]:.1f})"
                        )

                    elif ocr_distance >= base_ocr_distance + self.config.vanishing_point.ocr_distance_interval:
                        # 間隔到達（Rule 4）
                        reference_frames.append({
                            'frame_idx': frame_idx,
                            'ocr_distance': ocr_distance,
                            'dark_center': dark_center
                        })
                        base_ocr_distance = ocr_distance
                        base_ocr_frame_idx = frame_idx
                        self.logger.info(
                            f"基準フレーム記録: frame={frame_idx}, "
                            f"OCR={ocr_distance:.1f}mm, "
                            f"dark_center=({dark_center[0]:.1f}, {dark_center[1]:.1f})"
                        )

            # 前フレーム情報を更新
            if ocr_distance is not None:
                prev_ocr_distance = ocr_distance
                prev_ocr_success = True
                prev_ocr_frame_idx = frame_idx
            else:
                prev_ocr_success = False

            # 基準フレーム記録後はコピー済みなので元フレームを解放
            # （基準フレーム以外は次のループで上書きされるため明示的解放不要）

        cap.release()

        # BUG-011修正: OCR距離検証の統計情報をログ出力
        ocr_discarded_count = ocr_raw_success_count - ocr_validated_success_count
        discard_rate = ocr_discarded_count / ocr_raw_success_count * 100 if ocr_raw_success_count > 0 else 0.0

        self.logger.info(
            f"OCR距離検証統計:\n"
            f"  - 全フレーム数: {total_frames}\n"
            f"  - OCR成功（検証前）: {ocr_raw_success_count}\n"
            f"  - OCR成功（検証後）: {ocr_validated_success_count}\n"
            f"  - 検証により破棄: {ocr_discarded_count}\n"
            f"  - 破棄率: {discard_rate:.2f}%（{ocr_discarded_count}/{ocr_raw_success_count}）"
        )

        self.logger.info(
            f"基準フレーム収集完了: {len(reference_frames)}フレーム"
        )

        return reference_frames

    def _estimate_feature_based_vp_between_references(
        self,
        reference_frames: List[Dict],
        video_path: str
    ) -> Dict[int, Dict]:
        """基準フレーム間で特徴点ベースVP推定
        
        Args:
            reference_frames: 基準フレームリスト
            video_path: 動画ファイルパス
        
        Returns:
            feature_vp_data: フレームIDごとの特徴点VPデータ
                {
                    frame_idx: {
                        'success': bool,
                        'vp_x': float,
                        'vp_y': float,
                        'inlier_count': int,
                        'inlier_ratio': float,
                        'dz_mm': float
                    }
                }
        """
        feature_vp_data = {}
        
        if self.vp_estimator is None:
            self.logger.warning("特徴点ベースVP推定器が無効です")
            return feature_vp_data
        
        for i in range(1, len(reference_frames)):
            prev_ref = reference_frames[i-1]
            curr_ref = reference_frames[i]
            
            # 動画から基準フレームを再読み込み
            cap = cv2.VideoCapture(video_path)
            
            # 前フレーム読み込み
            cap.set(cv2.CAP_PROP_POS_FRAMES, prev_ref['frame_idx'])
            ret1, prev_frame = cap.read()
            if not ret1:
                self.logger.warning(f"フレーム{prev_ref['frame_idx']}の読み込み失敗")
                cap.release()
                continue
            
            # 現フレーム読み込み
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_ref['frame_idx'])
            ret2, curr_frame = cap.read()
            if not ret2:
                self.logger.warning(f"フレーム{curr_ref['frame_idx']}の読み込み失敗")
                cap.release()
                continue
            
            # 特徴点マッチング
            try:
                # カメラパラメータ準備
                camera_params = {
                    'f': self.transformer.camera.f,
                    'center': (self.transformer.camera.cx, self.transformer.camera.cy),
                    'pipe_diameter': self.transformer.pipe_radius * 2,
                    'image_width': self.transformer.camera.image_width,
                    'image_height': self.transformer.camera.image_height
                }

                prev_points, curr_points, status = self.cylindrical_matcher.extract_and_match(
                    prev_frame, curr_frame,
                    camera_params=camera_params
                )

                # 静止状態チェック
                if status == "STATIC_FRAME":
                    self.logger.info("Phase1: 静止状態フレーム、スキップ")
                    cap.release()
                    continue
                elif status == "FAILED" or prev_points is None:
                    self.logger.warning("Phase1: 特徴点抽出失敗、スキップ")
                    cap.release()
                    continue

                
                
                # ミスマッチフィルタ（Phase1用）
                if prev_points is not None and curr_points is not None and len(prev_points) >= 3:
                    valid_mask = self._filter_mismatched_movements(prev_points, curr_points)
                    if np.sum(valid_mask) >= 3:
                        prev_points = prev_points[valid_mask]
                        curr_points = curr_points[valid_mask]
                        self.logger.debug(
                            f"Phase1ミスマッチフィルタ適用後: {len(prev_points)}点"
                        )
                    else:
                        self.logger.warning(
                            f"Phase1ミスマッチフィルタ: 特徴点不足 {np.sum(valid_mask)}"
                        )
                
                if prev_points is None or len(prev_points) < 20:
                    self.logger.warning(
                        f"フレーム{curr_ref['frame_idx']}: 特徴点不足"
                    )
                    cap.release()
                    continue
                
                # 特徴点ベースVP推定
                # 画像中心を取得
                image_center = (camera_params['center'][0], camera_params['center'][1])
                
                vp_result = self.vp_estimator.estimate(prev_points, curr_points, image_center=image_center)
                
                # dz計算
                dz_mm = (curr_ref['ocr_distance'] or 0) - (prev_ref['ocr_distance'] or 0)
                
                feature_vp_data[curr_ref['frame_idx']] = {
                    'success': vp_result.success,
                    'vp_x': vp_result.vp_x if vp_result.success else None,
                    'vp_y': vp_result.vp_y if vp_result.success else None,
                    'inlier_count': vp_result.inlier_count,
                    'inlier_ratio': vp_result.inlier_ratio,
                    'dz_mm': dz_mm,
                    'failure_reason': vp_result.failure_reason if not vp_result.success else None
                }
                
                if vp_result.success:
                    self.logger.info(
                        f"フレーム{curr_ref['frame_idx']}: 特徴点VP成功 "
                        f"vp=({vp_result.vp_x:.1f}, {vp_result.vp_y:.1f}), "
                        f"inliers={vp_result.inlier_count}/{len(prev_points)} ({vp_result.inlier_ratio:.2%}), "
                        f"dz={dz_mm:.1f}mm"
                    )
                else:
                    self.logger.debug(
                        f"フレーム{curr_ref['frame_idx']}: 特徴点VP失敗 "
                        f"reason={vp_result.failure_reason}, dz={dz_mm:.1f}mm"
                    )
            
                
                # フレーム解放
                del prev_frame, curr_frame
                cap.release()
            except Exception as e:
                self.logger.error(
                    f"フレーム{curr_ref['frame_idx']}で特徴点マッチングエラー: {e}",
                    exc_info=True
                )
                # VideoCapture解放
                cap.release()
        
        return feature_vp_data

    def collect_vanishing_points_with_features(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        ocr_roi_ratio: Tuple[float, float, float, float],
        ocr_tesseract_config: str,
        ocr_preprocessing_enabled: bool,
        read_ocr: bool = True,
        max_distance_increment_mm: float = 10.0
    ) -> Dict:
        """Phase1: 暗部重心VP + 累積10mm間隔での特徴点ベースVP推定
        
        Args:
            video_path: 動画ファイルパス
            start_frame: 開始フレーム
            end_frame: 終了フレーム
            ocr_roi_ratio: OCR読み取りROI領域 (x1, y1, x2, y2)
            ocr_tesseract_config: Tesseract設定文字列
            ocr_preprocessing_enabled: OCR前処理の有効化フラグ
            read_ocr: OCR読み取りを実行するか（デフォルト: True）
                False の場合、OCR読み取りをスキップし、ocr_distance=None を設定
        
        Returns:
            vp_collection_result: {
                'vp_data': フレームごとのVPデータ,
                'smoothed_vp': スムージング済みVP,
                'search_ranges': 探索範囲制約
            }
        """
        from src.ocr_utils import extract_distance_from_frame
        
        self.logger.info("=" * 80)
        self.logger.info("Phase1: VP収集開始（累積10mm間隔特徴点VP付き）")
        self.logger.info("=" * 80)
        
        # Step 1: 基準フレーム収集
        reference_frames = self._collect_reference_frames(
            video_path, start_frame, end_frame,
            ocr_roi_ratio, ocr_tesseract_config, ocr_preprocessing_enabled,
            max_distance_increment_mm=max_distance_increment_mm
        )
        
        if len(reference_frames) < self.config.vanishing_point.min_reference_frames:
            self.logger.warning(
                f"基準フレーム不足: {len(reference_frames)} < {self.config.vanishing_point.min_reference_frames}, "
                "暗部重心VPのみ使用"
            )
            # 従来の処理にフォールバック - この場合はエラーとして処理
            raise ValueError(
                f"基準フレーム不足: {len(reference_frames)} < {self.config.vanishing_point.min_reference_frames}"
            )
        
        # Step 2: 基準フレーム間で特徴点ベースVP推定
        feature_vp_data = self._estimate_feature_based_vp_between_references(
            reference_frames=reference_frames,
            video_path=video_path
        )
        
        # Step 3: 全フレームでVPデータを構築
        cap = cv2.VideoCapture(video_path)
        vp_data = []
        
        for frame_idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 暗部重心VP（全フレーム）
            try:
                dark_center = self.detect_dark_region_centroid(
                    frame,
                    threshold=self.config.pose.dark_threshold,
                    radius_limit=min(
                        self.transformer.camera.image_height,
                        self.transformer.camera.image_width
                    ) // 2 * self.config.pose.dark_region_radius_limit_ratio if hasattr(self.config.pose, 'dark_region_radius_limit_ratio') else None,
                    target_area=self.config.pose.target_dark_area if self.config.pose.use_adaptive_threshold else None,
                    threshold_range=self.config.pose.threshold_search_range if self.config.pose.use_adaptive_threshold else (1, 100)
                )
            except DarkRegionNotFoundError:
                camera = self.transformer.camera
                dark_center = (camera.cx, camera.cy)
                self.logger.warning(
                    f"暗部重心VP検出失敗、レンズ中心を使用 ({camera.cx:.1f}, {camera.cy:.1f})"
                )
            
            # OCR距離（read_ocr=Trueの場合のみ）
            if read_ocr:
                from src.ocr_utils import OCRConfig as OCRConfigUtils
                _ocr_config = OCRConfigUtils(
                    psm_mode=OCRConfigUtils._extract_psm_mode(ocr_tesseract_config),
                    preprocessing_enabled=ocr_preprocessing_enabled,
                )
                ocr_distance, _ = extract_distance_from_frame(
                    frame,
                    ocr_roi_ratio,
                    config=_ocr_config
                )
            else:
                ocr_distance = None
            
            frame_vp = {
                'frame_idx': frame_idx,
                'dark_center': dark_center,
                'ocr_distance': ocr_distance
            }
            
            # 特徴点VP（基準フレームのみ）
            if frame_idx in feature_vp_data:
                frame_vp['feature_vp'] = feature_vp_data[frame_idx]
            
            vp_data.append(frame_vp)
        
        cap.release()
        
        # Step 4: VP統合とスムージング
        # 特徴点VPがある場合は加重平均、ない場合は暗部重心VP
        integrated_vp = []
        for frame_vp in vp_data:
            dark_x, dark_y = frame_vp['dark_center']
            
            if 'feature_vp' in frame_vp and frame_vp['feature_vp']['success']:
                # 加重平均（inlier_ratioで重み付け）
                feature_data = frame_vp['feature_vp']
                weight = feature_data['inlier_ratio']
                
                integrated_x = (1 - weight) * dark_x + weight * feature_data['vp_x']
                integrated_y = (1 - weight) * dark_y + weight * feature_data['vp_y']
            else:
                # 暗部重心VPのみ
                integrated_x, integrated_y = dark_x, dark_y
            
            integrated_vp.append((integrated_x, integrated_y))
        
        # Step 5: Savitzky-Golay平滑化（ダミー実装 - 実際の実装は別途必要）
        # TODO: 実際のスムージング処理を実装
        # numpy.ndarray型に変換
        if integrated_vp:  # リストが空でない場合
            smoothed_vp = np.array(integrated_vp, dtype=np.float64)
        else:
            smoothed_vp = np.empty((0, 2), dtype=np.float64)
        
        # Step 6: 姿勢範囲計算（ダミー実装 - 実際の実装は別途必要）
        # TODO: 実際の探索範囲計算を実装
        search_ranges = {}  # とりあえず空辞書を返す
        
        self.logger.info("Phase1完了: VP収集と探索範囲計算完了")
        
        return {
            'vp_data': vp_data,
            'smoothed_vp': smoothed_vp,
            'search_ranges': search_ranges,
            'feature_vp_success_count': sum(
                1 for fvp in feature_vp_data.values() if fvp['success']
            ),
            'feature_vp_total_count': len(feature_vp_data)
        }

    def _u_frame_xy_feature_residual(
        self,
        world_prev: np.ndarray,
        curr_points: np.ndarray,
        x0: float,
        y0: float,
        z0: float,
        dz: float,
        roll_0: float,
        yaw_0: float,
        pitch_0: float,
        yaw2: float,
        pitch2: float,
    ) -> np.ndarray:
        """U Mode A: フレーム残差を (dz, dpitch) の同一予測で説明する。

        初期 (-90,90,90) は (0,0,90) と同一姿勢。pitch が周方向（車体 roll）。
        yaw は光軸まわり捩れで roll と同じ軸のため固定（pitch への相殺漏れを防ぐ）。
        逆投影は world_to_pixel と同一の R_c2w 視線を使う。
        """
        pos_z = np.array([x0, y0, z0 + dz], dtype=float)
        pred = self.transformer.world_to_pixel(
            world_prev, pos_z, roll_0, yaw2, pitch2
        )
        res_v = curr_points[:, 1] - pred[:, 1]
        res_u = curr_points[:, 0] - pred[:, 0]
        return np.concatenate([res_v, res_u])

    def _estimate_motion_with_reference(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        camera_state: Dict[str, Any],
        camera_params: Dict[str, Any],
        constraints: Optional[Dict[str, Tuple[float, float]]],
        run_reference: Dict[str, float],
        center_prior: Optional[Any],
        estimation_mode: Optional[str],
        hard_bounds: Optional[Any],
    ) -> Dict[str, float]:
        """run_reference 中心。特徴点残差と prior を同一 least_squares で相互補正する。"""
        from src.validation.geometry import (
            mode_a_estimate_yaw_pitch,
            mode_a_use_frame_xy_residual,
            wrap_angle_rad,
        )

        mode = (estimation_mode or "A").upper()
        x0, y0, z0 = [float(v) for v in camera_state["position"]]
        roll_0, yaw_0, pitch_0 = [float(v) for v in camera_state["orientation"]]
        x_ref = float(run_reference["x_ref_mm"])
        y_ref = float(run_reference["y_ref_mm"])
        roll_ref = float(run_reference["roll_ref_rad"])
        yaw_ref = float(run_reference["yaw_ref_rad"])
        pitch_ref = float(run_reference["pitch_ref_rad"])
        run_id = str(run_reference.get("run_id") or "")

        estimate_xy = mode == "C"
        estimate_roll = mode == "C"
        if mode == "C":
            estimate_yaw, estimate_pitch = True, True
            use_frame_xy = False
        else:
            estimate_yaw, estimate_pitch = mode_a_estimate_yaw_pitch(run_id)
            use_frame_xy = mode_a_use_frame_xy_residual(run_id)
        param_names = []
        bounds_lower = []
        bounds_upper = []
        initial_guess = []

        def _add(name, lo, hi, init=0.0):
            param_names.append(name)
            bounds_lower.append(lo)
            bounds_upper.append(hi)
            initial_guess.append(init)

        max_dx = self.config.motion.max_dx
        max_dy = self.config.motion.max_dy
        max_dz = self.config.motion.max_dz
        max_droll = np.radians(self.config.motion.max_droll)
        max_dyaw = np.radians(self.config.motion.max_dtheta)
        max_dpitch = np.radians(self.config.motion.max_dphi)

        if estimate_xy:
            dx_b = constraints.get("dx", (-max_dx, max_dx)) if constraints else (-max_dx, max_dx)
            dy_b = constraints.get("dy", (-max_dy, max_dy)) if constraints else (-max_dy, max_dy)
            _add("dx", dx_b[0], dx_b[1])
            _add("dy", dy_b[0], dy_b[1])
        dz_b = constraints.get("dz", (0.0, max_dz)) if constraints else (0.0, max_dz)
        _add("dz", dz_b[0], dz_b[1], (dz_b[0] + dz_b[1]) / 2.0)
        if estimate_roll:
            dr_b = constraints.get("droll", (-max_droll, max_droll)) if constraints else (-max_droll, max_droll)
            _add("droll", dr_b[0], dr_b[1])
        if estimate_yaw:
            dyaw_b = constraints.get("dyaw", (-max_dyaw, max_dyaw)) if constraints else (-max_dyaw, max_dyaw)
            _add("dyaw", dyaw_b[0], dyaw_b[1])
        if estimate_pitch:
            dp_b = constraints.get("dpitch", (-max_dpitch, max_dpitch)) if constraints else (-max_dpitch, max_dpitch)
            _add("dpitch", dp_b[0], dp_b[1])
        n_priors = (
            (2 if estimate_xy else 0)
            + (1 if estimate_roll else 0)
            + (1 if estimate_yaw else 0)
            + (1 if estimate_pitch else 0)
        )
        n_feat = (2 * len(prev_points)) if use_frame_xy else len(prev_points)

        try:
            unproject = (
                self.transformer.pixel_to_world_R_c2w
                if use_frame_xy
                else self.transformer.pixel_to_world
            )
            p0 = unproject(
                prev_points,
                np.array([x0, y0, z0], dtype=float),
                roll_0, yaw_0, pitch_0,
            )
        except CylinderIntersectionError as exc:
            raise MotionEstimationError(f"前フレーム特徴点の円筒交点に失敗: {exc}") from exc

        scale = 1.0
        sigmas = {
            "x": 1.0, "y": 2.0, "roll": np.radians(3.0),
            "yaw": np.radians(5.0), "pitch": np.radians(8.0),
        }
        if center_prior is not None:
            scale = float(getattr(center_prior, "scale", 1.0))
            sigmas["x"] = float(getattr(center_prior, "sigma_x_mm", sigmas["x"]))
            sigmas["y"] = float(getattr(center_prior, "sigma_y_mm", sigmas["y"]))
            sigmas["roll"] = np.radians(float(getattr(center_prior, "sigma_roll_deg", 3.0)))
            sigmas["yaw"] = np.radians(float(getattr(center_prior, "sigma_yaw_deg", 5.0)))
            sigmas["pitch"] = np.radians(float(getattr(center_prior, "sigma_pitch_deg", 8.0)))

        def unpack(params):
            d = dict(zip(param_names, params))
            dx = d.get("dx", 0.0)
            dy = d.get("dy", 0.0)
            dz = d.get("dz", 0.0)
            droll = d.get("droll", 0.0)
            dyaw = d.get("dyaw", 0.0)
            dpitch = d.get("dpitch", 0.0)
            return dx, dy, dz, droll, dyaw, dpitch

        def residuals(params, frozen=None):
            dx, dy, dz, droll, dyaw, dpitch = unpack(params)
            if frozen:
                dx = frozen.get("dx", dx)
                dy = frozen.get("dy", dy)
                dz = frozen.get("dz", dz)
                droll = frozen.get("droll", droll)
                dyaw = frozen.get("dyaw", dyaw)
                dpitch = frozen.get("dpitch", dpitch)
            pos1 = np.array([x0 + dx, y0 + dy, z0 + dz], dtype=float)
            roll2 = roll_0 + droll if estimate_roll else roll_ref
            yaw2 = yaw_0 + dyaw if estimate_yaw else yaw_ref
            pitch2 = pitch_0 + dpitch if estimate_pitch else pitch_ref
            try:
                if use_frame_xy:
                    feat = self._u_frame_xy_feature_residual(
                        p0, curr_points, x0, y0, z0, dz,
                        roll_0, yaw_0, pitch_0, yaw2, pitch2,
                    )
                else:
                    p1 = self.transformer.pixel_to_world(
                        curr_points, pos1, roll2, yaw2, pitch2
                    )
                    feat = np.linalg.norm(p0 - p1, axis=1)
            except Exception:
                return np.full(n_feat + n_priors, 1e6)
            priors = []
            if estimate_xy:
                priors.append(scale * (pos1[0] - x_ref) / max(sigmas["x"], 1e-6))
                priors.append(scale * (pos1[1] - y_ref) / max(sigmas["y"], 1e-6))
            if estimate_roll:
                priors.append(scale * wrap_angle_rad(roll2 - roll_ref) / max(sigmas["roll"], 1e-6))
            if estimate_yaw:
                priors.append(scale * wrap_angle_rad(yaw2 - yaw_ref) / max(sigmas["yaw"], 1e-6))
            if estimate_pitch:
                priors.append(scale * wrap_angle_rad(pitch2 - pitch_ref) / max(sigmas["pitch"], 1e-6))
            if not priors:
                return feat
            return np.concatenate([feat, np.array(priors, dtype=float)])

        result = least_squares(
            residuals, initial_guess, bounds=(bounds_lower, bounds_upper), loss="soft_l1"
        )
        motion = {
            "dx": 0.0, "dy": 0.0, "dz": 0.0,
            "droll": 0.0, "dtheta": 0.0, "dphi": 0.0, "dyaw": 0.0, "dpitch": 0.0,
        }
        raw = dict(zip(param_names, result.x))
        motion["dx"] = float(raw.get("dx", 0.0))
        motion["dy"] = float(raw.get("dy", 0.0))
        motion["dz"] = float(raw.get("dz", 0.0))
        motion["droll"] = float(raw.get("droll", 0.0))
        motion["dyaw"] = float(raw.get("dyaw", 0.0))
        motion["dpitch"] = float(raw.get("dpitch", 0.0))
        motion["dtheta"] = motion["dyaw"]
        motion["dphi"] = motion["dpitch"]
        motion["prior_norm"] = float(np.linalg.norm(result.fun[n_feat:]))
        motion["residual_rms"] = float(np.sqrt(np.mean(result.fun[:n_feat] ** 2)))

        pos = np.array([x0, y0, z0])
        ori = np.array([roll_0, yaw_0, pitch_0])
        constrained, bound_hit = self._apply_reference_hard_bounds(
            motion, pos, ori, run_reference, hard_bounds,
            estimate_xy, estimate_roll, estimate_yaw, estimate_pitch,
        )
        if any(bound_hit.values()):
            frozen = {}
            if bound_hit.get("x") or bound_hit.get("y"):
                frozen["dx"] = constrained["dx"]
                frozen["dy"] = constrained["dy"]
            if bound_hit.get("roll"):
                frozen["droll"] = constrained["droll"]
            if bound_hit.get("yaw"):
                frozen["dyaw"] = constrained["dyaw"]
            if bound_hit.get("pitch"):
                frozen["dpitch"] = constrained["dpitch"]
            result2 = least_squares(
                lambda p: residuals(p, frozen=frozen),
                [constrained.get(n, 0.0) if n != "dyaw" else constrained["dyaw"]
                 for n in param_names],
                bounds=(bounds_lower, bounds_upper),
                loss="soft_l1",
            )
            raw2 = dict(zip(param_names, result2.x))
            for k, v in raw2.items():
                if k not in frozen:
                    if k == "dyaw":
                        constrained["dyaw"] = float(v)
                        constrained["dtheta"] = float(v)
                    elif k == "dpitch":
                        constrained["dpitch"] = float(v)
                        constrained["dphi"] = float(v)
                    else:
                        constrained[k] = float(v)
            constrained, bound_hit = self._apply_reference_hard_bounds(
                constrained, pos, ori, run_reference, hard_bounds,
                estimate_xy, estimate_roll, estimate_yaw, estimate_pitch,
            )
        constrained["bound_hit"] = bound_hit
        constrained["prior_norm"] = motion.get("prior_norm", 0.0)
        constrained["residual_rms"] = motion.get("residual_rms", 0.0)
        return constrained

    def _apply_reference_hard_bounds(
        self,
        motion: Dict[str, float],
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        run_reference: Dict[str, float],
        hard_bounds: Optional[Any],
        estimate_xy: bool,
        estimate_roll: bool,
        estimate_yaw: bool = True,
        estimate_pitch: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        from src.validation.geometry import wrap_angle_rad
        out = dict(motion)
        hit = {"x": False, "y": False, "roll": False, "yaw": False, "pitch": False}
        max_x = getattr(hard_bounds, "max_x_dev_mm", 5.0) if hard_bounds else 5.0
        max_y = getattr(hard_bounds, "max_y_dev_mm", 10.0) if hard_bounds else 10.0
        max_roll = np.radians(getattr(hard_bounds, "max_roll_dev_deg", 15.0) if hard_bounds else 15.0)
        max_yaw = np.radians(getattr(hard_bounds, "max_yaw_dev_deg", 20.0) if hard_bounds else 20.0)
        max_pitch = np.radians(getattr(hard_bounds, "max_pitch_dev_deg", 25.0) if hard_bounds else 25.0)
        x_ref = run_reference["x_ref_mm"]
        y_ref = run_reference["y_ref_mm"]
        roll_ref = run_reference["roll_ref_rad"]
        yaw_ref = run_reference["yaw_ref_rad"]
        pitch_ref = run_reference["pitch_ref_rad"]
        x_new = current_position[0] + out.get("dx", 0.0)
        y_new = current_position[1] + out.get("dy", 0.0)
        roll_new = current_orientation[0] + out.get("droll", 0.0)
        yaw_new = current_orientation[1] + out.get("dyaw", out.get("dtheta", 0.0))
        pitch_new = current_orientation[2] + out.get("dpitch", out.get("dphi", 0.0))
        if estimate_xy:
            if abs(x_new - x_ref) > max_x:
                hit["x"] = True
                x_new = np.clip(x_new, x_ref - max_x, x_ref + max_x)
                out["dx"] = float(x_new - current_position[0])
            if abs(y_new - y_ref) > max_y:
                hit["y"] = True
                y_new = np.clip(y_new, y_ref - max_y, y_ref + max_y)
                out["dy"] = float(y_new - current_position[1])
        if estimate_roll:
            droll = wrap_angle_rad(roll_new - roll_ref)
            if abs(droll) > max_roll:
                hit["roll"] = True
                roll_new = roll_ref + np.clip(droll, -max_roll, max_roll)
                out["droll"] = float(roll_new - current_orientation[0])
        if estimate_yaw:
            dyaw = wrap_angle_rad(yaw_new - yaw_ref)
            if abs(dyaw) > max_yaw:
                hit["yaw"] = True
                yaw_new = yaw_ref + np.clip(dyaw, -max_yaw, max_yaw)
                out["dyaw"] = float(yaw_new - current_orientation[1])
                out["dtheta"] = out["dyaw"]
        else:
            out["dyaw"] = 0.0
            out["dtheta"] = 0.0
        if estimate_pitch:
            dpitch = wrap_angle_rad(pitch_new - pitch_ref)
            if abs(dpitch) > max_pitch:
                hit["pitch"] = True
                pitch_new = pitch_ref + np.clip(dpitch, -max_pitch, max_pitch)
                out["dpitch"] = float(pitch_new - current_orientation[2])
                out["dphi"] = out["dpitch"]
        else:
            out["dpitch"] = 0.0
            out["dphi"] = 0.0
        return out, hit

    def _constrain_camera_pose_to_reference(
        self,
        motion: Dict[str, float],
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        frame_num: int,
        run_reference: Dict[str, float],
        hard_bounds: Optional[Any] = None,
    ) -> Dict[str, float]:
        constrained, hit = self._apply_reference_hard_bounds(
            motion, current_position, current_orientation,
            run_reference, hard_bounds, True, True,
        )
        if any(hit.values()):
            self.logger.warning(
                f"フレーム{frame_num}: run_reference hard bound hit={hit}"
            )
        constrained["bound_hit"] = hit
        return constrained


# ============================================================================
# フェーズ2タスク1.1: 消失点スムージング補正機能
# ============================================================================

def collect_vanishing_points(
    frames: list,
    estimator: CameraEstimator,
    roll_angles: Optional[np.ndarray] = None,
    threshold: int = 30,
    radius_limit: Optional[float] = None,
    target_area: Optional[int] = None,
    threshold_range: Tuple[int, int] = (1, 100)
) -> np.ndarray:
    """全フレームの消失点座標を収集

    Args:
        frames: 全フレーム画像のリスト（各要素は np.ndarray BGR画像）
        estimator: カメラ推定器
        roll_angles: 各フレームのロール角（ラジアン）、Noneの場合は全て0
        threshold: 暗部判定閾値（0-255）
        radius_limit: 暗部検出半径制限（ピクセル）
        target_area: 目標暗部ピクセル数（適応的閾値選択用）
        threshold_range: 適応的閾値探索範囲 (min, max)

    Returns:
        vanishing_points: 消失点座標配列 (N, 2) [x, y]
            検出失敗したフレームはNaNを格納

    Raises:
        ValueError: framesが空リストの場合

    処理の流れ:
        1. 各フレームでdetect_dark_region_centroid()を呼び出し
        2. 検出失敗時はNaNを記録
        3. 全フレームの消失点座標をndarrayで返す

    Example:
        >>> frames = [frame1, frame2, frame3]
        >>> estimator = CameraEstimator(config, transformer)
        >>> vp = collect_vanishing_points(frames, estimator, target_area=2000)
        >>> print(vp.shape)  # (3, 2)
        >>> print(vp[0])     # [960.5, 540.2] or [nan, nan]
    """
    if not frames:
        raise ValueError("framesが空リストです")

    n_frames = len(frames)
    vanishing_points = np.full((n_frames, 2), np.nan, dtype=np.float64)

    # ロール角が指定されていない場合は全て0
    if roll_angles is None:
        roll_angles = np.zeros(n_frames, dtype=np.float64)

    logger = logging.getLogger(__name__)

    for i, frame in enumerate(frames):
        try:
            centroid_x, centroid_y = estimator.detect_dark_region_centroid(
                frame, threshold, radius_limit, target_area, threshold_range
            )
            vanishing_points[i, 0] = centroid_x
            vanishing_points[i, 1] = centroid_y
            logger.debug(f"フレーム{i}: 消失点 ({centroid_x:.2f}, {centroid_y:.2f})")
        except DarkRegionNotFoundError as e:
            logger.warning(f"フレーム{i}: 消失点検出失敗 - {e}")
            # NaNのまま

    # 統計情報をログ出力
    n_valid = np.sum(~np.isnan(vanishing_points[:, 0]))
    logger.info(
        f"消失点収集完了: {n_valid}/{n_frames}フレーム成功 "
        f"({n_valid/n_frames*100:.1f}%)"
    )

    return vanishing_points

def collect_vanishing_points_and_ocr_distances(
    video_path: 'Path',
    start_frame: int,
    end_frame: int,
    estimator: 'CameraEstimator',
    ocr_roi_ratio: Tuple[float, float, float, float],
    config: 'EstimationConfig',
    read_ocr: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    ocr_config: Optional['OCRConfig'] = None,
    max_distance_increment_mm: float = 10.0
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """消失点とOCR距離を同時に収集（1回の動画読み込みで処理）

    Args:
        video_path: 動画ファイルパス
        start_frame: 開始フレーム番号
        end_frame: 終了フレーム番号
        estimator: CameraEstimatorインスタンス
        ocr_roi_ratio: OCR読み取り領域比率 (left, top, width, height)
        config: 推定設定
        read_ocr: OCR読み取りを実行するか（デフォルト: True）
            False の場合、OCR読み取りをスキップし、ocr_distances=None, success=None を返す
        progress_callback: フレーム単位の進捗コールバック関数（オプション）
            progress_callback(current_frame, total_frames)
            - current_frame: 処理済みフレーム数（1からtotal_framesまで）
            - total_frames: 総フレーム数
        ocr_config: OCR設定パラメータ（Noneの場合はデフォルト設定を使用）

    Returns:
        Tuple[vanishing_points, ocr_distances, success]:
            - vanishing_points: 消失点座標配列 (N, 2)
            - ocr_distances: OCR距離配列 (N,)、read_ocr=False の場合は None
            - success: OCR成功フラグ配列 (N,)、read_ocr=False の場合は None
    """
    import cv2
    import logging
    from src.ocr_utils import extract_distance_from_frame

    logger = logging.getLogger(__name__)
    n_frames = end_frame - start_frame  # 左閉右開区間 [start_frame, end_frame)
    vanishing_points = np.full((n_frames, 2), np.nan, dtype=np.float64)

    # OCR読み取りが有効な場合のみ配列を初期化
    if read_ocr:
        ocr_distances = np.full(n_frames, np.nan, dtype=np.float64)
        success_flags = np.zeros(n_frames, dtype=bool)
        # BUG-011修正: OCR距離検証用の状態管理
        prev_ocr_distance = None
        prev_ocr_success = False
        prev_ocr_frame_idx = None  # 直前のOCR成功フレームインデックス
        base_ocr_distance = None
        base_ocr_frame_idx = 0
    else:
        ocr_distances = None
        success_flags = None
    
    # 動画を開く
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"動画ファイルを開けません: {video_path}")
    
    # 開始フレームに移動
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 適応的閾値選択パラメータ
    use_adaptive = config.pose.use_adaptive_threshold
    target_area = config.pose.target_dark_area if use_adaptive else None
    threshold_range = config.pose.threshold_search_range if use_adaptive else (1, 100)
    radius_limit_ratio = config.pose.dark_region_radius_limit_ratio
    
    # フレームサイズ取得（radius_limit計算用）
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"最初のフレームを読み込めません: {video_path}")
    
    frame_height, frame_width = first_frame.shape[:2]
    radius_limit = None
    if radius_limit_ratio is not None and radius_limit_ratio > 0:
        radius_limit = min(frame_width, frame_height) // 2 * radius_limit_ratio

    # 連続明るさ検出カウンター（Phase 3: 連続検出最適化）
    consecutive_bright_count = 0
    skip_brightness_check = False

    # TASK-22: 平均値計算用の累積変数（brightness_fallback_method="average"用）
    vp_sum_x = 0.0  # 有効な消失点X座標の合計
    vp_sum_y = 0.0  # 有効な消失点Y座標の合計
    vp_valid_count = 0  # 有効な消失点のカウント
    brightness_fallback_method = config.pose.brightness_fallback_method

    # TASK-22: フォールバック値を計算するヘルパー関数
    def _get_fallback_vanishing_point(fallback_point: Tuple[float, float]) -> Tuple[float, float]:
        """フォールバック消失点を取得（brightness_fallback_methodに基づく）"""
        if brightness_fallback_method == "average" and vp_valid_count > 0:
            # 平均値を使用
            return (vp_sum_x / vp_valid_count, vp_sum_y / vp_valid_count)
        else:
            # レンズ中心を使用（現行方式、またはまだ有効な消失点がない場合）
            return fallback_point

    # 最初のフレームを処理
    if skip_brightness_check:
        # 明るさチェックをスキップして直接フォールバック値を使用
        camera = estimator.transformer.camera
        fallback_x, fallback_y = _get_fallback_vanishing_point((camera.cx, camera.cy))
        vanishing_points[0, 0] = fallback_x
        vanishing_points[0, 1] = fallback_y
        logger.info(f"フレーム0: スキップモード - フォールバック({fallback_x:.2f}, {fallback_y:.2f})を消失点に設定")
    else:
        try:
            centroid_x, centroid_y = estimator.detect_dark_region_centroid(
                first_frame,
                threshold=config.pose.dark_threshold,
                target_area=target_area,
                threshold_range=threshold_range,
                radius_limit=radius_limit
            )
            vanishing_points[0, 0] = centroid_x
            vanishing_points[0, 1] = centroid_y
            consecutive_bright_count = 0  # 正常検出時はカウンターリセット
            # TASK-22: 有効な消失点を累積
            vp_sum_x += centroid_x
            vp_sum_y += centroid_y
            vp_valid_count += 1
        except DarkRegionNotFoundError as e:
            # TASK-22: フォールバック方式に基づいて消失点を設定
            fallback_x, fallback_y = _get_fallback_vanishing_point(e.fallback_point)
            vanishing_points[0, 0] = fallback_x
            vanishing_points[0, 1] = fallback_y
            consecutive_bright_count += 1
            method_desc = "平均値" if brightness_fallback_method == "average" and vp_valid_count > 0 else "レンズ中心"
            logger.info(f"フレーム0: 明るさ検出により消失点を{method_desc}({fallback_x:.2f}, {fallback_y:.2f})に設定 (連続{consecutive_bright_count}回)")

            # 連続検出閾値に達したらスキップモードに移行
            if consecutive_bright_count >= config.pose.consecutive_brightness_detection_threshold:
                skip_brightness_check = True
                logger.warning(
                    f"連続{consecutive_bright_count}フレームで明るさ検出が発生しました。"
                    f"以降は明るさチェックをスキップしてフォールバック値を消失点として使用します。"
                )
    
    # OCR距離読み取り（read_ocr=Trueの場合のみ）
    # 前回成功した閾値を追跡（検証リトライで優先使用）
    last_successful_threshold = None

    if read_ocr:
        try:
            # 期待距離範囲を計算（prev_ocr_distanceベースを優先）
            expected_range = None
            if prev_ocr_distance is not None and prev_ocr_frame_idx is not None:
                gap_from_prev = start_frame - prev_ocr_frame_idx
                expected_range = (
                    prev_ocr_distance,
                    prev_ocr_distance + max_distance_increment_mm * gap_from_prev
                )
            elif base_ocr_distance is not None:
                frame_gap = start_frame - base_ocr_frame_idx
                expected_range = (
                    base_ocr_distance,
                    base_ocr_distance + max_distance_increment_mm * frame_gap
                )

            distance_raw, used_threshold = extract_distance_from_frame(
                first_frame,
                ocr_roi_ratio,
                config=ocr_config,
                expected_range_mm=expected_range,
                preferred_threshold=last_successful_threshold
            )
            if used_threshold is not None:
                last_successful_threshold = used_threshold

            # BUG-011修正: OCR距離の妥当性検証
            if prev_ocr_distance is not None and prev_ocr_frame_idx is not None:
                gap_for_validate = start_frame - prev_ocr_frame_idx
                ref_distance = prev_ocr_distance
            elif base_ocr_distance is not None:
                gap_for_validate = start_frame - base_ocr_frame_idx
                ref_distance = base_ocr_distance
            else:
                gap_for_validate = 1
                ref_distance = None
            distance = estimator.validate_ocr_distance(
                frame_idx=start_frame,
                ocr_distance=distance_raw,
                prev_ocr_distance=prev_ocr_distance,
                prev_ocr_success=prev_ocr_success,
                base_ocr_distance=ref_distance,
                frame_gap=gap_for_validate,
                max_increment=max_distance_increment_mm
            )

            if distance is not None and not np.isnan(distance):
                ocr_distances[0] = distance
                success_flags[0] = True
                # 状態更新
                prev_ocr_distance = distance
                prev_ocr_success = True
                prev_ocr_frame_idx = start_frame
                if base_ocr_distance is None:
                    base_ocr_distance = distance
                    base_ocr_frame_idx = start_frame
            else:
                prev_ocr_success = False
        except Exception as e:
            # OCR失敗時はスキップ（既にNaN/Falseで初期化済み）
            logger.debug(f"フレーム{start_frame}のOCR失敗: {e}")
            prev_ocr_success = False
    
    # 進捗更新（最初のフレーム完了）
    if progress_callback:
        try:
            progress_callback(1, n_frames)
        except Exception as e:
            logger.warning(f"Progress callback failed at frame {start_frame}: {e}")
    
    # 残りのフレームを処理
    for i in range(1, n_frames):
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"フレーム{start_frame + i}の読み込み失敗")
            break
        
        # 消失点検出
        if skip_brightness_check:
            # 明るさチェックをスキップして直接フォールバック値を使用
            camera = estimator.transformer.camera
            fallback_x, fallback_y = _get_fallback_vanishing_point((camera.cx, camera.cy))
            vanishing_points[i, 0] = fallback_x
            vanishing_points[i, 1] = fallback_y
            # スキップモード時はログを減らす（100フレームごとに出力）
            if i % 100 == 0:
                logger.debug(f"フレーム{i}: スキップモード - フォールバックを消失点に設定")
        else:
            try:
                centroid_x, centroid_y = estimator.detect_dark_region_centroid(
                    frame,
                    threshold=config.pose.dark_threshold,
                    target_area=target_area,
                    threshold_range=threshold_range,
                    radius_limit=radius_limit
                )
                vanishing_points[i, 0] = centroid_x
                vanishing_points[i, 1] = centroid_y
                consecutive_bright_count = 0  # 正常検出時はカウンターリセット
                # TASK-22: 有効な消失点を累積
                vp_sum_x += centroid_x
                vp_sum_y += centroid_y
                vp_valid_count += 1
            except DarkRegionNotFoundError as e:
                # TASK-22: フォールバック方式に基づいて消失点を設定
                fallback_x, fallback_y = _get_fallback_vanishing_point(e.fallback_point)
                vanishing_points[i, 0] = fallback_x
                vanishing_points[i, 1] = fallback_y
                consecutive_bright_count += 1
                method_desc = "平均値" if brightness_fallback_method == "average" and vp_valid_count > 0 else "レンズ中心"
                logger.info(f"フレーム{i}: 明るさ検出により消失点を{method_desc}({fallback_x:.2f}, {fallback_y:.2f})に設定 (連続{consecutive_bright_count}回)")

                # 連続検出閾値に達したらスキップモードに移行
                if consecutive_bright_count >= config.pose.consecutive_brightness_detection_threshold:
                    skip_brightness_check = True
                    logger.warning(
                        f"連続{consecutive_bright_count}フレームで明るさ検出が発生しました。"
                        f"以降は明るさチェックをスキップしてフォールバック値を消失点として使用します。"
                    )
        
        # OCR距離読み取り（read_ocr=Trueの場合のみ）
        if read_ocr:
            try:
                # 期待距離範囲を計算（prev_ocr_distanceベースを優先）
                current_frame_idx = start_frame + i
                expected_range = None
                if prev_ocr_distance is not None and prev_ocr_frame_idx is not None:
                    gap_from_prev = current_frame_idx - prev_ocr_frame_idx
                    expected_range = (
                        prev_ocr_distance,
                        prev_ocr_distance + max_distance_increment_mm * gap_from_prev
                    )
                elif base_ocr_distance is not None:
                    frame_gap = current_frame_idx - base_ocr_frame_idx
                    expected_range = (
                        base_ocr_distance,
                        base_ocr_distance + max_distance_increment_mm * frame_gap
                    )

                distance_raw, used_threshold = extract_distance_from_frame(
                    frame,
                    ocr_roi_ratio,
                    config=ocr_config,
                    expected_range_mm=expected_range,
                    preferred_threshold=last_successful_threshold
                )
                if used_threshold is not None:
                    last_successful_threshold = used_threshold

                # BUG-011修正: OCR距離の妥当性検証（prev_ocr_distanceベースを優先）
                if prev_ocr_distance is not None and prev_ocr_frame_idx is not None:
                    gap_for_validate = current_frame_idx - prev_ocr_frame_idx
                    ref_distance = prev_ocr_distance
                elif base_ocr_distance is not None:
                    gap_for_validate = current_frame_idx - base_ocr_frame_idx
                    ref_distance = base_ocr_distance
                else:
                    gap_for_validate = 1
                    ref_distance = None
                distance = estimator.validate_ocr_distance(
                    frame_idx=current_frame_idx,
                    ocr_distance=distance_raw,
                    prev_ocr_distance=prev_ocr_distance,
                    prev_ocr_success=prev_ocr_success,
                    base_ocr_distance=ref_distance,
                    frame_gap=gap_for_validate,
                    max_increment=max_distance_increment_mm
                )

                if distance is not None and not np.isnan(distance):
                    ocr_distances[i] = distance
                    success_flags[i] = True
                    # 状態更新
                    prev_ocr_distance = distance
                    prev_ocr_success = True
                    prev_ocr_frame_idx = current_frame_idx
                    # 基準フレームの更新判定（10mm間隔到達時）
                    if base_ocr_distance is not None:
                        if distance >= base_ocr_distance + 10.0:
                            base_ocr_distance = distance
                            base_ocr_frame_idx = current_frame_idx
                else:
                    prev_ocr_success = False
            except Exception as e:
                # OCR失敗時はスキップ（既にNaN/Falseで初期化済み）
                logger.debug(f"フレーム{start_frame + i}のOCR失敗: {e}")
                prev_ocr_success = False
        
        # 進捗更新
        if progress_callback:
            try:
                progress_callback(i + 1, n_frames)
            except Exception as e:
                logger.warning(f"Progress callback failed at frame {start_frame + i}: {e}")
    
    cap.release()
    
    return vanishing_points, ocr_distances, success_flags





def detect_outliers_iqr(
    points: np.ndarray,
    k: float = 1.5
) -> np.ndarray:
    """IQR法による異常値検出

    Args:
        points: 座標配列 (N, 2) [x, y] または (N, 1) [x]
        k: IQR倍率（デフォルト1.5、Tukeyの基準）

    Returns:
        mask: 正常値マスク (N,) bool配列（True=正常値、False=異常値）

    処理の流れ:
        1. x座標、y座標それぞれについてQ1, Q3を計算
        2. IQR = Q3 - Q1
        3. [Q1 - k*IQR, Q3 + k*IQR] の範囲を正常値とする
        4. NaN値は異常値（False）とする

    Note:
        Tukeyの外れ値検出法:
        - k=1.5: 通常の外れ値
        - k=3.0: 極端な外れ値

    Example:
        >>> points = np.array([[100, 200], [102, 198], [500, 600], [101, 201]])
        >>> mask = detect_outliers_iqr(points, k=1.5)
        >>> print(mask)  # [True, True, False, True]

        >>> # 1次元配列も対応
        >>> points_1d = np.array([[100], [102], [500], [101]])
        >>> mask = detect_outliers_iqr(points_1d, k=1.5)
        >>> print(mask)  # [True, True, False, True]
    """
    if points.shape[0] == 0:
        return np.array([], dtype=bool)

    # NaN値を除外して統計を計算
    if points.shape[1] == 1:
        # 1次元配列の場合
        valid_mask = ~np.isnan(points[:, 0])
    else:
        # 2次元配列の場合
        valid_mask = ~np.isnan(points[:, 0]) & ~np.isnan(points[:, 1])
    
    if np.sum(valid_mask) < 4:
        # 有効点が4点未満の場合、四分位数が計算できないため全てTrueを返す
        logger = logging.getLogger(__name__)
        logger.warning(
            f"有効点数が不足（{np.sum(valid_mask)}点）、IQR法をスキップ"
        )
        return valid_mask.copy()
    
    mask = valid_mask.copy()

    # x座標のIQR
    x_valid = points[valid_mask, 0]
    Q1_x = np.percentile(x_valid, 25)
    Q3_x = np.percentile(x_valid, 75)
    IQR_x = Q3_x - Q1_x
    lower_x = Q1_x - k * IQR_x
    upper_x = Q3_x + k * IQR_x

    # x座標が範囲内の点を正常値とする
    mask &= (points[:, 0] >= lower_x) & (points[:, 0] <= upper_x)

    # y座標のIQR（2次元配列の場合のみ）
    if points.shape[1] == 2:
        y_valid = points[valid_mask, 1]
        Q1_y = np.percentile(y_valid, 25)
        Q3_y = np.percentile(y_valid, 75)
        IQR_y = Q3_y - Q1_y
        lower_y = Q1_y - k * IQR_y
        upper_y = Q3_y + k * IQR_y

        # x, y両方が範囲内の点を正常値とする
        mask &= (points[:, 1] >= lower_y) & (points[:, 1] <= upper_y)

        logger = logging.getLogger(__name__)
        n_outliers = np.sum(valid_mask) - np.sum(mask)
        logger.info(
            f"IQR法: {n_outliers}個の異常値を検出 "
            f"(k={k:.2f}, x:[{lower_x:.1f}, {upper_x:.1f}], "
            f"y:[{lower_y:.1f}, {upper_y:.1f}])"
        )
    else:
        # 1次元配列の場合
        logger = logging.getLogger(__name__)
        n_outliers = np.sum(valid_mask) - np.sum(mask)
        logger.info(
            f"IQR法: {n_outliers}個の異常値を検出 "
            f"(k={k:.2f}, x:[{lower_x:.1f}, {upper_x:.1f}])"
        )
    
    return mask




def detect_outliers_local_iqr(
    vp: np.ndarray,
    window_size: int = 50,
    k: float = 1.5,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """スライディングウィンドウIQR法による異常値検出

    各フレームを中心に ±window_size/2 の範囲でIQR統計値を計算し、
    その局所統計に基づいて異常判定を行います。

    Args:
        vp: 消失点座標配列 (N, 2) [x, y]
        window_size: ウィンドウサイズ（フレーム数）
        k: IQR倍率（デフォルト1.5、Tukeyの基準）
        logger: ロガー（Noneの場合は取得）

    Returns:
        mask: 正常値マスク (N,) bool配列（True=正常値、False=異常値）

    処理の流れ:
        1. 各フレームiについて、ウィンドウ範囲 [i-window_size/2, i+window_size/2] を決定
        2. ウィンドウ内のx座標、y座標それぞれについてQ1, Q3を計算
        3. IQR = Q3 - Q1
        4. フレームiの値が [Q1 - k*IQR, Q3 + k*IQR] 範囲外なら異常と判定
        5. NaN値は異常値（False）とする

    Note:
        グローバル統計ベース判定と異なり、処理範囲に依存せず、
        各フレーム周辺の局所的な統計値のみで判定するため、
        処理範囲が変わっても同じフレームは同じ判定結果を得る。

    Example:
        >>> # 短い範囲でも長い範囲でも同じフレームは同じ判定
        >>> points_short = vp[2380:2410]  # 30フレーム
        >>> points_long = vp[1800:2800]   # 1000フレーム
        >>> mask_short = detect_outliers_local_iqr(points_short, window_size=50)
        >>> mask_long = detect_outliers_local_iqr(points_long, window_size=50)
        >>> # フレーム2399の判定は両方で一致する
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if vp.shape[0] == 0:
        return np.array([], dtype=bool)

    n_frames = vp.shape[0]

    # NaN値を除外
    valid_mask = ~np.isnan(vp[:, 0]) & ~np.isnan(vp[:, 1])

    # 初期化: 全て正常値とする
    mask = valid_mask.copy()

    outlier_count = 0

    # 各フレームについてスライディングウィンドウで判定
    for i in range(n_frames):
        if not valid_mask[i]:
            # NaN値はスキップ
            continue

        # ウィンドウ範囲を決定
        half_window = window_size // 2
        start = max(0, i - half_window)
        end = min(n_frames, i + half_window + 1)

        # ウィンドウ内の有効点を抽出
        window_valid = valid_mask[start:end]
        window_points = vp[start:end]

        # 有効点が4点未満の場合は判定スキップ（四分位数が計算できない）
        if np.sum(window_valid) < 4:
            continue

        # X座標のIQR計算
        x_valid = window_points[window_valid, 0]
        Q1_x = np.percentile(x_valid, 25)
        Q3_x = np.percentile(x_valid, 75)
        IQR_x = Q3_x - Q1_x
        lower_x = Q1_x - k * IQR_x
        upper_x = Q3_x + k * IQR_x

        # フレームiのx座標が範囲外なら異常
        if vp[i, 0] < lower_x or vp[i, 0] > upper_x:
            mask[i] = False
            outlier_count += 1
            continue

        # Y座標のIQR計算
        y_valid = window_points[window_valid, 1]
        Q1_y = np.percentile(y_valid, 25)
        Q3_y = np.percentile(y_valid, 75)
        IQR_y = Q3_y - Q1_y
        lower_y = Q1_y - k * IQR_y
        upper_y = Q3_y + k * IQR_y

        # フレームiのy座標が範囲外なら異常
        if vp[i, 1] < lower_y or vp[i, 1] > upper_y:
            mask[i] = False
            outlier_count += 1

    # ログ出力
    logger.info(
        f"局所IQR法: {outlier_count}個の異常値を検出 "
        f"(window_size={window_size}, k={k:.2f})"
    )

    return mask


def detect_outliers_local_3sigma(
    vp: np.ndarray,
    window_size: int = 50,
    sigma: float = 3.0,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """スライディングウィンドウ3σ法による異常値検出

    各フレームを中心に ±window_size/2 の範囲で平均・標準偏差を計算し、
    その局所統計に基づいて異常判定を行います。

    Args:
        vp: 消失点座標配列 (N, 2) [x, y]
        window_size: ウィンドウサイズ（フレーム数）
        sigma: 標準偏差倍率（デフォルト3.0）
        logger: ロガー（Noneの場合は取得）

    Returns:
        mask: 正常値マスク (N,) bool配列（True=正常値、False=異常値）

    処理の流れ:
        1. 各フレームiについて、ウィンドウ範囲 [i-window_size/2, i+window_size/2] を決定
        2. ウィンドウ内のx座標、y座標それぞれについて平均μと標準偏差σを計算
        3. フレームiの値が [μ - sigma*σ, μ + sigma*σ] 範囲外なら異常と判定
        4. NaN値は異常値（False）とする

    Note:
        グローバル統計ベース判定と異なり、処理範囲に依存せず、
        各フレーム周辺の局所的な統計値のみで判定するため、
        処理範囲が変わっても同じフレームは同じ判定結果を得る。
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if vp.shape[0] == 0:
        return np.array([], dtype=bool)

    n_frames = vp.shape[0]

    # NaN値を除外
    valid_mask = ~np.isnan(vp[:, 0]) & ~np.isnan(vp[:, 1])

    # 初期化: 全て正常値とする
    mask = valid_mask.copy()

    outlier_count = 0

    # 各フレームについてスライディングウィンドウで判定
    for i in range(n_frames):
        if not valid_mask[i]:
            continue

        # ウィンドウ範囲を決定
        half_window = window_size // 2
        start = max(0, i - half_window)
        end = min(n_frames, i + half_window + 1)

        # ウィンドウ内の有効点を抽出
        window_valid = valid_mask[start:end]
        window_points = vp[start:end]

        # 有効点が2点未満の場合は判定スキップ（標準偏差が計算できない）
        if np.sum(window_valid) < 2:
            continue

        # X座標の平均・標準偏差計算
        x_valid = window_points[window_valid, 0]
        mean_x = np.mean(x_valid)
        std_x = np.std(x_valid, ddof=1)

        # 標準偏差がゼロの場合の対処
        if std_x < 1e-10:
            std_x = 1e-10  # ゼロ除算回避

        lower_x = mean_x - sigma * std_x
        upper_x = mean_x + sigma * std_x

        # フレームiのx座標が範囲外なら異常
        if vp[i, 0] < lower_x or vp[i, 0] > upper_x:
            mask[i] = False
            outlier_count += 1
            continue

        # Y座標の処理
        y_valid = window_points[window_valid, 1]
        mean_y = np.mean(y_valid)
        std_y = np.std(y_valid, ddof=1)

        if std_y < 1e-10:
            std_y = 1e-10

        lower_y = mean_y - sigma * std_y
        upper_y = mean_y + sigma * std_y

        # フレームiのy座標が範囲外なら異常
        if vp[i, 1] < lower_y or vp[i, 1] > upper_y:
            mask[i] = False
            outlier_count += 1

    # ログ出力
    logger.info(
        f"局所3σ法: {outlier_count}個の異常値を検出 "
        f"(window_size={window_size}, sigma={sigma:.1f})"
    )

    return mask


def detect_outliers_3sigma(
    points: np.ndarray,
    sigma: float = 3.0
) -> np.ndarray:
    """3σ法による異常値検出

    Args:
        points: 座標配列 (N, 2) [x, y] または (N, 1) [x]
        sigma: 標準偏差倍率（デフォルト3.0）

    Returns:
        mask: 正常値マスク (N,) bool配列（True=正常値、False=異常値）

    処理の流れ:
        1. x座標、y座標それぞれについて平均μと標準偏差σを計算
        2. [μ - sigma*σ, μ + sigma*σ] の範囲を正常値とする
        3. NaN値は異常値（False）とする

    Note:
        正規分布を仮定した場合:
        - σ=2.0: 約95.4%のデータを含む
        - σ=3.0: 約99.7%のデータを含む

    Example:
        >>> points = np.array([[100, 200], [102, 198], [500, 600], [101, 201]])
        >>> mask = detect_outliers_3sigma(points, sigma=3.0)
        >>> print(mask)  # [True, True, False, True]

        >>> # 1次元配列も対応
        >>> points_1d = np.array([[100], [102], [500], [101]])
        >>> mask = detect_outliers_3sigma(points_1d, sigma=3.0)
        >>> print(mask)  # [True, True, False, True]
    """
    if points.shape[0] == 0:
        return np.array([], dtype=bool)

    # NaN値を除外して統計を計算
    if points.shape[1] == 1:
        # 1次元配列の場合
        valid_mask = ~np.isnan(points[:, 0])
    else:
        # 2次元配列の場合
        valid_mask = ~np.isnan(points[:, 0]) & ~np.isnan(points[:, 1])
    
    if np.sum(valid_mask) < 2:
        # 有効点が2点未満の場合、標準偏差が計算できないため全てTrueを返す
        logger = logging.getLogger(__name__)
        logger.warning(
            f"有効点数が不足（{np.sum(valid_mask)}点）、3σ法をスキップ"
        )
        return valid_mask.copy()
    
    mask = valid_mask.copy()

    # x座標の平均と標準偏差
    x_valid = points[valid_mask, 0]
    mean_x = np.mean(x_valid)
    std_x = np.std(x_valid, ddof=1)  # 不偏標準偏差

    # 標準偏差がゼロの場合の対処
    logger = logging.getLogger(__name__)
    if std_x < 1e-10:
        # x座標がほぼ同じ値の場合、全て正常値とする
        logger.warning(f"x座標の標準偏差がほぼゼロ（{std_x:.2e}）")
        std_x = 1e-10  # ゼロ除算回避

    # 範囲計算
    lower_x = mean_x - sigma * std_x
    upper_x = mean_x + sigma * std_x

    # x座標が範囲内の点を正常値とする
    mask &= (points[:, 0] >= lower_x) & (points[:, 0] <= upper_x)

    # y座標の処理（2次元配列の場合のみ）
    if points.shape[1] == 2:
        # y座標の平均と標準偏差
        y_valid = points[valid_mask, 1]
        mean_y = np.mean(y_valid)
        std_y = np.std(y_valid, ddof=1)

        if std_y < 1e-10:
            logger.warning(f"y座標の標準偏差がほぼゼロ（{std_y:.2e}）")
            std_y = 1e-10

        lower_y = mean_y - sigma * std_y
        upper_y = mean_y + sigma * std_y

        # x, y両方が範囲内の点を正常値とする
        mask &= (points[:, 1] >= lower_y) & (points[:, 1] <= upper_y)

        n_outliers = np.sum(valid_mask) - np.sum(mask)
        logger.info(
            f"3σ法: {n_outliers}個の異常値を検出 "
            f"(sigma={sigma:.1f}, x:[{lower_x:.1f}, {upper_x:.1f}], "
            f"y:[{lower_y:.1f}, {upper_y:.1f}])"
        )
    else:
        # 1次元配列の場合
        n_outliers = np.sum(valid_mask) - np.sum(mask)
        logger.info(
            f"3σ法: {n_outliers}個の異常値を検出 "
            f"(sigma={sigma:.1f}, x:[{lower_x:.1f}, {upper_x:.1f}])"
        )
    
    return mask


def detect_outliers_by_magnitude(
    vp: np.ndarray,
    threshold: float,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """消失点座標の変動量ベースで異常値を検出

    フレーム間のユークリッド距離を計算し、閾値を超える変動を異常として検出

    Args:
        vp: 消失点座標 (n_frames, 2) [X, Y]
        threshold: 変動量閾値(px)
        logger: ロガー

    Returns:
        マスク配列 (n_frames,) True=正常, False=異常

    アルゴリズム:
        1. フレーム間のユークリッド距離を計算
           delta[i] = √((x[i]-x[i-1])² + (y[i]-y[i-1])²)
        2. delta > threshold の場合、異常と判定
        3. 最初のフレームは正常と判定（delta=0）

    Example:
        >>> vp = np.array([[550, 540], [620, 535], [590, 545]])
        >>> mask = detect_outliers_by_magnitude(vp, threshold=50.0)
        >>> print(mask)  # [True, False, True]
    """
    n_frames = vp.shape[0]
    mask = np.ones(n_frames, dtype=bool)

    if n_frames < 2:
        return mask

    # フレーム間変動量を計算
    delta = np.sqrt(np.sum(np.diff(vp, axis=0)**2, axis=1))

    # 変動量が閾値を超えるフレームを異常と判定
    outlier_indices = np.where(delta > threshold)[0] + 1
    mask[outlier_indices] = False

    # ログ出力
    if logger is not None and len(outlier_indices) > 0:
        logger.info(
            f"変動量ベース異常判定: {len(outlier_indices)}フレームを異常と検出 "
            f"（閾値: {threshold:.1f}px）"
        )
        for idx in outlier_indices:
            logger.warning(
                f"消失点の急激な変動を検出: frame={idx}, "
                f"delta={delta[idx-1]:.2f}px > threshold={threshold:.1f}px"
            )

        # 統計情報
        logger.info(
            f"変動量統計: 平均={np.mean(delta):.2f}px, "
            f"最大={np.max(delta):.2f}px, "
            f"標準偏差={np.std(delta):.2f}px"
        )

    return mask


def apply_vector_outlier_logic(
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """X, Y座標の異常判定を統合（ベクトル判定）

    X, Yのいずれかが異常の場合、両方を異常として扱う

    Args:
        mask_x: X座標のマスク (n_frames,) True=正常, False=異常
        mask_y: Y座標のマスク (n_frames,) True=正常, False=異常
        logger: ロガー

    Returns:
        統合マスク配列 (n_frames,) True=正常, False=異常

    ロジック:
        mask_vector = mask_x AND mask_y
        （両方が正常の場合のみ、ベクトル全体が正常）

    Example:
        >>> mask_x = np.array([True, False, True])
        >>> mask_y = np.array([True, True, True])
        >>> mask = apply_vector_outlier_logic(mask_x, mask_y)
        >>> print(mask)  # [True, False, True]
    """
    mask_vector = mask_x & mask_y

    # ログ出力
    if logger is not None:
        x_only_outliers = np.where(~mask_x & mask_y)[0]
        y_only_outliers = np.where(mask_x & ~mask_y)[0]

        if len(x_only_outliers) > 0:
            logger.info(
                f"ベクトル異常判定（X異常）: {len(x_only_outliers)}フレームで "
                f"X座標が異常 → (X,Y)全体を破棄"
            )

        if len(y_only_outliers) > 0:
            logger.info(
                f"ベクトル異常判定（Y異常）: {len(y_only_outliers)}フレームで "
                f"Y座標が異常 → (X,Y)全体を破棄"
            )

    return mask_vector


def _detect_vp_segments(
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    segment_threshold: float = 30.0
) -> list:
    """有効VP座標列から不連続点を検出しセグメント境界を返す。

    連続する有効点間のユークリッド距離がthresholdを超える箇所を
    セグメント境界とする。隣接する2点以下の短いセグメントは
    前後の安定セグメントに統合する。

    Args:
        valid_x: 有効点のx座標配列
        valid_y: 有効点のy座標配列
        segment_threshold: セグメント分割閾値（ピクセル）

    Returns:
        segments: [(start, end), ...] 各セグメントの開始・終了インデックス
    """
    n = len(valid_x)
    if n < 2:
        return [(0, n)]

    # 連続する有効点間のユークリッド距離
    dx = np.diff(valid_x)
    dy = np.diff(valid_y)
    deltas = np.sqrt(dx ** 2 + dy ** 2)

    # 閾値を超える箇所をセグメント境界とする
    boundary_indices = np.where(deltas > segment_threshold)[0]

    if len(boundary_indices) == 0:
        return [(0, n)]

    # セグメント構築
    raw_segments = []
    prev_end = 0
    for bi in boundary_indices:
        raw_segments.append((prev_end, bi + 1))
        prev_end = bi + 1
    raw_segments.append((prev_end, n))

    # 短いセグメント（3点以下）を前後の安定セグメントに統合
    MIN_SEGMENT_LEN = 4
    merged_segments = []
    for seg in raw_segments:
        seg_len = seg[1] - seg[0]
        if seg_len < MIN_SEGMENT_LEN and merged_segments:
            # 前のセグメントに統合
            prev = merged_segments[-1]
            merged_segments[-1] = (prev[0], seg[1])
        else:
            merged_segments.append(seg)

    return merged_segments


def smooth_with_savgol(
    points: np.ndarray,
    mask: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2
) -> np.ndarray:
    """Savitzky-Golayフィルタによる消失点の平滑化（セグメント分割対応）

    緩やかに変化する消失点座標のトレンドを捉えるための平滑化。
    VP座標に不連続点（管路形状変化等）がある場合、不連続点で
    データをセグメント分割し、各セグメントに独立してフィルタを
    適用する。これにより、Savgolの多項式フィッティングが不連続点を
    跨いで前方に漏洩する問題を防止する。

    Args:
        points: 消失点座標配列 (N, 2) [x, y]
        mask: 有効値マスク (N,) True=正常値
        window_length: フィルタウィンドウ長（奇数、デフォルト11）
        polyorder: 多項式次数（デフォルト2）

    Returns:
        smoothed: 平滑化後の座標配列 (N, 2) [x, y]

    処理の流れ:
        1. 有効な点のみを抽出
        2. 不連続点を検出してセグメントに分割
        3. 各セグメントに対して独立にSavitzky-Golayフィルタを適用
        4. 無効な点（mask=False）はセグメント内で線形補間して埋める

    Note:
        Savitzky-Golayフィルタは、スプライン補間と異なり、
        代表点を通ることよりも滑らかさを優先する。
        消失点座標が「緩やかに変化する」仮定に適している。

        設計方針:
        - 消失点は粗い姿勢方向推定の初期値として使用
        - 個々の点に厳密にフィットさせる必要はない
        - 滑らかなトレンドを捉えることが重要
        - 不連続点を跨ぐ平滑化はスムージングアーティファクトを生むため回避する
    """
    from scipy.signal import savgol_filter

    n_frames = points.shape[0]
    smoothed = points.copy()

    logger = logging.getLogger(__name__)

    # 有効点の抽出
    valid_indices = np.where(mask)[0]
    n_valid = len(valid_indices)

    # window_lengthは奇数である必要がある
    if window_length % 2 == 0:
        window_length += 1

    if n_valid < window_length:
        logger.warning(
            f"有効点数({n_valid})がウィンドウ長({window_length})未満、"
            f"Savitzky-Golayフィルタをスキップ"
        )
        return smoothed

    valid_x = points[mask, 0]
    valid_y = points[mask, 1]

    # セグメント分割: 不連続点を検出
    segments = _detect_vp_segments(valid_x, valid_y, segment_threshold=30.0)

    smoothed_x = valid_x.copy()
    smoothed_y = valid_y.copy()

    if len(segments) > 1:
        logger.info(
            f"VP不連続点検出: {len(segments)}セグメントに分割 "
            f"(境界: {[s[0] for s in segments[1:]]})"
        )

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start

        # セグメントに適用可能なウィンドウ長を決定
        wl = window_length
        if seg_len < wl:
            wl = seg_len
            if wl % 2 == 0:
                wl -= 1
            if wl < polyorder + 2 or wl < 3:
                # セグメントが短すぎ: 元データを維持
                continue

        seg_x = valid_x[seg_start:seg_end]
        seg_y = valid_y[seg_start:seg_end]

        try:
            smoothed_x[seg_start:seg_end] = savgol_filter(
                seg_x, wl, polyorder, mode='nearest'
            )
        except Exception as e:
            logger.warning(
                f"セグメント[{seg_start}:{seg_end}]のx座標平滑化失敗: {e}"
            )

        try:
            smoothed_y[seg_start:seg_end] = savgol_filter(
                seg_y, wl, polyorder, mode='nearest'
            )
        except Exception as e:
            logger.warning(
                f"セグメント[{seg_start}:{seg_end}]のy座標平滑化失敗: {e}"
            )

    # 有効点位置に平滑化結果を設定
    smoothed[mask, 0] = smoothed_x
    smoothed[mask, 1] = smoothed_y

    # 無効点（mask=False）はセグメント境界を考慮して補間
    invalid_mask = ~mask
    if np.sum(invalid_mask) > 0:
        invalid_indices = np.where(invalid_mask)[0]
        smoothed[invalid_indices, 0] = np.interp(
            invalid_indices, valid_indices, smoothed_x
        )
        smoothed[invalid_indices, 1] = np.interp(
            invalid_indices, valid_indices, smoothed_y
        )

    logger.info(
        f"Savitzky-Golay平滑化適用: window_length={window_length}, "
        f"polyorder={polyorder}, {n_valid}/{n_frames}点を使用"
        + (f", {len(segments)}セグメント" if len(segments) > 1 else "")
    )

    return smoothed


def smooth_with_moving_average(
    points: np.ndarray,
    mask: np.ndarray,
    window: int = 5
) -> np.ndarray:
    """移動平均による消失点の平滑化（ジャンプ境界対策）

    スプライン補間の前処理として、局所的なジャンプを均すために使用。

    Args:
        points: 消失点座標配列 (N, 2) [x, y]
        mask: 有効値マスク (N,) True=正常値
        window: 移動平均ウィンドウサイズ（デフォルト5フレーム）

    Returns:
        smoothed: 平滑化後の座標配列 (N, 2) [x, y]

    処理の流れ:
        1. 有効な点のみを抽出
        2. 各座標に対して移動平均を適用（mode='same'でサイズ維持）
        3. 無効な点（mask=False）は線形補間で埋める

    Note:
        numpy.convolveを使用した単純な移動平均。
        スプライン補間前に適用することで、局所的な大きな変化を平滑化する。
    """
    n_frames = points.shape[0]
    smoothed = points.copy()

    logger = logging.getLogger(__name__)

    # 有効点の抽出
    valid_indices = np.where(mask)[0]
    n_valid = len(valid_indices)

    if n_valid < window:
        logger.warning(
            f"有効点数({n_valid})がウィンドウ長({window})未満、"
            f"移動平均をスキップ"
        )
        return smoothed

    # 移動平均カーネル
    kernel = np.ones(window) / window

    # x座標の移動平均
    valid_x = points[mask, 0]
    smoothed_x = np.convolve(valid_x, kernel, mode='same')

    # y座標の移動平均
    valid_y = points[mask, 1]
    smoothed_y = np.convolve(valid_y, kernel, mode='same')

    # 有効点位置に平滑化結果を設定
    smoothed[mask, 0] = smoothed_x
    smoothed[mask, 1] = smoothed_y

    # 無効点（mask=False）は線形補間で埋める
    invalid_mask = ~mask
    if np.sum(invalid_mask) > 0:
        invalid_indices = np.where(invalid_mask)[0]
        smoothed[invalid_indices, 0] = np.interp(invalid_indices, valid_indices, smoothed_x)
        smoothed[invalid_indices, 1] = np.interp(invalid_indices, valid_indices, smoothed_y)

    logger.info(
        f"移動平均平滑化適用: window={window}, {n_valid}/{n_frames}点を使用"
    )

    return smoothed


def smooth_vanishing_points_savgol(
    vanishing_points: np.ndarray,
    outlier_mask: Optional[np.ndarray] = None,
    window_length: int = 11,
    polyorder: int = 2
) -> np.ndarray:
    """Savitzky-Golay平滑化による消失点スムージング（推奨）

    消失点座標が「緩やかに変化する」仮定に基づく平滑化。
    個々の点に厳密にフィットさせるのではなく、滑らかなトレンドを捉える。

    Args:
        vanishing_points: 消失点座標配列 (N, 2) [x, y]
        outlier_mask: 異常値マスク (N,) True=正常値
            Noneの場合は全て正常値とする
        window_length: フィルタウィンドウ長（奇数、デフォルト11）
        polyorder: 多項式次数（デフォルト2）

    Returns:
        smoothed_points: 平滑化後の座標配列 (N, 2) [x, y]

    Raises:
        ValueError: 正常値が不足してフィルタを適用できない

    処理の流れ:
        1. 異常値とNaN値を除外
        2. Savitzky-Golayフィルタで平滑化
        3. 無効な点は線形補間で埋める

    Note:
        設計方針:
        - 消失点は粗い姿勢方向推定の初期値として使用
        - 最小二乗法での精緻な推定のガイドとして機能
        - 個々の点に厳密にフィットさせる必要はない
        - 滑らかなトレンドを捉えることが重要

        Savitzky-Golayフィルタの利点:
        - 局所的な多項式フィッティング
        - 異常値の影響を受けにくい
        - スプライン補間のような数値不安定性がない
        - ウィンドウ分割不要

    Example:
        >>> vp = np.array([[960, 540], [np.nan, np.nan], [965, 542], ...])
        >>> mask = np.array([True, False, True, ...])
        >>> smoothed = smooth_vanishing_points_savgol(vp, mask, window_length=11, polyorder=2)
    """
    n_frames = vanishing_points.shape[0]

    # 異常値マスクが指定されていない場合は全て正常値
    if outlier_mask is None:
        outlier_mask = np.ones(n_frames, dtype=bool)

    # NaN値も除外
    valid_mask = (
        outlier_mask &
        ~np.isnan(vanishing_points[:, 0]) &
        ~np.isnan(vanishing_points[:, 1])
    )

    n_valid = np.sum(valid_mask)

    logger = logging.getLogger(__name__)

    # window_lengthは奇数である必要がある
    if window_length % 2 == 0:
        window_length += 1

    if n_valid < window_length:
        raise ValueError(
            f"正常値が不足: {n_valid}点 < {window_length}点（ウィンドウ長）"
        )

    # Savitzky-Golay平滑化を適用
    smoothed_points = smooth_with_savgol(
        vanishing_points, valid_mask, window_length, polyorder
    )

    logger.info(f"Savitzky-Golay平滑化完了: 全{n_frames}フレーム")

    return smoothed_points


def smooth_vanishing_points_spline(
    vanishing_points: np.ndarray,
    outlier_mask: Optional[np.ndarray] = None,
    smoothing_factor: float = 0.0,
    k: int = 3,
    window_size: Optional[int] = None,
    use_moving_average_preprocess: bool = False,
    moving_average_window: int = 5
) -> np.ndarray:
    """スプライン補間による消失点スムージング

    Args:
        vanishing_points: 消失点座標配列 (N, 2) [x, y]
        outlier_mask: 異常値マスク (N,) True=正常値
            Noneの場合は全て正常値とする
        smoothing_factor: スムージング係数（0=補間、>0=平滑化）
            - 0: 正常値を全て通る補間スプライン
            - >0: スムーズな曲線（値が大きいほど滑らか）
        k: スプライン次数（1=線形、2=2次、3=3次、デフォルト3）
        window_size: ウィンドウサイズ（Noneの場合は全フレーム一括処理）
            長範囲のスプライン補間での数値不安定性を防ぐため、
            window_sizeを指定すると分割補間を実行
            推奨値: 20-30フレーム
        use_moving_average_preprocess: 移動平均前処理を使用するか（デフォルトFalse）
            生データのジャンプ境界を均すために、スプライン補間の前に移動平均を適用
        moving_average_window: 移動平均ウィンドウサイズ（デフォルト5フレーム）

    Returns:
        smoothed_points: スムージング後の座標配列 (N, 2) [x, y]

    Raises:
        ValueError: 正常値が不足（<k+1点）してスプラインを計算できない

    処理の流れ:
        1. 異常値とNaN値を除外
        2. [オプション] 移動平均前処理でジャンプ境界を均す
        3. window_sizeが指定されている場合、ウィンドウ分割補間
           - 各ウィンドウで独立したスプライン補間
           - ウィンドウ境界では線形ブレンド（3フレーム幅）
        4. x座標、y座標それぞれにスプライン補間
        5. 全フレームインデックスでスムージング座標を計算

    Note:
        scipy.interpolate.splrep/splevを使用
        - splrep: スプライン係数を計算
        - splev: 補間値を評価

        ウィンドウ分割補間は長範囲スプライン補間の数値不安定性対策
        （100フレーム以上で境界条件の影響により発散する問題を解決）

        移動平均前処理は、消失点のフレーム間ジャンプ対策
        （暗部検出の失敗などで発生する局所的な大きな変化を平滑化）

    Example:
        >>> vp = np.array([[960, 540], [np.nan, np.nan], [965, 542]])
        >>> mask = np.array([True, False, True])
        >>> smoothed = smooth_vanishing_points_spline(vp, mask, k=1)
        >>> print(smoothed[1])  # [962.5, 541.0]（線形補間）

        >>> # ウィンドウ分割補間（100フレーム以上推奨）
        >>> smoothed = smooth_vanishing_points_spline(vp, mask, k=3, window_size=30)

        >>> # ジャンプ境界対策（移動平均前処理）
        >>> smoothed = smooth_vanishing_points_spline(
        ...     vp, mask, k=3, window_size=30,
        ...     use_moving_average_preprocess=True, moving_average_window=5
        ... )
    """
    from scipy.interpolate import splrep, splev

    n_frames = vanishing_points.shape[0]

    # 異常値マスクが指定されていない場合は全て正常値
    if outlier_mask is None:
        outlier_mask = np.ones(n_frames, dtype=bool)

    # NaN値も除外
    valid_mask = (
        outlier_mask &
        ~np.isnan(vanishing_points[:, 0]) &
        ~np.isnan(vanishing_points[:, 1])
    )

    # 移動平均前処理（ジャンプ境界対策）
    input_points = vanishing_points
    if use_moving_average_preprocess:
        input_points = smooth_with_moving_average(
            vanishing_points, valid_mask, moving_average_window
        )

    n_valid = np.sum(valid_mask)

    logger = logging.getLogger(__name__)

    # デバッグ: 最初の数フレームの有効性を確認
    n_check = min(50, n_frames)
    n_valid_first = np.sum(valid_mask[:n_check])
    logger.info(
        f"スプライン補間開始: 全{n_frames}フレーム、有効点{n_valid}個 "
        f"(最初{n_check}フレーム中の有効点: {n_valid_first}個)"
    )

    if n_valid < k + 1:
        raise ValueError(
            f"正常値が不足: {n_valid}点 < {k+1}点（スプライン次数k={k}）"
        )

    # ウィンドウ分割補間
    if window_size is not None and n_frames > window_size:
        logger.info(
            f"ウィンドウ分割スプライン補間: {n_frames}フレームを"
            f"window_size={window_size}で分割 (k={k}, s={smoothing_factor})"
        )
        return _smooth_vanishing_points_windowed(
            vanishing_points=input_points,
            valid_mask=valid_mask,
            smoothing_factor=smoothing_factor,
            k=k,
            window_size=window_size
        )

    # 従来の一括補間
    # 有効なフレームインデックスと座標
    valid_indices = np.where(valid_mask)[0]
    valid_x = input_points[valid_mask, 0]
    valid_y = input_points[valid_mask, 1]

    logger.info(
        f"スプライン補間: {n_valid}/{n_frames}点を使用 "
        f"(k={k}, s={smoothing_factor})"
    )

    # x座標のスプライン補間
    try:
        tck_x = splrep(valid_indices, valid_x, s=smoothing_factor, k=k)
        all_indices = np.arange(n_frames)
        smoothed_x = splev(all_indices, tck_x)
    except Exception as e:
        logger.error(f"x座標のスプライン補間失敗: {e}")
        raise ValueError(f"x座標のスプライン補間失敗: {e}")

    # y座標のスプライン補間
    try:
        tck_y = splrep(valid_indices, valid_y, s=smoothing_factor, k=k)
        smoothed_y = splev(all_indices, tck_y)
    except Exception as e:
        logger.error(f"y座標のスプライン補間失敗: {e}")
        raise ValueError(f"y座標のスプライン補間失敗: {e}")

    # 結果を構築
    smoothed_points = np.column_stack((smoothed_x, smoothed_y))

    # 元がNaNだった位置はNaNを維持（オプション）
    # 今回は全フレームで補間値を返す仕様とする
    # もし元のNaNを維持したい場合は以下を有効化:
    # nan_mask = np.isnan(vanishing_points[:, 0])
    # smoothed_points[nan_mask, :] = np.nan

    logger.info(f"スプライン補間完了: 全{n_frames}フレーム")

    return smoothed_points


def _smooth_vanishing_points_windowed(
    vanishing_points: np.ndarray,
    valid_mask: np.ndarray,
    smoothing_factor: float,
    k: int,
    window_size: int
) -> np.ndarray:
    """ウィンドウ分割スプライン補間（内部実装）

    長範囲スプライン補間の数値不安定性を防ぐため、
    window_size単位でスプライン補間を分割実行し、
    ウィンドウ境界で線形ブレンドを行う。

    Args:
        vanishing_points: 消失点座標配列 (N, 2) [x, y]
        valid_mask: 有効値マスク (N,) True=正常値
        smoothing_factor: スムージング係数
        k: スプライン次数
        window_size: ウィンドウサイズ（フレーム数）

    Returns:
        smoothed_points: スムージング後の座標配列 (N, 2) [x, y]

    処理の流れ:
        1. window_size単位でウィンドウを分割（オーバーラップ5フレーム）
        2. 各ウィンドウで独立したスプライン補間を実行
        3. ウィンドウ境界で線形ブレンド（3フレーム幅）
        4. 最終ウィンドウは余りフレームを全て含める

    Example:
        100フレーム、window_size=30の場合：
        - Window 0: frames 0-34 (30+5オーバーラップ)
        - Window 1: frames 30-64 (5+30+5)
        - Window 2: frames 60-99 (5+40、余りを含む)

        境界ブレンド:
        - frames 30-32: window0とwindow1を線形ブレンド
        - frames 60-62: window1とwindow2を線形ブレンド
    """
    from scipy.interpolate import splrep, splev

    logger = logging.getLogger(__name__)
    n_frames = vanishing_points.shape[0]

    # ブレンド幅（ウィンドウ境界での遷移フレーム数）
    blend_width = 3
    # オーバーラップ幅（隣接ウィンドウとの重複フレーム数）
    overlap = 5

    # ウィンドウ境界を計算
    window_starts = []
    window_ends = []

    start = 0
    while start < n_frames:
        # 次のウィンドウ開始位置を計算
        next_start = start + window_size

        # 残りフレーム数を確認
        remaining = n_frames - start

        if remaining <= window_size:
            # 最後のウィンドウ：余りを全て含める
            window_starts.append(start)
            window_ends.append(n_frames)
            break
        else:
            # 次のウィンドウが作られた場合の残りを計算
            next_remaining = n_frames - next_start

            if next_remaining < window_size // 2:
                # 次のウィンドウが短すぎる（window_size/2未満）：現在のウィンドウを拡張
                window_starts.append(start)
                window_ends.append(n_frames)
                break
            else:
                # 通常のウィンドウ：window_size + overlap
                window_starts.append(start)
                end = start + window_size + overlap
                window_ends.append(end)
                start = next_start

    n_windows = len(window_starts)
    logger.info(
        f"ウィンドウ分割: {n_windows}個のウィンドウ "
        f"(window_size={window_size}, overlap={overlap}, blend_width={blend_width})"
    )

    # 各ウィンドウで補間を実行
    window_results = []

    for i, (w_start, w_end) in enumerate(zip(window_starts, window_ends)):
        logger.info(f"ウィンドウ{i}: frames {w_start}-{w_end-1}, size={w_end-w_start}")

        # ウィンドウ内の有効マスクとデータ
        window_valid_mask = valid_mask[w_start:w_end]
        window_vp = vanishing_points[w_start:w_end]

        n_valid_window = np.sum(window_valid_mask)
        logger.info(f"ウィンドウ{i}: 有効点数={n_valid_window}/{w_end-w_start}")

        # ウィンドウ内で異常値を再判定（局所的な分布に基づく）
        # 全体統計では異常値と判定されても、局所的には正常な場合がある
        window_mask_iqr = detect_outliers_iqr(window_vp, k=1.5)
        window_mask_3sigma = detect_outliers_3sigma(window_vp)
        window_local_mask = window_mask_iqr & window_mask_3sigma

        # 全体マスクと局所マスクの論理和（どちらかで正常なら使用）
        window_combined_mask = window_valid_mask | window_local_mask

        n_combined = np.sum(window_combined_mask)
        if n_combined > n_valid_window:
            logger.info(
                f"ウィンドウ{i}: 局所異常値判定により "
                f"{n_combined - n_valid_window}点を追加使用"
            )
            window_valid_mask = window_combined_mask
            n_valid_window = n_combined

        # 有効点が不足している場合の処理
        if n_valid_window < k + 1:
            logger.warning(
                f"ウィンドウ{i}: 有効点不足({n_valid_window}<{k+1})、"
                f"線形補間で代用"
            )
            # 線形補間で代用
            window_smoothed = _linear_interpolate_window(
                window_vp, window_valid_mask
            )
        else:
            # スプライン補間を実行
            valid_indices = np.where(window_valid_mask)[0]
            valid_x = window_vp[window_valid_mask, 0]
            valid_y = window_vp[window_valid_mask, 1]

            try:
                # x座標
                tck_x = splrep(valid_indices, valid_x, s=smoothing_factor, k=k)
                window_indices = np.arange(w_end - w_start)
                smoothed_x = splev(window_indices, tck_x)

                # y座標
                tck_y = splrep(valid_indices, valid_y, s=smoothing_factor, k=k)
                smoothed_y = splev(window_indices, tck_y)

                window_smoothed = np.column_stack((smoothed_x, smoothed_y))

                logger.info(f"ウィンドウ{i}: スプライン補間成功")
            except Exception as e:
                logger.warning(
                    f"ウィンドウ{i}: スプライン補間失敗({e})、"
                    f"線形補間で代用"
                )
                window_smoothed = _linear_interpolate_window(
                    window_vp, window_valid_mask
                )
                logger.info(
                    f"ウィンドウ{i}: 線形補間完了 "
                    f"(NaN数: {np.sum(np.isnan(window_smoothed))})"
                )

        # NaNチェック
        n_nan = np.sum(np.isnan(window_smoothed))
        if n_nan > 0:
            logger.warning(
                f"ウィンドウ{i}: 補間結果にNaNが{n_nan}個含まれています "
                f"(total={window_smoothed.size})"
            )

        window_results.append({
            'start': w_start,
            'end': w_end,
            'data': window_smoothed
        })

    # ウィンドウ結果を結合（境界でブレンド）
    smoothed_points = np.zeros((n_frames, 2), dtype=np.float64)

    for i, result in enumerate(window_results):
        w_start = result['start']
        w_end = result['end']
        w_data = result['data']

        if i == 0:
            # 最初のウィンドウ：そのままコピー
            smoothed_points[w_start:w_end] = w_data
        else:
            # 2番目以降：前ウィンドウとブレンド
            prev_result = window_results[i - 1]
            prev_end = prev_result['end']

            # ブレンド範囲を計算
            blend_start = w_start
            blend_end = min(w_start + blend_width, w_end, prev_end)

            if blend_start < blend_end:
                # ブレンド領域
                blend_len = blend_end - blend_start

                # 前ウィンドウのデータ（グローバルインデックス）
                prev_data = prev_result['data']
                prev_blend_data = prev_data[blend_start - prev_result['start']:
                                            blend_end - prev_result['start']]

                # 現ウィンドウのデータ（ローカルインデックス）
                curr_blend_data = w_data[0:blend_end - blend_start]

                # 線形ブレンド重み
                alpha = np.linspace(0, 1, blend_len).reshape(-1, 1)

                # ブレンド実行
                blended = (1 - alpha) * prev_blend_data + alpha * curr_blend_data
                smoothed_points[blend_start:blend_end] = blended

                # ブレンド後の領域をコピー
                if blend_end < w_end:
                    offset = blend_end - w_start
                    smoothed_points[blend_end:w_end] = w_data[offset:]
            else:
                # ブレンド不要（ギャップがある場合）
                smoothed_points[w_start:w_end] = w_data

    logger.info(
        f"ウィンドウ分割スプライン補間完了: {n_windows}個のウィンドウで"
        f"{n_frames}フレームを処理"
    )

    return smoothed_points


def _linear_interpolate_window(
    window_vp: np.ndarray,
    valid_mask: np.ndarray
) -> np.ndarray:
    """ウィンドウ内の線形補間（スプライン失敗時の代替処理）

    Args:
        window_vp: ウィンドウ内の消失点座標 (M, 2)
        valid_mask: 有効値マスク (M,)

    Returns:
        interpolated: 線形補間後の座標 (M, 2)
    """
    m_frames = window_vp.shape[0]
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        # 有効点がない場合は全てNaN
        return np.full((m_frames, 2), np.nan)

    if len(valid_indices) == 1:
        # 有効点が1点のみ：その値で埋める
        return np.tile(window_vp[valid_indices[0]], (m_frames, 1))

    # 線形補間
    valid_x = window_vp[valid_mask, 0]
    valid_y = window_vp[valid_mask, 1]

    all_indices = np.arange(m_frames)
    interp_x = np.interp(all_indices, valid_indices, valid_x)
    interp_y = np.interp(all_indices, valid_indices, valid_y)

    return np.column_stack((interp_x, interp_y))


def constrain_angle(
    angle: float,
    max_abs_angle_deg: Optional[float],
    angle_name: str,
    frame_idx: int,
    logger: logging.Logger
) -> float:
    """角度を最大絶対値で制約する
    
    Args:
        angle: 制約対象の角度（ラジアン）
        max_abs_angle_deg: 最大絶対値（度）、Noneの場合は制約なし
        angle_name: 角度の名称（"yaw" or "pitch"）
        frame_idx: フレームインデックス（ログ用）
        logger: ロガーインスタンス
        
    Returns:
        制約後の角度（ラジアン）
        
    Note:
        - max_abs_angle_deg がNoneまたは0以下の場合、制約なし
        - 制約が適用された場合、DEBUGログを出力
    """
    if max_abs_angle_deg is None or max_abs_angle_deg <= 0:
        return angle
    
    angle_deg = np.degrees(angle)
    max_abs = max_abs_angle_deg
    
    if abs(angle_deg) > max_abs:
        constrained_angle_deg = np.sign(angle_deg) * max_abs
        constrained_angle_rad = np.radians(constrained_angle_deg)
        
        logger.debug(
            f"フレーム{frame_idx}: {angle_name}角制約適用 "
            f"({angle_deg:.2f}° → {constrained_angle_deg:.2f}°)"
        )
        
        return constrained_angle_rad
    
    return angle


def compute_pitch_yaw_ranges(
    smoothed_centroids: np.ndarray,
    estimator: CameraEstimator,
    roll_angles: Optional[np.ndarray] = None,
    range_margin: float = 0.05,
    vp_config: Optional['VanishingPointConfig'] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """スムージング後の消失点からピッチ角・ヨー角範囲を計算
    
    Args:
        smoothed_centroids: スムージング後の消失点座標 (N, 2) [x, y]
        estimator: カメラ推定器（estimate_camera_poseを使用）
        roll_angles: 各フレームのロール角（ラジアン）、Noneの場合は全て0
        range_margin: 範囲マージン（ラジアン、デフォルト±0.05 rad ≈ ±2.86°）
        vp_config: 消失点設定（角度制約用）、Noneの場合は制約なし
    
    Returns:
        dpitch_ranges: ピッチ角変化量範囲 (N, 2) [dpitch_min, dpitch_max]
        dyaw_ranges: ヨー角変化量範囲 (N, 2) [dyaw_min, dyaw_max]
    
    処理の流れ:
        1. 各フレームの消失点座標から現在のピッチ角・ヨー角を推定
        2. フレーム間の差分からdpitch, dyawを計算
        3. ±range_marginの範囲を設定
        4. 最初のフレームは基準となるため、範囲は広めに設定
    
    Note:
        フレームiのdpitch[i] = pitch[i] - pitch[i-1]
        最初のフレーム(i=0)のdpitch[0]は計算できないため、
        [-range_margin*2, range_margin*2]のような広い範囲を設定
    
    Example:
        >>> smoothed = np.array([[960, 540], [962, 538], [964, 536]])
        >>> estimator = CameraEstimator(config, transformer)
        >>> dp_ranges, dy_ranges = compute_pitch_yaw_ranges(smoothed, estimator)
        >>> print(dp_ranges.shape)  # (3, 2)
        >>> print(dp_ranges[1])     # [-0.01, 0.03]（例）
    """
    n_frames = smoothed_centroids.shape[0]
    
    if n_frames == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    # ロール角が指定されていない場合は全て0
    if roll_angles is None:
        roll_angles = np.zeros(n_frames, dtype=np.float64)
    
    logger = logging.getLogger(__name__)
    
    # 各フレームのピッチ角・ヨー角を推定
    pitch_angles = np.full(n_frames, np.nan, dtype=np.float64)
    yaw_angles = np.full(n_frames, np.nan, dtype=np.float64)
    
    for i in range(n_frames):
        cx, cy = smoothed_centroids[i]
        
        # NaNの場合はスキップ
        if np.isnan(cx) or np.isnan(cy):
            logger.warning(f"フレーム{i}: 消失点がNaN、角度推定スキップ")
            continue
        
        # 消失点から姿勢を推定（簡易版）
        # estimate_camera_poseはフレーム画像を必要とするため、
        # ここでは座標から直接角度を計算
        camera = estimator.transformer.camera
        
        # 画像座標からカメラ平面座標へ
        dx = cx - camera.cx
        dy = cy - camera.cy
        x_cam = dx
        y_cam = -dy  # カメラ座標系では上が+
        
        # ロール補正（画像を-roll回転）
        roll = roll_angles[i]
        c, s = np.cos(-roll), np.sin(-roll)
        x_u = c * x_cam + s * y_cam
        y_u = -s * x_cam + c * y_cam
        
        # 魚眼カメラの角度計算
        from src.coordinate_transform import FisheyeCamera, PinholeCamera
        
        if isinstance(camera, FisheyeCamera):
            r = np.hypot(x_u, y_u) + 1e-12
            theta = r / camera.f
            phi_fish = np.arctan2(y_u, x_u)
            vx = np.sin(theta) * np.cos(phi_fish)
            vy = np.sin(theta) * np.sin(phi_fish)
            vz = np.cos(theta)
        elif isinstance(camera, PinholeCamera):
            vx = x_u
            vy = y_u
            vz = camera.f
            norm = np.linalg.norm([vx, vy, vz]) + 1e-12
            vx, vy, vz = vx / norm, vy / norm, vz / norm
        else:
            logger.warning(f"未対応のカメラモデル: {type(camera)}")
            continue
        
        # ヨー（右+）、ピッチ（上+）を算出
        yaw = -np.arctan2(vx, vz)
        pitch = -np.arctan2(vy, np.hypot(vx, vz))
        
        
        # Task 12: 角度制約の適用
        # TASK-17: max_abs_*パラメータはestimator.configから取得
        if vp_config is not None:
            yaw = constrain_angle(
                yaw, estimator.config.max_abs_yaw_degrees, "yaw", i, logger
            )
            pitch = constrain_angle(
                pitch, estimator.config.max_abs_pitch_degrees, "pitch", i, logger
            )
        
        yaw_angles[i] = yaw
        pitch_angles[i] = pitch
        
        logger.debug(
            f"フレーム{i}: yaw={np.degrees(yaw):.2f}°, "
            f"pitch={np.degrees(pitch):.2f}°"
        )
    
    # フレーム間差分を計算
    dpitch_ranges = np.full((n_frames, 2), np.nan, dtype=np.float64)
    dyaw_ranges = np.full((n_frames, 2), np.nan, dtype=np.float64)
    
    for i in range(n_frames):
        if i == 0:
            # 最初のフレームは基準となるため、広めの範囲を設定
            dpitch_ranges[i] = [-range_margin * 3, range_margin * 3]
            dyaw_ranges[i] = [-range_margin * 3, range_margin * 3]
        else:
            # フレーム間差分
            if not np.isnan(pitch_angles[i]) and not np.isnan(pitch_angles[i-1]):
                dpitch = pitch_angles[i] - pitch_angles[i-1]
                dpitch_ranges[i] = [dpitch - range_margin, dpitch + range_margin]
            else:
                # どちらかがNaNの場合は広い範囲
                dpitch_ranges[i] = [-range_margin * 3, range_margin * 3]
            
            if not np.isnan(yaw_angles[i]) and not np.isnan(yaw_angles[i-1]):
                dyaw = yaw_angles[i] - yaw_angles[i-1]
                dyaw_ranges[i] = [dyaw - range_margin, dyaw + range_margin]
            else:
                dyaw_ranges[i] = [-range_margin * 3, range_margin * 3]
    
    logger.info(
        f"ピッチ角・ヨー角範囲計算完了: {n_frames}フレーム "
        f"(margin=±{np.degrees(range_margin):.2f}°)"
    )
    
    
    # Task 12: 角度制約の統計ログ
    # TASK-17: max_abs_*パラメータはestimator.configから取得
    if vp_config is not None:
        if estimator.config.max_abs_yaw_degrees is not None and estimator.config.max_abs_yaw_degrees > 0:
            yaw_constrained_count = np.sum(np.abs(np.degrees(yaw_angles)) >= estimator.config.max_abs_yaw_degrees * 0.99)
            logger.info(
                f"ヨー角制約: ±{estimator.config.max_abs_yaw_degrees:.1f}°、"
                f"制約適用フレーム数: {yaw_constrained_count}/{n_frames}"
            )

        if estimator.config.max_abs_pitch_degrees is not None and estimator.config.max_abs_pitch_degrees > 0:
            pitch_constrained_count = np.sum(np.abs(np.degrees(pitch_angles)) >= estimator.config.max_abs_pitch_degrees * 0.99)
            logger.info(
                f"ピッチ角制約: ±{estimator.config.max_abs_pitch_degrees:.1f}°、"
                f"制約適用フレーム数: {pitch_constrained_count}/{n_frames}"
            )
    
    return dpitch_ranges, dyaw_ranges


# ============================================================================
# フェーズ2タスク1.3: 探索範囲絞り込み統合機能
# ============================================================================

def compute_frame_constraints(
    video_path: 'Path',
    start_frame: int,
    end_frame: int,
    estimator: CameraEstimator,
    ocr_roi_ratio: Tuple[float, float, float, float],
    config: EstimationConfig,
    ocr_tesseract_config: str,
    ocr_preprocessing_enabled: bool,
    ocr_enabled: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_distance_increment_mm: float = 10.0
) -> Tuple[Dict[int, Dict[str, Tuple[float, float]]], Optional[np.ndarray], Optional[np.ndarray]]:

    """全フレームの探索範囲制約を一括計算（逐次読み込み版）

    Args:
        video_path: 動画ファイルパス
        start_frame: 開始フレーム番号
        end_frame: 終了フレーム番号
        estimator: カメラ推定器
        ocr_roi_ratio: OCR読み取りROI領域
        config: 推定設定
        ocr_tesseract_config: Tesseract OCR設定文字列
        ocr_preprocessing_enabled: OCR前処理の有効/無効
        progress_callback: フレーム単位の進捗コールバック関数（オプション）
            progress_callback(current_frame, total_frames)
            - current_frame: 処理済みフレーム数（1からtotal_framesまで）
            - total_frames: 総フレーム数
    Returns:
        Tuple[constraints_dict, z_positions, vanishing_points]:
            - constraints_dict: フレームインデックスごとの制約辞書
                {
                    0: {'dz': (5.0, 8.0), 'dyaw': (-0.01, 0.01), ...},
                    1: {'dz': (6.0, 9.0), 'dyaw': (-0.02, 0.02), ...},
                    ...
                }
            - z_positions: 各フレームのOCR距離推定値（z*）配列、shape (N,)
                OCR制約が無効の場合はNone
            - vanishing_points: 各フレームの消失点座標（スムージング済み）、shape (N, 2)
                消失点制約が無効の場合はNone

    処理の流れ:
        1. 消失点収集とスムージング（タスク1.1機能使用）
        2. OCR距離収集と高精度推定（タスク1.2機能使用）
        3. フレームごとのdpitch, dyaw, dz範囲を計算
        4. マージン適用して制約辞書を構築

    Note:
        この関数はmain.pyのPhase1処理で呼び出され、
        Phase2（カラーマップ生成）で各フレームの制約を参照する。

    Example:
        >>> frames = [frame1, frame2, frame3]
        >>> estimator = CameraEstimator(config, transformer)
        >>> ocr_roi = (0.1, 0.2, 0.7, 0.9)
        >>> constraints, z_pos = compute_frame_constraints(frames, estimator, ocr_roi, config)
        >>> print(constraints[0])  # {'dz': (5.0, 8.0), 'dyaw': (-0.01, 0.01), ...}
        >>> print(z_pos[0])  # 702.5 (OCR距離推定値 z*)
    """
    from src.ocr_utils import (
        collect_ocr_distances,
        compute_average_speed,
        estimate_high_precision_z_positions,  # 既存手法（バックアップ用に残す）
        estimate_positions_offset_moving_average,  # 新手法
        compute_z_ranges,
        OCRConfig as OCRConfigUtils
    )

    logger = logging.getLogger(__name__)
    n_frames = end_frame - start_frame  # 左閉右開区間 [start_frame, end_frame)

    # OCRConfig変換: config.py側の設定をocr_utils側のOCRConfigに反映
    _ocr_config = OCRConfigUtils(
        psm_mode=OCRConfigUtils._extract_psm_mode(ocr_tesseract_config),
        preprocessing_enabled=ocr_preprocessing_enabled,
    )

    # 消失点推定値の初期化（Phase2で使用）
    vanishing_points_output = None

    # ステップ1: 消失点スムージング
    if config.use_vanishing_point_constraints:
        logger.info("消失点収集とスムージング開始...")

        # 適応的閾値選択パラメータを取得
        use_adaptive = config.pose.use_adaptive_threshold
        target_area = config.pose.target_dark_area if use_adaptive else None
        threshold_range = config.pose.threshold_search_range if use_adaptive else (1, 100)
        dark_threshold = config.pose.dark_threshold

        # 暗部検出の半径制限を計算（外周のブラックエリアを除外）
        # レガシープログラムの max_radius_px = radius - crip_margin_px と同等
        camera = estimator.transformer.camera
        radius_limit = min(camera.image_height, camera.image_width) // 2 * config.pose.dark_region_radius_limit_ratio

        logger.info(
            f"暗部検出パラメータ: use_adaptive={use_adaptive}, "
            f"target_area={target_area}, radius_limit={radius_limit:.1f}px"
        )

        # BUG-006修正: enable_feature_vp_in_phase1 → enable_feature_based_vp に変更
        # 消失点とOCR距離を収集（新旧メソッド切り替え）
        if config.vanishing_point.enable_feature_based_vp:
            logger.info("Phase1: 特徴点ベースVP推定を使用（enable_feature_based_vp=true）")
            try:
                vp_result = estimator.collect_vanishing_points_with_features(
                    video_path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    ocr_roi_ratio=ocr_roi_ratio,
                    ocr_tesseract_config=ocr_tesseract_config,
                    ocr_preprocessing_enabled=ocr_preprocessing_enabled,
                    max_distance_increment_mm=max_distance_increment_mm
                )
                
                # 返り値から必要なデータを取得
                vp = vp_result['smoothed_vp']
                # OCR距離はvp_dataから抽出（既存のフォーマットに合わせる）
                ocr_dist = np.array([frame_data.get('ocr_distance_mm', 0.0) 
                                     for frame_data in vp_result['vp_data']])
                success = np.array([frame_data.get('ocr_success', False) 
                                   for frame_data in vp_result['vp_data']])
                
                # 成功率をログ出力
                feature_success = vp_result.get('feature_vp_success_count', 0)
                feature_total = vp_result.get('feature_vp_total_count', 0)
                feature_rate = (feature_success / max(feature_total, 1)) * 100
                logger.info(
                    f"特徴点ベースVP推定成功率: "
                    f"{feature_success}/{feature_total} ({feature_rate:.1f}%)"
                )
                
            except Exception as e:
                logger.error(f"特徴点ベースVP推定失敗: {e}", exc_info=True)
                logger.warning("従来方式にフォールバックします")
                # フォールバック: 従来方式を使用
                vp, ocr_dist, success = collect_vanishing_points_and_ocr_distances(
                    video_path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    estimator=estimator,
                    ocr_roi_ratio=ocr_roi_ratio,
                    config=config,
                    read_ocr=ocr_enabled,
                    progress_callback=progress_callback,
                    ocr_config=_ocr_config,
                    max_distance_increment_mm=max_distance_increment_mm
                )
        else:
            logger.info("Phase1: 暗部中心法による消失点推定を使用（enable_feature_based_vp=false）")
            vp, ocr_dist, success = collect_vanishing_points_and_ocr_distances(
                video_path=video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                estimator=estimator,
                ocr_roi_ratio=ocr_roi_ratio,
                config=config,
                read_ocr=ocr_enabled,
                progress_callback=progress_callback,
                ocr_config=_ocr_config,
                max_distance_increment_mm=max_distance_increment_mm
            )
        # 異常値除去（分布ベース + 変動量ベース + ベクトル判定）

        # BUG-012修正: 変動量ベースの異常判定（ベクトル全体で判定）
        vp_magnitude_threshold = getattr(
            config.vanishing_point,
            'vp_magnitude_threshold',
            50.0
        )
        mask_magnitude = detect_outliers_by_magnitude(
            vp,
            threshold=vp_magnitude_threshold,
            logger=logger
        )

        # 🆕 BUG-013修正: 局所統計ベース異常判定（処理範囲に依存しない）
        outlier_window_size = getattr(
            config.vanishing_point,
            'vanishing_point_outlier_window_size',
            50
        )

        # 局所IQR法と局所3σ法による異常判定
        mask_iqr = detect_outliers_local_iqr(
            vp,
            window_size=outlier_window_size,
            logger=logger
        )

        mask_3sigma = detect_outliers_local_3sigma(
            vp,
            window_size=outlier_window_size,
            logger=logger
        )

        # 両方の判定で正常と判定された点のみ正常とする
        mask_distribution = mask_iqr & mask_3sigma

        # 分布ベースと変動量ベースを統合（論理積）
        mask = mask_distribution & mask_magnitude

        # 統計情報
        n_outliers_dist = np.sum(~mask_distribution)
        n_outliers_mag = np.sum(~mask_magnitude)
        n_outliers_total = np.sum(~mask)

        logger.info(
            f"消失点異常判定統計:\n"
            f"  - 全フレーム数: {n_frames}\n"
            f"  - 異常判定（分布ベース）: {n_outliers_dist}フレーム（{n_outliers_dist/n_frames*100:.2f}%）\n"
            f"  - 異常判定（変動量ベース）: {n_outliers_mag}フレーム（{n_outliers_mag/n_frames*100:.2f}%）\n"
            f"  - 異常判定（統合）: {n_outliers_total}フレーム（{n_outliers_total/n_frames*100:.2f}%）"
        )

        try:
            # スムージング手法の選択
            if config.vanishing_point_smoothing_method == "savgol":
                # Savitzky-Golay平滑化（推奨）
                smoothed_vp = smooth_vanishing_points_savgol(
                    vp, mask,
                    window_length=config.savgol_window_length,
                    polyorder=config.savgol_polyorder
                )
            else:
                # スプライン補間（旧方式）
                smoothed_vp = smooth_vanishing_points_spline(
                    vp, mask,
                    smoothing_factor=config.vanishing_point_smoothing_factor,
                    k=config.vanishing_point_spline_order,
                    window_size=config.vanishing_point_window_size,
                    use_moving_average_preprocess=config.use_moving_average_preprocess,
                    moving_average_window=config.moving_average_window
                )

            dpitch_ranges, dyaw_ranges = compute_pitch_yaw_ranges(
                smoothed_vp, estimator,
                range_margin=config.pitch_yaw_range_margin,
                vp_config=config.vanishing_point
            )
            vanishing_points_output = smoothed_vp  # Phase2に渡す
            logger.info(f"消失点スムージング完了: method={config.vanishing_point_smoothing_method}")
        except ValueError as e:
            logger.warning(f"消失点スムージング失敗: {e}、広い範囲を使用")
            dpitch_ranges = np.full((n_frames, 2), [-0.1, 0.1])
            dyaw_ranges = np.full((n_frames, 2), [-0.1, 0.1])
    else:
        # 制約なし: 広い範囲を設定
        logger.info("消失点制約は無効化されています")
        dpitch_ranges = np.full((n_frames, 2), [-0.1, 0.1])
        dyaw_ranges = np.full((n_frames, 2), [-0.1, 0.1])
        
        # Bug 6修正に伴う追加処理: OCRデータ収集
        # ocr.enabled=true の場合、消失点制約なしでもOCRデータを収集
        if ocr_enabled:
            logger.info("OCRデータのみ収集します（消失点制約は使用しません）")
            # OCR距離のみを収集
            _, ocr_dist, success = collect_vanishing_points_and_ocr_distances(
                video_path=video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                estimator=estimator,
                ocr_roi_ratio=ocr_roi_ratio,
                config=config,
                read_ocr=True,  # OCR読み取りを有効化
                progress_callback=progress_callback,
                ocr_config=_ocr_config,
                max_distance_increment_mm=max_distance_increment_mm
            )
            vp = None  # 消失点は使用しない
        else:
            # OCR無効時の変数初期化
            vp = None
            ocr_dist = None
            success = None

    
    # ステップ2: OCR高精度Z位置推定
    z_positions_output = None  # Phase2/Phase3に返すz*配列
    # Bug 6修正: use_ocr_z_constraints に関わらずOCR高精度推定を実行
    # ocr.enabled=true の場合、Phase2での使用有無に関わらず z_positions を生成
    # Phase3補正のために常に推定値が必要（2025-10-27修正）
    if ocr_enabled:
        logger.info("OCR距離収集と高精度推定開始...")

        # ocr_distがNoneの場合はエラー
        if ocr_dist is None or success is None:
            logger.error(
                "ocr_enabled=True だが OCR読み取り結果がありません。"
                "collect_vanishing_points_*() の read_ocr パラメータを確認してください。"
            )
            raise ValueError("OCR読み取り結果が存在しません")
        
        try:
            # ocr_dist, successは既に統合関数で取得済み

            # OCR読取り値を0基準化（レガシー仕様への回帰）
            # Issue 2 & 3 修正: Phase2とPhase3の座標系を統一
            ocr_offset = None
            ocr_dist_normalized = ocr_dist.copy()  # 元データを保持

            success_indices = np.where(success)[0]
            if len(success_indices) > 0:
                # 最初の成功OCR読取り値をオフセットとする
                ocr_offset = ocr_dist[success_indices[0]]
                ocr_dist_normalized = ocr_dist - ocr_offset

                logger.info(
                    f"OCR読取り値を0基準化（レガシー仕様）: "
                    f"offset={ocr_offset:.1f}mm, "
                    f"range=[{ocr_dist_normalized[success_indices[0]]:.1f}, "
                    f"{ocr_dist_normalized[success_indices[-1]]:.1f}]mm"
                )
            else:
                logger.warning("OCR成功フレームが0件のため、0基準化をスキップ")

            # 設定に応じて推定手法を切り替え
            if config.use_offset_moving_average:
                logger.info("オフセット移動平均距離推定法を使用")
                z_positions = estimate_positions_offset_moving_average(
                    ocr_dist_normalized,
                    success,
                    window_size=config.offset_window_size,
                    overlap=config.offset_overlap,
                    e_resolution=config.offset_e_resolution
                )
            else:
                logger.info("平均速度ベース距離推定法を使用")
                avg_speed = compute_average_speed(ocr_dist_normalized, success)
                z_positions = estimate_high_precision_z_positions(ocr_dist_normalized, success, avg_speed)

            # BUG-004修正: use_ocr_z_constraintsがtrueの場合のみOCR制約を適用
            if config.use_ocr_z_constraints:
                # OCR推定値からdz範囲を計算
                dz_ranges = compute_z_ranges(z_positions, tolerance=config.z_range_tolerance)
                logger.info("Phase1: OCR制約に基づくdz範囲を計算（use_ocr_z_constraints=true）")
            else:
                # OCR制約なし: 広い範囲を設定
                max_dz = estimator.config.motion.max_dz
                dz_ranges = np.full((n_frames, 2), [0.0, max_dz])
                logger.info(f"Phase1: 広いdz範囲を設定（use_ocr_z_constraints=false）: [0, {max_dz}]mm")
            z_positions_output = z_positions  # Phase2に渡す
            logger.info("OCR高精度推定完了（0基準）")

            # Phase1完了後: OCR距離推定値とdz範囲を出力
            logger.info("=" * 80)
            logger.info("Phase1完了: OCR距離推定値（z*）とdz範囲")
            logger.info("=" * 80)
            for i in range(min(10, n_frames)):  # 最初の10フレームを出力
                if success[i]:
                    if config.use_ocr_z_constraints:
                        # OCR制約有効時: 制約範囲を明示
                        logger.info(
                            f"フレーム{i}: OCR={ocr_dist[i]:.1f}mm, "
                            f"z*={z_positions[i]:.2f}mm, "
                            f"dz=[{dz_ranges[i][0]:.2f}, {dz_ranges[i][1]:.2f}]mm（OCR制約）"
                        )
                    else:
                        # OCR制約無効時: 参考値として表示
                        logger.info(
                            f"フレーム{i}: OCR={ocr_dist[i]:.1f}mm（参考値）, "
                            f"z*={z_positions[i]:.2f}mm（Phase3用）, "
                            f"dz=[{dz_ranges[i][0]:.2f}, {dz_ranges[i][1]:.2f}]mm（広い範囲）"
                        )
                else:
                    if config.use_ocr_z_constraints:
                        logger.info(
                            f"フレーム{i}: OCR=失敗, "
                            f"z*={z_positions[i]:.2f}mm, "
                            f"dz=[{dz_ranges[i][0]:.2f}, {dz_ranges[i][1]:.2f}]mm（OCR制約）"
                        )
                    else:
                        logger.info(
                            f"フレーム{i}: OCR=失敗, "
                            f"z*={z_positions[i]:.2f}mm（Phase3用）, "
                            f"dz=[{dz_ranges[i][0]:.2f}, {dz_ranges[i][1]:.2f}]mm（広い範囲）"
                        )
            if n_frames > 10:
                logger.info("...")
                for i in range(max(10, n_frames - 3), n_frames):  # 最後の3フレームを出力
                    if success[i]:
                        if config.use_ocr_z_constraints:
                            logger.info(
                                f"フレーム{i}: OCR={ocr_dist[i]:.1f}mm, "
                                f"z*={z_positions[i]:.2f}mm, "
                                f"dz=[{dz_ranges[i][0]:.2f}, {dz_ranges[i][1]:.2f}]mm（OCR制約）"
                            )
                        else:
                            logger.info(
                                f"フレーム{i}: OCR={ocr_dist[i]:.1f}mm（参考値）, "
                                f"z*={z_positions[i]:.2f}mm（Phase3用）, "
                                f"dz=[{dz_ranges[i][0]:.2f}, {dz_ranges[i][1]:.2f}]mm（広い範囲）"
                            )
                    else:
                        if config.use_ocr_z_constraints:
                            logger.info(
                                f"フレーム{i}: OCR=失敗, "
                                f"z*={z_positions[i]:.2f}mm, "
                                f"dz=[{dz_ranges[i][0]:.2f}, {dz_ranges[i][1]:.2f}]mm（OCR制約）"
                            )
                        else:
                            logger.info(
                                f"フレーム{i}: OCR=失敗, "
                                f"z*={z_positions[i]:.2f}mm（Phase3用）, "
                                f"dz=[{dz_ranges[i][0]:.2f}, {dz_ranges[i][1]:.2f}]mm（広い範囲）"
                            )
            logger.info("=" * 80)
        except Exception as e:
            logger.warning(f"OCR高精度推定失敗: {e}、広い範囲を使用")
            max_dz = estimator.config.motion.max_dz
            dz_ranges = np.full((n_frames, 2), [0.0, max_dz])
    else:
        # OCR無効: 既存のmax_dzを使用
        logger.info("OCRが無効のため、OCR距離推定をスキップします")
        max_dz = estimator.config.motion.max_dz
        dz_ranges = np.full((n_frames, 2), [0.0, max_dz])

    # Phase2でのOCR制約使用状況をログ出力
    if config.use_ocr_z_constraints:
        if z_positions_output is not None:
            logger.info("Phase2でOCR Z位置制約を使用します（use_ocr_z_constraints=true）")
        else:
            logger.warning("use_ocr_z_constraints=true ですが、OCRデータが存在しません")
    else:
        if z_positions_output is not None:
            logger.info("Phase2ではOCR Z位置制約を使用しません（use_ocr_z_constraints=false）")
            logger.info("OCRデータはPhase3展開画像補正で使用されます")
        else:
            logger.info("OCR Z位置制約は無効化されています")

    # ステップ3: 制約辞書構築
    # x, yの累積値を追跡（初期値は0）
    x_accumulated = 0.0
    y_accumulated = 0.0
    max_x_limit = 1.0  # x, yの累積値制限 ±1.0mm
    max_y_limit = 1.0
    max_dx_step = 0.5  # dx, dyの1フレーム当たりの変動幅 ±0.5mm
    max_dy_step = 0.5

    constraints_dict = {}
    for i in range(n_frames):
        # dx, dyの制約：変動幅±0.5mm と 累積値制限±1.0mm の両方を満たす範囲
        dx_lower = max(-max_dx_step, -max_x_limit - x_accumulated)
        dx_upper = min(max_dx_step, max_x_limit - x_accumulated)
        dy_lower = max(-max_dy_step, -max_y_limit - y_accumulated)
        dy_upper = min(max_dy_step, max_y_limit - y_accumulated)

        constraints_dict[i] = {
            'dx': (dx_lower, dx_upper),
            'dy': (dy_lower, dy_upper),
            'dz': tuple(dz_ranges[i]),
            'droll': (-np.radians(estimator.config.motion.max_droll),
                      np.radians(estimator.config.motion.max_droll)),
            'dyaw': tuple(dyaw_ranges[i]),
            'dpitch': tuple(dpitch_ranges[i]),
        }

        # 次のフレームのために、中央値で累積値を更新（実際の値は実行時に決まる）
        # ここでは0と仮定（最小二乗法の初期値が0なので）
        x_accumulated += 0.0
        y_accumulated += 0.0

    logger.info(f"全{n_frames}フレームの探索範囲制約計算完了")
    logger.info(f"dx, dy制約: 変動幅±{max_dx_step}mm, 累積値制限±{max_x_limit}mm")

    return constraints_dict, z_positions_output, vanishing_points_output


def compute_distance_constraints_only(
    video_path: Optional['Path'] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    estimator: Optional[CameraEstimator] = None,
    ocr_roi_ratio: Tuple[float, float, float, float] = (0.0, 0.05, 0.0, 0.2),
    config: Optional[EstimationConfig] = None,
    ocr_tesseract_config: str = "--psm 7 -c tessedit_char_whitelist=0123456789.m",
    ocr_preprocessing_enabled: bool = False,
    frames: Optional[List[np.ndarray]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_distance_increment_mm: float = 10.0,
    known_z_mm: Optional[np.ndarray] = None,
) -> Tuple[Dict[int, Dict[str, Tuple[float, float]]], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """OCR-only の距離制約。暗部・輝度・VP 系列は呼び出さない。

    known_z_mm を渡すと Tesseract を呼ばず、その値を OCR 読み取り結果として使う
    （第1段階: 仮想動画の生成距離を代用する）。

    Returns:
        constraints_dict, z_positions, ocr_dist, success
    """
    from src.ocr_utils import (
        extract_distance_from_frame,
        OCRTextNotFoundError,
        OCRConfig as OCRConfigUtils,
        compute_average_speed,
        estimate_high_precision_z_positions,
        estimate_positions_offset_moving_average,
        compute_z_ranges,
    )

    logger = logging.getLogger(__name__)
    if config is None:
        if estimator is None:
            raise ValueError("estimator または config が必要です")
        config = estimator.config

    _ocr_config = OCRConfigUtils(
        psm_mode=OCRConfigUtils._extract_psm_mode(ocr_tesseract_config),
        preprocessing_enabled=ocr_preprocessing_enabled,
    )

    loaded: List[np.ndarray] = []
    ocr_from_video = False
    if known_z_mm is not None:
        known = np.asarray(known_z_mm, dtype=float).reshape(-1)
        if frames is not None and len(frames) > 0 and known.size != len(frames):
            raise ValueError(
                f"known_z_mm の長さ ({known.size}) がフレーム数 ({len(frames)}) と一致しません"
            )
        n_frames = int(known.size)
        ocr_dist = known.copy()
        success = np.isfinite(ocr_dist)
        logger.info(
            f"既知zをOCR代用: {n_frames}フレーム, success={int(np.sum(success))}"
        )
        if progress_callback:
            progress_callback(n_frames, n_frames)
    elif frames is not None:
        loaded = frames
        n_frames = len(loaded)
        ocr_dist = np.full(n_frames, np.nan, dtype=float)
        success = np.zeros(n_frames, dtype=bool)
        for i, frame in enumerate(loaded):
            try:
                distance, _ = extract_distance_from_frame(
                    frame, ocr_roi_ratio, _ocr_config
                )
                ocr_dist[i] = distance
                success[i] = True
            except OCRTextNotFoundError:
                pass
            except Exception as exc:
                logger.debug(f"OCR-only frame {i} failed: {exc}")
            if progress_callback:
                progress_callback(i + 1, n_frames)
    elif video_path is not None:
        ocr_from_video = True
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"動画を開けません: {video_path}")
        ocr_vals: List[float] = []
        ok_flags: List[bool] = []
        try:
            idx = 0
            start = int(start_frame or 0)
            while idx < start:
                if not cap.grab():
                    break
                idx += 1
            while end_frame is None or idx < int(end_frame):
                ok, frame = cap.read()
                if not ok:
                    break
                dist = np.nan
                succeeded = False
                try:
                    dist, _ = extract_distance_from_frame(
                        frame, ocr_roi_ratio, _ocr_config
                    )
                    succeeded = True
                except OCRTextNotFoundError:
                    pass
                except Exception as exc:
                    logger.debug(f"OCR-only frame {idx} failed: {exc}")
                ocr_vals.append(dist)
                ok_flags.append(succeeded)
                idx += 1
                if progress_callback:
                    progress_callback(len(ocr_vals), len(ocr_vals))
        finally:
            cap.release()
        n_frames = len(ocr_vals)
        ocr_dist = np.asarray(ocr_vals, dtype=float)
        success = np.asarray(ok_flags, dtype=bool)
    else:
        raise ValueError("video_path または frames が必要です")

    z_positions = None
    if n_frames == 0:
        return {}, None, ocr_dist, success

    ocr_dist_normalized = ocr_dist.copy()
    success_indices = np.where(success)[0]
    if len(success_indices) > 0:
        ocr_offset = ocr_dist[success_indices[0]]
        ocr_dist_normalized = ocr_dist - ocr_offset

    try:
        if config.use_offset_moving_average:
            z_positions = estimate_positions_offset_moving_average(
                ocr_dist_normalized, success,
                window_size=config.offset_window_size,
                overlap=config.offset_overlap,
                e_resolution=config.offset_e_resolution,
            )
        else:
            avg_speed = compute_average_speed(ocr_dist_normalized, success)
            z_positions = estimate_high_precision_z_positions(
                ocr_dist_normalized, success, avg_speed
            )
    except Exception as exc:
        logger.warning(f"OCR高精度推定失敗: {exc}")
        z_positions = np.nan_to_num(ocr_dist_normalized, nan=0.0)

    max_dz = estimator.config.motion.max_dz if estimator is not None else 10.0
    if config.use_ocr_z_constraints and z_positions is not None:
        dz_ranges = compute_z_ranges(z_positions, tolerance=config.z_range_tolerance)
    else:
        dz_ranges = np.full((n_frames, 2), [0.0, max_dz])

    constraints_dict = {}
    for i in range(n_frames):
        constraints_dict[i] = {
            "dx": (-0.5, 0.5),
            "dy": (-0.5, 0.5),
            "dz": tuple(dz_ranges[i]),
            "droll": (-np.radians(1.0), np.radians(1.0)),
            "dyaw": (-np.radians(2.0), np.radians(2.0)),
            "dpitch": (-np.radians(2.0), np.radians(2.0)),
        }
    logger.info(f"OCR-only制約: {n_frames}フレーム, success={int(np.sum(success))}")
    return constraints_dict, z_positions, ocr_dist, success


