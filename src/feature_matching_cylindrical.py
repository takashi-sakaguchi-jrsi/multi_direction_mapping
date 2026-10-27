"""円筒座標系特徴点マッチングモジュール

管内カメラカーシミュレーション用の円筒座標系に基づく特徴点マッチング機能を提供します。
通常のフレーム座標系ではなく、円筒座標系に変換してから特徴点マッチングを行うことで、
カメラの姿勢変化に対してロバストなマッチングを実現します。

主な機能:
- フレーム画像を円筒座標系に変換
- 前処理(平均除去、ハイパスフィルタ、ハニング窓)
- 全画像でORB特徴点検出
- マッチング後に空間分散フィルタ適用(円筒座標グリッド)
- RANSACによるアフィン変換推定
- 円筒座標からフレーム座標への逆変換

座標系の定義:
- 円筒座標系: theta(円周角、0〜360度) × z(管路延長方向、mm)
- theta=0: 画像上部、時計回りに増加
- z: カメラ進行方向が正
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

import cv2
import numpy as np

from src.coordinate_transform import CoordinateTransformer, R_w2c


# ============================================================================
# ロガー設定
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# 設定データクラス
# ============================================================================

@dataclass
class CylindricalMatchingConfig:
    """円筒座標系マッチング設定
    
    Attributes:
        # 円筒座標変換パラメータ
        theta_resolution: theta方向の解像度(ピクセル)
        z_resolution: z方向の解像度(ピクセル)
        inner_radius_ratio: 内側半径比率(0〜1、円筒変換対象範囲)
        outer_radius_ratio: 外側半径比率(0〜1、円筒変換対象範囲)
        z_range_mm: z方向の範囲(mm)
        
        # 前処理パラメータ
        enable_mean_removal: 平均除去を有効化
        enable_highpass_filter: ハイパスフィルタを有効化
        highpass_kernel_size: ハイパスフィルタのカーネルサイズ
        highpass_sigma: ハイパスフィルタのガウス標準偏差
        enable_hanning_window: ハニング窓を有効化
        hanning_window_ratio: ハニング窓の縁の割合(0〜0.5)
        
        # 特徴点抽出パラメータ(新方式)
        orb_max_features: ORB検出の最大特徴点数(マッチング前)
        match_max_features: マッチング後の最大特徴点数
        spatial_filter_enabled: 空間分散フィルタの有効化
        spatial_grid_rows: 空間分散フィルタのグリッド行数(θ方向)
        spatial_grid_cols: 空間分散フィルタのグリッド列数(z方向)
        
        # マッチングパラメータ
        match_method: マッチング手法('BF' or 'FLANN')
        match_cross_check: クロスチェックを有効化
        
        # RANSACパラメータ
        ransac_iterations: RANSAC反復回数
        ransac_min_samples: RANSAC最小サンプル数
        ransac_inlier_threshold: RANSACインライア閾値(ピクセル)
        min_inlier_ratio: 最小インライア比率(0〜1)
        min_inlier_count: 最小インライア数
        
        # Task 4.2.9: 静止状態フレームスキップパラメータ
        static_frame_detection_enabled: 静止状態フレームスキップの有効/無効
            True (デフォルト): 静止状態フレームを検出してスキップ
            False: 全フレームを処理
            
            静止状態フレームとは、フレーム間でカメラがほとんど移動していない状態を指します。
            特徴点座標がほぼ一致するため、移動ベクトルが微小(< 2.0px)となり、
            座標誤差の影響が相対的に巨大化し、ベクトル方向が不安定(ランダム方向を向く)になります。
            その結果、レンズ中心点距離フィルタで全点除外され、除外率99%超(残存点5-9点のみ)となります。
            
            この機能を有効化することで、不安定なフレームペアを事前に除外し、
            除外率を89%から70%程度まで改善します。
        
        static_frame_movement_threshold: 静止状態判定の移動量閾値(ピクセル)
            デフォルト: 2.0ピクセル

            全特徴点の平均移動量がこの閾値以下の場合、静止状態と判定します。

            デフォルト値の根拠:
                - Phase 2実験より、2.0px以下の移動では方向推定が不安定
                - 姿勢推定に必要な最小移動量
                - 経験的に、座標誤差(±0.5px程度)の影響が顕著になる境界
            
            調整指針:
                - 1.0px以下: 非常に厳格、多くのフレームがスキップされる可能性
                - 2.0px (推奨): バランス良好
                - 3.0px以上: 緩い、不安定なフレームも処理される可能性
        
        static_frame_min_points: 静止状態判定に必要な最小特徴点数
            デフォルト: 10点
            
            マッチング点数がこの値未満の場合、移動量に関わらず
            特徴点不足として処理をスキップします。
            
            デフォルト値の根拠:
                - 最小二乗法による姿勢推定には最低10点程度必要
                - RANSACのmin_inlier_countと整合
            
            調整指針:
                - 5点以下: 推定精度が著しく低下するリスク
                - 10点 (推奨): 安定した推定が可能
                - 20点以上: 厳格、スキップされるフレームが増加
        
        z_axis: 円筒画像の行方向。tan=R·tan(ζ)（旧）、physical=管軸距離 Z
        physical_vector_filter_enabled: ΔZ/Δθ の側視型 IQR。physical 軸で使う
        physical_iqr_multiplier: ΔZ/Δθ IQR 倍率
        physical_outlier_min_points: IQR を掛ける最小点数
        physical_max_dz_mm: 点の |ΔZ| 上限、かつ推定 dz の上限（既定 20 mm）
        physical_max_dtheta_deg: 点の |Δθ| 上限（既定 5°）。解析クリップではない
        reject_reverse_travel: 推定 dz が後退なら失敗（点は捨てない）
        reverse_dz_tolerance_mm: 後退判定の許容 (mm)
        physical_max_dyaw_deg: マッチ 1 次から取る dyaw のフレーム間上限（超過は推定失敗）
        physical_max_dpitch_deg: マッチ 1 次から取る dpitch のフレーム間上限
        physical_min_points_for_attitude: これ未満で大きな Δyaw/Δpitch は積まない（dz は積む）
        physical_untrusted_attitude_deg: 低点数のとき姿勢を捨てる閾値（度）
        physical_max_applied_attitude_deg: 点数に関係なく姿勢を捨てる上限（度）。微小振動は残す
        physical_attitude_source: 展開に積む yaw/pitch。match=3DoF、
            discard=3DoF推定は残し適用は消失点、dz_only=推定もdzのみ
        exclude_subpixel_matches: 円筒上で変位が閾値未満の点を落とすか。
            既定 False。True だと停止点（正しい微小移動）が落ち、継ぎ目誤マッチが多数派になり得る。
        min_pixel_displacement: exclude_subpixel_matches=True のときの最小変位（px）

        # 位相相関パラメータ
        phase_correlation_enabled: 位相相関によるdz/droll推定の有効/無効
            True: 円筒座標画像の位相相関でdz/drollを推定し、特徴点最適化を2DOFに縮小
            False (デフォルト): 従来の4DOF/6DOF特徴点最適化を使用
        
        phase_correlation_response_threshold: 位相相関の信頼度閾値
            デフォルト: 0.05
            この値以下の場合、位相相関結果を破棄し従来の4DOF最適化にフォールバック
        
        phase_correlation_gaussian_blur_size: 位相相関前処理のガウシアンブラーカーネルサイズ
            デフォルト: 5
            0に設定するとガウシアンブラーを無効化
    """
    
    # 円筒座標変換パラメータ
    theta_resolution: int = 720
    z_resolution: int = 500
    inner_radius_ratio: float = 0.3
    outer_radius_ratio: float = 0.815
    z_range_mm: float = 1000.0
    
    # 前処理パラメータ
    enable_mean_removal: bool = False
    enable_highpass_filter: bool = False
    highpass_kernel_size: int = 5
    highpass_sigma: float = 1.0
    enable_hanning_window: bool = False
    hanning_window_ratio: float = 0.1
    
    # 特徴点抽出パラメータ(新方式)
    orb_max_features: int = 2000
    match_max_features: int = 200  # Task 4.2.4C: フィルタリング強化のため増強
    spatial_filter_enabled: bool = True
    spatial_grid_rows: int = 8
    spatial_grid_cols: int = 8
    
    # マッチングパラメータ
    match_method: str = 'BF'
    match_cross_check: bool = True
    
    # RANSACパラメータ
    ransac_iterations: int = 1000
    ransac_min_samples: int = 3
    ransac_inlier_threshold: float = 3.0
    min_inlier_ratio: float = 0.3
    min_inlier_count: int = 10
    
    # Task 4.2.9: 静止状態フレームスキップパラメータ
    static_frame_detection_enabled: bool = True
    static_frame_movement_threshold: float = 2.0
    static_frame_min_points: int = 10

    z_axis: str = "tan"
    physical_vector_filter_enabled: bool = False
    physical_iqr_multiplier: float = 1.5
    physical_outlier_min_points: int = 4
    physical_max_dz_mm: float = 20.0
    physical_max_dtheta_deg: float = 5.0
    reject_reverse_travel: bool = True
    reverse_dz_tolerance_mm: float = 0.5
    physical_max_dyaw_deg: float = 2.0
    physical_max_dpitch_deg: float = 2.0
    physical_min_points_for_attitude: int = 20
    physical_untrusted_attitude_deg: float = 0.8
    physical_max_applied_attitude_deg: float = 1.2
    physical_attitude_source: str = "match"
    exclude_subpixel_matches: bool = False
    min_pixel_displacement: float = 0.5

    # 位相相関パラメータ
    phase_correlation_enabled: bool = False
    phase_correlation_response_threshold: float = 0.05
    phase_correlation_gaussian_blur_size: int = 5

# ============================================================================
# CylindricalFeatureMatcher クラス
# ============================================================================

class CylindricalFeatureMatcher:
    """円筒座標系特徴点マッチャー
    
    フレーム画像を円筒座標系に変換してから特徴点マッチングを行い、
    カメラの移動量と姿勢変化量を推定します。
    
    新方式:
        1. 円筒座標変換 + 前処理
        2. 全画像でORB検出
        3. マッチング
        4. 空間分散フィルタ(円筒座標グリッド)
        5. フレーム座標に逆変換
    
    Attributes:
        transformer: 座標変換器
        config: マッチング設定
        logger: ロガー
    """
    
    def __init__(
        self,
        transformer: CoordinateTransformer,
        config: CylindricalMatchingConfig,
        feature_filtering_config: Optional[Any] = None,
        static_frame_detection_config: Optional[Any] = None,
        pixels_per_mm: float = 1.0
    ):
        """初期化

        Args:
            transformer: 座標変換器
            config: マッチング設定
            feature_filtering_config: 特徴点フィルタリング設定
            static_frame_detection_config: 停止フレーム検出設定 (REFACTOR-002で追加)
            pixels_per_mm: カラーマップの解像度（ピクセル/mm）、デフォルト1.0
                第1段階停止判定でピクセル→mm変換に使用 (TASK-4.2.11d Phase 2)
        """
        self.transformer = transformer
        self.config = config
        self.logger = logger
        self.feature_filtering_config = feature_filtering_config
        # R不変な円筒座標画像のため、z_rangeをR/R_defaultでスケーリング
        # _create_cylindrical_grid呼び出し時にcamera_paramsから計算・キャッシュ
        self._effective_z_range_mm = config.z_range_mm
        self._z_near_mm = 50.0
        self._z_far_mm = 500.0

        # TASK-4.2.11d Phase 2: pixels_per_mmを保存（停止判定のピクセル→mm変換用）
        self.pixels_per_mm = pixels_per_mm

        # REFACTOR-002: 停止フレーム検出設定を優先的に使用
        # 後方互換性のため、config.static_frame_*もサポート
        if static_frame_detection_config is not None:
            self.static_frame_detection_config = static_frame_detection_config
        else:
            # 旧方式（config内のパラメータ）をラップ
            from dataclasses import dataclass
            @dataclass
            class LegacyStaticFrameConfig:
                enabled: bool
                movement_threshold: float
                min_points: int

            self.static_frame_detection_config = LegacyStaticFrameConfig(
                enabled=config.static_frame_detection_enabled,
                movement_threshold=config.static_frame_movement_threshold,
                min_points=config.static_frame_min_points
            )
        
        # マッチャーの初期化
        if config.match_method == 'BF':
            self.matcher = cv2.BFMatcher(
                cv2.NORM_HAMMING,
                crossCheck=config.match_cross_check
            )
        elif config.match_method == 'FLANN':
            # FLANNパラメータ(ORBはバイナリ記述子なのでLSH使用)
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Invalid match_method: {config.match_method}")
        
        logger.info(
            f"CylindricalFeatureMatcher initialized(新方式): "
            f"method={config.match_method}, "
            f"orb_features={config.orb_max_features}, "
            f"match_features={config.match_max_features}, "
            f"spatial_filter={config.spatial_filter_enabled}, "
            f"grid={config.spatial_grid_rows}x{config.spatial_grid_cols}, "
            f"z_axis={config.z_axis}"
        )

    def _use_physical_z(self) -> bool:
        return str(getattr(self.config, "z_axis", "tan") or "tan").lower() == "physical"

    def _unwrap_pose(
        self, camera_params: Dict[str, Any]
    ) -> Optional[Tuple[np.ndarray, float, float, float]]:
        """前フレーム姿勢。yaw/pitch キーがあるときだけ管 (θ,Z) をその姿勢で投影する。"""
        if not self._use_physical_z():
            return None
        if camera_params.get("yaw") is None and camera_params.get("pitch") is None:
            return None
        cam_pos = np.asarray(
            camera_params.get("cam_pos", (0.0, 0.0, 0.0)), dtype=float
        ).reshape(3)
        roll = float(camera_params.get("roll", 0.0) or 0.0)
        yaw = float(camera_params.get("yaw", 0.0) or 0.0)
        pitch = float(camera_params.get("pitch", 0.0) or 0.0)
        return cam_pos, roll, yaw, pitch
    
    def extract_and_match(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        camera_params: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """フレーム間の特徴点抽出とマッチング(新方式)
        
        処理フロー:
        1. 円筒座標変換(前処理含む)
        2. 全画像でORB特徴点検出
        3. マッチング
        3.5. 物理 (ΔZ, Δθ) フィルタ（静止スキップはしない）
        4. 空間分散フィルタ(円筒座標グリッド)
        5. フレーム座標に逆変換
        
        Args:
            prev_frame: 前フレーム画像 (H, W, 3) BGR
            curr_frame: 現フレーム画像 (H, W, 3) BGR
            camera_params: カメラパラメータ
                - 'f': float - 焦点距離 (ピクセル)
                - 'center': Tuple[float, float] - 画像中心 (u, v)
                - 'pipe_diameter': float - 管径 (mm)
                - 'image_width': int - 画像幅
                - 'image_height': int - 画像高さ
        
        Returns:
            prev_points: 前フレームのマッチ点座標 (N, 2)、失敗時はNone
            curr_points: 現フレームのマッチ点座標 (N, 2)、失敗時はNone
            status: str - 処理結果ステータス
                - "SUCCESS": 成功（mean|ΔZ|≈0 でも推定へ進める）
                - "FAILED": 特徴点抽出またはマッチング失敗
        """
        try:
            self.logger.info("=== 円筒座標系特徴点マッチング開始(新方式) ===")
            
            # Phase 1: 円筒座標変換(前処理含む)
            self.logger.debug("Phase 1: 円筒座標変換 + 前処理")
            maps = self._build_remap_maps(prev_frame.shape, camera_params)
            prev_cylindrical, theta_grid, z_grid = self._transform_and_preprocess(
                prev_frame, camera_params, maps=maps
            )
            curr_cylindrical, _, _ = self._transform_and_preprocess(
                curr_frame, camera_params, maps=maps
            )
            
            # Phase 1.5: 位相相関によるdz/droll推定（有効時のみ）
            self._last_phase_correlation_result = None
            if self.config.phase_correlation_enabled:
                try:
                    dz_mm, droll_deg, response = self.estimate_shift_phase_correlation(
                        prev_cylindrical, curr_cylindrical
                    )
                    self._last_phase_correlation_result = {
                        'dz_mm': dz_mm,
                        'droll_deg': droll_deg,
                        'response': response
                    }
                    self.logger.info(
                        f"位相相関推定: dz={dz_mm:.3f}mm, droll={droll_deg:.4f}°, "
                        f"response={response:.4f}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"位相相関推定でエラー: {e}", exc_info=True
                    )

            # Phase 2: 全画像でORB特徴点検出
            self.logger.debug("Phase 2: 全画像でORB特徴点検出")
            prev_kp, prev_des = self._extract_features_full(
                prev_cylindrical, camera_params
            )
            curr_kp, curr_des = self._extract_features_full(
                curr_cylindrical, camera_params
            )
            
            if prev_des is None or curr_des is None:
                self.logger.warning("特徴点検出に失敗")
                return None, None, "FAILED"
            
            # Phase 3: マッチング
            self.logger.debug("Phase 3: マッチング")
            matches = self._match_features(prev_des, curr_des)
            
            # 最低2点あればマッチング層は通過させる
            # 推定に必要な最低点数の判断は推定器側の責務
            MIN_MATCH_POINTS = 2
            if len(matches) < MIN_MATCH_POINTS:
                self.logger.warning(f"マッチング点数が不足: {len(matches)}点")
                return None, None, "FAILED"
            
            # マッチング点の円筒座標を取得（ピクセル座標）
            prev_points_pixel = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            curr_points_pixel = np.float32([curr_kp[m.trainIdx].pt for m in matches])

            prev_points_pixel, curr_points_pixel = filter_subpixel_displacement(
                prev_points_pixel,
                curr_points_pixel,
                enabled=bool(getattr(self.config, "exclude_subpixel_matches", False)),
                min_pixel_displacement=float(
                    getattr(self.config, "min_pixel_displacement", 0.5)
                ),
                logger=self.logger,
                min_inlier_count=int(self.config.min_inlier_count),
            )

            # (x, y) を (theta, z) に変換
            prev_points_cyl = self._pixel_to_cylindrical(prev_points_pixel)
            curr_points_cyl = self._pixel_to_cylindrical(curr_points_pixel)

            self.logger.info(f"マッチング成功: {len(prev_points_cyl)}点")
            if self._use_physical_z():
                n_before = len(prev_points_cyl)
                prev_points_cyl, curr_points_cyl = filter_physical_motion_vectors(
                    prev_points_cyl,
                    curr_points_cyl,
                    iqr_multiplier=self.config.physical_iqr_multiplier,
                    min_points=self.config.physical_outlier_min_points,
                    max_dz_mm=self.config.physical_max_dz_mm,
                    max_dtheta_deg=self.config.physical_max_dtheta_deg,
                )
                self.logger.info(
                    f"物理 (ΔZ, Δθ) フィルタ: {n_before} → {len(prev_points_cyl)} 点"
                )
                if len(prev_points_cyl) < 2:
                    self.logger.warning("物理フィルタ後の点数が不足")
                    return None, None, "FAILED"
            else:

                self.logger.debug("Phase 4: 空間分散フィルタ")
                prev_points_cyl, curr_points_cyl = self._apply_spatial_filter(
                    prev_points_cyl, curr_points_cyl
                )

                if self.feature_filtering_config is not None:
                    if self.feature_filtering_config.magnitude_filter_enabled or \
                       self.feature_filtering_config.direction_filter_enabled:
                        try:
                            matches_list = [(prev_points_cyl, curr_points_cyl)]
                            filtered_matches, filtering_stats = self._apply_advanced_filtering(
                                matches_list, self.feature_filtering_config,
                                theta_grid, z_grid, camera_params
                            )
                            if len(filtered_matches) > 0 and len(filtered_matches[0][0]) > 0:
                                prev_points_cyl, curr_points_cyl = filtered_matches[0]
                                self.logger.info(
                                    f"フィルタリングパイプライン適用: "
                                    f"{filtering_stats['original_count']} → {filtering_stats['final_count']} 点 "
                                    f"({filtering_stats['total_removal_rate']:.1f}%除去)"
                                )
                            else:
                                self.logger.warning("フィルタリング後に点が0になりました。元の点を使用します。")
                        except Exception as e:
                            self.logger.warning(
                                f"フィルタリングパイプライン適用でエラー: {e}",
                                exc_info=True,
                            )

            self.logger.debug("Phase 5: フレーム座標に逆変換")
            prev_points_frame, curr_points_frame = \
                self._invert_paired_to_frame_coordinates(
                    prev_points_cyl, curr_points_cyl,
                    theta_grid, z_grid, camera_params
                )

            self._last_prev_points_cyl = prev_points_cyl
            self._last_curr_points_cyl = curr_points_cyl

            if prev_points_frame is None or curr_points_frame is None:
                if self._use_physical_z() and len(prev_points_cyl) >= 2:
                    self.logger.warning(
                        "フレーム座標への逆変換に失敗。物理 (θ, Z) 点は保持して続行"
                    )
                    return prev_points_cyl, curr_points_cyl, "SUCCESS"
                self.logger.warning("フレーム座標への逆変換に失敗")
                return None, None, "FAILED"

            self.logger.info(
                f"=== 円筒座標系特徴点マッチング完了: "
                f"prev={len(prev_points_frame)}点, curr={len(curr_points_frame)}点 ==="
            )
            return prev_points_frame, curr_points_frame, "SUCCESS"
        
        except Exception as e:
            self.logger.error(f"円筒座標系特徴点マッチングでエラー: {e}", exc_info=True)
            return None, None, "FAILED"
    
    def _transform_and_preprocess(
        self,
        frame: np.ndarray,
        camera_params: Dict[str, Any],
        maps: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """円筒座標変換と前処理を実行"""
        cyl_image, theta_grid, z_grid = self._to_cylindrical(
            frame, camera_params, maps=maps
        )
        cyl_image_processed = self._preprocess(cyl_image)
        return cyl_image_processed, theta_grid, z_grid
    
    def _extract_features_full(
        self,
        cylindrical_image: np.ndarray,
        camera_params: Dict[str, Any]
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """円筒座標画像から全画像でORB特徴点を検出
        
        Args:
            cylindrical_image: 円筒座標変換後の画像 (H, W) uint8
            camera_params: カメラパラメータ(使用しない、互換性のため)
        
        Returns:
            keypoints: 検出されたキーポイントリスト
            descriptors: 記述子 (N, 32) uint8、検出失敗時はNone
        """
        # ORB特徴点検出器を作成
        orb = cv2.ORB_create(
            nfeatures=self.config.orb_max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # 特徴点検出
        keypoints, descriptors = orb.detectAndCompute(cylindrical_image, None)
        
        if keypoints is None or len(keypoints) == 0:
            self.logger.warning("円筒座標画像でORB特徴点が検出されませんでした")
            return [], None
        
        self.logger.info(f"円筒座標画像でORB特徴点を検出: {len(keypoints)}点")
        
        return keypoints, descriptors
    
    def _match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
    ) -> List[cv2.DMatch]:
        """特徴量マッチング
        
        Args:
            desc1: 記述子1 (N1, 32)
            desc2: 記述子2 (N2, 32)
        
        Returns:
            matches: マッチング結果リスト
        """
        if len(desc1) == 0 or len(desc2) == 0:
            self.logger.warning("記述子が空です")
            return []
        
        if self.config.match_cross_check:
            # クロスチェック有効の場合
            matches_raw = self.matcher.match(desc1, desc2)
            # 距離でソート
            matches = sorted(matches_raw, key=lambda x: x.distance)
        else:
            # knnMatch + ratio test
            matches_raw = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for m_n in matches_raw:
                if len(m_n) == 2:
                    m, n = m_n
                    # Lowe's ratio test (threshold=0.7)
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            # 距離でソート
            matches = sorted(good_matches, key=lambda x: x.distance)
        
        self.logger.debug(f"特徴量マッチング: {len(matches)}組")
        
        return matches
    
    def _pixel_to_cylindrical(
        self,
        points_pixel: np.ndarray
    ) -> np.ndarray:
        """円筒座標画像のピクセル座標を円筒座標(theta, z)に変換
        
        Args:
            points_pixel: ピクセル座標 (N, 2) [x, y]
        
        Returns:
            points_cylindrical: 円筒座標 (N, 2) [theta_deg, z_mm]
        """
        x = points_pixel[:, 0]
        y = points_pixel[:, 1]

        theta_deg = (x / self.config.theta_resolution) * 360.0
        z_denom = max(self.config.z_resolution - 1, 1)
        t = np.clip(y / z_denom, 0.0, 1.0)
        if self._use_physical_z():
            z_mm = self._z_far_mm + t * (self._z_near_mm - self._z_far_mm)
        else:
            z_mm = t * self._effective_z_range_mm

        return np.column_stack([theta_deg, z_mm])
    
    def _apply_spatial_filter(
        self,
        prev_points_cylindrical: np.ndarray,
        curr_points_cylindrical: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """マッチング後の空間分散フィルタ(円筒座標グリッド)
        
        円筒座標空間(theta, z)でグリッド分割し、各グリッドから均等に点を選出します。
        これにより、特定の領域に点が偏ることを防ぎます。
        
        Args:
            prev_points_cylindrical: 前フレームの円筒座標点 (N, 2) [theta, z]
            curr_points_cylindrical: 現フレームの円筒座標点 (N, 2) [theta, z]
        
        Returns:
            filtered_prev: フィルタ後の前フレーム円筒座標点 (M, 2)
            filtered_curr: フィルタ後の現フレーム円筒座標点 (M, 2)
        """
        if not self.config.spatial_filter_enabled:
            return prev_points_cylindrical, curr_points_cylindrical
        
        n_points = len(prev_points_cylindrical)
        if n_points <= self.config.match_max_features:
            return prev_points_cylindrical, curr_points_cylindrical
        
        # theta, z座標を抽出
        theta = prev_points_cylindrical[:, 0]
        z = prev_points_cylindrical[:, 1]
        
        # 自動スケーリング: 極端な外れ値の影響を除外
        theta_min, theta_max = np.percentile(theta, [1, 99])
        z_min, z_max = np.percentile(z, [1, 99])
        
        # グリッド分割
        theta_edges = np.linspace(theta_min, theta_max, self.config.spatial_grid_rows + 1)
        z_edges = np.linspace(z_min, z_max, self.config.spatial_grid_cols + 1)
        
        theta_bins = np.digitize(theta, theta_edges) - 1
        z_bins = np.digitize(z, z_edges) - 1
        
        # 優先度(中心からの距離)
        theta_center = (theta_min + theta_max) / 2
        z_center = (z_min + z_max) / 2
        priority = (theta - theta_center)**2 + (z - z_center)**2
        
        # グリッドごとにグループ化
        grid = {}
        for i, (tb, zb) in enumerate(zip(theta_bins, z_bins)):
            if 0 <= tb < self.config.spatial_grid_rows and 0 <= zb < self.config.spatial_grid_cols:
                key = (tb, zb)
                grid.setdefault(key, []).append((priority[i], i))
        
        # 各グリッド内で優先度順にソート
        for key in grid:
            grid[key].sort()
        
        # ラウンドロビン方式で各グリッドから選出
        selected_indices = []
        while len(selected_indices) < self.config.match_max_features:
            any_added = False
            for key in sorted(grid.keys()):
                if grid[key]:
                    _, idx = grid[key].pop(0)
                    selected_indices.append(idx)
                    any_added = True
                    if len(selected_indices) == self.config.match_max_features:
                        break
            if not any_added:
                break
        
        selected_indices = np.array(selected_indices)
        
        self.logger.info(
            f"空間分散フィルタ適用: {n_points}点 → {len(selected_indices)}点 "
            f"(grid={self.config.spatial_grid_rows}×{self.config.spatial_grid_cols})"
        )
        
        return prev_points_cylindrical[selected_indices], curr_points_cylindrical[selected_indices]
    
    def _invert_to_frame_coordinates(
        self,
        points_cylindrical: np.ndarray,
        theta_grid: np.ndarray,
        z_grid: np.ndarray,
        camera_params: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """円筒座標からフレーム座標への逆変換
        
        Args:
            points_cylindrical: 円筒座標点群 (N, 2) [theta_deg, z_mm]
            theta_grid: theta座標2Dメッシュ(ラジアン、(z_res, theta_res))
            z_grid: z座標2Dメッシュ(mm、(z_res, theta_res))
            camera_params: カメラパラメータ辞書
        
        Returns:
            points_frame: フレーム座標点群 (N, 2) [u, v], 範囲外はNaN
            valid_mask: 有効点マスク (N,) bool
        """
        # カメラパラメータ取得
        img_h = camera_params.get('image_height', 1080)
        img_w = camera_params.get('image_width', 1920)
        f = camera_params.get('f', 300.0)
        center = camera_params.get('center', (img_w / 2.0, img_h / 2.0))
        center_x, center_y = center
        pipe_diameter = camera_params.get('pipe_diameter', 250.0)
        R = pipe_diameter / 2.0
        
        # ベクトル化変換
        theta_deg = points_cylindrical[:, 0]
        z_mm = points_cylindrical[:, 1]
        
        theta_rad = np.deg2rad(theta_deg)
        pose = self._unwrap_pose(camera_params)
        if pose is not None:
            cam_pos, roll, yaw, pitch = pose
            X = R * np.cos(theta_rad)
            Y = R * np.sin(theta_rad)
            Z = cam_pos[2] + z_mm
            pts = np.column_stack([X, Y, Z])
            pix_ideal = self.transformer.world_to_pixel(
                pts, cam_pos, roll, yaw, pitch, apply_distortion=False
            )
        else:
            if self._use_physical_z():
                z_safe = np.maximum(z_mm, 1e-3)
                zeta = np.arctan2(R, z_safe)
            else:
                zeta = np.arctan2(z_mm, R)

            x_cam = np.sin(zeta) * np.cos(theta_rad)
            y_cam = -np.sin(zeta) * np.sin(theta_rad)
            z_cam = np.cos(zeta)
            u_id, v_id = self.transformer.camera.cam_to_img(x_cam, y_cam, z_cam)
            pix_ideal = np.column_stack([u_id, v_id])

        pix = self.transformer.distort_ideal_pixels(pix_ideal)
        u, v = pix[:, 0], pix[:, 1]
        
        # 範囲内チェック（マスクとして返す）
        valid_mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        valid_mask = valid_mask & self.transformer.kannala_rising_branch_mask(pix_ideal)
        
        # 全点を返し、範囲外はNaN
        points_frame = np.stack([u, v], axis=-1).astype(np.float32)
        points_frame[~valid_mask] = np.nan
        
        n_valid = np.sum(valid_mask)
        if n_valid == 0:
            self.logger.warning("有効なフレーム座標点が見つかりませんでした")
            return None, valid_mask
        
        self.logger.debug(
            f"円筒→フレーム座標変換: {len(points_cylindrical)}点 → {n_valid}点有効"
        )
        
        return points_frame, valid_mask

    def _invert_paired_to_frame_coordinates(
        self,
        prev_points_cyl: np.ndarray,
        curr_points_cyl: np.ndarray,
        theta_grid: np.ndarray,
        z_grid: np.ndarray,
        camera_params: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """prev/currペアの1-to-1対応を維持して円筒→フレーム座標変換
        
        _invert_to_frame_coordinatesを個別に呼ぶと、範囲外フィルタリングで
        異なる点が除外されペアリングが崩壊する。本メソッドは両方の
        有効マスクをANDで結合し、対応関係を保証する。
        
        Args:
            prev_points_cyl: 前フレーム円筒座標 (N, 2) [theta_deg, z_mm]
            curr_points_cyl: 現フレーム円筒座標 (N, 2) [theta_deg, z_mm]
            theta_grid: theta座標2Dメッシュ
            z_grid: z座標2Dメッシュ
            camera_params: カメラパラメータ辞書
        
        Returns:
            prev_frame: 前フレームのフレーム座標 (M, 2), M≤N
            curr_frame: 現フレームのフレーム座標 (M, 2), M≤N
            両方Noneの場合は有効点なし
        """
        prev_frame, prev_mask = self._invert_to_frame_coordinates(
            prev_points_cyl, theta_grid, z_grid, camera_params
        )
        curr_frame, curr_mask = self._invert_to_frame_coordinates(
            curr_points_cyl, theta_grid, z_grid, camera_params
        )
        
        # 両方が有効な点のみ残す
        combined_mask = prev_mask & curr_mask
        
        if not np.any(combined_mask):
            self.logger.warning("ペア逆変換: 両方有効な点がありません")
            return None, None
        
        n_prev_only = int(np.sum(prev_mask & ~curr_mask))
        n_curr_only = int(np.sum(curr_mask & ~prev_mask))
        if n_prev_only > 0 or n_curr_only > 0:
            self.logger.debug(
                f"ペア逆変換: 片側のみ有効な点を除外 "
                f"(prev側のみ={n_prev_only}, curr側のみ={n_curr_only})"
            )
        
        return prev_frame[combined_mask], curr_frame[combined_mask]
    
    def _build_remap_maps(
        self,
        frame_shape: Tuple[int, ...],
        camera_params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """円筒 (θ,Z) → 魚眼画素の remap。両フレームで同じ地図を使う。"""
        img_h, img_w = frame_shape[:2]
        center = camera_params.get("center", (img_w / 2.0, img_h / 2.0))
        center_x, center_y = center
        theta_grid, z_grid, eta_grid, zeta_grid = self._create_cylindrical_grid(camera_params)
        pose = self._unwrap_pose(camera_params)
        invalid_behind = None
        if pose is not None:
            cam_pos, roll, yaw, pitch = pose
            pipe_diameter = camera_params.get("pipe_diameter", 250.0)
            radius = float(pipe_diameter) / 2.0
            world = np.stack(
                [
                    radius * np.cos(theta_grid),
                    radius * np.sin(theta_grid),
                    cam_pos[2] + z_grid,
                ],
                axis=-1,
            )
            pix_ideal = self.transformer.world_to_pixel(
                world.reshape(-1, 3),
                cam_pos,
                roll,
                yaw,
                pitch,
                apply_distortion=False,
            )
            pts_cam = (world.reshape(-1, 3) - cam_pos) @ R_w2c(roll, pitch, yaw)
            invalid_behind = pts_cam[:, 2].reshape(theta_grid.shape) <= 1e-6
        else:
            x_cam = np.sin(zeta_grid) * np.cos(eta_grid)
            y_cam = -np.sin(zeta_grid) * np.sin(eta_grid)
            z_cam = np.cos(zeta_grid)
            camera = self.transformer.camera
            u_flat, v_flat = camera.cam_to_img(
                x_cam.ravel(), y_cam.ravel(), z_cam.ravel()
            )
            pix_ideal = np.column_stack([u_flat, v_flat])

        pix = self.transformer.distort_ideal_pixels(pix_ideal)
        u = pix[:, 0].reshape(theta_grid.shape)
        v = pix[:, 1].reshape(theta_grid.shape)

        dist_from_center = np.hypot(u - center_x, v - center_y)
        max_radius = min(img_h, img_w) / 2.0
        inner_radius = max_radius * self.config.inner_radius_ratio
        outer_radius = max_radius * self.config.outer_radius_ratio
        invalid_mask = (dist_from_center < inner_radius) | (dist_from_center > outer_radius)
        invalid_mask |= (u < 0) | (u >= img_w) | (v < 0) | (v >= img_h)
        invalid_mask |= ~self.transformer.kannala_rising_branch_mask(pix_ideal).reshape(
            theta_grid.shape
        )
        if invalid_behind is not None:
            invalid_mask |= invalid_behind
        u = np.where(invalid_mask, -1, u)
        v = np.where(invalid_mask, -1, v)
        return u.astype(np.float32), v.astype(np.float32), theta_grid, z_grid

    def _to_cylindrical(
        self,
        frame: np.ndarray,
        camera_params: Dict[str, Any],
        maps: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """フレーム画像を円筒座標系に変換"""
        if maps is None:
            maps = self._build_remap_maps(frame.shape, camera_params)
        map_u, map_v, theta_grid, z_grid = maps
        cyl_image = cv2.remap(
            frame,
            map_u,
            map_v,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return cyl_image, theta_grid, z_grid


    def _preprocess(self, cyl_image: np.ndarray) -> np.ndarray:
        """円筒画像の前処理
        
        Args:
            cyl_image: 円筒座標系画像(BGR or グレースケール、(z_res, theta_res, [3]))
        
        Returns:
            processed: 前処理済み画像(グレースケール、(z_res, theta_res))
        """
        # ステップ1: グレースケール化(必要に応じて)
        if len(cyl_image.shape) == 3:
            gray = cv2.cvtColor(cyl_image, cv2.COLOR_BGR2GRAY)
            self.logger.debug(f"前処理: BGRからグレースケールに変換 shape={gray.shape}")
        else:
            gray = cyl_image.copy()
            self.logger.debug(f"前処理: グレースケール画像を使用 shape={gray.shape}")
        
        processed = gray.astype(np.float32)
        
        # ステップ2: 平均除去(DC成分削除)
        if self.config.enable_mean_removal:
            # 有効ピクセル(>0)の平均を計算
            valid_mask = processed > 0
            if np.sum(valid_mask) > 0:
                mean_val = np.mean(processed[valid_mask])
                # 平均値を減算し、+128でオフセット
                processed = np.where(valid_mask, processed - mean_val + 128.0, 0.0)
                # 0-255にクリップ
                processed = np.clip(processed, 0, 255)
                self.logger.debug(
                    f"前処理: 平均除去完了 mean={mean_val:.2f}, "
                    f"有効ピクセル={np.sum(valid_mask)}/{valid_mask.size}"
                )
            else:
                self.logger.warning("前処理: 有効ピクセルが見つかりませんでした(平均除去をスキップ)")
        
        # ステップ3: LoGフィルタ(Laplacian of Gaussian)
        if self.config.enable_highpass_filter:
            # uint8に変換(GaussianBlur用)
            temp = processed.astype(np.uint8)
            
            # ガウシアンブラーでローパスフィルタ
            blurred = cv2.GaussianBlur(
                temp,
                (self.config.highpass_kernel_size, self.config.highpass_kernel_size),
                self.config.highpass_sigma
            )
            
            # ラプラシアンで2階微分
            laplacian = cv2.Laplacian(blurred, cv2.CV_32F)
            
            # 絶対値を取り、0-255にクリップ
            processed = np.abs(laplacian)
            processed = np.clip(processed, 0, 255)
            
            self.logger.debug(
                f"前処理: LoGフィルタ完了 kernel_size={self.config.highpass_kernel_size}, "
                f"sigma={self.config.highpass_sigma:.2f}"
            )
        
        # ステップ4: Hanning窓(z方向のみ)
        if self.config.enable_hanning_window:
            z_res, theta_res = processed.shape
            
            # z方向のHanning窓を生成
            hanning_z = np.hanning(z_res)
            
            # 端部のみに適用(中央部は1.0)
            edge_size = int(z_res * self.config.hanning_window_ratio)
            if edge_size > 0:
                # 中央部を1.0に設定
                hanning_z[edge_size:-edge_size] = 1.0
            
            # z方向にブロードキャスト
            window = hanning_z[:, np.newaxis]  # (z_res, 1)
            
            # 窓関数を適用
            processed = processed * window
            
            self.logger.debug(
                f"前処理: Hanning窓適用完了 z方向={z_res}, "
                f"edge_size={edge_size}px ({self.config.hanning_window_ratio*100:.1f}%)"
            )
        
        # uint8に変換して返す
        processed = processed.astype(np.uint8)
        
        self.logger.debug(f"前処理完了: 出力shape={processed.shape}, dtype={processed.dtype}")
        
        return processed

    def estimate_shift_phase_correlation(
        self,
        prev_cyl_gray: np.ndarray,
        curr_cyl_gray: np.ndarray
    ) -> Tuple[float, float, float]:
        """位相相関によるdz(mm)・droll(deg)推定

        円筒座標画像間の位相相関を計算し、z方向シフト（前進量）と
        theta方向シフト（ロール回転量）を推定する。

        円筒座標画像では:
        - 垂直方向(y軸)のシフト = z方向の移動(前進)
        - 水平方向(x軸)のシフト = theta方向の回転(ロール)

        Args:
            prev_cyl_gray: 前フレームの前処理済みグレースケール円筒画像 (z_res, theta_res)
            curr_cyl_gray: 現フレームの前処理済みグレースケール円筒画像 (z_res, theta_res)

        Returns:
            dz_mm: z方向移動量（mm）。正の値は前進を示す。
            droll_deg: roll回転量（度）
            response: 信頼度（0〜1）。1に近いほど相関ピークが明瞭。
        """
        # float32変換
        prev_f32 = prev_cyl_gray.astype(np.float32)
        curr_f32 = curr_cyl_gray.astype(np.float32)

        # オプションのガウシアンブラー（ノイズ軽減）
        blur_size = self.config.phase_correlation_gaussian_blur_size
        if blur_size > 0:
            if blur_size % 2 == 0:
                blur_size += 1  # 奇数に調整
            prev_f32 = cv2.GaussianBlur(prev_f32, (blur_size, blur_size), 0)
            curr_f32 = cv2.GaussianBlur(curr_f32, (blur_size, blur_size), 0)

        # Hanning窓を作成（位相相関の精度向上）
        hann_window = cv2.createHanningWindow(
            (prev_f32.shape[1], prev_f32.shape[0]), cv2.CV_32F
        )

        # 位相相関
        (shift_x, shift_y), response = cv2.phaseCorrelate(
            prev_f32, curr_f32, hann_window
        )

        # ピクセルシフトを物理量に変換
        # 符号規約: phaseCorrelateはcurrがprevに対してどれだけシフトしたかを返す。
        # カメラ前進（正のdz）→円筒画像上でコンテンツが下にシフト→shift_yが正→dzはshift_y
        # カメラロール回転→コンテンツが同方向にシフト→drollはshift_x
        z_range_mm = self._effective_z_range_mm
        z_resolution = self.config.z_resolution
        theta_resolution = self.config.theta_resolution

        if self._use_physical_z():
            dz_mm = shift_y * ((self._z_far_mm - self._z_near_mm) / max(z_resolution, 1))
        else:
            dz_mm = shift_y * (z_range_mm / z_resolution)
        droll_deg = shift_x * (360.0 / theta_resolution)

        self.logger.debug(
            f"位相相関結果: shift=({shift_x:.3f}, {shift_y:.3f})px, "
            f"dz={dz_mm:.3f}mm, droll={droll_deg:.4f}°, response={response:.4f}"
        )

        return dz_mm, droll_deg, response
    
    def _apply_advanced_filtering(
        self,
        matches: List[Tuple[np.ndarray, np.ndarray]],
        feature_filtering_config: Any,
        theta_grid: np.ndarray,
        z_grid: np.ndarray,
        camera_params: Dict[str, Any]
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        """
        Task 4.2.4C: フィルタリングパイプライン（円筒座標系）
        
        処理フロー:
        1. ベクトル長さフィルタ（IQR法）
        2. θ-セクター方向フィルタ（IQR法）
        
        Parameters
        ----------
        matches : List[Tuple[np.ndarray, np.ndarray]]
            [(prev_pts_cyl, curr_pts_cyl), ...] 円筒座標マッチ点リスト
        feature_filtering_config : FeatureFilteringConfig
            フィルタリング設定
        
        Returns
        -------
        filtered_matches : List[Tuple[np.ndarray, np.ndarray]]
            フィルタリング後のマッチ点リスト
        stats : Dict[str, Any]
            統計情報
        """
        pipeline_stats = {
            'original_count': 0,
            'after_magnitude_filter': 0,
            'after_direction_filter': 0,
            'final_count': 0,
            'total_removal_rate': 0.0,
            'magnitude_stats': {},
            'direction_stats': {}
        }
        
        # Step 0: 元の点数
        if matches and len(matches) > 0:
            pipeline_stats['original_count'] = sum(len(prev) for prev, _ in matches)
        else:
            return matches, pipeline_stats
        
        filtered_matches = matches
        
        # Step 1: ベクトル長さフィルタ
        if feature_filtering_config.magnitude_filter_enabled:
            filtered_matches, mag_stats = filter_by_magnitude_outliers(
                filtered_matches,
                method=feature_filtering_config.magnitude_filter_method,
                iqr_multiplier=feature_filtering_config.magnitude_iqr_multiplier,
                absolute_max=feature_filtering_config.magnitude_absolute_max
            )
            pipeline_stats['magnitude_stats'] = mag_stats
            pipeline_stats['after_magnitude_filter'] = mag_stats.get('filtered_count', 0)
            
            if pipeline_stats['after_magnitude_filter'] == 0:
                self.logger.warning("長さフィルタで全ての点が除外されました。元の点を使用します。")
                return matches, pipeline_stats

        # Step 1.5: VP円フィルタ（フレーム座標系で実行）
        if feature_filtering_config.vp_circle_filter_enabled:
            before_vp_circle = sum(len(prev) for prev, _ in filtered_matches)
            
            # フィルタリング後のマッチリストを統合
            all_prev_cyl = np.vstack([prev for prev, curr in filtered_matches])
            all_curr_cyl = np.vstack([curr for prev, curr in filtered_matches])
            
            # 円筒座標→フレーム座標に逆変換（ペアリング維持）
            prev_points_frame, prev_valid = self._invert_to_frame_coordinates(
                all_prev_cyl, theta_grid, z_grid, camera_params
            )
            curr_points_frame, curr_valid = self._invert_to_frame_coordinates(
                all_curr_cyl, theta_grid, z_grid, camera_params
            )
            
            # 両方が有効な点のみを対象にする
            both_valid = prev_valid & curr_valid
            
            if prev_points_frame is not None and curr_points_frame is not None and np.any(both_valid):
                # フレーム座標でVP円制約チェック
                vp_center = (
                    self.transformer.camera.cx,
                    self.transformer.camera.cy
                )
                
                # 有効点のみ抽出してVP円チェック
                prev_valid_frame = prev_points_frame[both_valid]
                curr_valid_frame = curr_points_frame[both_valid]
                
                # フレーム座標でVP円制約チェック（移動ベクトル単位）
                vp_mask = self._check_vp_circle_constraint_frame(
                    prev_valid_frame,
                    curr_valid_frame,
                    vp_center,
                    feature_filtering_config.vp_circle_radius
                )
                # マスクを元の円筒座標に適用
                # both_validのTrue位置にvp_maskを埋め込む
                combined_mask = np.zeros(len(all_prev_cyl), dtype=bool)
                combined_mask[both_valid] = vp_mask
                filtered_prev_cyl = all_prev_cyl[combined_mask]
                filtered_curr_cyl = all_curr_cyl[combined_mask]
                
                # マッチリスト形式に戻す
                if len(filtered_prev_cyl) > 0:
                    filtered_matches = [(filtered_prev_cyl, filtered_curr_cyl)]
                else:
                    filtered_matches = []
                
                after_vp_circle = len(filtered_prev_cyl)
                vp_circle_filtered_count = before_vp_circle - after_vp_circle
                
                self.logger.info(
                    f"VP円フィルタ適用: {before_vp_circle}点 → {after_vp_circle}点 "
                    f"(除外: {vp_circle_filtered_count}点, "
                    f"半径: {feature_filtering_config.vp_circle_radius:.1f}px)"
                )
                
                # パイプライン統計に追加
                pipeline_stats['after_vp_circle_filter'] = after_vp_circle
                pipeline_stats['vp_circle_filtered'] = vp_circle_filtered_count
            else:
                self.logger.warning("フレーム座標への逆変換に失敗、VP円フィルタをスキップ")
        
        else:
            pipeline_stats['after_magnitude_filter'] = pipeline_stats['original_count']
        
        # Step 2: θ-セクター方向フィルタ（レンズ中心点距離フィルタ統合版）
        if feature_filtering_config.direction_filter_enabled:
            filtered_matches, dir_stats = filter_by_direction_outliers(
                filtered_matches,
                n_sectors=feature_filtering_config.direction_n_sectors,
                method='iqr',
                iqr_multiplier=feature_filtering_config.direction_iqr_multiplier,
                min_points_per_sector=feature_filtering_config.direction_min_points_per_sector,
                # レンズ中心点距離フィルタのパラメータ
                # 注: レンズ中心点は画像中心(cx, cy)を使用（center_offsetは将来のキャリブレーションで反映予定）
                vp_circle_filter_enabled=feature_filtering_config.vp_circle_filter_enabled,
                vp_circle_radius=feature_filtering_config.vp_circle_radius,
                vp_circle_center=(
                    self.transformer.camera.cx,
                    self.transformer.camera.cy
                )
            )
            pipeline_stats['direction_stats'] = dir_stats
            pipeline_stats['after_direction_filter'] = dir_stats.get('filtered_count', 0)
            
            if pipeline_stats['after_direction_filter'] == 0:
                self.logger.warning("方向フィルタで全ての点が除外されました。元の点を使用します。")
                return matches, pipeline_stats
        else:
            pipeline_stats['after_direction_filter'] = pipeline_stats['after_magnitude_filter']
        
        # Step 3: 最終統計
        if filtered_matches and len(filtered_matches) > 0:
            pipeline_stats['final_count'] = sum(len(prev) for prev, _ in filtered_matches)
        else:
            pipeline_stats['final_count'] = 0
        
        # 総除去率計算
        if pipeline_stats['original_count'] > 0:
            removed = pipeline_stats['original_count'] - pipeline_stats['final_count']
            pipeline_stats['total_removal_rate'] = (removed / pipeline_stats['original_count']) * 100
        
        return filtered_matches, pipeline_stats



    def _create_cylindrical_grid(
        self,
        camera_params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """円筒座標グリッドの生成
        
        Args:
            camera_params: カメラパラメータ辞書
        
        Returns:
            theta_grid: theta座標2Dメッシュ(ラジアン、(z_res, theta_res))
            z_grid: z座標2Dメッシュ(mm、(z_res, theta_res))
            eta_grid: カメラ座標系η角2Dメッシュ(ラジアン、(z_res, theta_res))
            zeta_grid: カメラ座標系ζ角2Dメッシュ(ラジアン、(z_res, theta_res))
        """
        # パラメータ取得
        pipe_diameter = camera_params.get('pipe_diameter', 250.0)
        R = pipe_diameter / 2.0

        theta_1d = np.linspace(0, 2 * np.pi, self.config.theta_resolution, endpoint=False)

        if self._use_physical_z():
            z_near, z_far = self._physical_z_limits(camera_params, R)
            self._z_near_mm = z_near
            self._z_far_mm = z_far
            self._effective_z_range_mm = z_far - z_near
            z_1d = np.linspace(z_far, z_near, self.config.z_resolution, endpoint=True)
            theta_grid, z_grid = np.meshgrid(theta_1d, z_1d, indexing='xy')
            eta_grid = theta_grid.copy()
            z_safe = np.maximum(z_grid, 1e-3)
            zeta_grid = np.arctan2(R, z_safe)
            self.logger.debug(
                f"円筒座標グリッド生成(physical): theta_res={self.config.theta_resolution}, "
                f"z_res={self.config.z_resolution}, R={R:.1f}mm, "
                f"Z=[{z_near:.1f}, {z_far:.1f}]mm"
            )
            return theta_grid, z_grid, eta_grid, zeta_grid

        R_DEFAULT = 125.0
        effective_z_range = self.config.z_range_mm * R / R_DEFAULT
        self._effective_z_range_mm = effective_z_range

        z_1d = np.linspace(0, effective_z_range, self.config.z_resolution, endpoint=True)

        theta_grid, z_grid = np.meshgrid(theta_1d, z_1d, indexing='xy')

        eta_grid = theta_grid.copy()
        zeta_grid = np.arctan2(z_grid, R)

        self.logger.debug(
            f"円筒座標グリッド生成: theta_res={self.config.theta_resolution}, "
            f"z_res={self.config.z_resolution}, R={R:.1f}mm, "
            f"z_range={effective_z_range:.1f}mm (config={self.config.z_range_mm:.1f}mm)"
        )

        return theta_grid, z_grid, eta_grid, zeta_grid

    def _physical_z_limits(
        self,
        camera_params: Dict[str, Any],
        pipe_radius: float,
    ) -> Tuple[float, float]:
        """inner/outer_radius_ratio を管軸距離 Z の遠/近に読み替える。"""
        img_h = camera_params.get('image_height', 1080)
        img_w = camera_params.get('image_width', 1920)
        f = float(camera_params.get('f', 300.0))
        max_radius = min(img_h, img_w) / 2.0
        inner_r = max(1.0, max_radius * float(self.config.inner_radius_ratio))
        outer_r = max(inner_r + 1.0, max_radius * float(self.config.outer_radius_ratio))
        f = max(f, 1.0)
        zeta_near = float(np.clip(outer_r / f, 1e-3, np.pi / 2.0 - 1e-3))
        zeta_far = float(np.clip(inner_r / f, 1e-3, np.pi / 2.0 - 1e-3))
        z_near = float(pipe_radius / np.tan(zeta_near))
        z_far = float(pipe_radius / np.tan(zeta_far))
        if z_far <= z_near:
            z_far = z_near * 4.0
        z_near = float(np.clip(z_near, 10.0, 5000.0))
        z_far = float(np.clip(z_far, z_near + 20.0, 10000.0))
        return z_near, z_far



    def _check_vp_circle_constraint_frame(
        self,
        prev_points_frame: np.ndarray,
        curr_points_frame: np.ndarray,
        vp_center: Tuple[float, float],
        vp_radius: float
    ) -> np.ndarray:
        """
        フレーム座標系でのVP円制約チェック

        移動ベクトルの後方延長線がVP中心距離圏内を通るかチェックします。

        Args:
            prev_points_frame: 前フレームの特徴点 (N, 2) - (u, v) in pixels
            curr_points_frame: 現フレームの特徴点 (N, 2) - (u, v) in pixels
            vp_center: VP中心座標 (cx, cy) [ピクセル]
            vp_radius: VP円の半径 [ピクセル]

        Returns:
            各点ペアが制約を満たすかのマスク (N,)
            True: 移動ベクトルの後方延長線がVP中心距離圏内を通る
            False: 制約を満たさない

        Notes:
            - 移動ベクトルが(du, dv) = curr - prevで定義される
            - curr点を通り方向(du, dv)の直線を定義
            - VP中心から直線までの垂直距離 <= vp_radius かつ
            - VP中心が移動ベクトルの後方（逆方向）にあることをチェック
            - 後方方向判定は内積 dot(du, dv, cx-u2, cy-v2) < 0 で実施

        Examples:
            >>> prev = np.array([[100, 100], [200, 200]])
            >>> curr = np.array([[110, 110], [210, 210]])
            >>> center = (50.0, 50.0)  # VP中心
            >>> radius = 10.0
            >>> mask = self._check_vp_circle_constraint_frame(prev, curr, center, radius)
            >>> # 移動ベクトルの後方延長線がVP中心近くを通る点のみTrue
        """
        cx, cy = vp_center
        valid_mask = np.zeros(len(prev_points_frame), dtype=bool)

        for i, ((u1, v1), (u2, v2)) in enumerate(zip(prev_points_frame, curr_points_frame)):
            # 移動ベクトル
            du = u2 - u1
            dv = v2 - v1

            # 直線の式: a*u + b*v + c = 0
            # 点(u2, v2)を通り、方向ベクトル(du, dv)の直線
            a = dv
            b = -du
            c = du * v2 - dv * u2

            # 正規化
            norm = np.sqrt(a**2 + b**2)
            if norm < 1e-10:
                # 移動量がほぼゼロ（静止状態）→除外
                valid_mask[i] = False
                continue

            # VP中心から直線までの距離
            distance = abs(a * cx + b * cy + c) / norm

            # 方向判定: VP中心が移動ベクトルの後方（逆方向）にあるか
            to_center_u = cx - u2
            to_center_v = cy - v2
            dot_product = du * to_center_u + dv * to_center_v

            # 後方延長側（dot < 0）かつ 距離が閾値以内
            is_minus_side = (dot_product < 0)
            is_within_radius = (distance < vp_radius)

            valid_mask[i] = (is_minus_side and is_within_radius)

        return valid_mask

# ============================================================================
# フィルタリング関数
# ============================================================================

def _wrap_deg(delta: np.ndarray) -> np.ndarray:
    """角度差を (-180, 180] に折り返す。"""
    return (np.asarray(delta, dtype=float) + 180.0) % 360.0 - 180.0


def _iqr_keep(values: np.ndarray, multiplier: float, min_points: int, floor: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < min_points:
        return np.ones(n, dtype=bool)
    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = max(float(q3 - q1), float(floor))
    lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
    return (values >= lo) & (values <= hi)


def fit_dc_first_harmonic(
    theta_deg: np.ndarray,
    values: np.ndarray,
) -> Dict[str, Any]:
    """値 ≈ dc + a cosθ + b sinθ。yaw/pitch は 1 次、roll は DC。"""
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    y = np.asarray(values, dtype=float).reshape(-1)
    n = int(y.size)
    if n < 3:
        dc = float(np.median(y)) if n else 0.0
        return {
            "dc": dc,
            "cos": 0.0,
            "sin": 0.0,
            "residual": y - dc,
            "model": np.full(n, dc, dtype=float),
        }
    A = np.column_stack([np.ones(n), np.cos(theta), np.sin(theta)])
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    model = A @ coeff
    return {
        "dc": float(coeff[0]),
        "cos": float(coeff[1]),
        "sin": float(coeff[2]),
        "residual": y - model,
        "model": model,
    }


def filter_subpixel_displacement(
    prev_points_pixel: np.ndarray,
    curr_points_pixel: np.ndarray,
    *,
    enabled: bool = False,
    min_pixel_displacement: float = 0.5,
    logger: Optional[logging.Logger] = None,
    min_inlier_count: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """円筒上の微小変位点を落とす。既定は落とさない。

    enabled=True のときだけ、変位が min_pixel_displacement 以下の点を除外する。
    停止点まで落とすと継ぎ目誤マッチが多数派になり得る。
    """
    if prev_points_pixel is None or curr_points_pixel is None:
        return prev_points_pixel, curr_points_pixel
    if not enabled or float(min_pixel_displacement) <= 0.0:
        return prev_points_pixel, curr_points_pixel
    log = logger if logger is not None else logging.getLogger(__name__)
    pixel_displacements = np.linalg.norm(
        np.asarray(curr_points_pixel, dtype=float)
        - np.asarray(prev_points_pixel, dtype=float),
        axis=1,
    )
    valid_mask = pixel_displacements > float(min_pixel_displacement)
    n_same = int(np.sum(~valid_mask))
    n_move = int(np.sum(valid_mask))
    if n_same > 0:
        log.debug(
            f"同一位置マッチ検出: {n_same}点 "
            f"(全{len(prev_points_pixel)}点中、"
            f"除外率={100.0 * n_same / max(len(prev_points_pixel), 1):.1f}%)"
        )
    if n_move > 0:
        if n_move < int(min_inlier_count):
            log.info(
                f"変位有り点が少数({n_move}点 < 推奨{min_inlier_count}点)だが"
                "ゼロ変位点を除外して続行"
            )
        return prev_points_pixel[valid_mask], curr_points_pixel[valid_mask]
    log.info(
        f"全マッチ点({len(prev_points_pixel)}点)が同一位置、"
        "静止フレームの可能性が高い"
    )
    return prev_points_pixel, curr_points_pixel


def filter_physical_motion_vectors(
    prev_cyl: np.ndarray,
    curr_cyl: np.ndarray,
    *,
    iqr_multiplier: float = 1.5,
    min_points: int = 4,
    max_dz_mm: float = 20.0,
    max_dtheta_deg: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """物理 (θ, Z) 上の誤マッチを落とす。後退 DC では全点捨てしない。

    点ごとの |ΔZ|≤max_dz_mm、|Δθ|≤max_dtheta_deg のあと、残差 IQR。
    マッチ失敗対応でこの点ゲートを外すと、IQR が誤マッチの 1 次を残して蛇腹になる。
    """
    if prev_cyl is None or curr_cyl is None or len(prev_cyl) == 0:
        return prev_cyl, curr_cyl
    prev_cyl = np.asarray(prev_cyl, dtype=float)
    curr_cyl = np.asarray(curr_cyl, dtype=float)
    dtheta = _wrap_deg(curr_cyl[:, 0] - prev_cyl[:, 0])
    dZ = curr_cyl[:, 1] - prev_cyl[:, 1]
    keep = (
        (np.abs(dZ) <= abs(float(max_dz_mm)))
        & (np.abs(dtheta) <= abs(float(max_dtheta_deg)))
    )
    prev_cyl, curr_cyl = prev_cyl[keep], curr_cyl[keep]
    if len(prev_cyl) < min_points:
        return prev_cyl, curr_cyl
    dZ = curr_cyl[:, 1] - prev_cyl[:, 1]
    fit_z = fit_dc_first_harmonic(prev_cyl[:, 0], dZ)
    z_keep = _iqr_keep(fit_z["residual"], iqr_multiplier, min_points, floor=0.2)
    if int(np.sum(z_keep)) >= 2:
        prev_cyl, curr_cyl = prev_cyl[z_keep], curr_cyl[z_keep]
    if len(prev_cyl) < min_points:
        return prev_cyl, curr_cyl
    dtheta = _wrap_deg(curr_cyl[:, 0] - prev_cyl[:, 0])
    fit_th = fit_dc_first_harmonic(prev_cyl[:, 0], dtheta)
    th_keep = _iqr_keep(fit_th["residual"], iqr_multiplier, min_points, floor=0.2)
    if int(np.sum(th_keep)) >= 2:
        prev_cyl, curr_cyl = prev_cyl[th_keep], curr_cyl[th_keep]
    return prev_cyl, curr_cyl


PHYSICAL_ATTITUDE_SOURCES = ("match", "discard", "dz_only")
_PHYSICAL_ATTITUDE_ALIASES = {
    "match": "match",
    "discard": "discard",
    "vp": "discard",
    "vp_discard": "discard",
    "dz_only": "dz_only",
    "dz-only": "dz_only",
    "dzonly": "dz_only",
}


def normalize_physical_attitude_source(
    value: Any,
    default: str = "match",
) -> str:
    """front.physical_attitude_source を match / discard / dz_only に正規化する。"""
    if value is None or str(value).strip() == "":
        return default
    key = str(value).strip().lower()
    if key not in _PHYSICAL_ATTITUDE_ALIASES:
        allowed = ", ".join(PHYSICAL_ATTITUDE_SOURCES)
        raise ValueError(
            f"front.physical_attitude_source は {allowed} のいずれかです: {value!r}"
        )
    return _PHYSICAL_ATTITUDE_ALIASES[key]


def apply_physical_mapping_attitude(
    motion: Dict[str, float],
    source: Any,
    yaw0: float,
    pitch0: float,
    yaw_est: Optional[float],
    pitch_est: Optional[float],
) -> Dict[str, float]:
    """推定 Δyaw/Δpitch を残し、discard/dz_only では消失点との差分を積む。"""
    out = dict(motion)
    out["dtheta_match"] = float(out.get("dtheta", 0.0))
    out["dphi_match"] = float(out.get("dphi", 0.0))
    src = normalize_physical_attitude_source(source)
    if src == "match":
        return out
    if yaw_est is not None:
        out["dtheta"] = float(yaw_est) - float(yaw0)
    else:
        out["dtheta"] = 0.0
    if pitch_est is not None:
        out["dphi"] = float(pitch_est) - float(pitch0)
    else:
        out["dphi"] = 0.0
    return out


def clip_front_physical_estimate(
    result: Dict[str, Any],
    *,
    max_dz_mm: float = 20.0,
    max_dyaw_deg: float = 2.0,
    max_dpitch_deg: float = 2.0,
    reject_reverse: bool = True,
    reverse_tol_mm: float = 0.5,
    dz_bounds: Optional[Tuple[float, float]] = None,
    min_points: int = 4,
    min_points_for_attitude: int = 20,
    untrusted_attitude_deg: float = 0.8,
    max_applied_attitude_deg: float = 1.2,
    clip_attitude: bool = True,
) -> Dict[str, Any]:
    """解析結果が物理上限を超えたら無効にする。飽和値は積まない。

    点は捨てない。カメラ前進が dz>0。上限超過・後退・点数不足は success=False。
    低点数の大きな Δyaw/Δpitch は 0 にする（dz は残す。スキップしない）。
    """
    out = dict(result)
    if not out.get("success"):
        return out
    dz = float(out.get("dz", 0.0))
    dyaw = float(out.get("dyaw", 0.0))
    dpitch = float(out.get("dpitch", 0.0))
    out["dz_raw"] = dz
    out["dyaw_raw"] = dyaw
    out["dpitch_raw"] = dpitch
    n = int(out.get("n_points", 0))
    max_dz = abs(float(max_dz_mm))
    max_yaw = np.radians(float(max_dyaw_deg))
    max_pitch = np.radians(float(max_dpitch_deg))
    eps = 1e-9
    if n < int(min_points):
        out["success"] = False
        out["failure_reason"] = "too_few_points"
        return out
    if abs(dz) > max_dz + eps:
        out["success"] = False
        out["failure_reason"] = "dz_exceeded_clip"
        return out
    if clip_attitude:
        if abs(dyaw) > max_yaw + eps or abs(dpitch) > max_pitch + eps:
            out["success"] = False
            out["failure_reason"] = "attitude_exceeded_clip"
            return out
    if reject_reverse and dz < -abs(float(reverse_tol_mm)):
        out["success"] = False
        out["failure_reason"] = "reverse_travel"
        return out
    if dz_bounds is not None:
        dz = float(np.clip(dz, float(dz_bounds[0]), float(dz_bounds[1])))
    drop_att = False
    drop_reason = None
    if clip_attitude:
        dyaw_deg = abs(np.degrees(dyaw))
        dpitch_deg = abs(np.degrees(dpitch))
        att_deg = max(dyaw_deg, dpitch_deg)
        max_applied = abs(float(max_applied_attitude_deg))
        untrusted = abs(float(untrusted_attitude_deg))
        n_att = int(min_points_for_attitude)
        if att_deg > max_applied + eps:
            drop_att = True
            drop_reason = "attitude_too_large"
        elif n < n_att and att_deg > untrusted + eps:
            drop_att = True
            drop_reason = "low_n_large_attitude"
    if drop_att:
        dyaw = 0.0
        dpitch = 0.0
        out["attitude_suppressed"] = True
        out["attitude_suppress_reason"] = drop_reason
    else:
        out["attitude_suppressed"] = False
    out["dz"] = dz
    out["droll"] = 0.0
    out["droll_deg"] = 0.0
    out["dyaw"] = dyaw
    out["dpitch"] = dpitch
    return out


def estimate_front_physical_option2(
    prev_cyl: np.ndarray,
    curr_cyl: np.ndarray,
    pipe_radius_mm: float = 125.0,
) -> Dict[str, Any]:
    """管 (θ, Z) から dz とフレーム間 Δyaw/Δpitch。droll は 0。

    ΔZ の DC が dz。1 次は dyaw/dpitch。Δθ の DC は roll に積まない。
    """
    prev_cyl = np.asarray(prev_cyl, dtype=float)
    curr_cyl = np.asarray(curr_cyl, dtype=float)
    n = int(len(prev_cyl))
    fail = {
        "dz": 0.0,
        "droll": 0.0,
        "droll_deg": 0.0,
        "dyaw": 0.0,
        "dpitch": 0.0,
        "n_points": n,
        "success": False,
    }
    if n < 3:
        return fail
    dZ = curr_cyl[:, 1] - prev_cyl[:, 1]
    dtheta_deg = _wrap_deg(curr_cyl[:, 0] - prev_cyl[:, 0])
    fit_z = fit_dc_first_harmonic(prev_cyl[:, 0], dZ)
    fit_th = fit_dc_first_harmonic(prev_cyl[:, 0], dtheta_deg)
    theta = np.deg2rad(prev_cyl[:, 0])
    Zs = np.maximum(prev_cyl[:, 1], 1e-3)
    radius = max(float(pipe_radius_mm), 1e-3)
    c = np.cos(theta)
    s = np.sin(theta)
    kZ = radius + (Zs * Zs) / radius
    kth = Zs / radius
    dth = np.deg2rad(dtheta_deg)
    A = np.zeros((2 * n, 3), dtype=float)
    b = np.zeros(2 * n, dtype=float)
    A[:n, 0] = -1.0
    A[:n, 1] = -kZ * c
    A[:n, 2] = -kZ * s
    b[:n] = dZ
    A[n:, 1] = -kth * s * radius
    A[n:, 2] = kth * c * radius
    b[n:] = dth * radius
    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
    dz = float(coeff[0])
    dyaw = float(coeff[1])
    dpitch = float(coeff[2])
    if not np.isfinite(dz):
        return fail
    return {
        "dz": dz,
        "droll": 0.0,
        "droll_deg": 0.0,
        "dyaw": dyaw,
        "dpitch": dpitch,
        "n_points": n,
        "success": True,
        "dZ_dc": float(fit_z["dc"]),
        "dZ_std": float(np.std(fit_z["residual"])),
        "dtheta_dc": float(fit_th["dc"]),
        "dtheta_cos": float(fit_th["cos"]),
        "dtheta_sin": float(fit_th["sin"]),
        "dtheta_std": float(np.std(fit_th["residual"])),
    }


def estimate_front_physical_dz_only(
    prev_cyl: np.ndarray,
    curr_cyl: np.ndarray,
    pipe_radius_mm: float = 125.0,
) -> Dict[str, Any]:
    """管 (θ, Z) から dz のみ。Δyaw/Δpitch は推定しない。

    ΔZ = −dz の 1 列 LS。1 次はモデルに入れないので、θ が偏ると dz に漏れる。
    """
    prev_cyl = np.asarray(prev_cyl, dtype=float)
    curr_cyl = np.asarray(curr_cyl, dtype=float)
    n = int(len(prev_cyl))
    fail = {
        "dz": 0.0,
        "droll": 0.0,
        "droll_deg": 0.0,
        "dyaw": 0.0,
        "dpitch": 0.0,
        "n_points": n,
        "success": False,
    }
    if n < 1:
        return fail
    dZ = curr_cyl[:, 1] - prev_cyl[:, 1]
    dtheta_deg = _wrap_deg(curr_cyl[:, 0] - prev_cyl[:, 0])
    fit_z = fit_dc_first_harmonic(prev_cyl[:, 0], dZ)
    fit_th = fit_dc_first_harmonic(prev_cyl[:, 0], dtheta_deg)
    A = -np.ones((n, 1), dtype=float)
    coeff, *_ = np.linalg.lstsq(A, dZ, rcond=None)
    dz = float(coeff[0])
    if not np.isfinite(dz):
        return fail
    return {
        "dz": dz,
        "droll": 0.0,
        "droll_deg": 0.0,
        "dyaw": 0.0,
        "dpitch": 0.0,
        "n_points": n,
        "success": True,
        "dZ_dc": float(fit_z["dc"]),
        "dZ_std": float(np.std(fit_z["residual"])),
        "dtheta_dc": float(fit_th["dc"]),
        "dtheta_cos": float(fit_th["cos"]),
        "dtheta_sin": float(fit_th["sin"]),
        "dtheta_std": float(np.std(fit_th["residual"])),
    }


def estimate_dz_droll_from_physical(
    prev_cyl: np.ndarray,
    curr_cyl: np.ndarray,
) -> Dict[str, Any]:
    """後方互換。中身は案②（droll は常に 0）。"""
    return estimate_front_physical_option2(prev_cyl, curr_cyl)


def should_skip_static_frame(
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    threshold: float = 2.0,
    min_points: int = 10,
    pixels_per_mm: float = 1.0
) -> Tuple[bool, Dict[str, Any]]:
    """静止状態フレームをスキップするか判定

    全特徴点の平均移動量を計算し、閾値以下の場合は静止状態と判定します。
    
    静止状態フレームの問題点:
        - 特徴点座標がほぼ一致 → 移動ベクトルが微小
        - 座標誤差の影響が相対的に巨大化
        - ベクトル方向が不安定(ランダム方向を向く)
        - レンズ中心点距離フィルタで全点除外 → 除外率99%超
    
    Args:
        prev_points: 前フレーム特徴点 (N, 2) [x, y] ピクセル座標
        curr_points: 現フレーム特徴点 (N, 2) [x, y] ピクセル座標
        threshold: 移動量閾値（mm、デフォルト2.0）
            TASK-4.2.11d Phase 1で単位をピクセル→mmに変更
        min_points: 最小点数（これ以下は信頼性が低いため判定しない）
        pixels_per_mm: カラーマップの解像度（ピクセル/mm、デフォルト1.0）
            ピクセル座標での移動量をmm単位に変換するために使用
            TASK-4.2.11d Phase 2で追加
    
    Returns:
        should_skip: bool
            True - スキップすべき(静止状態)
            False - 処理すべき(移動あり)
        stats: Dict[str, Any]
            統計情報:
                - n_points: int - 特徴点数
                - avg_movement_px: float - 平均移動量(ピクセル)
                - avg_movement_mm: float - 平均移動量(mm)
                - median_movement_px: float - 中央値移動量(ピクセル)
                - max_movement_px: float - 最大移動量(ピクセル)
                - min_movement_px: float - 最小移動量(ピクセル)
                - std_movement_px: float - 移動量標準偏差(ピクセル)
                - threshold_mm: float - 使用した閾値(mm)
                - pixels_per_mm: float - 使用した解像度(ピクセル/mm)
                - decision: str - 判定理由("static", "moving", "insufficient_points")
    
    Raises:
        ValueError: prev_pointsとcurr_pointsの形状が一致しない場合
    
    Examples:
        >>> prev_pts = np.array([[100, 200], [300, 400]])
        >>> curr_pts = np.array([[101, 201], [302, 403]])
        >>> should_skip, stats = should_skip_static_frame(prev_pts, curr_pts, threshold=2.0)
        >>> print(f"Skip: {should_skip}, Avg movement: {stats['avg_movement']:.2f}px")
        Skip: True, Avg movement: 1.58px
    
    Notes:
        - 特徴点数が0の場合は、安全側に倒してTrue(スキップ)を返す
        - 円筒座標の場合、フレーム座標に逆変換後の座標で判定する
        - 閾値2.0pxは経験的な値(Phase 2実験より)
    """
    # 入力検証: 空配列チェック
    if len(prev_points) == 0 or len(curr_points) == 0:
        logger.debug("特徴点数が0のため、静止状態と判定(スキップ)")
        return True, {
            'n_points': 0,
            'avg_movement_px': 0.0,
            'avg_movement_mm': 0.0,
            'median_movement_px': 0.0,
            'max_movement_px': 0.0,
            'min_movement_px': 0.0,
            'std_movement_px': 0.0,
            'threshold_mm': threshold,
            'pixels_per_mm': pixels_per_mm,
            'decision': 'insufficient_points'
        }
    
    # 入力検証: 形状チェック
    if prev_points.shape != curr_points.shape:
        raise ValueError(
            f"prev_points と curr_points の形状が一致しません: "
            f"{prev_points.shape} != {curr_points.shape}"
        )
    
    if prev_points.ndim != 2 or prev_points.shape[1] != 2:
        raise ValueError(
            f"特徴点配列は (N, 2) の形状である必要があります: "
            f"shape={prev_points.shape}"
        )
    
    # 移動量計算: L2ノルム
    movements = np.linalg.norm(curr_points - prev_points, axis=1)
    
    # 統計計算
    avg_movement = float(np.mean(movements))
    median_movement = float(np.median(movements))
    max_movement = float(np.max(movements))
    min_movement = float(np.min(movements))
    std_movement = float(np.std(movements))
    n_points = int(len(movements))
    
    # TASK-4.2.11d Phase 2: ピクセル→mm変換
    avg_movement_px = avg_movement
    avg_movement_mm = avg_movement_px / pixels_per_mm
    
    # 判定: 平均移動量(mm)が閾値以下なら静止状態
    should_skip = (avg_movement_mm <= threshold)
    decision = "static" if should_skip else "moving"
    
    # ログ出力（第1段階判定を明記）
    if should_skip:
        logger.info(
            f"フレーム停止検出（第1段階判定）: "
            f"平均移動量={avg_movement_mm:.2f}mm <= 閾値={threshold:.2f}mm "
            f"({avg_movement_px:.2f}px / {pixels_per_mm:.2f}px/mm) "
            f"(n={n_points}点, median={median_movement:.2f}px, max={max_movement:.2f}px) → スキップ"
        )
    else:
        logger.debug(
            f"移動フレーム: "
            f"平均移動量={avg_movement_mm:.2f}mm > 閾値={threshold:.2f}mm "
            f"({avg_movement_px:.2f}px / {pixels_per_mm:.2f}px/mm) "
            f"(n={n_points}点, median={median_movement:.2f}px, max={max_movement:.2f}px) → 処理続行"
        )
    
    # 統計情報
    stats = {
        'n_points': n_points,
        'avg_movement_px': avg_movement_px,
        'avg_movement_mm': avg_movement_mm,
        'median_movement_px': median_movement,
        'max_movement_px': max_movement,
        'min_movement_px': min_movement,
        'std_movement_px': std_movement,
        'threshold_mm': threshold,
        'pixels_per_mm': pixels_per_mm,
        'decision': decision
    }
    
    return should_skip, stats



def filter_by_magnitude_outliers(
    matches: List[Tuple[np.ndarray, np.ndarray]],
    method: str = 'iqr',
    iqr_multiplier: float = 1.5,
    absolute_max: float = 100.0
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """
    移動ベクトルの長さに基づく外れ値フィルタリング
    
    2段階フィルタ:
    1. 絶対値上限フィルタ: absolute_maxを超える変位を無条件除去（IQR歪み防止）
    2. IQR（四分位範囲）法: 統計的外れ値を除去
    
    Args:
        matches: 特徴点ペアのリスト [(prev_pts, curr_pts), ...]
                prev_pts, curr_pts: (N, 2) 円筒座標 [theta_deg, z_mm]
        method: 外れ値検出方法 ('iqr' または 'mad')
        iqr_multiplier: IQR倍率（1.5が標準、2.0で緩和、1.0で厳格）
        absolute_max: 絶対値上限（px）。これを超える変位は無条件で除去
    
    Returns:
        filtered_matches: フィルタ後の特徴点ペアリスト
        stats: 統計情報dict
    
    Raises:
        ValueError: methodが'iqr'または'mad'以外の場合
    """
    # パラメータバリデーション
    if method not in ['iqr', 'mad']:
        raise ValueError(f"Invalid method: {method}. Must be 'iqr' or 'mad'.")
    
    if iqr_multiplier <= 0:
        raise ValueError(f"iqr_multiplier must be positive, got {iqr_multiplier}")
    
    # エッジケース: 空のマッチリスト
    if len(matches) == 0:
        logger.warning("入力マッチが空です。フィルタリングをスキップします。")
        return matches, {
            'original_count': 0,
            'filtered_count': 0,
            'removed_count': 0,
            'removal_rate': 0.0
        }
    
    # Step 1: 移動ベクトルの長さを計算
    magnitudes = []
    for prev_pts, curr_pts in matches:
        dx = curr_pts[:, 0] - prev_pts[:, 0]
        dy = curr_pts[:, 1] - prev_pts[:, 1]
        mag = np.sqrt(dx**2 + dy**2)
        magnitudes.extend(mag)
    
    magnitudes = np.array(magnitudes)
    original_count = len(magnitudes)
    
    # エッジケース: 特徴点数が1点のみ
    if original_count <= 1:
        logger.debug("特徴点数が1点以下のため、フィルタリングをスキップします。")
        return matches, {
            'original_count': original_count,
            'filtered_count': original_count,
            'removed_count': 0,
            'removal_rate': 0.0,
            'magnitude_median': magnitudes[0] if original_count == 1 else 0.0
        }
    
    # Step 1.5: 絶対値上限フィルタ（IQR計算前に極端な外れ値を除去）
    abs_removed_count = 0
    if absolute_max > 0:
        pre_filtered_matches = []
        for prev_pts, curr_pts in matches:
            dx = curr_pts[:, 0] - prev_pts[:, 0]
            dy = curr_pts[:, 1] - prev_pts[:, 1]
            mag = np.sqrt(dx**2 + dy**2)
            mask = mag <= absolute_max
            n_removed = int(np.sum(~mask))
            abs_removed_count += n_removed
            if np.any(mask):
                pre_filtered_matches.append((
                    prev_pts[mask].copy(),
                    curr_pts[mask].copy()
                ))
        
        if abs_removed_count > 0:
            logger.debug(
                f"絶対値上限フィルタ: {abs_removed_count}点除去 "
                f"(閾値={absolute_max:.1f}px)"
            )
        
        if len(pre_filtered_matches) == 0:
            logger.warning(
                f"絶対値上限フィルタで全点除去 (閾値={absolute_max:.1f}px)"
            )
            return matches, {
                'original_count': original_count,
                'filtered_count': original_count,
                'removed_count': 0,
                'removal_rate': 0.0
            }
        
        matches = pre_filtered_matches
        
        # IQR用にmagnitudesを再計算
        magnitudes = []
        for prev_pts, curr_pts in matches:
            dx = curr_pts[:, 0] - prev_pts[:, 0]
            dy = curr_pts[:, 1] - prev_pts[:, 1]
            mag = np.sqrt(dx**2 + dy**2)
            magnitudes.extend(mag)
        magnitudes = np.array(magnitudes)
    
    # Step 2: IQR法で外れ値を検出
    q1 = np.percentile(magnitudes, 25)
    q3 = np.percentile(magnitudes, 75)
    iqr = q3 - q1
    
    # IQRが0の場合（全て同じ値）の処理
    if iqr < 1e-9:
        logger.debug("IQRが0です（全て同じ値）。外れ値なしと判定します。")
        filtered_count = len(magnitudes)
        return matches, {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'removed_count': original_count - filtered_count,
            'removal_rate': float((original_count - filtered_count) / original_count * 100) if original_count > 0 else 0.0,
            'magnitude_q1': float(q1),
            'magnitude_median': float(np.median(magnitudes)),
            'magnitude_q3': float(q3),
            'magnitude_iqr': float(iqr),
            'lower_bound': float(q1),
            'upper_bound': float(q3),
            'abs_removed_count': abs_removed_count
        }
    
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    
    logger.debug(
        f"IQR計算: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
        f"bounds=[{lower_bound:.2f}, {upper_bound:.2f}]"
    )
    
    # Step 3: フィルタリング
    filtered_matches = []
    iqr_removed_count = 0
    
    for prev_pts, curr_pts in matches:
        dx = curr_pts[:, 0] - prev_pts[:, 0]
        dy = curr_pts[:, 1] - prev_pts[:, 1]
        mag = np.sqrt(dx**2 + dy**2)
        
        # 有効なマスク
        mask = (mag >= lower_bound) & (mag <= upper_bound)
        
        if np.any(mask):
            filtered_matches.append((
                prev_pts[mask].copy(),
                curr_pts[mask].copy()
            ))
        
        iqr_removed_count += np.sum(~mask)
    
    # Step 4: 統計情報
    total_removed = abs_removed_count + iqr_removed_count
    filtered_count = sum(len(m[0]) for m in filtered_matches)
    
    if filtered_count == 0:
        logger.warning(
            f"ベクトル長さフィルタで全ての点が除外されました。"
            f"(元={original_count}点, bounds=[{lower_bound:.2f}, {upper_bound:.2f}])"
        )
    
    stats = {
        'original_count': original_count,
        'filtered_count': int(filtered_count),
        'removed_count': int(total_removed),
        'removal_rate': float((total_removed / original_count * 100) if original_count > 0 else 0.0),
        'magnitude_q1': float(q1),
        'magnitude_median': float(np.median(magnitudes)),
        'magnitude_q3': float(q3),
        'magnitude_iqr': float(iqr),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'abs_removed_count': abs_removed_count,
        'iqr_removed_count': int(iqr_removed_count)
    }
    
    logger.info(
        f"ベクトル長さフィルタ: {stats['original_count']}点 → {stats['filtered_count']}点 "
        f"(除外率={stats['removal_rate']:.1f}%"
        f"{f', 絶対値上限={abs_removed_count}点' if abs_removed_count > 0 else ''})"
    )
    
    return filtered_matches, stats


def _check_vp_circle_constraint(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    radius: float,
    center: Tuple[float, float]
) -> np.ndarray:
    """レンズ中心点距離制約チェック
    
    移動ベクトルの後方延長線がレンズ中心距離圏内を通るかチェック
    
    Parameters
    ----------
    prev_pts : np.ndarray
        前フレーム点 (N, 2) - 円筒座標(theta_deg, z_mm)
    curr_pts : np.ndarray
        現フレーム点 (N, 2) - 円筒座標(theta_deg, z_mm)
    radius : float
        レンズ中心点距離の閾値（ピクセル、円筒座標系では度単位として扱う）
    center : Tuple[float, float]
        レンズ中心点座標 (cx, cy) - 円筒座標系では(theta_center_deg, z_center_mm)
    
    Returns
    -------
    valid_mask : np.ndarray
        有効な点のマスク (N,) bool
    
    Notes
    -----
    円筒座標系では、ピクセル座標系とは異なり、theta(度)とz(mm)の
    スケールが異なるため、距離計算には注意が必要です。
    ここでは簡易的に、移動ベクトルの後方延長線が中心点の近傍を
    通るかをチェックします。
    """
    cx, cy = center
    valid_mask = np.zeros(len(prev_pts), dtype=bool)
    
    for i, ((theta1, z1), (theta2, z2)) in enumerate(zip(prev_pts, curr_pts)):
        # 移動ベクトル
        d_theta = theta2 - theta1
        d_z = z2 - z1
        
        # 直線の式: a*theta + b*z + c = 0
        # 点(theta2, z2)を通り、方向ベクトル(d_theta, d_z)の直線
        a = d_z
        b = -d_theta
        c = d_theta * z2 - d_z * theta2
        
        # 正規化
        norm = np.sqrt(a**2 + b**2)
        if norm < 1e-10:
            valid_mask[i] = False
            continue
        
        # 円中心から直線までの距離
        distance = abs(a * cx + b * cy + c) / norm
        
        # 方向判定: 円中心が移動ベクトルの後方（逆方向）にあるか
        to_center_theta = cx - theta2
        to_center_z = cy - z2
        dot_product = d_theta * to_center_theta + d_z * to_center_z
        
        # 後方延長側（dot < 0）かつ 距離が閾値以内
        is_minus_side = (dot_product < 0)
        is_within_radius = (distance < radius)
        
        valid_mask[i] = (is_minus_side and is_within_radius)
    
    return valid_mask


def filter_by_direction_outliers(
    matches: List[Tuple[np.ndarray, np.ndarray]],
    n_sectors: int = 8,
    method: str = 'iqr',
    iqr_multiplier: float = 1.5,
    min_points_per_sector: int = 5,
    # レンズ中心点距離フィルタパラメータ
    vp_circle_filter_enabled: bool = False,
    vp_circle_radius: float = 300.0,
    vp_circle_center: Tuple[float, float] = (640.0, 360.0)
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:

    """
    θ-セクター方向フィルタ + レンズ中心点距離フィルタ
    
    円筒座標θ（0-360度）を複数セクター（デフォルト8分割＝45°ごと）に分割し、
    各セクター内で移動ベクトルのΔθ方向の外れ値をIQR法で検出・除去します。
    さらに、レンズ中心点距離フィルタを前段として統合し、ミスマッチ特徴点を排除します。
    
    処理フロー:
        Step 0: データ統合
        ↓
        Step 1: レンズ中心点距離フィルタ（オプション）
            - 移動ベクトルの後方延長線がレンズ中心距離圏内を通るかチェック
            - 通らない点はミスマッチとして除外
        ↓
        Step 2: θ-セクター分割
        ↓
        Step 3: 各セクター内でΔθのIQR外れ値除去
        ↓
        Step 4: 統計情報計算
    
    Parameters
    ----------
    matches : List[Tuple[np.ndarray, np.ndarray]]
        [(prev_pts, curr_pts), ...] 形式のマッチング点リスト
        各配列の形状: (N, 2) - (theta_deg, z_mm)円筒座標
    n_sectors : int, default=8
        セクター分割数（8の場合、45°ごとに分割）
    method : str, default='iqr'
        外れ値検出手法（現在は'iqr'のみ対応）
    iqr_multiplier : float, default=1.5
        IQR倍率（1.5=標準、2.0=緩和、1.0=厳格）
    min_points_per_sector : int, default=5
        セクター内最小点数（これ以下の場合はフィルタリングスキップ）
    vp_circle_filter_enabled : bool, default=False
        レンズ中心点距離フィルタの有効/無効
    vp_circle_radius : float, default=300.0
        レンズ中心点距離の閾値（ピクセル）
    vp_circle_center : Tuple[float, float], default=(640.0, 360.0)
        レンズ中心点座標 (cx, cy)
    
    Returns
    -------
    filtered_matches : List[Tuple[np.ndarray, np.ndarray]]
        フィルタリング後のマッチング点リスト
    stats : Dict[str, Any]
        統計情報辞書:
            - original_count: int - 元の総点数
            - after_vp_circle_filter: int - レンズ中心点距離フィルタ後の点数
            - vp_circle_removed: int - レンズ中心点距離フィルタで除去された点数
            - filtered_count: int - 最終フィルタ後の総点数
            - removed_count: int - 除去された点数
            - removal_rate: float - 除去率（%）
            - n_sectors: int - セクター数
            - min_points_per_sector: int - 最小点数閾値
            - sectors_processed: int - 処理されたセクター数
            - sectors_skipped: int - スキップされたセクター数
            - per_sector_stats: List[Dict] - セクターごとの統計
            - delta_theta_q1: float - 全体のΔθ Q1
            - delta_theta_median: float - 全体のΔθ中央値
            - delta_theta_q3: float - 全体のΔθ Q3
            - delta_theta_iqr: float - 全体のΔθ IQR
            - delta_theta_std: float - 全体のΔθ標準偏差
    
    Raises
    ------
    ValueError
        n_sectors < 1、method != 'iqr'、iqr_multiplier <= 0、
        min_points_per_sector < 1 の場合
    
    Examples
    --------
    >>> matches = [(prev_pts1, curr_pts1), (prev_pts2, curr_pts2)]
    >>> filtered, stats = filter_by_direction_outliers(matches, n_sectors=8)
    >>> print(f"除去率: {stats['removal_rate']:.1f}%")
    >>> for s in stats['per_sector_stats']:
    ...     print(f"Sector {s['sector_id']}: {s['removed_count']}個除外")
    """
    # 入力検証
    if not matches or len(matches) == 0:
        logger.warning("入力マッチが空です。フィルタリングをスキップします。")
        return matches, {
            'original_count': 0,
            'filtered_count': 0,
            'removed_count': 0,
            'removal_rate': 0.0,
            'n_sectors': n_sectors,
            'min_points_per_sector': min_points_per_sector,
            'sectors_processed': 0,
            'sectors_skipped': 0,
            'per_sector_stats': [],
            'delta_theta_q1': 0.0,
            'delta_theta_median': 0.0,
            'delta_theta_q3': 0.0,
            'delta_theta_iqr': 0.0,
            'delta_theta_std': 0.0
        }
    
    if n_sectors < 1:
        raise ValueError(f"n_sectors must be >= 1, got {n_sectors}")
    
    if method != 'iqr':
        raise ValueError(f"Unsupported method: {method}")
    
    if iqr_multiplier <= 0:
        raise ValueError(f"iqr_multiplier must be > 0, got {iqr_multiplier}")
    
    if min_points_per_sector < 1:
        raise ValueError(f"min_points_per_sector must be >= 1, got {min_points_per_sector}")
    
    # Step 1: データ統合（全マッチグループを1つの配列に統合）
    all_prev_pts = np.vstack([prev for prev, curr in matches])
    all_curr_pts = np.vstack([curr for prev, curr in matches])

    original_count = len(all_prev_pts)

    # Note: VP円フィルタはフレーム座標版（Step 1.5）で実施済み
    # 円筒座標版は座標系ミスマッチにより削除（Task 4.2.11C）
    vp_circle_removed = 0
    
    # エッジケース: 特徴点数が1点のみ
    if original_count <= 1:
        logger.debug("特徴点数が1点以下のため、フィルタリングをスキップします。")
        return matches, {
            'original_count': original_count,
            'filtered_count': original_count,
            'removed_count': 0,
            'removal_rate': 0.0,
            'n_sectors': n_sectors,
            'min_points_per_sector': min_points_per_sector,
            'sectors_processed': 0,
            'sectors_skipped': n_sectors,
            'per_sector_stats': [],
            'delta_theta_q1': 0.0,
            'delta_theta_median': 0.0,
            'delta_theta_q3': 0.0,
            'delta_theta_iqr': 0.0,
            'delta_theta_std': 0.0
        }
    
    # Step 2: Δθ計算（循環性考慮）
    # theta_deg (0-360度) → theta_rad (0-2π)
    theta_prev_rad = np.deg2rad(all_prev_pts[:, 0])
    theta_curr_rad = np.deg2rad(all_curr_pts[:, 0])
    
    # Δθ = curr - prev（循環性を考慮し、[-π, π]に正規化）
    delta_theta = theta_curr_rad - theta_prev_rad
    delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))  # [-π, π]
    
    # 全体統計の計算（後で使用）
    delta_theta_all = delta_theta.copy()
    
    # Step 3: セクター分割
    sector_size = 2 * np.pi / n_sectors
    sector_indices = np.floor(theta_prev_rad / sector_size).astype(int) % n_sectors
    
    # Step 4: 外れ値検出用のマスク初期化（Trueは保持、Falseは除外）
    keep_mask = np.ones(len(all_prev_pts), dtype=bool)
    
    # セクター別統計を格納するリスト
    per_sector_stats = []
    sectors_processed = 0
    sectors_skipped = 0
    
    # Step 5: 各セクター内で方向外れ値除去
    for sector_id in range(n_sectors):
        # このセクターに属する点のマスク
        sector_mask = (sector_indices == sector_id)
        sector_delta_theta = delta_theta[sector_mask]
        
        # セクター内の点数
        n_points_in_sector = len(sector_delta_theta)
        
        # セクターのtheta範囲（度単位、可視化用）
        theta_range_deg = (sector_id * 360.0 / n_sectors, (sector_id + 1) * 360.0 / n_sectors)
        
        # 点数不足チェック
        if n_points_in_sector < min_points_per_sector:
            logger.debug(
                f"Sector {sector_id} ({theta_range_deg[0]:.0f}-{theta_range_deg[1]:.0f}°): "
                f"点数不足({n_points_in_sector}点 < {min_points_per_sector}点)、フィルタリングスキップ"
            )
            per_sector_stats.append({
                'sector_id': sector_id,
                'theta_range': theta_range_deg,
                'original_count': n_points_in_sector,
                'filtered_count': n_points_in_sector,
                'removed_count': 0,
                'delta_theta_q1': 0.0,
                'delta_theta_median': 0.0,
                'delta_theta_q3': 0.0,
                'delta_theta_iqr': 0.0,
                'lower_bound': 0.0,
                'upper_bound': 0.0
            })
            sectors_skipped += 1
            continue
        
        # IQR計算
        q1 = np.percentile(sector_delta_theta, 25)
        q3 = np.percentile(sector_delta_theta, 75)
        iqr = q3 - q1
        
        # IQRが0の場合（全て同じ方向）
        if iqr < 1e-9:
            logger.debug(
                f"Sector {sector_id}: IQRが0（全て同じ方向）、外れ値なしと判定"
            )
            per_sector_stats.append({
                'sector_id': sector_id,
                'theta_range': theta_range_deg,
                'original_count': n_points_in_sector,
                'filtered_count': n_points_in_sector,
                'removed_count': 0,
                'delta_theta_q1': float(q1),
                'delta_theta_median': float(np.median(sector_delta_theta)),
                'delta_theta_q3': float(q3),
                'delta_theta_iqr': float(iqr),
                'lower_bound': float(q1),
                'upper_bound': float(q3)
            })
            sectors_processed += 1
            continue
        
        # 外れ値境界の計算
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        
        # 外れ値検出（セクター内マスク）
        sector_inlier_mask = (sector_delta_theta >= lower_bound) & (sector_delta_theta <= upper_bound)
        
        # グローバルマスクに反映
        # sector_maskがTrueの位置のうち、sector_inlier_maskがFalseのものを除外
        sector_indices_array = np.where(sector_mask)[0]
        keep_mask[sector_indices_array[~sector_inlier_mask]] = False
        
        # 統計情報
        n_removed_in_sector = np.sum(~sector_inlier_mask)
        n_filtered_in_sector = n_points_in_sector - n_removed_in_sector
        
        logger.debug(
            f"Sector {sector_id} ({theta_range_deg[0]:.0f}-{theta_range_deg[1]:.0f}°): "
            f"{n_points_in_sector}点 → {n_filtered_in_sector}点 "
            f"(除外={n_removed_in_sector}点, {100*n_removed_in_sector/n_points_in_sector:.1f}%)"
        )
        
        per_sector_stats.append({
            'sector_id': sector_id,
            'theta_range': theta_range_deg,
            'original_count': int(n_points_in_sector),
            'filtered_count': int(n_filtered_in_sector),
            'removed_count': int(n_removed_in_sector),
            'delta_theta_q1': float(q1),
            'delta_theta_median': float(np.median(sector_delta_theta)),
            'delta_theta_q3': float(q3),
            'delta_theta_iqr': float(iqr),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        })
        sectors_processed += 1
    
    # Step 6: フィルタリング後のマッチリストを再構築
    # keep_maskに基づいて点を選別
    filtered_prev_pts = all_prev_pts[keep_mask]
    filtered_curr_pts = all_curr_pts[keep_mask]
    
    # マッチリスト形式に戻す（単一のマッチグループとして）
    if len(filtered_prev_pts) > 0:
        filtered_matches = [(filtered_prev_pts, filtered_curr_pts)]
    else:
        filtered_matches = []
        logger.warning(
            f"θ-セクター方向フィルタで全ての点が除外されました（元={original_count}点）"
        )
    
    # Step 7: 統計情報の集計
    filtered_count = len(filtered_prev_pts)
    removed_count = original_count - filtered_count
    removal_rate = (removed_count / original_count * 100) if original_count > 0 else 0.0
    
    # 全体のΔθ統計
    if len(delta_theta_all) > 0:
        delta_theta_q1_all = float(np.percentile(delta_theta_all, 25))
        delta_theta_median_all = float(np.median(delta_theta_all))
        delta_theta_q3_all = float(np.percentile(delta_theta_all, 75))
        delta_theta_iqr_all = delta_theta_q3_all - delta_theta_q1_all
        delta_theta_std_all = float(np.std(delta_theta_all))
    else:
        delta_theta_q1_all = 0.0
        delta_theta_median_all = 0.0
        delta_theta_q3_all = 0.0
        delta_theta_iqr_all = 0.0
        delta_theta_std_all = 0.0
    
    stats = {
        'original_count': int(original_count),
        'after_vp_circle_filter': int(original_count - vp_circle_removed),
        'vp_circle_removed': int(vp_circle_removed),
        'filtered_count': int(filtered_count),
        'removed_count': int(removed_count),
        'removal_rate': float(removal_rate),
        'n_sectors': int(n_sectors),
        'min_points_per_sector': int(min_points_per_sector),
        'sectors_processed': int(sectors_processed),
        'sectors_skipped': int(sectors_skipped),
        'per_sector_stats': per_sector_stats,
        'delta_theta_q1': delta_theta_q1_all,
        'delta_theta_median': delta_theta_median_all,
        'delta_theta_q3': delta_theta_q3_all,
        'delta_theta_iqr': delta_theta_iqr_all,
        'delta_theta_std': delta_theta_std_all
    }
    
    logger.info(
        f"θ-セクター方向フィルタ: {original_count}点 → {filtered_count}点 "
        f"(除外率={removal_rate:.1f}%, sectors={sectors_processed}/{n_sectors}処理)"
    )
    
    return filtered_matches, stats
