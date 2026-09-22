"""管内カメラカーシミュレーション用壁面画像生成処理

カメラ車による下水道管内検査作業の訓練用シミュレーターのための
カラーマップ生成処理のエントリーポイント。

動画ファイルからフレーム単位で以下の処理を実行:
1. 特徴点マッチングによるフレーム間移動量推定
2. 暗部検出によるカメラ姿勢推定
3. OCR距離読み取りとカメラ位置補正
4. カラーマップへの投影・合成
5. デバッグ可視化（オプション）

使用例:
    # デフォルト設定で実行
    $ python src/main.py --input data/input/videos/sample.mp4

    # カスタム設定で実行
    $ python src/main.py --input video.mp4 --config config.json

    # 処理範囲指定
    $ python src/main.py --input video.mp4 --start 100 --end 500

    # デバッグモード
    $ python src/main.py --input video.mp4 --debug

    # 主要パラメータ指定
    $ python src/main.py --input video.mp4 --aov 185.0 --pi 250.0
"""

# 標準ライブラリ
import argparse
import os
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from dataclasses import dataclass

# サードパーティライブラリ
import cv2
import numpy as np

# ローカルモジュール
from src.config import (
    Config,
    load_config,
)
from src.ocr_utils import (
    correct_camera_position_single_frame,
    OCRConfig,
)
from src.coordinate_transform import (
    FisheyeCamera,
    PinholeCamera,
    CoordinateTransformer,
    CameraModel,
)
from src.debug_visualizer import (
    DebugVisualizer,
    DebugConfig,
)
from src.feature_matching import FeatureMatcher
from src.camera_estimation import CameraEstimator
from src.colormap_generator import ColorMapGenerator
from src.progress_reporter import ProgressReporter
from src.image_size_correction import apply_camera_correction


# ============================================================================
# カスタム例外クラス
# ============================================================================

class MainProcessError(Exception):
    """メイン処理関連のベース例外"""
    pass


class VideoFileError(MainProcessError):
    """動画ファイル関連のエラー"""
    pass


class InitializationError(MainProcessError):
    """初期化エラー"""
    pass


# ============================================================================
# データクラス
# ============================================================================

@dataclass
class CameraState:
    """カメラの状態を表すデータクラス
    
    Attributes:
        position: カメラ位置 [x, y, z] (mm)
        orientation: カメラ姿勢 [roll, yaw, pitch] (ラジアン)
        frame_num: 現在のフレーム番号
        timestamp: タイムスタンプ（秒）
    """
    position: np.ndarray
    orientation: np.ndarray
    frame_num: int
    timestamp: float


# ============================================================================
# カラーマップ生成パイプライン
# ============================================================================

class ColorMapPipeline:
    """カラーマップ生成パイプライン
    
    動画ファイルからカラーマップを生成する全処理を管理します。
    各モジュール（特徴点マッチング、姿勢推定、カラーマップ生成等）を
    統合し、フレーム処理ループを制御します。
    
    Attributes:
        config: メイン設定
        logger: ロガー
        camera: カメラモデル（FisheyeCamera or PinholeCamera）
        camera_params: カメラパラメータ辞書
        transformer: 座標変換ユーティリティ
        debug_vis: デバッグ可視化ユーティリティ
        feature_matcher: 特徴点マッチング処理
        camera_estimator: カメラ姿勢・移動量推定
        colormap_gen: カラーマップ生成
        progress_rep: 進捗レポート
    """
    
    def __init__(self, config: Config):
        """初期化
        
        Args:
            config: メイン設定
            
        Raises:
            InitializationError: 初期化に失敗した場合
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # モジュール初期化（動画読み込み後に実施）
        self.camera: Optional[CameraModel] = None
        self.camera_params: Optional[Dict[str, Any]] = None
        self.transformer: Optional[CoordinateTransformer] = None
        self.debug_vis: Optional[DebugVisualizer] = None
        self.feature_matcher: Optional[FeatureMatcher] = None
        self.camera_estimator: Optional[CameraEstimator] = None
        self.colormap_gen: Optional[ColorMapGenerator] = None
        self.progress_rep: Optional[ProgressReporter] = None
    
    def _initialize_modules(self, first_frame: np.ndarray, total_frames: int) -> None:
        """モジュール初期化
        
        動画の最初のフレームを使用してカメラモデルとパラメータを初期化し、
        各処理モジュールをインスタンス化します。
        
        Args:
            first_frame: 動画の最初のフレーム (H, W, 3)
            total_frames: 総フレーム数
            
        Raises:
            InitializationError: 初期化に失敗した場合
        """
        try:
            # カメラモデルとパラメータを初期化
            self.camera, self.camera_params = self._initialize_camera_model(
                first_frame.shape
            )
            
            # 座標変換ユーティリティ
            pipe_radius = self.config.pipe.diameter_mm / 2.0
            self.transformer = CoordinateTransformer(
                self.camera, pipe_radius,
                calibration=self.config.camera._lens_calibration
            )
            
            # デバッグ可視化ユーティリティ
            self.debug_vis = DebugVisualizer(self.config.debug)
            
            # 特徴点マッチング
            self.feature_matcher = FeatureMatcher(
                self.config.estimation.feature_matching
            )
            
            # カメラ姿勢・移動量推定
            self.camera_estimator = CameraEstimator(
                self.config.estimation, self.transformer
            )
            
            # カラーマップ生成
            self.colormap_gen = ColorMapGenerator(
                self.config, self.transformer
            )
            
            # 進捗レポート
            if self.config.output.progress_path:
                self.progress_rep = ProgressReporter(
                    Path(self.config.output.progress_path.replace(
                    "{timestamp}", datetime.now().strftime("%Y%m%d_%H%M%S")
                ).replace(
                    "{process_id}", str(os.getpid())
                )), total_frames
                )
                self.progress_rep.initialize()
            
            self.logger.info("全モジュールの初期化が完了しました")
            
        except Exception as e:
            raise InitializationError(f"モジュール初期化に失敗しました: {e}") from e
    
    def _initialize_camera_model(
        self,
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[CameraModel, Dict[str, Any]]:
        """カメラモデルとパラメータを初期化（Phase 3: v2.0対応）

        LensCalibration v2.0形式のキャリブレーション結果を優先的に使用し、
        存在しない場合はv1.0形式（FOV+offset）にフォールバックします。

        Args:
            frame_shape: フレーム形状 (height, width, channels)

        Returns:
            (camera_model, camera_params): カメラモデルと計算済みパラメータ

        Raises:
            ValueError: カメラモデルが不正な場合
        """
        height, width = frame_shape[:2]

        # ========================================
        # Step 1: v2.0形式の値の有効性チェック
        # ========================================
        has_valid_v2_calibration = (
            self.config.camera.fx > 0 and
            self.config.camera.fy > 0 and
            self.config.camera.cx > 0 and
            self.config.camera.cy > 0
        )

        # ========================================
        # Step 2: 使用パラメータの決定
        # ========================================
        if has_valid_v2_calibration:
            # ⭐ v2.0形式を優先: 直接値を使用
            calib_width = self.config.camera.calibrated_image_width
            calib_height = self.config.camera.calibrated_image_height

            # 画像サイズ確認
            if calib_width > 0 and calib_height > 0:
                if width == calib_width and height == calib_height:
                    # サイズ一致: v2.0値を直接使用
                    f = self.config.camera.fx  # fx を使用
                    cx = self.config.camera.cx
                    cy = self.config.camera.cy
                    is_from_calibration = True
                    calibration_mode = "v2.0_direct"

                    self.logger.info(
                        f"カメラモデル初期化: v2.0形式（直接値使用）\n"
                        f"  f={f:.2f}px, cx={cx:.1f}, cy={cy:.1f}\n"
                        f"  frame_size=({width}x{height})"
                    )
                else:
                    # サイズ不一致: 変換タイプを自動検出して適切な補正を適用
                    correction = apply_camera_correction(
                        fx=self.config.camera.fx,
                        cx=self.config.camera.cx,
                        cy=self.config.camera.cy,
                        calib_width=calib_width,
                        calib_height=calib_height,
                        frame_width=width,
                        frame_height=height,
                        logger=self.logger
                    )
                    
                    f = correction.f
                    cx = correction.cx
                    cy = correction.cy
                    is_from_calibration = True
                    
                    # キャリブレーションモードを変換タイプに応じて設定
                    if correction.transformation_type == "trim":
                        calibration_mode = "v2.0_trimmed"
                    elif correction.transformation_type == "resize":
                        calibration_mode = "v2.0_scaled"
                    else:
                        calibration_mode = "v2.0_corrected"
                    
                    # ログ出力は apply_camera_correction 内で実施済み
            else:
                # キャリブレーションサイズ不明: v2.0値を直接使用（警告付き）
                f = self.config.camera.fx
                cx = self.config.camera.cx
                cy = self.config.camera.cy
                is_from_calibration = True
                calibration_mode = "v2.0_direct_no_size"

                self.logger.warning(
                    f"カメラモデル初期化: v2.0形式（画像サイズ不明）\n"
                    f"  キャリブレーション時の画像サイズが記録されていません。\n"
                    f"  v2.0値を直接使用しますが、サイズ不一致の可能性があります。\n"
                    f"  f={f:.2f}px, cx={cx:.1f}, cy={cy:.1f}"
                )
        else:
            # ========================================
            # フォールバック: v1.0形式（従来方式）
            # ========================================
            theta_max = np.radians(self.config.camera.fov_degrees / 2.0)
            cx = width // 2 + self.config.camera.center_offset_x
            cy = height // 2 + self.config.camera.center_offset_y
            radius = min(cx, cy)
            f = radius / theta_max
            is_from_calibration = False
            calibration_mode = "v1.0_fallback"

            self.logger.info(
                f"カメラモデル初期化: v1.0形式（FOV+offsetから計算）\n"
                f"  fov={self.config.camera.fov_degrees}°, "
                f"offset=({self.config.camera.center_offset_x}, {self.config.camera.center_offset_y})\n"
                f"  f={f:.2f}px, cx={cx:.1f}, cy={cy:.1f}"
            )

        # ========================================
        # Step 3: カメラモデル作成
        # ========================================
        center = (cx, cy)
        camera_params_model = self.config.camera.camera_model or "fisheye"
        if camera_params_model == "pinhole":
            camera = PinholeCamera(
                f=f,
                cx=cx,
                cy=cy,
                image_width=width,
                image_height=height
            )
        else:
            camera_params_model = "fisheye"
            camera = FisheyeCamera(
                f=f,
                cx=cx,
                cy=cy,
                image_width=width,
                image_height=height
            )

        # ========================================
        # Step 4: camera_params辞書作成（拡張）
        # ========================================
        camera_params = {
            'f': f,
            'center': center,
            'radius': min(cx, cy),
            'model': camera_params_model,
            # ⭐ Phase 3追加: キャリブレーション情報
            'from_calibration': is_from_calibration,
            'calibration_mode': calibration_mode
        }

        return camera, camera_params
    
    def _process_frame_pair(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        camera_state: CameraState
    ) -> np.ndarray:
        """フレームペア処理（移動量推定）
        
        前フレームと現フレームから特徴点マッチングとカメラ姿勢推定を行い、
        カメラの移動量を計算します。
        
        Args:
            prev_frame: 前フレーム (H, W, 3)
            curr_frame: 現フレーム (H, W, 3)
            camera_state: カメラ状態
            
        Returns:
            movement: 移動量 [dx, dy, dz, droll, dyaw, dpitch] (mm, ラジアン)
        """
        # (1) 暗部検出による姿勢推定
        yaw_est = None
        pitch_est = None

        if self.config.estimation.pose.use_vanishing_point:
            try:
                # 適応的閾値選択パラメータを取得
                use_adaptive = self.config.estimation.pose.use_adaptive_threshold
                target_area = self.config.estimation.pose.target_dark_area if use_adaptive else None
                threshold_range = tuple(self.config.estimation.pose.threshold_search_range) if use_adaptive else (1, 100)
                dark_threshold = self.config.estimation.pose.dark_threshold

                # 暗部検出の半径制限を計算（外周のブラックエリアを除外）
                radius_limit = (self.camera.image_width / 2.0) * self.config.estimation.pose.dark_region_radius_limit_ratio

                yaw_est, pitch_est = self.camera_estimator.estimate_camera_pose(
                    curr_frame,
                    camera_state.orientation[0],  # roll
                    threshold=dark_threshold,
                    radius_limit=radius_limit,
                    target_area=target_area,
                    threshold_range=threshold_range
                )

                if yaw_est is not None and pitch_est is not None:
                    # 推定姿勢で更新
                    camera_state.orientation[1] = yaw_est
                    camera_state.orientation[2] = pitch_est

                    self.logger.debug(
                        f"姿勢推定成功: yaw={np.degrees(yaw_est):.2f}°, "
                        f"pitch={np.degrees(pitch_est):.2f}°"
                    )
            except Exception as e:
                self.logger.warning(f"暗部検出による姿勢推定失敗: {e}")
        
        # (2) 特徴点マッチング
        prev_points, curr_points = self.feature_matcher.detect_and_match(
            prev_frame, curr_frame, self.camera_params
        )
        
        # マッチング失敗時は移動なしとする
        if prev_points is None or curr_points is None:
            self.logger.warning("特徴点マッチング失敗、移動量をゼロとします")
            return np.zeros(6)
        
        # (3) カメラ状態に推定姿勢を追加（CameraEstimator.estimate_motionで使用）
        camera_state_dict = {
            'position': camera_state.position.copy(),
            'orientation': camera_state.orientation.copy(),
        }
        
        if yaw_est is not None:
            camera_state_dict['yaw_estimated'] = yaw_est
        if pitch_est is not None:
            camera_state_dict['phi_estimated'] = pitch_est
        
        # (4) 移動量推定
        try:
            motion = self.camera_estimator.estimate_motion(
                prev_points, curr_points, camera_state_dict, self.camera_params
            )
            
            # 辞書形式からnumpy配列に変換
            movement = np.array([
                motion['dx'],
                motion['dy'],
                motion['dz'],
                motion['droll'],
                motion['dtheta'],
                motion['dphi']
            ])
            
            self.logger.debug(
                f"移動量推定: dx={motion['dx']:.2f}, dy={motion['dy']:.2f}, "
                f"dz={motion['dz']:.2f}, droll={np.degrees(motion['droll']):.2f}°"
            )
            
            return movement
            
        except Exception as e:
            self.logger.warning(f"移動量推定失敗: {e}, 移動量をゼロとします")
            return np.zeros(6)
    
    def _process_ocr_correction(
        self,
        curr_frame: np.ndarray,
        camera_state: CameraState,
        movement: np.ndarray,
        prev_z_corrected: Optional[float],
        prev_z2_ocr: Optional[float]
    ) -> Tuple[float, float]:
        """OCR距離読み取りとカメラ位置補正
        
        Args:
            curr_frame: 現フレーム (H, W, 3)
            camera_state: カメラ状態
            movement: 移動量 [dx, dy, dz, droll, dyaw, dpitch]
            prev_z_corrected: 前フレームの補正後z座標 (mm)
            prev_z2_ocr: 前フレームのOCR読み取り距離 (mm)
            
        Returns:
            (z_star, z2_ocr): 補正後z座標、OCR読み取り距離 (mm)
        """
        if not self.config.ocr.enabled:
            # OCR無効時は推定値をそのまま使用
            z_star = camera_state.position[2]
            z2_ocr = z_star
            return z_star, z2_ocr
        
        # OCR距離読み取り・補正
        ocr_roi = tuple(self.config.camera.ocr_roi_ratio)
        ocr_config = OCRConfig(
            threshold=100,
            psm_mode=6,
            retry_thresholds=[100, 120, 80, 150]
        )
        
        z1_ocr = (prev_z_corrected if prev_z_corrected is not None else 0.0) + movement[2]
        
        try:
            z_star, z2_ocr = correct_camera_position_single_frame(
                curr_frame,
                z1_ocr,
                ocr_roi,
                prev_z_corrected,
                prev_z2_ocr,
                ocr_config
            )
        except Exception as e:
            self.logger.warning(f"OCR補正失敗: {e}, 推定値を使用")
            z_star = camera_state.position[2]
            z2_ocr = prev_z2_ocr if prev_z2_ocr is not None else z_star
        
        return z_star, z2_ocr
    
    def _generate_colormap_frame(
        self,
        curr_frame: np.ndarray,
        camera_state: CameraState,
        color_map_img: Optional[np.ndarray]
    ) -> np.ndarray:
        """カラーマップ生成（1フレーム追加）
        
        Args:
            curr_frame: 現フレーム (H, W, 3)
            camera_state: カメラ状態
            color_map_img: 既存のカラーマップ画像（Noneの場合は新規作成）
            
        Returns:
            更新されたカラーマップ画像
        """
        # カメラ状態を辞書形式に変換
        camera_state_dict = {
            'position': camera_state.position,
            'orientation': camera_state.orientation
        }
        
        # カラーマップに1フレーム追加
        color_map_img = self.colormap_gen.add_frame(
            curr_frame,
            camera_state_dict,
            self.camera_params,
            color_map_img,
            max_dz=self.config.estimation.motion.max_dz
        )
        
        return color_map_img
    
    def _debug_visualize(
        self,
        curr_frame: np.ndarray,
        camera_state: CameraState,
        movement: np.ndarray,
        z2_ocr: float,
        z_star: float,
        z0_ocr: float
    ) -> bool:
        """デバッグ可視化
        
        Args:
            curr_frame: 現フレーム (H, W, 3)
            camera_state: カメラ状態
            movement: 移動量 [dx, dy, dz, droll, dyaw, dpitch]
            z2_ocr: OCR読み取り距離 (mm)
            z_star: 補正後z座標 (mm)
            z0_ocr: 初期OCR距離 (mm)
            
        Returns:
            True: 処理継続、False: ユーザー中断（'q'キー押下）
        """
        if not self.config.debug.enabled:
            return True
        
        debug_frame = curr_frame.copy()
        
        # グリッド描画
        pipe_radius = self.config.pipe.diameter_mm / 2.0
        debug_frame = self.debug_vis.draw_grid(
            debug_frame,
            self.camera_params,
            pipe_radius,
            camera_state.position,
            camera_state.orientation[0],
            camera_state.orientation[1],
            camera_state.orientation[2]
        )
        
        # テキスト情報オーバーレイ
        info = {
            'frame_num': camera_state.frame_num,
            'cam_pos': camera_state.position,
            'cam_orient': camera_state.orientation,
            'movement': movement,
            'ocr_distance': z2_ocr - z0_ocr,
            'corrected_distance': z_star
        }
        debug_frame = self.debug_vis.add_text_overlay(debug_frame, info)
        
        # フレーム表示
        if not self.debug_vis.show_frame("Camera Car Simulator", debug_frame):
            self.logger.info("ユーザーによる中断（'q'キー押下）")
            return False
        
        # コンソール出力
        self.debug_vis.print_frame_info(
            camera_state.frame_num,
            movement,
            np.concatenate([camera_state.position, camera_state.orientation]),
            z2_ocr - z0_ocr,
            z_star
        )
        
        return True
    
    def run(self) -> None:
        """メイン処理実行
        
        動画ファイルを読み込み、フレーム処理ループを実行します。
        
        Raises:
            VideoFileError: 動画ファイルの読み込みに失敗した場合
            MainProcessError: 処理中にエラーが発生した場合
        """
        # 動画読み込み
        video_path = self.config.input.video_path
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise VideoFileError(f"動画ファイルを開けません: {video_path}")
        
        # 動画情報取得
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"動画情報: {width}x{height}, {fps:.2f}fps, {total_frames}フレーム")
        
        # 処理範囲決定
        start_frame = self.config.input.start_frame
        end_frame = (
            self.config.input.end_frame 
            if self.config.input.end_frame is not None 
            else total_frames
        )
        
        self.logger.info(f"処理範囲: フレーム {start_frame} 〜 {end_frame}")
        
        # 最初のフレームを読み込んでモジュール初期化
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise VideoFileError("最初のフレームを読み込めません")
        
        self._initialize_modules(first_frame, total_frames)
        
        # カメラ状態初期化
        camera_state = CameraState(
            position=np.array([
                self.config.initial_position.x,
                self.config.initial_position.y,
                self.config.initial_position.z
            ]),
            orientation=np.array([
                np.radians(self.config.initial_position.theta),
                np.radians(self.config.initial_position.phi),
                0.0  # roll（今後、Configに追加予定）
            ]),
            frame_num=start_frame,
            timestamp=start_frame / fps
        )
        
        # OCR補正用変数
        prev_z_corrected: Optional[float] = None
        prev_z2_ocr: Optional[float] = None
        z0_ocr = 0.0
        
        # フレーム処理ループ
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        prev_frame: Optional[np.ndarray] = None
        color_map_img: Optional[np.ndarray] = None
        
        self.logger.info("フレーム処理ループを開始します")
        
        for frame_num in range(start_frame, end_frame):
            # フレーム読み取り
            ret, curr_frame = cap.read()
            if not ret:
                self.logger.warning(f"フレーム {frame_num} の読み込みに失敗、処理を終了")
                break
            
            camera_state.frame_num = frame_num
            camera_state.timestamp = frame_num / fps
            
            # (1) フレームペア処理（移動量推定）
            if prev_frame is not None:
                movement = self._process_frame_pair(prev_frame, curr_frame, camera_state)
                
                # 姿勢更新
                camera_state.position += movement[:3]
                camera_state.orientation += movement[3:]
            else:
                movement = np.zeros(6)
            
            # (2) OCR距離読み取り・補正
            z_star, z2_ocr = self._process_ocr_correction(
                curr_frame, camera_state, movement, prev_z_corrected, prev_z2_ocr
            )
            
            # 初期OCR距離記録（最初のフレームのみ）
            if frame_num == start_frame:
                z0_ocr = z2_ocr
            
            prev_z_corrected = z_star
            prev_z2_ocr = z2_ocr
            
            # (3) カラーマップ生成
            color_map_img = self._generate_colormap_frame(
                curr_frame, camera_state, color_map_img
            )
            
            # (4) デバッグ可視化
            if not self._debug_visualize(
                curr_frame, camera_state, movement, z2_ocr, z_star, z0_ocr
            ):
                # ユーザー中断
                break
            
            # (5) 進捗更新（10フレームごと）
            if self.progress_rep and (frame_num - start_frame + 1) % 10 == 0:
                current_frame_info = {
                    "frame_index": frame_num,
                    "camera_position": camera_state.position.tolist(),
                    "camera_orientation": camera_state.orientation.tolist()
                }
                self.progress_rep.update(
                    frame_num - start_frame + 1,
                    current_frame_info
                )
            
            # 次フレーム用に保存
            prev_frame = curr_frame
            
            # 進捗ログ（100フレームごと）
            if (frame_num - start_frame + 1) % 100 == 0:
                processed = frame_num - start_frame + 1
                total = end_frame - start_frame
                progress_percent = (processed / total) * 100
                self.logger.info(
                    f"進捗: {processed}/{total} フレーム ({progress_percent:.1f}%)"
                )
        
        # リソース解放
        cap.release()
        if self.debug_vis:
            self.debug_vis.close()
        
        # カラーマップ保存
        self._save_colormap(color_map_img)
        
        # 完了通知
        if self.progress_rep:
            self.progress_rep.finalize(success=True)
        
        self.logger.info("カラーマップ生成処理が完了しました")
    
    def _save_colormap(self, color_map_img: Optional[np.ndarray]) -> None:
        """カラーマップを保存
        
        Args:
            color_map_img: カラーマップ画像
        """
        if color_map_img is None:
            self.logger.warning("カラーマップ画像がNullのため保存をスキップ")
            return
        
        # タイムスタンプ生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 出力パス生成（{timestamp}プレースホルダを置換）
        output_path = Path(
            self.config.output.colormap_path.format(timestamp=timestamp)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存（matplotlibでカラースケール保持）
        try:
            import matplotlib.pyplot as plt
            plt.imsave(str(output_path), color_map_img)
            self.logger.info(f"カラーマップを保存しました: {output_path}")
        except ImportError:
            # matplotlibがない場合はcv2で保存
            cv2.imwrite(str(output_path), color_map_img)
            self.logger.info(
                f"カラーマップを保存しました（cv2使用）: {output_path}"
            )


# ============================================================================
# コマンドライン引数解析
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数をパース
    
    Returns:
        パース結果
    """
    parser = argparse.ArgumentParser(
        description='管内カメラカーシミュレーション用カラーマップ生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で実行
  python src/main.py --input data/input/videos/sample.mp4

  # カスタム設定で実行
  python src/main.py --input video.mp4 --config config.json

  # 処理範囲指定
  python src/main.py --input video.mp4 --start 100 --end 500

  # デバッグモード
  python src/main.py --input video.mp4 --debug

  # 主要パラメータ指定
  python src/main.py --input video.mp4 --aov 185.0 --pi 250.0
        """
    )
    
    # 必須引数
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='入力動画ファイルパス'
    )
    
    # オプション引数（設定ファイル）
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='設定ファイルパス（JSON形式）'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='出力カラーマップパス'
    )
    
    # 処理範囲
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='処理開始フレーム番号'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        help='処理終了フレーム番号'
    )
    
    # デバッグモード
    parser.add_argument(
        '--debug',
        action='store_true',
        help='デバッグモードを有効化'
    )
    
    # Legacy実装との互換性のため、主要パラメータをCLI引数でも受け付ける
    parser.add_argument(
        '--aov',
        type=float,
        help='画角（度）'
    )
    
    parser.add_argument(
        '--pi',
        type=float,
        help='管径（mm）'
    )
    
    # 今後、必要に応じて他のlegacyパラメータも追加
    # --cx, --cy, --outer, --dx, --dy, --dz, --dr, --dp, --dt, etc.
    
    return parser.parse_args()


# ============================================================================
# ログ設定
# ============================================================================

def setup_logging(config: Config) -> None:
    """ログ設定を初期化
    
    Args:
        config: メイン設定
    """
    log_level = getattr(logging, config.logging.level)
    log_format = config.logging.format
    
    handlers = []
    
    # ファイルハンドラ
    if config.logging.file:
        log_file = Path(config.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_file, encoding='utf-8')
        )
    
    # コンソールハンドラ
    if config.logging.console_output:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )


# ============================================================================
# メインエントリーポイント
# ============================================================================

def main() -> int:
    """メインエントリーポイント
    
    Returns:
        終了コード（0: 成功、1: 失敗）
    """
    try:
        # 引数パース
        args = parse_arguments()
        
        # 設定読み込み
        config = load_config(str(args.config) if args.config else None)
        
        # CLI引数で設定を上書き
        if args.input:
            config.input.video_path = str(args.input)
        if args.output:
            config.output.colormap_path = str(args.output)
        if args.start is not None:
            config.input.start_frame = args.start
        if args.end is not None:
            config.input.end_frame = args.end
        if args.debug:
            config.debug.enabled = True
        if args.aov is not None:
            config.camera.fov_degrees = args.aov
        if args.pi is not None:
            config.pipe.diameter_mm = args.pi
        
        # バリデーション
        config.validate()
        
        # ログ設定
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 60)
        logger.info("管内カメラカーシミュレーション用壁面画像生成処理")
        logger.info("=" * 60)
        logger.info(f"入力動画: {config.input.video_path}")
        logger.info(f"画角: {config.camera.fov_degrees}度")
        logger.info(f"管径: {config.pipe.diameter_mm}mm")
        logger.info(f"デバッグモード: {config.debug.enabled}")
        logger.info("=" * 60)
        
        # パイプライン実行
        pipeline = ColorMapPipeline(config)
        pipeline.run()
        
        logger.info("=" * 60)
        logger.info("正常終了")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n処理が中断されました（Ctrl+C）", file=sys.stderr)
        return 1
        
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        logging.getLogger(__name__).exception("予期しないエラーが発生しました")
        return 1


if __name__ == '__main__':
    sys.exit(main())
