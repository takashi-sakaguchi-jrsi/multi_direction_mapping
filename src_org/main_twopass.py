"""管内カメラカーシミュレーション用壁面画像生成処理（2パスフロー版）

カメラ車による下水道管内検査作業の訓練用シミュレーターのための
カラーマップ生成処理のエントリーポイント。

2パス処理フロー:
  Phase1: 全フレーム事前処理（消失点スムージング + OCR高精度推定）
  Phase2: カラーマップ生成（制約付き推定）

動画ファイルからフレーム単位で以下の処理を実行:
1. Phase1: 全フレームの消失点検出とOCR読み取り
2. Phase1: 探索範囲制約の計算
3. Phase2: 制約付き移動量推定
4. Phase2: カラーマップへの投影・合成
5. Phase2: デバッグ可視化（オプション）

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
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime
from collections import deque
from dataclasses import dataclass

# サードパーティライブラリ
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    convert_pixel_to_pose,
)
from src.debug_visualizer import (
    DebugVisualizer,
    DebugConfig,
)
from src.feature_matching import FeatureMatcher
from src.camera_estimation import CameraEstimator, compute_frame_constraints
from src.colormap_generator import ColorMapGenerator
from src.colormap_correction import ColormapCorrector, ColormapCorrectionError
from src.progress_reporter import ProgressReporter
from src.camera_utils import compute_fisheye_focal_length
from src.auto_tuner import AutoTuner
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
# カラーマップ生成パイプライン（2パス処理版）
# ============================================================================

class ColorMapPipelineTwoPass:
    """カラーマップ生成パイプライン（2パス処理版）
    
    動画ファイルからカラーマップを生成する全処理を管理します。
    Phase1で全フレームを事前処理し、Phase2でカラーマップを生成します。
    
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
        self._timestamp: Optional[str] = None  # BUG-008修正: 処理開始時タイムスタンプ
        
        # モジュール初期化（動画読み込み後に実施）
        self.camera: Optional[CameraModel] = None
        self.camera_params: Optional[Dict[str, Any]] = None
        self.transformer: Optional[CoordinateTransformer] = None
        self.debug_vis: Optional[DebugVisualizer] = None
        self.feature_matcher: Optional[FeatureMatcher] = None
        self.camera_estimator: Optional[CameraEstimator] = None
        self.colormap_gen: Optional[ColorMapGenerator] = None
        self.progress_rep: Optional[ProgressReporter] = None

        # OCRエンジンパス設定（環境変数展開対応: %USERPROFILE%, $HOME 等）
        if self.config.ocr.tesseract_cmd:
            import pytesseract
            resolved = os.path.expandvars(self.config.ocr.tesseract_cmd)
            pytesseract.pytesseract.tesseract_cmd = resolved
            self.logger.info(f"Tesseractパス設定: {resolved}")

    def _get_video_total_frames(self) -> int:
        """動画の総フレーム数を取得
        
        動画ファイルをOpenCVで開き、総フレーム数を取得します。
        この関数は`end_frame=None`の場合に呼び出され、動画全体のフレーム数を
        自動検出するために使用されます。
        
        Returns:
            動画の総フレーム数
            
        Raises:
            FileNotFoundError: 動画ファイルが存在しない場合
            RuntimeError: 動画ファイルを開けない場合
            ValueError: 総フレーム数が0または異常値の場合
            
        Note:
            - OpenCVのCAP_PROP_FRAME_COUNTを使用して総フレーム数を取得
            - 動画ファイルは開いた後すぐにクローズされる（メモリ節約）
            - 検出した総フレーム数はログに記録される
        """
        video_path = str(self.config.input.video_path)
        
        # ファイル存在チェック
        if not os.path.exists(video_path):
            error_msg = f"動画ファイルが見つかりません: {video_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 動画を開く
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"動画ファイルを開けません（コーデックの問題の可能性）: {video_path}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # 総フレーム数を取得
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # フレーム数の妥当性チェック
        if total_frames == 0:
            error_msg = f"動画の総フレーム数が0です: {video_path}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if total_frames < 0:
            error_msg = f"動画の総フレーム数が負の値です: {total_frames}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if total_frames > 10000000:  # 1000万フレーム以上は異常値として扱う
            self.logger.warning(
                f"動画の総フレーム数が異常に大きい値です: {total_frames} "
                "(処理に時間がかかる可能性があります)"
            )
        
        self.logger.info(f"動画の総フレーム数を検出: {total_frames}")
        return total_frames

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
                self.config.estimation,
                self.transformer,
                pixels_per_mm=self.config.colormap.pixels_per_mm
            )
            
            # カラーマップ生成
            self.colormap_gen = ColorMapGenerator(
                self.config, self.transformer
            )
            
            # 進捗レポート
            # ProgressReporter初期化（外部から設定済みでなければ生成）
            if self.progress_rep is None and self.config.output.progress_path:
                # end_frameがNoneの場合は動画の総フレーム数を自動検出
                effective_end_frame = self.config.input.end_frame
                if effective_end_frame is None:
                    effective_end_frame = self._get_video_total_frames()
                
                # 処理対象のフレーム数を計算
                total_frames = effective_end_frame - self.config.input.start_frame + 1
                
                self.progress_rep = ProgressReporter(
                    output_path=Path(self.config.output.progress_path.replace(
                        '{process_id}',
                        self._timestamp
                    )),
                    total_frames=total_frames
                )

                # 4ステップモードで初期化
                steps = [
                    {"step_id": 1, "name_ja": "パラメータ調整", "name_en": "Parameter Tuning"},
                    {"step_id": 2, "name_ja": "フレーム解析", "name_en": "Frame Analysis"},
                    {"step_id": 3, "name_ja": "カラーマップ生成", "name_en": "Colormap Generation"},
                    {"step_id": 4, "name_ja": "カラーマップ補正", "name_en": "Colormap Correction"}
                ]
                self.progress_rep.initialize_steps(steps)
                self.logger.info("ProgressReporter初期化完了（4ステップモード）")

            
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
            f = compute_fisheye_focal_length(
                self.config.camera.fov_degrees,
                (height, width)
            )
            cx = width // 2 + self.config.camera.center_offset_x
            cy = height // 2 + self.config.camera.center_offset_y
            is_from_calibration = False
            calibration_mode = "v1.0_fallback"

            self.logger.info(
                f"カメラモデル初期化: v1.0形式（{self.config.camera.camera_model}モデル）\n"
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
            from src.coordinate_transform import PinholeCamera
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
            'calibration_mode': calibration_mode,
            # 管径: 円筒座標グリッド生成に必要（デフォルト250mmと乖離する場合に過大評価の原因となる）
            'pipe_diameter': self.config.pipe.diameter_mm,
        }

        return camera, camera_params
    
    def _compute_constraints_phase1(
        self,
        video_path: Path,
        start_frame: int,
        end_frame: int,
        progress_callback=None
    ) -> Tuple[Dict[int, Dict[str, Tuple[float, float]]], Optional[np.ndarray], Optional[np.ndarray]]:
        """Phase1: 全フレームの探索範囲制約を計算

        Args:
            video_path: 動画ファイルパス
            start_frame: 開始フレーム番号
            end_frame: 終了フレーム番号


        Returns:
            Tuple[constraints_dict, z_positions]:
                - constraints_dict: フレームインデックスごとの制約辞書
                - z_positions: 各フレームのOCR距離推定値（z*）配列

        Raises:
            MainProcessError: 制約計算失敗
        """
        self.logger.info("=" * 80)
        self.logger.info("Phase1: 全フレーム事前処理開始")
        self.logger.info("=" * 80)

        ocr_roi_ratio = tuple(self.config.camera.ocr_roi_ratio)

        try:
            constraints_dict, z_positions, vanishing_points = compute_frame_constraints(
                video_path=video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                estimator=self.camera_estimator,
                ocr_roi_ratio=ocr_roi_ratio,
                config=self.config.estimation,
                ocr_tesseract_config=self.config.ocr.tesseract_config,
                ocr_preprocessing_enabled=self.config.ocr.preprocessing_enabled,
                ocr_enabled=self.config.ocr.enabled,
                progress_callback=progress_callback,
                max_distance_increment_mm=self.config.ocr.max_distance_increment_mm
            )


            # Phase1終了時に消失点履歴を保存（中間ファイル保存有効時のみ）
            if self.config.debug.enabled and self.config.debug.save_intermediate and vanishing_points is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_dir = Path("data/output/debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_file = debug_dir / f"vp_smooth_phase1_{timestamp}.npy"
                np.save(debug_file, vanishing_points)
                self.logger.info(
                    f"Phase1消失点履歴を保存: {debug_file} ({len(vanishing_points)}フレーム)"
                )
            self.logger.info("Phase1完了: 探索範囲制約計算完了")

            return constraints_dict, z_positions, vanishing_points

        except Exception as e:
            self.logger.error(f"Phase1で制約計算失敗: {e}")
            # フォールバック: 広い範囲を設定
            n_frames = end_frame - start_frame
            fallback_constraints = {}
            max_dz = self.config.estimation.motion.max_dz
            for i in range(n_frames):
                fallback_constraints[i] = {
                    'dz': (0.0, max_dz),
                    'dyaw': (-0.1, 0.1),
                    'dpitch': (-0.1, 0.1),
                }

            self.logger.warning("Phase1失敗、広い範囲を使用します")
            return fallback_constraints, None, None
    

    def _compute_ocr_local_avg_dz(
        self,
        z_positions: Optional[np.ndarray],
        window: int = 50
    ) -> Optional[np.ndarray]:
        """OCR距離値から局所的な平均dz（mm/frame）を計算

        Args:
            z_positions: Phase1のOCR距離推定値配列 (N,) [正規化済み]
            window: 平均化ウィンドウサイズ（フレーム数）

        Returns:
            局所平均dz配列 (N,)。OCR無効の場合はNone
        """
        if z_positions is None or len(z_positions) < 2:
            return None

        # OCRデータが実質無い場合（全て0）
        if np.all(z_positions == 0):
            return None

        # フレーム間dz
        dz_per_frame = np.diff(z_positions)  # (N-1,)

        # 負のdz（逆走）を0にクランプ（前進のみ想定）
        dz_per_frame = np.maximum(dz_per_frame, 0.0)

        # 移動平均で局所平均を計算
        from scipy.ndimage import uniform_filter1d
        local_avg = uniform_filter1d(dz_per_frame, size=min(window, len(dz_per_frame)))

        # z_positionsと同じ長さに揃える（末尾を複製）
        local_avg = np.concatenate([local_avg, [local_avg[-1]]])

        return local_avg

    def _generate_colormap_phase2(
        self,
        video_path: Path,
        actual_start_frame: int,
        end_frame: int,
        user_start_frame: int,
        fps: float,
        constraints_dict: Dict[int, Dict[str, Tuple[float, float]]],
        initial_z_star: Optional[float] = None,
        z_positions: Optional[np.ndarray] = None,
        vanishing_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Phase2: カラーマップ生成

        Args:
            video_path: 動画ファイルパス
            actual_start_frame: 実際の処理開始フレーム番号（user_start_frame - 1）
            end_frame: 終了フレーム番号
            user_start_frame: ユーザー指定の開始フレーム番号（カラーマップ展開開始）
            fps: フレームレート
            constraints_dict: Phase1で計算した制約辞書
            initial_z_star: 開始フレーム（user_start_frame）のOCR距離推定値（z*）
            z_positions: 各フレームのOCR距離推定値（z*）配列

        Returns:
            color_map_img: 生成されたカラーマップ画像

        Raises:
            MainProcessError: カラーマップ生成失敗
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Phase2: カラーマップ生成開始")
        self.logger.info("=" * 80)

        # 初期z座標の決定（Phase1のz*を優先）


        # BUG-003修正: use_ocr_z_constraintsパラメータチェック追加（箇所2: 初期z値設定）
        if self.config.estimation.use_ocr_z_constraints and initial_z_star is not None:
            initial_z = initial_z_star
            self.logger.info(
                f"初期z座標をPhase1のOCR距離推定値に設定: z={initial_z:.2f}mm"
            )
        else:
            initial_z = self.config.initial_position.z
            self.logger.info(
                f"初期z座標をconfig設定値に設定: z={initial_z:.2f}mm"
            )


        # Phase1のVP推定値から初期yaw/pitchを計算
        try:
            # 開始-1フレームのVP推定値（スムージング済み）
            vp_x, vp_y = vanishing_points[0]

            # VP座標からyaw/pitchを計算
            initial_yaw, initial_pitch = convert_pixel_to_pose(
                vp_x, vp_y, self.camera, np.radians(self.config.initial_position.theta)
            )

            self.logger.info(
                f"初期姿勢角をVP推定値で初期化: yaw={np.degrees(initial_yaw):.2f}°, "
                f"pitch={np.degrees(initial_pitch):.2f}°"
            )
        except (IndexError, ValueError, TypeError) as e:
            # VP推定失敗時はconfig値を使用
            initial_yaw = np.radians(self.config.initial_position.theta)
            initial_pitch = np.radians(self.config.initial_position.phi)
            self.logger.warning(
                f"VP推定値からの初期化に失敗、config値を使用: {e}"
            )

        # カメラ状態初期化（actual_start_frameで初期化）
        camera_state = CameraState(
            position=np.array([
                self.config.initial_position.x,
                self.config.initial_position.y,
                initial_z
            ]),
            orientation=np.array([
                np.radians(self.config.initial_position.theta),  # roll
                initial_yaw,   # yaw - VP推定値から計算
                initial_pitch  # pitch - VP推定値から計算
            ]),
            frame_num=actual_start_frame,
            timestamp=actual_start_frame / fps
        )

        # 動画再オープン
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoFileError(f"動画ファイルを再オープンできません: {video_path}")

        # 実際の処理開始フレーム（actual_start_frame）から読み込み開始
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start_frame)
        
        prev_frame: Optional[np.ndarray] = None
        color_map_img: Optional[np.ndarray] = None
        first_valid_frame_found = False  # 開始フレーム先送り判定用

        # dz_minクリップ統計
        dz_min_clipped_count = 0
        ocr_constraint_violation_count = 0

        # カラーマップ生成統計
        colormap_generated_count = 0
        skipped_static_count = 0
        skipped_failed_count = 0
        skipped_dz_zero_count = 0

        # フレームごとのデータ記録用リスト
        frame_data_list = []

        # 前フレームのz*を保持（dz*計算用）
        # 初期値をPhase1のactual_start_frameのz*で初期化
        # BUG-003修正: use_ocr_z_constraintsパラメータチェック追加（箇所3: z_star_prev初期化）
        if self.config.estimation.use_ocr_z_constraints and \
           z_positions is not None and len(z_positions) > 0:
            z_star_prev = z_positions[0]  # actual_start_frameのz*
        else:
            z_star_prev = None

        self.logger.info("フレーム処理ループを開始します")

        # 前フレームdz（適応的dz制約・dz_hint用）
        prev_dz = None
        # フォールバックdz用の履歴バッファ（直近10フレームの中央値をスキップ時に使用）
        dz_history: deque = deque(maxlen=10)

        # OCR局所平均dz計算（Hybrid Stage 1用）
        ocr_local_avg_dz = None
        if self.config.estimation.motion.hybrid_two_stage_enabled:
            ocr_local_avg_dz = self._compute_ocr_local_avg_dz(z_positions)
            if ocr_local_avg_dz is not None:
                self.logger.info(
                    f"OCR局所平均dz計算完了: "
                    f"mean={np.mean(ocr_local_avg_dz):.2f}mm/frame, "
                    f"min={np.min(ocr_local_avg_dz):.2f}, max={np.max(ocr_local_avg_dz):.2f}"
                )
            else:
                self.logger.info("OCRデータなし: Hybrid Stage 1はmax_dz制約なしで動作")

        _step3_last_pct = -1.0

        for frame_num in range(actual_start_frame, end_frame):
            # フレーム読み取り
            ret, curr_frame = cap.read()
            if not ret:
                self.logger.warning(f"フレーム {frame_num} の読み込みに失敗、処理を終了")
                break

            camera_state.frame_num = frame_num
            camera_state.timestamp = frame_num / fps

            # フレームインデックス（actual_start_frameを基準とした0始まり）
            frame_idx = frame_num - actual_start_frame

            # (0) 進捗更新（1%単位で間引き、continue前に実行して確実に更新）
            if self.progress_rep:
                progress = ((frame_num - actual_start_frame + 1) / (end_frame - actual_start_frame)) * 100.0
                if (progress - _step3_last_pct >= 1.0
                        or frame_num >= end_frame - 1):
                    self.progress_rep.update_step(3, progress, {
                        "frame_index": frame_num,
                        "camera_position": camera_state.position.tolist(),
                        "camera_orientation": camera_state.orientation.tolist()
                    })
                    _step3_last_pct = progress

            # 現フレームのz*を取得（Phase1で計算済み、OCR推定値）
            # ★dz制約計算とCSV出力の両方で使用
            z_star_ocr = None  # OCR推定値（Phase1で計算）
            # BUG-007修正: Excel出力とPhase3補正のため、use_ocr_z_constraintsに関わらずOCRデータを取得
            if z_positions is not None and frame_idx < len(z_positions):
                z_star_ocr = z_positions[frame_idx]

            # Phase1のスムージング済み消失点から姿勢角を計算（移動量推定用）
            yaw_est_phase1 = None
            pitch_est_phase1 = None
            if vanishing_points is not None and frame_idx < len(vanishing_points):
                vp_x, vp_y = vanishing_points[frame_idx]  # (x, y)座標
                roll = camera_state.orientation[0]  # 現在のロール角
                # convert_pixel_to_pose()で正しい計算（魚眼補正+ロール補正）
                yaw_est_phase1, pitch_est_phase1 = convert_pixel_to_pose(
                    vp_x, vp_y, self.camera, roll
                )

            # 87フレーム前後で詳細ログ（デバッグ用）
            if frame_num in [85, 86, 87, 88, 89]:
                vp_smooth = vanishing_points[frame_idx] if vanishing_points is not None and frame_idx < len(vanishing_points) else None
                if vp_smooth is not None:
                    self.logger.info(
                        f"Frame {frame_num}: "
                        f"vp_smooth=({vp_smooth[0]:.1f}, {vp_smooth[1]:.1f}), "
                        f"yaw={np.degrees(camera_state.orientation[1]):.2f}deg, "
                        f"pitch={np.degrees(camera_state.orientation[2]):.2f}deg"
                    )

            # (1) フレームペア処理（移動量推定）
            if prev_frame is not None:
                # 前フレームの最終z推定値を取得
                z_prev = camera_state.position[2]

                # dx, dy, dz範囲を動的に計算
                frame_idx_in_dict = frame_num - actual_start_frame
                if frame_idx_in_dict in constraints_dict:
                    frame_constraints = constraints_dict[frame_idx_in_dict].copy()
                else:
                    frame_constraints = {}

                # dx, dyの累積値制約を更新
                max_x_limit = self.config.estimation.motion.max_x
                max_y_limit = self.config.estimation.motion.max_y
                max_dx_step = self.config.estimation.motion.max_dx
                max_dy_step = self.config.estimation.motion.max_dy
                current_x = camera_state.position[0]
                current_y = camera_state.position[1]

                dx_lower = max(-max_dx_step, -max_x_limit - current_x)
                dx_upper = min(max_dx_step, max_x_limit - current_x)
                dy_lower = max(-max_dy_step, -max_y_limit - current_y)
                dy_upper = min(max_dy_step, max_y_limit - current_y)

                frame_constraints['dx'] = (dx_lower, dx_upper)
                frame_constraints['dy'] = (dy_lower, dy_upper)

                # drollの累積値制約を更新
                max_roll_limit = np.radians(self.config.estimation.motion.max_roll)
                max_droll_step = np.radians(self.config.estimation.motion.max_droll)
                current_roll = camera_state.orientation[0]

                droll_lower = max(-max_droll_step, -max_roll_limit - current_roll)
                droll_upper = min(max_droll_step, max_roll_limit - current_roll)

                frame_constraints['droll'] = (droll_lower, droll_upper)

                # dyaw, dpitchの累積値制約を更新
                # ★消失点推定値を中心とする場合の処理
                max_yaw_limit = np.radians(self.config.estimation.motion.max_yaw)
                max_pitch_limit = np.radians(self.config.estimation.motion.max_pitch)
                max_dyaw_step = np.radians(self.config.estimation.motion.max_dtheta)
                max_dpitch_step = np.radians(self.config.estimation.motion.max_dphi)
                current_yaw = camera_state.orientation[1]
                current_pitch = camera_state.orientation[2]

                # 消失点推定値を中心とする場合
                if (self.config.estimation.motion.use_vanishing_point_yaw or
                    self.config.estimation.motion.use_vanishing_point_pitch) and vanishing_points is not None:
                    # Phase1で計算済みの消失点を使用
                    vp_x, vp_y = vanishing_points[frame_idx]  # (x, y)座標
                    roll = camera_state.orientation[0]  # 現在のロール角
                    # convert_pixel_to_pose()で正しい計算（魚眼補正+ロール補正）
                    yaw_vp, pitch_vp = convert_pixel_to_pose(
                        vp_x, vp_y, self.camera, roll
                    )

                    if self.config.estimation.motion.use_vanishing_point_yaw and yaw_vp is not None:
                        # yaw制約: yaw_vp ± max_yaw_limit
                        dyaw_lower = max(-max_dyaw_step, yaw_vp - max_yaw_limit - current_yaw)
                        dyaw_upper = min(max_dyaw_step, yaw_vp + max_yaw_limit - current_yaw)
                    else:
                        # 0を中心とする従来の制約
                        dyaw_lower = max(-max_dyaw_step, -max_yaw_limit - current_yaw)
                        dyaw_upper = min(max_dyaw_step, max_yaw_limit - current_yaw)

                    if self.config.estimation.motion.use_vanishing_point_pitch and pitch_vp is not None:
                        # pitch制約: pitch_vp ± max_pitch_limit
                        dpitch_lower = max(-max_dpitch_step, pitch_vp - max_pitch_limit - current_pitch)
                        dpitch_upper = min(max_dpitch_step, pitch_vp + max_pitch_limit - current_pitch)
                    else:
                        # 0を中心とする従来の制約
                        dpitch_lower = max(-max_dpitch_step, -max_pitch_limit - current_pitch)
                        dpitch_upper = min(max_dpitch_step, max_pitch_limit - current_pitch)
                else:
                    # 0を中心とする従来の制約
                    dyaw_lower = max(-max_dyaw_step, -max_yaw_limit - current_yaw)
                    dyaw_upper = min(max_dyaw_step, max_yaw_limit - current_yaw)
                    dpitch_lower = max(-max_dpitch_step, -max_pitch_limit - current_pitch)
                    dpitch_upper = min(max_dpitch_step, max_pitch_limit - current_pitch)

                frame_constraints['dyaw'] = (dyaw_lower, dyaw_upper)
                frame_constraints['dpitch'] = (dpitch_lower, dpitch_upper)

                # dz範囲を動的に計算
                # ★2つの制約を同時に適用:
                #   1. dzステップ制約: dz* ± tolerance（フレーム間移動量制約）
                #   2. Z累積値制約: z* ± tolerance（OCR推定値からのズレ制限）
                # BUG-003修正: use_ocr_z_constraintsパラメータチェック追加（箇所1: dz制約計算 - 最重要）
                if self.config.estimation.use_ocr_z_constraints and \
                   z_star_ocr is not None and z_star_prev is not None:
                    tolerance = self.config.estimation.z_range_tolerance
                    current_z = camera_state.position[2]  # 現在のZ累積値

                    # 制約1: dzステップ制約（dz* ± tolerance）
                    dz_star = z_star_ocr - z_star_prev  # OCR推定値の差分
                    dz_step_min = dz_star - tolerance
                    dz_step_max = dz_star + tolerance

                    # 制約2: Z累積値制約（z* ± tolerance）
                    # current_z + dz が [z_star_ocr - tolerance, z_star_ocr + tolerance] に収まる
                    dz_cumulative_min = (z_star_ocr - tolerance) - current_z
                    dz_cumulative_max = (z_star_ocr + tolerance) - current_z

                    # 2つの制約の共通範囲を取得
                    dz_min = max(0.0, dz_step_min, dz_cumulative_min)  # 負にならないよう保護
                    dz_max = min(dz_step_max, dz_cumulative_max)

                    # 制約が矛盾していないかチェック
                    if dz_min > dz_max:
                        self.logger.warning(
                            f"フレーム{frame_num}: dz制約が矛盾 - "
                            f"dz_step=[{dz_step_min:.2f}, {dz_step_max:.2f}]mm, "
                            f"dz_cumulative=[{dz_cumulative_min:.2f}, {dz_cumulative_max:.2f}]mm, "
                            f"current_z={current_z:.2f}mm, z*={z_star_ocr:.2f}mm"
                        )
                        # フォールバック: ステップ制約のみ使用
                        dz_min = max(0.0, dz_step_min)
                        dz_max = dz_step_max

                    frame_constraints['dz'] = (dz_min, dz_max)

                    # 詳細ログ出力（DEBUGレベル）
                    self.logger.debug(
                        f"フレーム{frame_num}: dz制約計算 - "
                        f"current_z={current_z:.2f}mm, z*={z_star_ocr:.2f}mm, "
                        f"dz*={dz_star:.2f}mm, "
                        f"dz_step=[{dz_step_min:.2f}, {dz_step_max:.2f}]mm, "
                        f"dz_cumulative=[{dz_cumulative_min:.2f}, {dz_cumulative_max:.2f}]mm, "
                        f"dz最終=[{dz_min:.2f}, {dz_max:.2f}]mm"
                    )
                else:
                    # BUG-004修正: OCR制約無効時はPhase1の制約辞書からdzのみ取得
                    # ★BUG-008修正: frame_constraints全体を上書きせず、dzのみ更新
                    # dx/dy/droll/dyaw/dpitchは既に計算済み（行720-792）
                    frame_idx_in_dict = frame_num - actual_start_frame
                    phase1_constraints = constraints_dict.get(frame_idx_in_dict, None)

                    if phase1_constraints and 'dz' in phase1_constraints:
                        frame_constraints['dz'] = phase1_constraints['dz']
                        self.logger.debug(
                            f"フレーム{frame_num}: Phase1制約を使用 - "
                            f"dz=[{phase1_constraints['dz'][0]:.2f}, {phase1_constraints['dz'][1]:.2f}]mm"
                        )
                    else:
                        # Phase1制約がない場合はデフォルト値を使用
                        frame_constraints['dz'] = (0.0, self.config.estimation.motion.max_dz)
                        self.logger.debug(
                            f"フレーム{frame_num}: デフォルトdz制約を使用 - "
                            f"dz=[0.00, {self.config.estimation.motion.max_dz:.2f}]mm"
                        )

                # 適応的dz制約: 前フレームの結果から探索範囲を絞る
                if self.config.estimation.motion.dz_adaptive_bounds_enabled and prev_dz is not None:
                    margin = self.config.estimation.motion.dz_adaptive_margin
                    adaptive_lower = max(0.0, prev_dz - margin)
                    adaptive_upper = prev_dz + margin
                    current_lower, current_upper = frame_constraints['dz']
                    new_lower = max(current_lower, adaptive_lower)
                    new_upper = min(current_upper, adaptive_upper)
                    if new_lower < new_upper:
                        frame_constraints['dz'] = (new_lower, new_upper)
                        self.logger.debug(
                            f"フレーム{frame_num}: 適応的dz制約適用 - prev_dz={prev_dz:.2f}mm, "
                            f"dz=[{new_lower:.2f}, {new_upper:.2f}]mm"
                        )

                # Phase1制約情報をログ出力（DEBUGレベル）
                if frame_constraints:
                    # dzは常に存在
                    log_msg = f"フレーム{frame_num}: 制約 - dz=[{frame_constraints['dz'][0]:.2f}, {frame_constraints['dz'][1]:.2f}]mm"

                    # dyawとdpitchは動的計算時は存在しない
                    if 'dyaw' in frame_constraints:
                        log_msg += f", dyaw=[{np.degrees(frame_constraints['dyaw'][0]):.2f}, {np.degrees(frame_constraints['dyaw'][1]):.2f}]deg"
                    if 'dpitch' in frame_constraints:
                        log_msg += f", dpitch=[{np.degrees(frame_constraints['dpitch'][0]):.2f}, {np.degrees(frame_constraints['dpitch'][1]):.2f}]deg"

                    self.logger.debug(log_msg)

                # OCR局所平均dzを取得（Hybrid Stage 1用）
                ocr_local_dz_value = None
                if ocr_local_avg_dz is not None and frame_idx < len(ocr_local_avg_dz):
                    ocr_local_dz_value = ocr_local_avg_dz[frame_idx]

                movement_raw, movement_constrained, yaw_est, pitch_est, motion, status = self._process_frame_pair_with_constraints(
                    prev_frame, curr_frame, camera_state, frame_constraints,
                    yaw_est_phase1, pitch_est_phase1,
                    dz_hint=prev_dz,
                    ocr_local_dz=ocr_local_dz_value,
                    prev_dz=prev_dz
                )

                # ★静止状態チェック（最重要）
                if status == "STATIC_FRAME":
                    self.logger.info(f"フレーム{frame_num}: 静止状態のためスキップ")
                    if not first_valid_frame_found:
                        self.logger.info(
                            f"フレーム{frame_num}: 開始フレーム先送り（有効フレーム未検出のため基準フレームを更新）"
                        )
                        prev_frame = curr_frame
                    skipped_static_count += 1
                    continue

                # ★特徴点抽出失敗チェック（新規追加）
                if status == "FAILED":
                    self.logger.warning(f"フレーム{frame_num}: 特徴点抽出失敗のためスキップ")
                    if not first_valid_frame_found:
                        self.logger.info(
                            f"フレーム{frame_num}: 開始フレーム先送り（有効フレーム未検出のため基準フレームを更新）"
                        )
                    prev_frame = curr_frame  # 連鎖スキップ防止: 常に基準フレームを更新
                    # フォールバックdz: 直近の成功フレームのdz中央値でz座標を進める
                    if first_valid_frame_found and len(dz_history) > 0:
                        fallback_dz = float(np.median(list(dz_history)))
                        camera_state.position[2] += fallback_dz
                        self.logger.info(
                            f"フレーム{frame_num}: フォールバックdz適用 "
                            f"dz={fallback_dz:.2f}mm (直近{len(dz_history)}フレームの中央値)"
                        )
                        prev_dz = fallback_dz
                    # スキップフレームのExcel記録
                    z_star_skip = z_positions[frame_idx] if z_positions is not None and frame_idx < len(z_positions) else None
                    dark_cx = vanishing_points[frame_idx][0] if vanishing_points is not None and frame_idx < len(vanishing_points) else None
                    dark_cy = vanishing_points[frame_idx][1] if vanishing_points is not None and frame_idx < len(vanishing_points) else None
                    frame_data_list.append({
                        'frame_num': frame_num,
                        'x': camera_state.position[0],
                        'y': camera_state.position[1],
                        'z': camera_state.position[2],
                        'roll': camera_state.orientation[0],
                        'yaw': camera_state.orientation[1],
                        'pitch': camera_state.orientation[2],
                        'yaw_vanishing': None,
                        'pitch_vanishing': None,
                        'z_ocr': z_star_skip,
                        'dark_center_x': dark_cx,
                        'dark_center_y': dark_cy,
                        'vp_feature_success': None,
                        'vp_feature_x': None,
                        'vp_feature_y': None,
                        'vp_feature_inlier_count': None,
                        'vp_feature_inlier_ratio': None,
                    })
                    skipped_failed_count += 1
                    continue


                # ★第2段階判定: Z方向移動量ベースの停止判定（TASK-4.2.11d Phase 3）
                # 特徴点ミスマッチで「50px離れているがdz=0mm」を検出
                if self.config.estimation.static_frame_detection.enabled:
                    dz = movement_raw[2]  # Z方向移動量（mm）- 補正前の値で判定
                    threshold = self.config.estimation.static_frame_detection.movement_threshold
                    if abs(dz) <= threshold:
                        self.logger.info(
                            f"フレーム{frame_num}: Z方向停止検出（第2段階判定: dzベース） "
                            f"dz={dz:.2f}mm <= {threshold}mm"
                        )
                        if not first_valid_frame_found:
                            self.logger.info(
                                f"フレーム{frame_num}: 開始フレーム先送り（有効フレーム未検出のため基準フレームを更新）"
                            )
                        prev_frame = curr_frame  # 連鎖スキップ防止: 常に基準フレームを更新
                        # フォールバックdz: 直近の成功フレームのdz中央値でz座標を進める
                        if first_valid_frame_found and len(dz_history) > 0:
                            fallback_dz = float(np.median(list(dz_history)))
                            camera_state.position[2] += fallback_dz
                            self.logger.info(
                                f"フレーム{frame_num}: フォールバックdz適用 "
                                f"dz={fallback_dz:.2f}mm (直近{len(dz_history)}フレームの中央値)"
                            )
                            prev_dz = fallback_dz
                        # スキップフレームのExcel記録
                        z_star_skip = z_positions[frame_idx] if z_positions is not None and frame_idx < len(z_positions) else None
                        dark_cx = vanishing_points[frame_idx][0] if vanishing_points is not None and frame_idx < len(vanishing_points) else None
                        dark_cy = vanishing_points[frame_idx][1] if vanishing_points is not None and frame_idx < len(vanishing_points) else None
                        frame_data_list.append({
                            'frame_num': frame_num,
                            'x': camera_state.position[0],
                            'y': camera_state.position[1],
                            'z': camera_state.position[2],
                            'roll': camera_state.orientation[0],
                            'yaw': camera_state.orientation[1],
                            'pitch': camera_state.orientation[2],
                            'yaw_vanishing': None,
                            'pitch_vanishing': None,
                            'z_ocr': z_star_skip,
                            'dark_center_x': dark_cx,
                            'dark_center_y': dark_cy,
                            'vp_feature_success': None,
                            'vp_feature_x': None,
                            'vp_feature_y': None,
                            'vp_feature_inlier_count': None,
                            'vp_feature_inlier_ratio': None,
                        })
                        skipped_dz_zero_count += 1
                        continue


                # 停止判定を通過 → 有効フレーム確定
                first_valid_frame_found = True

                # dz適応的制約: 次フレーム用にprev_dzを更新
                prev_dz = movement_raw[2]
                dz_history.append(prev_dz)

                # TASK-21: 移動量を詳細ログ出力（補正前と補正後）
                self.logger.debug(
                    f"フレーム{frame_num}: 移動量推定（補正前） - "
                    f"dx={movement_raw[0]:.2f}mm, dy={movement_raw[1]:.2f}mm, dz={movement_raw[2]:.2f}mm, "
                    f"droll={np.degrees(movement_raw[3]):.4f}deg"
                )
                self.logger.debug(
                    f"フレーム{frame_num}: 移動量推定（補正後） - "
                    f"dx={movement_constrained[0]:.2f}mm, dy={movement_constrained[1]:.2f}mm, dz={movement_constrained[2]:.2f}mm, "
                    f"droll={np.degrees(movement_constrained[3]):.4f}deg"
                )

                # 姿勢更新前のz座標を記録
                z_before = camera_state.position[2]

                # TASK-21: 姿勢更新（補正後の値を使用）
                camera_state.position += movement_constrained[:3]
                camera_state.orientation += movement_constrained[3:]

                # 更新後のz座標をログ出力
                z_after = camera_state.position[2]
                self.logger.debug(
                    f"フレーム{frame_num}: z座標更新 - "
                    f"z_before={z_before:.2f}mm → z_after={z_after:.2f}mm (dz={movement_constrained[2]:.2f}mm)"
                )
            else:
                movement_raw = np.zeros(6)
                movement_constrained = np.zeros(6)
                yaw_est = None
                pitch_est = None
                motion = None  # 初期フレームはmotion情報なし
                self.logger.info(
                    f"フレーム{frame_num}: 初期フレーム - "
                    f"初期位置 z={camera_state.position[2]:.2f}mm"
                )

            # (2) OCR距離取得
            # Phase2ではPhase1で計算済みのz*を使用（OCRは再実行しない）
            # BUG-003修正: use_ocr_z_constraintsパラメータチェック追加（箇所5: z_star代入）
            # BUG-007修正: Excel出力とPhase3補正のため、常にz_star_ocrを使用
            # use_ocr_z_constraintsはPhase2のdz制約のみを制御する
            z_star = z_star_ocr

            # Phase2ではdz積算を使用（z*は参考値・制約チェック用のみ）
            if frame_num == user_start_frame:
                z_current = camera_state.position[2]
                if z_star is not None:
                    self.logger.debug(
                        f"フレーム{frame_num}: 開始フレーム、z={z_current:.2f}mm, z*(Phase1)={z_star:.2f}mm"
                    )
                else:
                    self.logger.debug(
                        f"フレーム{frame_num}: 開始フレーム、z={z_current:.2f}mm (OCR推定なし)"
                    )
            else:
                z_current = camera_state.position[2]
                if z_star is not None:
                    self.logger.debug(
                        f"フレーム{frame_num}: z={z_current:.2f}mm, z*(Phase1参考値)={z_star:.2f}mm"
                    )
                else:
                    self.logger.debug(
                        f"フレーム{frame_num}: z={z_current:.2f}mm (OCR推定なし)"
                    )

            # TASK-21: カラーマップ用の一時位置計算（補正前の累積値）
            if frame_num >= user_start_frame and prev_frame is not None:
                # 補正前の累積値 = 現在の補正済み累積値 + 補正前の移動量
                # （現在のcamera_state.positionは前フレームまでの補正済み累積値）
                temp_position = camera_state.position.copy() + movement_raw[:3]
                temp_orientation = camera_state.orientation.copy() + movement_raw[3:]
                camera_state_for_colormap = CameraState(
                    position=temp_position,
                    orientation=temp_orientation,
                    frame_num=camera_state.frame_num,
                    timestamp=camera_state.timestamp
                )
            else:
                camera_state_for_colormap = camera_state

            # (3) カラーマップ生成（user_start_frame以降のみ）
            if frame_num >= user_start_frame:
                color_map_img = self._generate_colormap_frame(
                    curr_frame, camera_state_for_colormap, color_map_img
                )
                colormap_generated_count += 1

                # (4) デバッグ可視化
                if not self._debug_visualize(
                    curr_frame, camera_state, movement_constrained, z_star
                ):
                    # ユーザー中断
                    break
            
            # (6) フレームデータを記録
            # 暗部重心のピクセル座標（Phase1）
            dark_center_x = vanishing_points[frame_idx][0] if vanishing_points is not None and frame_idx < len(vanishing_points) else None
            dark_center_y = vanishing_points[frame_idx][1] if vanishing_points is not None and frame_idx < len(vanishing_points) else None
            
            # 特徴点ベース消失点の座標と品質情報（Phase2）
            vp_feature_success = motion.get('vp_feature_based_success', False) if motion is not None else None
            vp_feature_x = motion.get('vp_feature_based_x', None) if motion is not None else None
            vp_feature_y = motion.get('vp_feature_based_y', None) if motion is not None else None
            vp_feature_inlier_count = motion.get('vp_feature_based_inlier_count', None) if motion is not None else None
            vp_feature_inlier_ratio = motion.get('vp_feature_based_inlier_ratio', None) if motion is not None else None
            
            frame_data = {
                'frame_num': frame_num,
                'x': camera_state.position[0],
                'y': camera_state.position[1],
                'z': camera_state.position[2],
                'roll': camera_state.orientation[0],
                'yaw': camera_state.orientation[1],
                'pitch': camera_state.orientation[2],
                'yaw_vanishing': yaw_est,
                'pitch_vanishing': pitch_est,
                'z_ocr': z_star,  # Phase1の高精度推定値、ない場合はNone（空欄）
                # 暗部重心（Phase1のスムージング済み消失点）
                'dark_center_x': dark_center_x,
                'dark_center_y': dark_center_y,
                # 特徴点ベース消失点（Phase2で推定）
                'vp_feature_success': vp_feature_success,
                'vp_feature_x': vp_feature_x,
                'vp_feature_y': vp_feature_y,
                'vp_feature_inlier_count': vp_feature_inlier_count,
                'vp_feature_inlier_ratio': vp_feature_inlier_ratio,
            }
            frame_data_list.append(frame_data)

            # 次フレーム用に保存
            prev_frame = curr_frame
            # BUG-003修正: use_ocr_z_constraintsパラメータチェック追加（箇所7: z_star_prev更新）
            if self.config.estimation.use_ocr_z_constraints:
                z_star_prev = z_star_ocr  # dz*計算のため前フレームのz*（OCR推定値）を保存

            # 進捗ログ（100フレームごと）
            if (frame_num - actual_start_frame + 1) % 100 == 0:
                processed = frame_num - actual_start_frame + 1
                total = end_frame - actual_start_frame
                progress_percent = (processed / total) * 100
                self.logger.info(
                    f"進捗: {processed}/{total} フレーム ({progress_percent:.1f}%)"
                )

        # 全フレーム停止判定チェック
        if not first_valid_frame_found:
            cap.release()
            if self.debug_vis:
                self.debug_vis.close()
            raise MainProcessError(
                "有効なフレームが見つかりませんでした。"
                "全フレームが停止フレームと判定されたため、カラーマップを生成できません。"
            )

        # リソース解放
        cap.release()
        if self.debug_vis:
            self.debug_vis.close()

        # 統計情報出力
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Phase2完了: 統計情報")
        self.logger.info("=" * 80)
        total_frames = end_frame - actual_start_frame
        total_skipped = skipped_static_count + skipped_failed_count + skipped_dz_zero_count
        self.logger.info(f"処理フレーム数: {total_frames}")
        self.logger.info(f"カラーマップ生成フレーム数: {colormap_generated_count}")
        self.logger.info(f"スキップフレーム数: {total_skipped}")
        self.logger.info(f"  - 静止状態: {skipped_static_count}")
        self.logger.info(f"  - 特徴点抽出失敗: {skipped_failed_count}")
        self.logger.info(f"  - Z方向停止: {skipped_dz_zero_count}")
        self.logger.info(f"dz_minが0にクリップされたフレーム数: {dz_min_clipped_count}")
        self.logger.info(f"OCR制約範囲逸脱数: {ocr_constraint_violation_count}")
        if total_frames > 0:
            self.logger.info(
                f"OCR制約範囲逸脱率: {ocr_constraint_violation_count / total_frames * 100:.2f}%"
            )
        self.logger.info("=" * 80)
        self.logger.info(f"X位置補正回数: {self.camera_estimator.stats.get('x_constrained_count', 0)}")
        self.logger.info(f"Y位置補正回数: {self.camera_estimator.stats.get('y_constrained_count', 0)}")
        self.logger.info(f"Yaw角補正回数: {self.camera_estimator.stats.get('yaw_constrained_count', 0)}")
        self.logger.info(f"Pitch角補正回数: {self.camera_estimator.stats.get('pitch_constrained_count', 0)}")
        self.logger.info(f"Roll角補正回数: {self.camera_estimator.stats.get('roll_constrained_count', 0)}")

        # Excel出力
        self._save_frame_data_to_excel(frame_data_list)

        return color_map_img
    
    def _process_frame_pair_with_constraints(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        camera_state: CameraState,
        constraints: Optional[Dict[str, Tuple[float, float]]],
        yaw_est: Optional[float] = None,
        pitch_est: Optional[float] = None,
        dz_hint: Optional[float] = None,
        ocr_local_dz: Optional[float] = None,
        prev_dz: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float], Optional[Dict[str, Any]], str]:
        """フレームペア処理（制約付き移動量推定）

        Args:
            prev_frame: 前フレーム (H, W, 3)
            curr_frame: 現フレーム (H, W, 3)
            camera_state: カメラ状態
            constraints: Phase1で計算した制約（Noneの場合は広い範囲）
            yaw_est: Phase1で計算したyaw推定値（ラジアン）
            pitch_est: Phase1で計算したpitch推定値（ラジアン）
            dz_hint: 前フレームのdz（初期値・探索範囲のヒント、mm）
            ocr_local_dz: OCR局所平均dz（mm/frame、Hybrid Stage 1用）
            prev_dz: 前フレームのdz（適応的制約用、mm）

        Returns:
            movement_raw: 補正前の移動量 [dx, dy, dz, droll, dyaw, dpitch] (mm, ラジアン)
            movement_constrained: 補正後の移動量 [dx, dy, dz, droll, dyaw, dpitch] (mm, ラジアン)
            yaw_est: 消失点から推定したyaw（Excel出力用、ラジアン）
            pitch_est: 消失点から推定したpitch（Excel出力用、ラジアン）
            motion: motion辞書
            status: "SUCCESS" | "STATIC_FRAME" | "FAILED"
        """
        # Phase1のスムージング済み消失点をそのまま使用
        if yaw_est is not None and pitch_est is not None:
            self.logger.debug(
                f"Phase1スムージング済み消失点使用: yaw={np.degrees(yaw_est):.2f}°, "
                f"pitch={np.degrees(pitch_est):.2f}°"
            )

        # (2) カメラ状態に推定姿勢を追加
        camera_state_dict = {
            'position': camera_state.position.copy(),
            'orientation': camera_state.orientation.copy(),
        }

        # Phase1から渡された姿勢角をestimate_motion_with_constraints()に渡すため設定
        if yaw_est is not None:
            camera_state_dict['yaw_estimated'] = yaw_est
        if pitch_est is not None:
            camera_state_dict['phi_estimated'] = pitch_est

        # (3) 特徴点マッチング + 移動量推定 (新API使用)
        try:
            # 🆕 estimate_motion_from_frames()を使用
            # 円筒座標マッチング or 従来のORBマッチングを自動選択し、
            # 移動量推定まで一貫して実行
            if self.config.estimation.use_flexible_estimation:
                # 柔軟なパラメータ推定（段階的テスト用）
                # 注: estimate_motion_from_frames()は内部でestimate_motion()を呼び出すため、
                # 制約はestimate_motion()側で処理される。
                # ここでは特徴点抽出のみを新APIで行い、移動量推定は既存ロジックを使用
                prev_points, curr_points, status = self._extract_features_from_frames(
                    prev_frame, curr_frame
                )

                # 静止状態チェック（最重要）
                if status == "STATIC_FRAME":
                    self.logger.info(f"フレーム{camera_state.frame_num}: 静止状態のためスキップ（柔軟推定）")
                    # 移動量ゼロを返す（基準フレーム更新しない想定）
                    return np.zeros(6), np.zeros(6), yaw_est, pitch_est, None, "STATIC_FRAME"


                # 特徴点抽出失敗チェック
                if status == "FAILED" or prev_points is None or curr_points is None:
                    self.logger.warning("特徴点マッチング失敗、移動量をゼロとします")
                    return np.zeros(6), np.zeros(6), yaw_est, pitch_est, None, "FAILED"

                motion = self.camera_estimator.estimate_motion_flexible(
                    prev_points, curr_points, camera_state_dict, self.camera_params,
                    constraints=constraints
                )
            else:
                # 制約付き移動量推定（既存の方法）
                prev_points, curr_points, status = self._extract_features_from_frames(
                    prev_frame, curr_frame
                )

                # 静止状態チェック（最重要）
                if status == "STATIC_FRAME":
                    self.logger.info(f"フレーム{camera_state.frame_num}: 静止状態のためスキップ（制約付き推定）")
                    # 移動量ゼロを返す（基準フレーム更新しない想定）
                    return np.zeros(6), np.zeros(6), yaw_est, pitch_est, None, "STATIC_FRAME"


                # 特徴点抽出失敗チェック
                if status == "FAILED" or prev_points is None or curr_points is None:
                    self.logger.warning("特徴点マッチング失敗、移動量をゼロとします")
                    return np.zeros(6), np.zeros(6), yaw_est, pitch_est, None, "FAILED"

                # 位相相関結果をチェック
                fixed_dz = None
                fixed_droll = None
                cyl_matcher = self.camera_estimator.cylindrical_matcher
                if cyl_matcher is not None:
                    phase_corr = getattr(cyl_matcher, '_last_phase_correlation_result', None)
                    if phase_corr is not None:
                        threshold = cyl_matcher.config.phase_correlation_response_threshold
                        if phase_corr['response'] >= threshold:
                            fixed_dz = phase_corr['dz_mm']
                            fixed_droll = np.radians(phase_corr['droll_deg'])
                            self.logger.info(
                                f"位相相関採用: dz={fixed_dz:.3f}mm, "
                                f"droll={phase_corr['droll_deg']:.4f}°, "
                                f"response={phase_corr['response']:.3f}"
                            )
                        else:
                            self.logger.debug(
                                f"位相相関信頼度不足: response={phase_corr['response']:.3f} "
                                f"< threshold={threshold}"
                            )

                # ハイブリッド2段階推定: 第1段階（円筒座標でdz推定）
                if (self.config.estimation.motion.hybrid_two_stage_enabled
                        and cyl_matcher is not None):
                    prev_cyl = getattr(cyl_matcher, '_last_prev_points_cyl', None)
                    curr_cyl = getattr(cyl_matcher, '_last_curr_points_cyl', None)
                    if (prev_cyl is not None and curr_cyl is not None
                            and len(prev_cyl) >= 2):
                        # Stage 1 dz bounds: OCR局所平均 × マージンを優先使用
                        dz_bounds_stage1 = None
                        if ocr_local_dz is not None and ocr_local_dz > 0:
                            multiplier = self.config.estimation.motion.hybrid_stage1_ocr_margin_multiplier
                            ocr_max_dz = ocr_local_dz * multiplier
                            dz_bounds_stage1 = (0.0, max(1.0, ocr_max_dz))
                            # 適応的制約もOCR bounds上で適用
                            if (self.config.estimation.motion.dz_adaptive_bounds_enabled
                                    and prev_dz is not None):
                                margin = self.config.estimation.motion.dz_adaptive_margin
                                adaptive_lower = max(0.0, prev_dz - margin)
                                adaptive_upper = prev_dz + margin
                                dz_bounds_stage1 = (
                                    max(dz_bounds_stage1[0], adaptive_lower),
                                    min(dz_bounds_stage1[1], adaptive_upper)
                                )
                            self.logger.debug(
                                f"Stage1 OCR bounds: ocr_dz={ocr_local_dz:.2f}, "
                                f"bounds=[{dz_bounds_stage1[0]:.2f}, {dz_bounds_stage1[1]:.2f}]"
                            )
                        else:
                            # OCRなし: 従来のconstraintsからfallback
                            dz_bounds_stage1 = constraints.get('dz') if constraints else None
                        stage1_result = self.camera_estimator.estimate_dz_from_cylindrical(
                            prev_cyl, curr_cyl,
                            dz_bounds=dz_bounds_stage1,
                            pipe_radius_mm=self.config.pipe.diameter_mm / 2.0
                        )
                        if stage1_result['success']:
                            fixed_dz = stage1_result['dz']
                            # fixed_droll=None のまま → 3DOFモード起動
                            self.logger.info(
                                f"ハイブリッド第1段階成功: dz={fixed_dz:.3f}mm"
                            )
                        else:
                            self.logger.warning(
                                "ハイブリッド第1段階失敗、従来方式にフォールバック"
                            )

                # Stage 1成功かつ点数2点: 略式推定（Stage 2スキップ）
                # Stage 2の3DOF最適化(dx,dy,droll)には最低3点必要
                n_match_points = len(prev_points) if prev_points is not None else 0
                if fixed_dz is not None and n_match_points < 3:
                    droll_stage1_deg = stage1_result.get('droll_composite', 0.0)
                    # 2点ではdroll推定が不安定なため、droll=0（前フレームからロール変化なし）を採用
                    droll_stage1_rad = 0.0
                    # 消失点推定からのyaw/pitch変化量を設定（通常パスと同じ原点）
                    yaw_0 = camera_state_dict['orientation'][1]
                    phi_0 = camera_state_dict['orientation'][2]
                    dtheta_shortcut = camera_state_dict.get('yaw_estimated', yaw_0) - yaw_0
                    dphi_shortcut = camera_state_dict.get('phi_estimated', phi_0) - phi_0
                    motion = {
                        'dx': 0.0, 'dy': 0.0, 'dz': fixed_dz,
                        'droll': droll_stage1_rad,
                        'dtheta': dtheta_shortcut, 'dphi': dphi_shortcut
                    }
                    self.logger.info(
                        f"略式推定(2点): Stage 2スキップ, "
                        f"dz={fixed_dz:.3f}mm, "
                        f"droll=0°(raw={droll_stage1_deg:.2f}°, 2点不安定のため前フレーム維持), "
                        f"dtheta={np.degrees(dtheta_shortcut):.4f}°, "
                        f"dphi={np.degrees(dphi_shortcut):.4f}°"
                    )
                else:
                    motion = self.camera_estimator.estimate_motion_with_constraints(
                        prev_points, curr_points, camera_state_dict, self.camera_params,
                        constraints=constraints,
                        fix_angles=self.config.estimation.motion.fix_angles_to_vanishing_point,
                        dz_hint=dz_hint,
                        fixed_dz=fixed_dz,
                        fixed_droll=fixed_droll
                    )

            # TASK-21: 補正前の移動量を保存
            motion_raw = motion.copy()

            # 姿勢制約を適用（最終防衛線）
            motion_constrained = self.camera_estimator._constrain_camera_pose(
                motion, camera_state.position, camera_state.orientation, camera_state.frame_num,
                camera_state=camera_state_dict
            )


            # TASK-21: 補正前と補正後の両方をnumpy配列に変換
            movement_raw = np.array([
                motion_raw['dx'],
                motion_raw['dy'],
                motion_raw['dz'],
                motion_raw['droll'],
                motion_raw['dtheta'],
                motion_raw['dphi']
            ])

            movement_constrained = np.array([
                motion_constrained['dx'],
                motion_constrained['dy'],
                motion_constrained['dz'],
                motion_constrained['droll'],
                motion_constrained['dtheta'],
                motion_constrained['dphi']
            ])

            self.logger.debug(
                f"移動量推定（補正後）: dx={motion_constrained['dx']:.2f}, dy={motion_constrained['dy']:.2f}, "
                f"dz={motion_constrained['dz']:.2f}, droll={np.degrees(motion_constrained['droll']):.2f}°"
            )

            return movement_raw, movement_constrained, yaw_est, pitch_est, motion_constrained, "SUCCESS"

        except Exception as e:
            self.logger.warning(f"移動量推定失敗: {e}, 移動量をゼロとします")
            return np.zeros(6), np.zeros(6), yaw_est, pitch_est, None, "FAILED"

    def _extract_features_from_frames(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """フレーム画像から特徴点を抽出（新規メソッド）

        円筒座標マッチング or 従来のORBマッチングを自動選択します。

        Args:
            prev_frame: 前フレーム画像 (H, W, 3) BGR
            curr_frame: 現フレーム画像 (H, W, 3) BGR

        Returns:
            prev_points: 前フレーム特徴点 (N, 2) or None
            curr_points: 現フレーム特徴点 (N, 2) or None
            status: "SUCCESS" | "STATIC_FRAME" | "FAILED"
        """
        try:
            # 円筒座標マッチング or 従来のORBマッチング
            if (self.camera_estimator.cylindrical_matcher is not None and
                self.config.estimation.feature_matching.use_cylindrical_matching):
                # 円筒座標マッチング
                prev_points, curr_points, status = self.camera_estimator.cylindrical_matcher.extract_and_match(
                    prev_frame, curr_frame, self.camera_params
                )
                if status == "SUCCESS" and prev_points is not None:
                    self.logger.debug(f"円筒座標マッチング: {len(prev_points)}点")
                elif status == "STATIC_FRAME":
                    self.logger.debug("円筒座標マッチング: 静止状態検出")
                else:
                    self.logger.debug("円筒座標マッチング: 失敗")
            else:
                # 従来のORBマッチング（フレーム座標系）
                # 従来のORBマッチング（フレーム座標系） - statusなし
                prev_points_orb, curr_points_orb = self.feature_matcher.detect_and_match(
                    prev_frame, curr_frame, self.camera_params
                )
                # ORBマッチングの結果をstatus形式に変換
                if prev_points_orb is not None and curr_points_orb is not None:
                    prev_points, curr_points, status = prev_points_orb, curr_points_orb, "SUCCESS"
                    self.logger.debug(f"従来のORBマッチング: {len(prev_points)}点")
                else:
                    prev_points, curr_points, status = None, None, "FAILED"
                    self.logger.debug("従来のORBマッチング: 失敗")

            return prev_points, curr_points, status

        except Exception as e:
            self.logger.warning(f"特徴点抽出失敗: {e}")
            return None, None, "FAILED"
    
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
            max_dz=self.config.estimation.motion.max_dz,
            alpha=self.config.colormap.alpha
        )
        
        return color_map_img
    
    def _debug_visualize(
        self,
        curr_frame: np.ndarray,
        camera_state: CameraState,
        movement: np.ndarray,
        z_star: Optional[float]
    ) -> bool:
        """デバッグ可視化
        
        Args:
            curr_frame: 現フレーム (H, W, 3)
            camera_state: カメラ状態
            movement: 移動量 [dx, dy, dz, droll, dyaw, dpitch]
            z_star: Phase1のOCR高精度推定値 (mm)
            
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
            'z_star': z_star  # Phase1の推定値
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
            z_star if z_star is not None else 0.0,
            None
        )
        
        return True
    
    def run(self) -> None:
        """メイン処理実行（2パスフロー）
        
        Phase1: 全フレーム事前処理
        Phase2: カラーマップ生成
        
        Raises:
            VideoFileError: 動画ファイルの読み込みに失敗した場合
            MainProcessError: 処理中にエラーが発生した場合
        """
        # BUG-008修正: 処理開始時に一度だけタイムスタンプを生成
        # 外部（main()）で設定済みの場合はそのまま使用
        if self._timestamp is None:
            self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"処理開始: タイムスタンプ={self._timestamp}")

        # 動画読み込み
        video_path = Path(self.config.input.video_path)
        
        if not video_path.exists():
            raise VideoFileError(f"動画ファイルが見つかりません: {video_path}")
        
        self.logger.info("=" * 80)
        self.logger.info("管内カメラカーシミュレーション - カラーマップ生成（2パスフロー）")
        self.logger.info("=" * 80)
        self.logger.info(f"動画ファイル: {video_path}")
        
        # 動画情報取得
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoFileError(f"動画ファイルを開けません: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"動画情報: {width}x{height}, {fps:.2f}fps, {total_frames}フレーム")

        # 処理範囲決定
        user_start_frame = self.config.input.start_frame  # ユーザー指定の開始フレーム
        end_frame = (
            self.config.input.end_frame
            if self.config.input.end_frame is not None
            else total_frames
        )

        # 実際の処理開始フレーム（user_start_frame - 1）
        # 開始フレームの移動量推定に必要な前フレームを確保
        actual_start_frame = max(0, user_start_frame - 1)

        self.logger.info(f"カラーマップ生成範囲: フレーム {user_start_frame} 〜 {end_frame}")
        if actual_start_frame < user_start_frame:
            self.logger.info(
                f"移動量推定のため、フレーム {actual_start_frame} から処理を開始します"
            )
        
        # 最初のフレームを読み込んでモジュール初期化
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise VideoFileError("最初のフレームを読み込めません")
        
        cap.release()

        # モジュール初期化（処理フレーム数は実際の処理範囲で計算）
        self._initialize_modules(first_frame, end_frame - actual_start_frame)

        # Step 1（パラメータ調整）はmain()内で処理済み
        # progress_repが外部から設定されている場合、Step 1は既に完了/スキップ済み
        # progress_repが_initialize_modules()で新規生成された場合のみフォールバック処理
        if self.progress_rep:
            step1 = self.progress_rep._get_step(1)
            if step1.status == 'pending':
                # main()経由でない呼び出し（テスト等）のフォールバック
                if self.config.auto_tune_enabled:
                    self.progress_rep.start_step(1)
                    self.progress_rep.complete_step(1, success=True)
                    self.logger.info("Step 1（パラメータ調整）: フォールバック完了処理")
                else:
                    self.progress_rep.skip_step(1, reason="auto_tune_enabled=false")
                    self.logger.info("Step 1（パラメータ調整）をスキップしました（auto_tune無効）")

        # ============================================================
        # Phase1: 全フレーム事前処理
        # ============================================================

        # Step 2開始（フレーム解析: VP抽出 + OCR読取）
        if self.progress_rep:
            self.progress_rep.start_step(2)

        # Step 2（フレーム解析）の進捗コールバック（1%単位で間引き）
        _step2_last_pct = [-1.0]

        def frame_analysis_progress_callback(current_frame: int, total_frames: int):
            """フレーム解析進捗コールバック（1%刻み）"""
            try:
                progress_percent = (current_frame / total_frames) * 100.0
                if (progress_percent - _step2_last_pct[0] >= 1.0
                        or current_frame >= total_frames):
                    self.progress_rep.update_step(2, progress_percent)
                    _step2_last_pct[0] = progress_percent
            except Exception as e:
                self.logger.warning(f"Step 2 progress update failed at frame {current_frame}: {e}")


        try:
            constraints_dict, z_positions, vanishing_points = self._compute_constraints_phase1(
                video_path, actual_start_frame, end_frame, frame_analysis_progress_callback
            )

            # Step 2完了時のdetails設定
            total_frames = end_frame - actual_start_frame + 1
            ocr_success_count = np.sum(z_positions != 0) if z_positions is not None else 0
            step2_details = {
                "total_frames": total_frames,
                "ocr_success_count": int(ocr_success_count) if z_positions is not None else 0
            }
            if self.progress_rep:
                self.progress_rep.complete_step(2, success=True, details=step2_details)


        except Exception as e:
            self.logger.error(f"Phase1制約計算失敗: {e}")
            if self.progress_rep:
                self.progress_rep.complete_step(2, success=False, error_message=str(e))

            # フォールバック処理（既存のまま）
            raise

        # Phase1で計算した実際の処理開始フレーム（actual_start_frame）のz*を取得
        # z_positions[0]はactual_start_frameのデータなので、インデックスは0
        initial_z_star = None
        if z_positions is not None and len(z_positions) > 0:
            initial_z_star = z_positions[0]  # actual_start_frameのz*
            self.logger.info(
                f"Phase1で計算した処理開始フレーム（{actual_start_frame}）のOCR距離推定値: z*={initial_z_star:.2f}mm"
            )

        # メモリ節約のためframesを解放（オプション）

        # ============================================================
        # Phase2: カラーマップ生成
        # ============================================================

        # Step 3開始（カラーマップ生成）
        if self.progress_rep:
            self.progress_rep.start_step(3)

        try:
            color_map_img = self._generate_colormap_phase2(
                video_path, actual_start_frame, end_frame, user_start_frame, fps,
                constraints_dict, initial_z_star, z_positions, vanishing_points
            )

            # Step 3完了
            if self.progress_rep:
                self.progress_rep.complete_step(3, success=True)

        except Exception as e:
            self.logger.error(f"Phase2カラーマップ生成失敗: {e}")
            if self.progress_rep:
                self.progress_rep.complete_step(3, success=False, error_message=str(e))
            raise
        
        # ============================================================
        # 出力
        # ============================================================
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("カラーマップ出力")
        self.logger.info("=" * 80)

        # タイムスタンプ生成（一時カラーマップ、Excel、最終カラーマップで共通使用）
        timestamp = self._timestamp  # BUG-008修正: 処理開始時タイムスタンプを使用

        # 一時カラーマップを保存（Phase2の出力）
        temporary_colormap_path = Path(
            self.config.output.temporary_colormap_path.format(timestamp=timestamp)
        )
        temporary_colormap_path.parent.mkdir(parents=True, exist_ok=True)

        if color_map_img is not None:
            plt.imsave(str(temporary_colormap_path), color_map_img)
            self.logger.info(f"一時カラーマップを保存しました: {temporary_colormap_path}")
        else:
            self.logger.error("カラーマップ画像がNullのため保存できません")
            raise MainProcessError("カラーマップ生成に失敗しました")

        # Excelパスを取得
        excel_path = Path(
            self.config.output.report_path.format(timestamp=timestamp)
        )

        # 最終カラーマップパス
        final_colormap_path = Path(
            self.config.output.colormap_path.format(timestamp=timestamp)
        )

        # ============================================================
        # Phase3: 展開画像補正
        # ============================================================

        # Step 4開始（カラーマップ補正）
        if self.progress_rep:
            self.progress_rep.start_step(4)

        try:
            self._correct_colormap_phase3(
                temporary_colormap_path,
                excel_path,
                final_colormap_path
            )

            # Step 4完了
            if self.progress_rep:
                self.progress_rep.complete_step(4, success=True)

        except Exception as e:
            self.logger.error(f"Phase3カラーマップ補正失敗: {e}")
            if self.progress_rep:
                self.progress_rep.complete_step(4, success=False, error_message=str(e))
            # フォールバック処理（既存のまま）
            pass  # エラー時も処理継続

        # 注: 4ステップモードではfinalize()は不要（各complete_step()で自動的に最終化）

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("処理完了")
        self.logger.info("=" * 80)

    def _correct_colormap_phase3(
        self,
        temporary_colormap_path: Path,
        excel_path: Path,
        final_colormap_path: Path
    ) -> bool:
        """Phase3: 展開画像補正

        Args:
            temporary_colormap_path: 補正前の一時カラーマップ
            excel_path: Excelファイル（z, z_ocr列）
            final_colormap_path: 補正後の最終カラーマップ

        Returns:
            補正成功時True
        """
        if not self.config.correction.enabled:
            self.logger.info("展開画像補正は無効です（correction.enabled=false）")
            # 補正なしの場合、temporary → finalにコピー
            import shutil
            if self.progress_rep:
                self.progress_rep.update_step(4, 50.0, {"reason": "correction_disabled"})
            shutil.copy(temporary_colormap_path, final_colormap_path)
            self.logger.info(f"一時カラーマップを最終カラーマップにコピーしました: {final_colormap_path}")
            # correction無効時は一時ファイルを削除しない（最終画像として保持）
            self.logger.info(
                f"展開画像補正が無効のため、一時展開画像は最終画像です: {temporary_colormap_path}"
            )
            return True

        # Phase3補正実行前の設定矛盾チェック
        if self.config.estimation.use_ocr_z_constraints and self.config.correction.enabled:
            self.logger.warning(
                "⚠️ 設定矛盾の可能性: use_ocr_z_constraints=true と correction.enabled=true が同時に有効です。\n"
                "Phase2でOCR制約により位置合わせ済みのため、Phase3補正の効果はほとんどありません。\n"
                "推奨設定: use_ocr_z_constraints=false + correction.enabled=true"
            )

        # OCRが無効の場合、Phase3補正をスキップ
        if not self.config.ocr.enabled:
            self.logger.warning(
                "OCRが無効のため（ocr.enabled=false）、展開画像補正をスキップします"
            )
            # OCR無効の場合、temporary → finalにコピー
            import shutil
            shutil.copy(temporary_colormap_path, final_colormap_path)
            self.logger.info(f"一時カラーマップを最終カラーマップにコピーしました: {final_colormap_path}")
            # OCR無効時は一時ファイルを削除しない（最終画像として保持）
            self.logger.info(
                f"OCRが無効のため、一時展開画像は最終画像です: {temporary_colormap_path}"
            )
            return True

        # OCRデータの存在確認
        try:
            import pandas as pd
            df = pd.read_excel(excel_path)
            z_ocr_col = 'Z_OCR (mm)'
            ocr_data = df[df[z_ocr_col].notna()]

            if len(ocr_data) == 0:
                self.logger.warning(
                    f"OCRデータが存在しないため、展開画像補正をスキップします（OCRデータ数: 0）"
                )
                # OCRデータなしの場合、temporary → finalにコピー
                import shutil
                shutil.copy(temporary_colormap_path, final_colormap_path)
                self.logger.info(f"一時カラーマップを最終カラーマップにコピーしました: {final_colormap_path}")
                # コピー後に一時ファイルを削除
                self._delete_temporary_colormap(temporary_colormap_path)
                return True

            self.logger.info(f"OCRデータ確認: {len(ocr_data)}個のOCR距離データを検出")

        except Exception as e:
            self.logger.error(f"OCRデータ確認エラー: {e}")
            # エラー時は補正をスキップ
            import shutil
            shutil.copy(temporary_colormap_path, final_colormap_path)
            self.logger.info(f"一時カラーマップを最終カラーマップにコピーしました: {final_colormap_path}")
            # コピー後に一時ファイルを削除
            self._delete_temporary_colormap(temporary_colormap_path)
            return True

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Phase3: 展開画像補正開始")
        self.logger.info("=" * 80)

        try:
            corrector = ColormapCorrector(
                logger=self.logger,
                ocr_interval_mm=self.config.correction.ocr_interval_mm,
                max_image_size=self.config.correction.max_image_size
            )

            success = corrector.correct_colormap(
                image_path=temporary_colormap_path,
                excel_path=excel_path,
                output_path=final_colormap_path,
                pixels_per_mm=self.colormap_gen.pixels_per_mm
            )

            if success:
                self.logger.info(f"展開画像補正が完了しました: {final_colormap_path}")
                
                # TASK-PROD-002: Phase3補正成功後に一時ファイルを削除
                self._delete_temporary_colormap(temporary_colormap_path)
            else:
                self.logger.error("展開画像補正に失敗しました")
                # 補正失敗時は一時ファイルを保持（デバッグ用）
                self.logger.warning(
                    f"展開画像補正失敗のため、一時展開画像を保持します: {temporary_colormap_path}"
                )

            return success

        except ColormapCorrectionError as e:
            self.logger.error(f"展開画像補正エラー: {e}")
            # エラー時は一時カラーマップをコピー
            import shutil
            shutil.copy(temporary_colormap_path, final_colormap_path)
            self.logger.warning(f"一時カラーマップを最終カラーマップにコピーしました（補正失敗時）: {final_colormap_path}")
            # Phase3失敗時は一時ファイルを保持（デバッグ用）
            self.logger.warning(
                f"展開画像補正失敗のため、一時展開画像を保持します: {temporary_colormap_path}"
            )
            return False

    def _delete_temporary_colormap(self, temporary_colormap_path: Path) -> None:
        """Phase3完了後に一時展開画像を削除する
        
        Notes:
            - config.colormap.delete_temporary_images=false の場合はスキップ
            - ファイルが存在しない場合はスキップ（INFOログ）
            - 削除失敗時は警告ログのみ（処理は継続）
            - Phase3補正失敗時、correction無効時は呼び出されない想定
        
        Args:
            temporary_colormap_path: 削除対象の一時カラーマップファイルパス
        
        Raises:
            なし（削除失敗は例外としない）
        """
        # 削除フラグチェック
        if not self.config.colormap.delete_temporary_images:
            self.logger.info(
                f"一時展開画像を保持します（設定により）: {temporary_colormap_path}"
            )
            return
        
        # ファイルパス取得
        temp_path = Path(temporary_colormap_path)
        
        # ファイル存在チェック
        if not temp_path.exists():
            self.logger.info(
                f"一時展開画像が存在しないためスキップします: {temp_path}"
            )
            return
        
        # 削除実行
        self.logger.info(f"Phase3補正完了: 一時展開画像を削除します: {temp_path}")
        try:
            temp_path.unlink()
            self.logger.info(f"一時展開画像を削除しました: {temp_path}")
        except Exception as e:
            # 削除失敗は処理を中断させない
            self.logger.warning(
                f"一時展開画像の削除に失敗しました: {temp_path} - {e}"
            )
            self.logger.warning(f"手動で削除してください: {temp_path}")

    def _save_colormap(self, color_map_img: Optional[np.ndarray]) -> None:
        """カラーマップを保存
        
        Args:
            color_map_img: カラーマップ画像
        """
        if color_map_img is None:
            self.logger.warning("カラーマップ画像がNullのため保存をスキップ")
            return
        
        # タイムスタンプ生成
        timestamp = self._timestamp  # BUG-008修正: 処理開始時タイムスタンプを使用
        
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

    def _save_frame_data_to_excel(self, frame_data_list: List[Dict[str, Any]]) -> None:
        """フレームデータをExcelファイルに保存

        Args:
            frame_data_list: フレームごとのデータリスト
        """
        if not frame_data_list:
            self.logger.warning("フレームデータが空のためExcel出力をスキップ")
            return

        # タイムスタンプ生成
        timestamp = self._timestamp  # BUG-008修正: 処理開始時タイムスタンプを使用

        # 出力パス生成（{timestamp}プレースホルダを置換）
        output_path = Path(
            self.config.output.report_path.format(timestamp=timestamp)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # DataFrameに変換
        df = pd.DataFrame(frame_data_list)

        # 角度を度数に変換
        df['roll_deg'] = np.degrees(df['roll'])
        df['yaw_deg'] = np.degrees(df['yaw'])
        df['pitch_deg'] = np.degrees(df['pitch'])
        df['yaw_vanishing_deg'] = df['yaw_vanishing'].apply(
            lambda x: np.degrees(x) if x is not None else None
        )
        df['pitch_vanishing_deg'] = df['pitch_vanishing'].apply(
            lambda x: np.degrees(x) if x is not None else None
        )

        # 列を並び替え
        df = df[[
            'frame_num',
            'x', 'y', 'z',
            'roll_deg', 'yaw_deg', 'pitch_deg',
            'yaw_vanishing_deg', 'pitch_vanishing_deg',
            'dark_center_x', 'dark_center_y',
            'vp_feature_success', 'vp_feature_x', 'vp_feature_y',
            'vp_feature_inlier_count', 'vp_feature_inlier_ratio',
            'z_ocr'
        ]]

        # 列名を日本語に変更
        df.columns = [
            'フレーム番号',
            'X (mm)', 'Y (mm)', 'Z (mm)',
            'Roll (度)', 'Yaw (度)', 'Pitch (度)',
            'Yaw_消失点 (度)', 'Pitch_消失点 (度)',
            '暗部重心X (px)', '暗部重心Y (px)',
            '特徴点VP_成功', '特徴点VP_X (px)', '特徴点VP_Y (px)',
            '特徴点VP_インライア数', '特徴点VP_インライア比率',
            'Z_OCR (mm)'
        ]

        # Excelファイルに保存
        try:
            df.to_excel(output_path, index=False, engine='openpyxl')
            self.logger.info(f"フレームデータをExcelに保存しました: {output_path}")
        except ImportError:
            # openpyxlがない場合はxlsxwriterを試す
            try:
                df.to_excel(output_path, index=False, engine='xlsxwriter')
                self.logger.info(f"フレームデータをExcelに保存しました（xlsxwriter使用）: {output_path}")
            except ImportError:
                # xlsxwriterもない場合はCSVで保存
                csv_path = output_path.with_suffix('.csv')
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                self.logger.info(f"Excel保存できないためCSVで保存しました: {csv_path}")
        except Exception as e:
            self.logger.error(f"Excel保存に失敗しました: {e}")


# ============================================================================
# コマンドライン引数解析
# ============================================================================

def int_or_none(value: str) -> Optional[int]:
    """コマンドライン引数の型変換関数（int または None）
    
    文字列"null"または"none"（大文字小文字を区別しない）をNoneに変換し、
    それ以外は整数に変換します。
    
    Args:
        value: 入力文字列
        
    Returns:
        整数値、または文字列が"null"/"none"の場合はNone
        
    Raises:
        argparse.ArgumentTypeError: 整数に変換できない値の場合
        
    Examples:
        >>> int_or_none("null")
        None
        >>> int_or_none("NULL")
        None
        >>> int_or_none("none")
        None
        >>> int_or_none("300")
        300
    """
    # "null"または"none"（大文字小文字無視）の場合はNoneを返す
    if value.lower() in ('null', 'none'):
        return None
    
    # それ以外は整数に変換を試みる
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"無効な値です: '{value}' (整数または'null'を指定してください)"
        )


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数をパース
    
    Returns:
        パース結果
    """
    parser = argparse.ArgumentParser(
        description='管内カメラカーシミュレーション用カラーマップ生成（2パスフロー版）',
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
    
    # 入力動画ファイル（設定ファイルでも指定可能）
    parser.add_argument(
        '--input',
        type=Path,
        default=None,
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
        default=None,
        help='処理開始フレーム番号（設定ファイル未指定時のデフォルト=1）'
    )

    parser.add_argument(
        '--end',
        type=int_or_none,
        default=None,
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

    # カラーマップ画質
    parser.add_argument(
        '--ppm',
        type=float,
        help='カラーマップ解像度（pixels/mm）。デフォルト=1.0'
    )

    # 距離推定手法の選択
    parser.add_argument(
        '--use-average-speed',
        action='store_true',
        help='平均速度ベース距離推定法を使用（デフォルトはオフセット移動平均法）'
    )

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
    
    # ファイルハンドラ（RotatingFileHandler でサイズ制限付きローテーション）
    if config.logging.file:
        from logging.handlers import RotatingFileHandler
        log_file = Path(config.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_file,
                maxBytes=config.logging.max_bytes,
                backupCount=config.logging.backup_count,
                encoding='utf-8'
            )
        )
    
    # コンソールハンドラ
    if config.logging.console_output:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )



def main_twopass(
    config: Config,
    progress_callback: Optional[Any] = None
) -> Dict[str, Any]:
    """Two-pass processing のラッパー関数（後方互換性用）
    
    Args:
        config: 設定オブジェクト
        progress_callback: 進捗コールバック関数（オプション）
    
    Returns:
        処理結果の辞書
        {
            "success": bool,
            "colormap_path": str,
            "excel_path": str,
            "error": Optional[str]
        }
    
    Raises:
        Exception: 処理中にエラーが発生した場合
        
    Example:
        >>> from src.config import Config
        >>> config = Config.from_json("config.json")
        >>> result = main_twopass(config)
        >>> if result["success"]:
        ...     print(f"カラーマップ: {result['colormap_path']}")
    """
    try:
        # パイプライン実行
        pipeline = ColorMapPipelineTwoPass(config)
        pipeline.run()
        
        # 成功結果を返す
        return {
            "success": True,
            "colormap_path": config.output.colormap_path,
            "excel_path": config.output.excel_path,
            "error": None
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Two-pass processing failed: {e}")
        
        # エラー結果を返す
        return {
            "success": False,
            "colormap_path": None,
            "excel_path": None,
            "error": str(e)
        }


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
            config.logging.level = "DEBUG"
        if args.aov is not None:
            config.camera.fov_degrees = args.aov
        if args.pi is not None:
            config.pipe.diameter_mm = args.pi
        if args.ppm is not None:
            config.colormap.pixels_per_mm = args.ppm
        if args.use_average_speed:
            config.estimation.use_offset_moving_average = False

        # バリデーション
        config.validate()
        
        # ログ設定
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 60)
        logger.info("管内カメラカーシミュレーション用壁面画像生成処理（2パスフロー版）")
        logger.info("=" * 60)
        logger.info(f"入力動画: {config.input.video_path}")
        logger.info(f"画角: {config.camera.fov_degrees}度")
        logger.info(f"管径: {config.pipe.diameter_mm}mm")
        logger.info(f"デバッグモード: {config.debug.enabled}")
        logger.info("=" * 60)
        
        # ========================================
        # ProgressReporter 早期生成
        # ========================================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        progress_rep: Optional[ProgressReporter] = None

        if config.output.progress_path:
            # 総フレーム数を取得
            video_path = str(config.input.video_path)
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                effective_end_frame = config.input.end_frame
                if effective_end_frame is None:
                    effective_end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                total_frames = effective_end_frame - config.input.start_frame + 1

                progress_rep = ProgressReporter(
                    output_path=Path(config.output.progress_path.replace(
                        '{process_id}', timestamp
                    )),
                    total_frames=total_frames
                )
                steps = [
                    {"step_id": 1, "name_ja": "パラメータ調整", "name_en": "Parameter Tuning"},
                    {"step_id": 2, "name_ja": "フレーム解析", "name_en": "Frame Analysis"},
                    {"step_id": 3, "name_ja": "カラーマップ生成", "name_en": "Colormap Generation"},
                    {"step_id": 4, "name_ja": "カラーマップ補正", "name_en": "Colormap Correction"}
                ]
                progress_rep.initialize_steps(steps)
                logger.info("ProgressReporter早期初期化完了（main()内）")
            else:
                cap.release()

        # ========================================
        # 自動チューニング機能（Phase Config-2.5）
        # ========================================
        
        auto_tune_start_time = None
        auto_tune_end_time = None

        if config.auto_tune_enabled:
            logger.info("=" * 80)
            logger.info("自動チューニング機能: 有効")
            logger.info(f"分析対象: {config.input.video_path}")
            logger.info(f"サンプルフレーム数: {config.auto_tune_sample_frames}")
            logger.info(f"フレームスキップ: {config.auto_tune_frame_skip}")
            logger.info("=" * 80)

            auto_tune_start_time = datetime.now().isoformat()

            # Step 1 開始
            if progress_rep:
                progress_rep.start_step(1)

            try:
                # AutoTuner初期化
                auto_tuner = AutoTuner(config, config.input.video_path, logger)

                # 進捗コールバック（1%単位で間引き）
                _step1_last_pct = [-1.0]  # mutableでクロージャから更新

                def auto_tune_progress_callback(processed: int, total: int) -> None:
                    """AutoTuner進捗コールバック（1%刻み）"""
                    if progress_rep:
                        try:
                            pct = (processed / total) * 100.0 if total > 0 else 0.0
                            if pct - _step1_last_pct[0] >= 1.0 or processed >= total:
                                progress_rep.update_step(1, pct)
                                _step1_last_pct[0] = pct
                        except Exception:
                            pass  # 進捗更新失敗は無視

                # 動画分析
                logger.info("動画特性分析を開始します...")
                video_stats = auto_tuner.analyze_video(
                    num_sample_frames=config.auto_tune_sample_frames,
                    frame_skip=config.auto_tune_frame_skip,
                    progress_callback=auto_tune_progress_callback
                )

                # パラメータ調整
                logger.info("パラメータ自動調整を開始します...")
                config = auto_tuner.tune_parameters(video_stats)

                logger.info("自動チューニング完了")
                logger.info("=" * 80)

                # Step 1 完了
                if progress_rep:
                    progress_rep.complete_step(1, success=True)

            except FileNotFoundError as e:
                logger.error(f"自動チューニング失敗（動画ファイルなし）: {e}")
                logger.warning("デフォルト設定で処理を続行します")
                if progress_rep:
                    progress_rep.complete_step(1, success=False, error_message=str(e))
            except Exception as e:
                logger.warning(f"自動チューニング失敗（予期しないエラー）: {e}")
                logger.warning("デフォルト設定で処理を続行します")
                if progress_rep:
                    progress_rep.complete_step(1, success=False, error_message=str(e))
                # 元のconfigで続行（フォールバック）

            auto_tune_end_time = datetime.now().isoformat()
        else:
            # auto_tune無効時はスキップ
            if progress_rep:
                progress_rep.skip_step(1, reason="auto_tune_enabled=false")

        # パイプライン実行
        pipeline = ColorMapPipelineTwoPass(config)
        pipeline.auto_tune_start_time = auto_tune_start_time
        pipeline.auto_tune_end_time = auto_tune_end_time
        pipeline.progress_rep = progress_rep
        pipeline._timestamp = timestamp
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
