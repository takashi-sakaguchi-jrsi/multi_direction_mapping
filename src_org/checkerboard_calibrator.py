"""チェッカーボードキャリブレーションモジュール（Fisheye/Pinhole Model）

このモジュールは、チェッカーボードパターンからカメラパラメータを推定します。
2つのカメラモデルに対応:
- Fisheye Model: cv2.fisheye.calibrate()を使用、魚眼レンズ（FOV > 180°）向け等距離射影モデル
- Pinhole Model: cv2.calibrateCamera()を使用、標準レンズ（FOV < 180°）向けブラウン歪みモデル

キャリブレーション結果はLensCalibration v2.0形式で保存され、
fx, fy, cx, cy, image_width, image_height, camera_model などのフィールドを含みます。

Author: Claude Code
Date: 2025-11-13
Last Modified: 2025-11-18 (TASK-15 Phase 3: Pinhole Model対応追加)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np


# ==================== カスタム例外 ====================


class CalibrationError(Exception):
    """Base exception for calibration errors"""
    pass


class InsufficientCornersError(CalibrationError):
    """Insufficient corners detected
    
    Attributes:
        detected_count: 検出されたフレーム数
        required_count: 必要なフレーム数
    """
    
    def __init__(self, message: str, detected_count: int = 0, required_count: int = 0):
        """
        Args:
            message: エラーメッセージ
            detected_count: 検出されたフレーム数
            required_count: 必要なフレーム数
        """
        super().__init__(message)
        self.detected_count = detected_count
        self.required_count = required_count


class CalibrationFailedError(CalibrationError):
    """Calibration failed with high RMS error
    
    Attributes:
        rms_error: 再投影誤差
        threshold: 閾値
    """
    
    def __init__(self, message: str, rms_error: float = 0.0, threshold: float = 0.0):
        """
        Args:
            message: エラーメッセージ
            rms_error: 再投影誤差
            threshold: 閾値
        """
        super().__init__(message)
        self.rms_error = rms_error
        self.threshold = threshold


# ==================== チェッカーボードキャリブレーションクラス ====================


class CheckerboardCalibrator:
    """チェッカーボードキャリブレーションクラス
    
    チェッカーボードパターンからカメラパラメータを推定します。
    
    Attributes:
        pattern_size: チェッカーボードパターンサイズ（横、縦）
        square_size: チェッカーボードの正方形サイズ（mm）
        camera_model: カメラモデル（"fisheye" or "pinhole"）
        calibration_flags: cv2.calibrateCamera()のフラグ
        logger: ロガー
    
    Example:
        >>> # Fisheye Model（魚眼レンズ）
        >>> calibrator = CheckerboardCalibrator(
        ...     pattern_size=(9, 6),
        ...     square_size=25.0,
        ...     camera_model="fisheye"
        ... )
        >>> frames = [cv2.imread(f"frame_{i}.png") for i in range(20)]
        >>> result = calibrator.calibrate_from_frames(frames, (1920, 1080))
        >>> calibrator.save_calibration(result, Path("calibration.json"))
        
        >>> # Pinhole Model（標準レンズ）
        >>> calibrator_pinhole = CheckerboardCalibrator(
        ...     pattern_size=(9, 6),
        ...     square_size=25.0,
        ...     camera_model="pinhole"
        ... )
        >>> result = calibrator_pinhole.calibrate_from_frames(frames, (1920, 1080))
    """
    
    def __init__(
        self,
        pattern_size: Tuple[int, int] = (9, 6),
        square_size: float = 25.0,
        camera_model: str = "fisheye",
        calibration_flags: int = 0,
        logger: Optional[logging.Logger] = None
    ):
        """初期化
        
        Args:
            pattern_size: チェッカーボードパターンサイズ（横、縦）デフォルト: (9, 6)
            square_size: チェッカーボードの正方形サイズ（mm）デフォルト: 25.0
            camera_model: カメラモデル（"fisheye" or "pinhole"）デフォルト: "fisheye"
            calibration_flags: cv2.calibrateCamera()のフラグ デフォルト: 0
            logger: ロガー（オプション）
        
        Raises:
            ValueError: パラメータが不正な場合
        """
        # パラメータバリデーション
        if not (isinstance(pattern_size, tuple) and len(pattern_size) == 2):
            raise ValueError(
                f"pattern_size must be a tuple of 2 integers, got {pattern_size}"
            )
        
        if pattern_size[0] <= 0 or pattern_size[1] <= 0:
            raise ValueError(
                f"pattern_size elements must be positive, got {pattern_size}"
            )
        
        if square_size <= 0:
            raise ValueError(f"square_size must be positive, got {square_size}")
        
        if camera_model not in ["fisheye", "pinhole"]:
            raise ValueError(
                f"camera_model must be 'fisheye' or 'pinhole', got {camera_model}"
            )
        
        # 属性を設定
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.camera_model = camera_model
        self.calibration_flags = calibration_flags
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(
            f"CheckerboardCalibrator initialized: "
            f"pattern_size={pattern_size}, "
            f"square_size={square_size}mm, "
            f"camera_model={camera_model}, "
            f"flags={calibration_flags}"
        )
    
    def detect_corners(
        self,
        frames: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """チェッカーボードコーナー検出
        
        各フレームでcv2.findChessboardCorners()を実行し、
        検出成功したフレームのみcv2.cornerSubPix()でサブピクセル精度に改善します。
        
        Args:
            frames: フレームリスト（BGR形式）
        
        Returns:
            検出成功フレームの[(object_points, image_points), ...]
            - object_points: 3D座標（z=0の平面）、shape=(N, 3)
            - image_points: 2D座標、shape=(N, 2)
        
        Raises:
            ValueError: framesが空の場合
        """
        if not frames:
            raise ValueError("frames list is empty")
        
        self.logger.info(f"コーナー検出開始: {len(frames)}フレーム")
        
        # 3Dオブジェクト座標を準備（z=0の平面）
        objp = np.zeros(
            (self.pattern_size[0] * self.pattern_size[1], 3),
            dtype=np.float32
        )
        objp[:, :2] = np.mgrid[
            0:self.pattern_size[0],
            0:self.pattern_size[1]
        ].T.reshape(-1, 2)
        objp *= self.square_size
        
        # 検出成功フレームのリスト
        object_points_list = []
        image_points_list = []
        
        # cornerSubPixの基準
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
        
        # 各フレームでコーナー検出
        for i, frame in enumerate(frames):
            # グレースケール変換
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # チェッカーボードコーナー検出
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret:
                # サブピクセル精度に改善
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    winSize=(11, 11),
                    zeroZone=(-1, -1),
                    criteria=criteria
                )
                
                object_points_list.append(objp)
                image_points_list.append(corners_refined)
                
                self.logger.debug(
                    f"フレーム {i}: コーナー検出成功 "
                    f"({self.pattern_size[0]}x{self.pattern_size[1]}={len(corners_refined)} points)"
                )
            else:
                self.logger.debug(f"フレーム {i}: コーナー検出失敗")
        
        # 検出成功率をログ出力
        success_rate = len(object_points_list) / len(frames) * 100
        self.logger.info(
            f"コーナー検出完了: {len(object_points_list)}/{len(frames)}フレーム "
            f"({success_rate:.1f}%)"
        )
        
        return list(zip(object_points_list, image_points_list))
    
    def calibrate(
        self,
        object_points_list: List[np.ndarray],
        image_points_list: List[np.ndarray],
        image_size: Tuple[int, int],
        fov_degrees: float = 185.0
    ) -> Dict[str, Any]:
        """カメラキャリブレーション実行（Fisheye/Pinhole Model）

        カメラモデル（self.camera_model）に応じて適切なOpenCV関数を使用します:
        - Fisheye Model: cv2.fisheye.calibrate()を使用、等距離射影モデル（FOV > 180°対応）
        - Pinhole Model: cv2.calibrateCamera()を使用、ブラウン歪みモデル（標準レンズ）

        Args:
            object_points_list: 3Dオブジェクト座標リスト
            image_points_list: 2D画像座標リスト
            image_size: 画像サイズ（幅、高さ）
            fov_degrees: 視野角（度）デフォルト: 185.0（Fisheyeモード時のみ、InitExtrinsics対策で使用）

        Returns:
            キャリブレーション結果辞書 {
                "camera_matrix": np.ndarray,  # 3x3カメラ行列 K
                "dist_coeffs": np.ndarray,    # 歪み係数（Fisheye: 4x1, Pinhole: 1x5）
                "rvecs": List[np.ndarray],    # 回転ベクトルリスト
                "tvecs": List[np.ndarray],    # 並進ベクトルリスト
                "rms_error": float            # 再投影誤差（RMS）
            }

        Raises:
            ValueError: 入力パラメータが不正な場合
            CalibrationFailedError: キャリブレーション処理が失敗した場合
        """
        if not object_points_list or not image_points_list:
            raise ValueError("object_points_list and image_points_list must not be empty")

        if len(object_points_list) != len(image_points_list):
            raise ValueError(
                f"object_points_list and image_points_list must have same length: "
                f"{len(object_points_list)} != {len(image_points_list)}"
            )

        if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError(f"image_size must be (width, height) with positive values, got {image_size}")

        # カメラモデルごとに処理を分岐
        if self.camera_model == "fisheye":
            return self._calibrate_fisheye(object_points_list, image_points_list, image_size, fov_degrees)
        elif self.camera_model == "pinhole":
            return self._calibrate_pinhole(object_points_list, image_points_list, image_size)
        else:
            raise ValueError(f"Unsupported camera_model: {self.camera_model}")
    
    def _calibrate_fisheye(
        self,
        object_points_list: List[np.ndarray],
        image_points_list: List[np.ndarray],
        image_size: Tuple[int, int],
        fov_degrees: float
    ) -> Dict[str, Any]:
        """魚眼レンズキャリブレーション実行（内部メソッド）
        
        cv2.fisheye.calibrate()を使用して等距離射影モデルでキャリブレーションを実行します。
        
        Args:
            object_points_list: 3Dオブジェクト座標リスト
            image_points_list: 2D画像座標リスト
            image_size: 画像サイズ（幅、高さ）
            fov_degrees: 視野角（度）
        
        Returns:
            キャリブレーション結果辞書（calibrate()のdocstring参照）
        """
        self.logger.info(
            f"魚眼キャリブレーション実行: {len(object_points_list)}フレーム, "
            f"画像サイズ={image_size}"
        )

        # cv2.fisheye.calibrate()用にデータ形状・型を変換
        # OpenCVのfisheye APIは、(N, 1, 3)の形状とfloat32/float64型を要求
        object_points_converted = []
        for objp in object_points_list:
            if objp.ndim == 2 and objp.shape[1] == 3:
                # (N, 3) → (N, 1, 3) に変換、float32を維持（検出時と同じ型）
                objp_reshaped = objp.reshape(-1, 1, 3)
                object_points_converted.append(objp_reshaped)
            else:
                # すでに正しい形状の場合はそのまま
                object_points_converted.append(objp)

        image_points_converted = []
        for imgp in image_points_list:
            # cv2.cornerSubPixの出力はfloat32の(N, 1, 2)形状
            image_points_converted.append(imgp)

        # 初期推定値の準備
        # カメラ行列K（初期値: FOVから焦点距離を推定）
        # 魚眼レンズの等距離射影モデル: r = f * θ
        # θ_max = FOV / 2 (rad), r_max = min(width, height) / 2
        # 魚眼レンズの有効領域は短辺を直径とする円（外側はブラック）
        # したがって: f = r_max / θ_max
        theta_max_rad = np.deg2rad(fov_degrees / 2)  # 視野角の半分（ラジアン）
        r_max = min(image_size[0], image_size[1]) / 2  # 短辺の半径（有効画像半径、pixels）
        initial_focal_length = r_max / theta_max_rad  # 初期焦点距離推定（FOVベース）

        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = initial_focal_length  # FOVベースの初期焦点距離
        K[0, 2] = image_size[0] / 2  # 主点X
        K[1, 2] = image_size[1] / 2  # 主点Y

        self.logger.debug(
            f"K行列初期値（Fisheye）: FOV={fov_degrees}° → 初期焦点距離={initial_focal_length:.2f} pixels "
            f"（有効半径={r_max:.0f}px、従来の推定={image_size[0]/2:.2f}pxから修正）"
        )

        # 歪み係数D（初期値: ゼロ、4パラメータ: k1, k2, k3, k4）
        D = np.zeros((4, 1), dtype=np.float64)

        # 回転・並進ベクトル（OpenCVに自動初期化させるためNoneを使用）
        # ゼロベクトルでの初期化はInitExtrinsicsエラーの原因となる
        rvecs = None
        tvecs = None

        # キャリブレーションフラグ（魚眼モデル用）
        # CALIB_CHECK_CONDを削除: Ill-conditioned matrixエラー回避
        # （Phase 2対応前に通っていたデータとの互換性を維持）
        # 第1試行: CALIB_RECOMPUTE_EXTRINSICを使用（高精度）
        # 失敗時: フラグを削除してリトライ（安定性優先）
        # CALIB_USE_INTRINSIC_GUESS: FOVベースの初期焦点距離を使用
        calibration_flags_strict = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_FIX_SKEW +
            cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        )
        calibration_flags_relaxed = (
            cv2.fisheye.CALIB_FIX_SKEW +
            cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        )

        # 魚眼キャリブレーション実行（条件付きリトライ）
        rms_error = None
        K_result = None
        D_result = None
        rvecs_result = None
        tvecs_result = None

        # 第1試行: 高精度モード（CALIB_RECOMPUTE_EXTRINSIC使用）
        try:
            self.logger.debug("キャリブレーション第1試行: 高精度モード（CALIB_RECOMPUTE_EXTRINSIC使用）")
            rms_error, K_result, D_result, rvecs_result, tvecs_result = cv2.fisheye.calibrate(
                object_points_converted,
                image_points_converted,
                image_size,
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags_strict,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
            self.logger.debug("第1試行成功: 高精度モードで完了")
        except cv2.error as e:
            error_msg = str(e)
            # InitExtrinsicsエラーの場合のみリトライ
            if "InitExtrinsics" in error_msg and "fabs(norm_u1) > 0" in error_msg:
                self.logger.warning(f"第1試行失敗: InitExtrinsicsエラー検出 - {error_msg}")
                self.logger.info("第2試行: 安定性優先モード（CALIB_RECOMPUTE_EXTRINSIC削除）でリトライ")
                try:
                    rms_error, K_result, D_result, rvecs_result, tvecs_result = cv2.fisheye.calibrate(
                        object_points_converted,
                        image_points_converted,
                        image_size,
                        K,
                        D,
                        rvecs,
                        tvecs,
                        calibration_flags_relaxed,
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                    )
                    self.logger.info("第2試行成功: 安定性優先モードで完了（RMS精度は低下する可能性があります）")
                except cv2.error as e2:
                    raise CalibrationFailedError(f"cv2.fisheye.calibrate()失敗（第2試行も失敗）: {e2}")
            else:
                # InitExtrinsicsエラー以外はそのまま例外を投げる
                raise CalibrationFailedError(f"cv2.fisheye.calibrate()失敗: {e}")

        # 結果を整形（後方互換性のため、変数名はcamera_matrix, dist_coeffsを使用）
        camera_matrix = K_result
        dist_coeffs = D_result

        # ログ出力
        self.logger.info(f"RMS再投影誤差（Fisheye）: {rms_error:.4f}")
        self.logger.info(f"カメラ行列:\n{camera_matrix}")
        self.logger.info(f"歪み係数（k1, k2, k3, k4）: {dist_coeffs.ravel()}")

        return {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "rvecs": rvecs_result,
            "tvecs": tvecs_result,
            "rms_error": float(rms_error)
        }
    
    def _calibrate_pinhole(
        self,
        object_points_list: List[np.ndarray],
        image_points_list: List[np.ndarray],
        image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """ピンホールカメラキャリブレーション実行（内部メソッド）
        
        cv2.calibrateCamera()を使用してブラウン歪みモデルでキャリブレーションを実行します。
        
        Args:
            object_points_list: 3Dオブジェクト座標リスト
            image_points_list: 2D画像座標リスト
            image_size: 画像サイズ（幅、高さ）
        
        Returns:
            キャリブレーション結果辞書（calibrate()のdocstring参照）
        """
        self.logger.info(
            f"ピンホールキャリブレーション実行: {len(object_points_list)}フレーム, "
            f"画像サイズ={image_size}"
        )

        # cv2.calibrateCamera()用のデータ形状変換は不要
        # OpenCVのpinhole APIは(N, 3)と(N, 2)形状を受け入れる
        # ただし、cornerSubPixの出力は(N, 1, 2)なのでreshapeが必要
        object_points_converted = []
        for objp in object_points_list:
            if objp.ndim == 2 and objp.shape[1] == 3:
                # (N, 3)形状はそのまま使用
                object_points_converted.append(objp)
            elif objp.ndim == 3 and objp.shape[1] == 1 and objp.shape[2] == 3:
                # (N, 1, 3) → (N, 3) に変換
                object_points_converted.append(objp.reshape(-1, 3))
            else:
                object_points_converted.append(objp)

        image_points_converted = []
        for imgp in image_points_list:
            if imgp.ndim == 3 and imgp.shape[1] == 1 and imgp.shape[2] == 2:
                # (N, 1, 2) → (N, 2) に変換
                image_points_converted.append(imgp.reshape(-1, 2))
            else:
                image_points_converted.append(imgp)

        # 初期推定値の準備
        # カメラ行列K（初期値: None でOpenCVに自動推定させる）
        K = None

        # 歪み係数D（初期値: None でOpenCVに自動推定させる）
        D = None

        # キャリブレーションフラグ（ピンホールモデル用）
        # CALIB_RATIONAL_MODEL: 高精度な有理歪みモデル（k1, k2, p1, p2, k3, k4, k5, k6）を使用
        flags = cv2.CALIB_RATIONAL_MODEL

        # ピンホールキャリブレーション実行
        # InitExtrinsicsエラーはピンホールでは発生しないため、リトライ不要
        try:
            self.logger.debug("ピンホールキャリブレーション実行: CALIB_RATIONAL_MODELフラグ使用")
            rms_error, K_result, D_result, rvecs_result, tvecs_result = cv2.calibrateCamera(
                object_points_converted,
                image_points_converted,
                image_size,
                K,  # None: OpenCVに自動推定させる
                D,  # None: OpenCVに自動推定させる
                flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        except cv2.error as e:
            raise CalibrationFailedError(f"cv2.calibrateCamera()失敗: {e}")

        # 結果を整形
        camera_matrix = K_result
        dist_coeffs = D_result  # (1, 5) or (1, 8) or (5,) or (8,) など

        # ログ出力（Pinhole歪み係数は k1, k2, p1, p2, k3, [k4, k5, k6]）
        dist_ravel = dist_coeffs.ravel()
        self.logger.info(f"RMS再投影誤差（Pinhole）: {rms_error:.4f}")
        self.logger.info(f"カメラ行列:\n{camera_matrix}")
        self.logger.info(
            f"歪み係数（k1, k2, p1, p2, k3, ...）: {dist_ravel[:5]} "
            f"（全{len(dist_ravel)}パラメータ）"
        )

        return {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "rvecs": rvecs_result,
            "tvecs": tvecs_result,
            "rms_error": float(rms_error)
        }
    
    def save_calibration(
        self,
        calibration_result: Dict[str, Any],
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        fov_degrees: float = 185.0,
        image_size: Optional[Tuple[int, int]] = None
    ) -> None:
        """キャリブレーション結果をLensCalibration v2.0形式で保存

        src/calibration.pyのsave_calibration()関数を使用して、
        LensCalibration v2.0形式でキャリブレーション結果を保存します。

        Args:
            calibration_result: calibrate()の戻り値
            output_path: 出力ファイルパス
            metadata: メタデータ（オプション）- タイムスタンプ、動画ファイル名など
            fov_degrees: 視野角（度）デフォルト: 185.0
            image_size: 画像サイズ（幅、高さ）オプション

        Raises:
            ValueError: calibration_resultが不正な場合
        """
        if "camera_matrix" not in calibration_result or "dist_coeffs" not in calibration_result:
            raise ValueError("calibration_result must contain 'camera_matrix' and 'dist_coeffs'")

        from datetime import datetime
        from src.calibration import save_calibration as save_calib_function, LensCalibration

        # カメラ行列と歪み係数を取得
        K = calibration_result["camera_matrix"]
        D = calibration_result["dist_coeffs"]
        rms_error = calibration_result["rms_error"]

        # 歪み係数を取得（モデルに応じて異なる）
        dist_ravel = D.ravel()

        # LensCalibrationオブジェクト生成
        # Fisheye Model (k1, k2, k3, k4) vs Pinhole Model (k1, k2, p1, p2, k3, ...)
        if metadata:
            lens_model = metadata.get(
                "lens_model",
                f"{self.camera_model}_checkerboard_{datetime.now().strftime('%Y%m%d')}"
            )
            notes = metadata.get("notes", "")
        else:
            lens_model = f"{self.camera_model}_checkerboard_{datetime.now().strftime('%Y%m%d')}"
            notes = ""

        if self.camera_model == "fisheye":
            # Fisheye Model: k1, k2, k3, k4
            calibration = LensCalibration(
                lens_model=lens_model,
                calibration_date=datetime.now().isoformat() + "Z",
                focal_length_mm=float(K[0, 0]),  # ピクセル単位（mm変換は後で実施予定）
                fov_degrees=fov_degrees,
                center_offset_x=0,  # 主点オフセットは後で計算
                center_offset_y=0,

                # v2.0フィールド
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                image_width=image_size[0] if image_size else 0,
                image_height=image_size[1] if image_size else 0,
                camera_model="fisheye",

                # Fisheye歪み係数（k1, k2, k3）+ k4フィールド
                radial_distortion_k1=float(dist_ravel[0]),
                radial_distortion_k2=float(dist_ravel[1]),
                radial_distortion_k3=float(dist_ravel[2]) if len(dist_ravel) > 2 else 0.0,
                radial_distortion_k4=float(dist_ravel[3]) if len(dist_ravel) > 3 else 0.0,
                tangential_distortion_p1=0.0,  # Fisheye Modelはp1, p2無し
                tangential_distortion_p2=0.0,

                calibration_rms_error=float(rms_error),
                notes=notes,
                calibration_version="2.0"
            )
        elif self.camera_model == "pinhole":
            # Pinhole Model: k1, k2, p1, p2, k3, [k4, k5, k6]
            calibration = LensCalibration(
                lens_model=lens_model,
                calibration_date=datetime.now().isoformat() + "Z",
                focal_length_mm=float(K[0, 0]),
                fov_degrees=fov_degrees,
                center_offset_x=0,
                center_offset_y=0,

                # v2.0フィールド
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                image_width=image_size[0] if image_size else 0,
                image_height=image_size[1] if image_size else 0,
                camera_model="pinhole",

                # Pinhole歪み係数（k1, k2, p1, p2, k3）
                radial_distortion_k1=float(dist_ravel[0]),
                radial_distortion_k2=float(dist_ravel[1]),
                radial_distortion_k3=float(dist_ravel[4]) if len(dist_ravel) > 4 else 0.0,
                radial_distortion_k4=0.0,  # PinholeではLensCalibration.k4は未使用
                tangential_distortion_p1=float(dist_ravel[2]) if len(dist_ravel) > 2 else 0.0,
                tangential_distortion_p2=float(dist_ravel[3]) if len(dist_ravel) > 3 else 0.0,

                calibration_rms_error=float(rms_error),
                notes=notes,
                calibration_version="2.0"
            )
        else:
            raise ValueError(f"Unsupported camera_model: {self.camera_model}")

        # LensCalibration形式で保存
        save_calib_function(calibration, str(output_path))

        self.logger.info(
            f"LensCalibration v2.0形式で保存完了（{self.camera_model}）: {output_path} "
            f"(fx={calibration.fx:.2f}, fy={calibration.fy:.2f}, RMS={rms_error:.4f})"
        )
    
    def load_calibration(self, input_path: Path) -> Dict[str, Any]:
        """キャリブレーション結果を読み込み

        LensCalibration v2.0形式のJSON、または旧形式のJSONを読み込み、
        NumPy配列に変換します。

        Args:
            input_path: 入力ファイルパス

        Returns:
            calibrate()と同じ形式の辞書 {
                "camera_matrix": np.ndarray,  # 3x3
                "dist_coeffs": np.ndarray,    # Fisheye: (4,1), Pinhole: (1,5)
                "rms_error": float
            }

        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: JSONの形式が不正な場合
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {input_path}")

        # LensCalibration v2.0形式で読み込み
        from src.calibration import load_calibration as load_calib_function

        try:
            calibration = load_calib_function(str(input_path))

            # カメラ行列を構築
            camera_matrix = np.array([
                [calibration.fx, 0, calibration.cx],
                [0, calibration.fy, calibration.cy],
                [0, 0, 1]
            ], dtype=np.float64)

            # 歪み係数を構築（camera_modelフィールドで判定）
            camera_model = getattr(calibration, "camera_model", "fisheye")  # デフォルトfisheye（後方互換性）
            
            if camera_model == "fisheye":
                # Fisheye Model: k1, k2, k3, k4
                dist_coeffs = np.array([
                    [calibration.radial_distortion_k1],
                    [calibration.radial_distortion_k2],
                    [calibration.radial_distortion_k3],
                    [getattr(calibration, "radial_distortion_k4", 0.0)]
                ], dtype=np.float64)
            elif camera_model == "pinhole":
                # Pinhole Model: k1, k2, p1, p2, k3
                dist_coeffs = np.array([[
                    calibration.radial_distortion_k1,
                    calibration.radial_distortion_k2,
                    calibration.tangential_distortion_p1,
                    calibration.tangential_distortion_p2,
                    calibration.radial_distortion_k3
                ]], dtype=np.float64)
            else:
                raise ValueError(f"Unsupported camera_model in calibration file: {camera_model}")

            rms_error = calibration.calibration_rms_error if calibration.calibration_rms_error is not None else 0.0
        except Exception as e:
            raise ValueError(f"Failed to load calibration file: {input_path} - {e}")
        
        self.logger.info(f"キャリブレーション結果を読み込み（{camera_model}）: {input_path}")
        self.logger.info(f"RMS再投影誤差: {rms_error:.4f}")
        
        return {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "rms_error": rms_error
        }
    
    def calibrate_from_frames(
        self,
        frames: List[np.ndarray],
        image_size: Tuple[int, int],
        fov_degrees: float = 185.0
    ) -> Dict[str, Any]:
        """フレームからキャリブレーション実行（統合メソッド）

        detect_corners() → calibrate() を順次実行します。

        Args:
            frames: フレームリスト（BGR形式）
            image_size: 画像サイズ（幅、高さ）
            fov_degrees: 視野角（度）デフォルト: 185.0（Fisheyeモード時のみ使用）

        Returns:
            calibrate()と同じ形式の辞書

        Raises:
            InsufficientCornersError: 検出成功フレーム数が10未満の場合
            CalibrationFailedError: RMS誤差が4.0を超える場合
        """
        self.logger.info("=" * 60)
        self.logger.info(f"フレームからキャリブレーション開始（{self.camera_model}）")
        self.logger.info("=" * 60)
        
        # 1. コーナー検出
        corners_list = self.detect_corners(frames)
        
        # 検出成功フレーム数チェック
        if len(corners_list) < 10:
            raise InsufficientCornersError(
                f"検出成功フレーム数が不足しています（最小: 10フレーム）",
                detected_count=len(corners_list),
                required_count=10
            )
        
        # object_points_listとimage_points_listに分離
        object_points_list = [obj_pts for obj_pts, _ in corners_list]
        image_points_list = [img_pts for _, img_pts in corners_list]

        # 2. キャリブレーション実行（FOVパラメータを渡す）
        result = self.calibrate(object_points_list, image_points_list, image_size, fov_degrees)
        
        # RMS誤差チェック
        if result["rms_error"] > 4.0:
            raise CalibrationFailedError(
                f"RMS誤差が閾値を超えています",
                rms_error=result["rms_error"],
                threshold=4.0
            )
        
        self.logger.info("=" * 60)
        self.logger.info("キャリブレーション完了")
        self.logger.info("=" * 60)
        
        return result
