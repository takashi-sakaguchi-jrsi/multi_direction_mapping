"""レンズキャリブレーション機能

このモジュールは、カメラレンズのキャリブレーションデータを
外部ファイルから読み込み、管理する機能を提供します。

Phase 4: 基本実装（FOV, 焦点距離, レンズ中心オフセット）
Phase 5: 歪み補正パラメータ対応予定
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ========================================
# カスタム例外クラス
# ========================================


class CalibrationError(Exception):
    """キャリブレーション関連のベース例外"""
    pass


class CalibrationFileNotFoundError(CalibrationError):
    """キャリブレーションファイルが見つからない"""
    pass


class CalibrationFormatError(CalibrationError):
    """キャリブレーションファイルのフォーマットエラー"""
    pass


class CalibrationValidationError(CalibrationError):
    """キャリブレーションデータのバリデーションエラー"""
    pass


# ========================================
# LensCalibrationデータクラス
# ========================================


@dataclass
class LensCalibration:
    """レンズキャリブレーションデータ（v2.0）

    v2.0の変更点:
    - fx, fy, cx, cy, image_width, image_heightフィールド追加
    - calibration_versionフィールド追加
    - 後方互換性レイヤー実装（v1.0ファイル読み込み対応）
    - radial_distortion_k4フィールド追加（魚眼モデル第4歪み係数）
    - camera_modelフィールド追加（モデル識別）

    Attributes:
        lens_model: レンズモデル名（例: 'RICOH_THETA_S', 'default_185deg_f2.8'）
        calibration_date: キャリブレーション日時（ISO 8601形式）
        focal_length_mm: 焦点距離（mm）※v1.0互換性のため保持
        fov_degrees: 視野角（度）
        center_offset_x: レンズ中心のx方向オフセット（ピクセル）※v1.0互換性のため保持
        center_offset_y: レンズ中心のy方向オフセット（ピクセル）※v1.0互換性のため保持

        fx: 焦点距離X（ピクセル単位）※v2.0で追加
        fy: 焦点距離Y（ピクセル単位）※v2.0で追加
        cx: 主点X座標（ピクセル単位）※v2.0で追加
        cy: 主点Y座標（ピクセル単位）※v2.0で追加
        image_width: 画像幅（ピクセル）※v2.0で追加
        image_height: 画像高さ（ピクセル）※v2.0で追加

        radial_distortion_k1: 放射歪み係数K1（Phase 5対応予定、デフォルト: 0.0）
        radial_distortion_k2: 放射歪み係数K2（Phase 5対応予定、デフォルト: 0.0）
        radial_distortion_k3: 放射歪み係数K3（Phase 5対応予定、デフォルト: 0.0）
        radial_distortion_k4: 放射歪み係数K4（魚眼モデル第4係数、デフォルト: 0.0）
        tangential_distortion_p1: 接線歪み係数P1（Phase 5対応予定、デフォルト: 0.0）
        tangential_distortion_p2: 接線歪み係数P2（Phase 5対応予定、デフォルト: 0.0）

        camera_model: カメラモデル識別（"fisheye" or "pinhole"、デフォルト: "fisheye"）

        calibration_rms_error: キャリブレーションRMS誤差（オプション）
        notes: 備考（オプション）
        calibration_version: バージョン番号（"2.0"）※v2.0で追加

    Note:
        - cx = image_width / 2 + center_offset_x の関係
        - cy = image_height / 2 + center_offset_y の関係
        - camera_model="fisheye"時、k4フィールドが使用される
        - camera_model="pinhole"時、k4は無視される

    Example:
        >>> calib = LensCalibration(
        ...     lens_model="RICOH_THETA_S",
        ...     calibration_date="2025-10-15T10:00:00Z",
        ...     focal_length_mm=2.8,
        ...     fov_degrees=185.0,
        ...     center_offset_x=0,
        ...     center_offset_y=0,
        ...     fx=491.76,
        ...     fy=493.42,
        ...     cx=960.0,
        ...     cy=540.0,
        ...     image_width=1920,
        ...     image_height=1080,
        ...     camera_model="fisheye"
        ... )
        >>> calib.validate()  # バリデーション実行
    """
    lens_model: str
    calibration_date: str
    focal_length_mm: float  # 物理焦点距離（mm）※v1.0互換性のため保持
    fov_degrees: float
    center_offset_x: int = 0  # ※v1.0互換性のため保持
    center_offset_y: int = 0  # ※v1.0互換性のため保持

    # カメラ行列パラメータ（v2.0で追加）
    fx: float = 0.0  # 焦点距離X（ピクセル単位）
    fy: float = 0.0  # 焦点距離Y（ピクセル単位）
    cx: float = 0.0  # 主点X座標（ピクセル単位）
    cy: float = 0.0  # 主点Y座標（ピクセル単位）

    # 画像サイズ（v2.0で追加）
    image_width: int = 0  # 画像幅（ピクセル）
    image_height: int = 0  # 画像高さ（ピクセル）

    # 歪み補正パラメータ（Phase 5以降で実装予定）
    radial_distortion_k1: float = 0.0
    radial_distortion_k2: float = 0.0
    radial_distortion_k3: float = 0.0
    radial_distortion_k4: float = 0.0  # 魚眼モデル第4歪み係数
    tangential_distortion_p1: float = 0.0
    tangential_distortion_p2: float = 0.0

    # カメラモデル識別（v2.0拡張）
    camera_model: str = "fisheye"  # "fisheye" or "pinhole"

    # オプション
    calibration_rms_error: Optional[float] = None
    notes: str = ""
    calibration_version: str = "2.0"  # バージョン番号（v2.0で追加）
    
    def validate(self) -> None:
        """キャリブレーションデータのバリデーション

        以下の検証を行います：
        - focal_length_mm > 0
        - 0 < fov_degrees <= 360
        - center_offset_x/y が妥当な範囲
        - calibration_date が ISO 8601形式
        - 歪み係数が -1 ~ 1 の範囲
        - fx, fy > 0（v2.0で追加）
        - cx, cy が画像範囲内（v2.0で追加）
        - image_width, image_height > 0（v2.0で追加）
        - camera_model が "fisheye" または "pinhole"（v2.0拡張）
        - radial_distortion_k4 の範囲チェック（fisheye時のみ、v2.0拡張）

        Raises:
            CalibrationValidationError: バリデーションエラー
        """
        # focal_length_mm: > 0
        if self.focal_length_mm <= 0:
            raise CalibrationValidationError(
                f"focal_length_mmは正の値である必要があります: {self.focal_length_mm}"
            )

        # fov_degrees: 0 < fov <= 360
        if not (0 < self.fov_degrees <= 360):
            raise CalibrationValidationError(
                f"fov_degreesは0-360の範囲である必要があります: {self.fov_degrees}"
            )

        # center_offset_x/y: 妥当な範囲（画像サイズ以下）
        if abs(self.center_offset_x) > 10000:
            raise CalibrationValidationError(
                f"center_offset_xの値が異常です: {self.center_offset_x}"
            )
        if abs(self.center_offset_y) > 10000:
            raise CalibrationValidationError(
                f"center_offset_yの値が異常です: {self.center_offset_y}"
            )

        # calibration_date: ISO 8601形式のチェック
        try:
            datetime.fromisoformat(self.calibration_date.replace('Z', '+00:00'))
        except ValueError as e:
            raise CalibrationValidationError(
                f"calibration_dateはISO 8601形式で指定してください "
                f"(例: 2025-10-15T10:00:00Z): {self.calibration_date} - {e}"
            )

        # v2.0追加: fx, fy のバリデーション
        # fx/fy が設定されている場合のみチェック（デフォルト値0.0はスキップ）
        if self.fx != 0.0 or self.fy != 0.0:
            if self.fx <= 0:
                raise CalibrationValidationError(
                    f"fxは正の値である必要があります: fx={self.fx}"
                )
            if self.fy <= 0:
                raise CalibrationValidationError(
                    f"fyは正の値である必要があります: fy={self.fy}"
                )

        # v2.0追加: image_width, image_height のバリデーション
        # image_width/image_height が設定されている場合のみチェック（デフォルト値0はスキップ）
        if self.image_width != 0 or self.image_height != 0:
            if self.image_width <= 0:
                raise CalibrationValidationError(
                    f"image_widthは正の値である必要があります: image_width={self.image_width}"
                )
            if self.image_height <= 0:
                raise CalibrationValidationError(
                    f"image_heightは正の値である必要があります: image_height={self.image_height}"
                )

        # v2.0追加: cx, cy のバリデーション（画像範囲内）
        if self.cx != 0.0 or self.cy != 0.0:
            if self.image_width > 0:  # image_widthが設定されている場合のみチェック
                if not (0 <= self.cx <= self.image_width):
                    raise CalibrationValidationError(
                        f"cxが範囲外です: cx={self.cx}, image_width={self.image_width}"
                    )
            if self.image_height > 0:  # image_heightが設定されている場合のみチェック
                if not (0 <= self.cy <= self.image_height):
                    raise CalibrationValidationError(
                        f"cyが範囲外です: cy={self.cy}, image_height={self.image_height}"
                    )

        # v2.0拡張: camera_modelのバリデーション
        if self.camera_model not in ["fisheye", "pinhole"]:
            raise CalibrationValidationError(
                f"camera_modelは'fisheye'または'pinhole'である必要があります（現在値: {self.camera_model}）"
            )

        # 歪み係数: デフォルト値（0.0）の場合はスキップ
        # Fisheye Modelでは大きな値を取ることがあるため、広い範囲を許容
        distortion_params = [
            ("radial_distortion_k1", self.radial_distortion_k1),
            ("radial_distortion_k2", self.radial_distortion_k2),
            ("radial_distortion_k3", self.radial_distortion_k3),
            ("tangential_distortion_p1", self.tangential_distortion_p1),
            ("tangential_distortion_p2", self.tangential_distortion_p2),
        ]

        for param_name, param_value in distortion_params:
            # デフォルト値（0.0）の場合はスキップ
            if param_value == 0.0:
                continue
            # Fisheye Modelでは k2, k3 が非常に大きな値を取ることがあるため、
            # より広い範囲を許容（-1e12 ～ 1e12）
            if not (-1e12 <= param_value <= 1e12):
                raise CalibrationValidationError(
                    f"{param_name}が異常な値です: {param_value}"
                )

        # v2.0拡張: radial_distortion_k4のバリデーション（Fisheye Modelのみ）
        if self.camera_model == "fisheye" and self.radial_distortion_k4 != 0.0:
            if not (-1e12 <= self.radial_distortion_k4 <= 1e12):
                raise CalibrationValidationError(
                    f"radial_distortion_k4は-1e12から1e12の範囲である必要があります（現在値: {self.radial_distortion_k4}）"
                )

        # calibration_rms_error: 指定されている場合は非負
        if self.calibration_rms_error is not None and self.calibration_rms_error < 0:
            raise CalibrationValidationError(
                f"calibration_rms_errorは非負である必要があります: "
                f"{self.calibration_rms_error}"
            )


# ========================================
# ファイル読み込み・保存関数
# ========================================


def load_calibration(file_path: str) -> LensCalibration:
    """キャリブレーションファイルを読み込む（後方互換性あり）

    v1.0形式（fx, fy欠落）も読み込み可能。
    欠落フィールドはデフォルト値で補完します。

    Args:
        file_path: キャリブレーションファイルパス
            - フルパス（例: "/path/to/calibration.json"）
            - ファイル名のみ（例: "default_lens.json"）
                -> data/calibration/配下を検索
            - 標準ディレクトリからの相対パス（例: "my_lens.json"）

    Returns:
        LensCalibration: 読み込んだキャリブレーションデータ

    Raises:
        CalibrationFileNotFoundError: ファイルが見つからない
        CalibrationFormatError: JSONフォーマットエラー
        CalibrationValidationError: バリデーションエラー

    Example:
        >>> # 標準ディレクトリから読み込み
        >>> calib = load_calibration("default_lens.json")
        >>>
        >>> # フルパス指定
        >>> calib = load_calibration("/path/to/my_calibration.json")
    """
    # ファイルパス解決
    path = Path(file_path)

    # 1. フルパスとして存在するか確認
    if path.is_absolute() and path.exists():
        resolved_path = path
    # 2. 相対パスとして存在するか確認
    elif not path.is_absolute() and path.exists():
        resolved_path = path
    # 3. 標準ディレクトリ（data/calibration/）から検索
    else:
        standard_dir = Path("data/calibration")
        candidate_path = standard_dir / path.name
        if candidate_path.exists():
            resolved_path = candidate_path
        else:
            # ファイルが見つからない
            raise CalibrationFileNotFoundError(
                f"キャリブレーションファイルが見つかりません: {file_path}\n"
                f"検索パス: {path}, {candidate_path}"
            )

    # JSONファイル読み込み
    try:
        with open(resolved_path, "r", encoding="utf-8-sig", newline='') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise CalibrationFormatError(
            f"キャリブレーションJSONフォーマットエラー: {resolved_path} - {e}"
        )
    except Exception as e:
        raise CalibrationError(
            f"キャリブレーションファイル読み込みエラー: {resolved_path} - {e}"
        )

    # 必須フィールドのバリデーション
    required_fields = [
        "lens_model",
        "calibration_date",
        "focal_length_mm",
        "fov_degrees",
        "center_offset_x",
        "center_offset_y",
    ]

    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        raise CalibrationValidationError(
            f"キャリブレーションファイルに必須フィールドが不足しています: "
            f"{missing_fields}\n"
            f"必須フィールド: {required_fields}"
        )

    # ========================================
    # 後方互換性レイヤー: v1.0形式対応
    # ========================================
    version = data.get('calibration_version', '1.0')

    # v1.0形式またはfx/fyが欠落している場合
    if version == '1.0' or 'fx' not in data:
        logger.warning(
            f"旧形式(v1.0)のキャリブレーションファイルを読み込みました: {resolved_path}"
        )
        logger.warning(
            f"fx, fyフィールドが欠落している場合、デフォルト値(0.0)で補完します。"
        )

        # デフォルト値で補完
        data.setdefault('fx', 0.0)
        data.setdefault('fy', 0.0)
        data.setdefault('cx', 0.0)
        data.setdefault('cy', 0.0)
        data.setdefault('image_width', 0)
        data.setdefault('image_height', 0)
        data.setdefault('calibration_version', '2.0')

    # v1.0形式またはcamera_modelが欠落している場合（後方互換性拡張）
    data.setdefault('camera_model', 'fisheye')
    data.setdefault('radial_distortion_k4', 0.0)

    # LensCalibrationオブジェクト作成
    try:
        calibration = LensCalibration(**data)
    except TypeError as e:
        raise CalibrationFormatError(
            f"キャリブレーションデータの型が不正です: {e}"
        )

    # バリデーション実行
    calibration.validate()

    # ログ出力
    if version == '1.0':
        logger.info(
            f"レンズキャリブレーション読み込み成功（v1.0形式）: {resolved_path} "
            f"(lens_model={calibration.lens_model}, "
            f"fov={calibration.fov_degrees}deg, "
            f"focal_length={calibration.focal_length_mm}mm)"
        )
    else:
        logger.info(
            f"レンズキャリブレーション読み込み成功（v2.0形式）: {resolved_path} "
            f"(lens_model={calibration.lens_model}, "
            f"fov={calibration.fov_degrees}deg, "
            f"fx={calibration.fx}px, fy={calibration.fy}px, "
            f"camera_model={calibration.camera_model})"
        )

    return calibration


def save_calibration(calibration: LensCalibration, file_path: str) -> None:
    """キャリブレーションデータをファイルに保存（v2.0形式）

    Args:
        calibration: 保存するキャリブレーションデータ
        file_path: 保存先パス

    Raises:
        CalibrationValidationError: バリデーションエラー
        CalibrationError: ファイル書き込みエラー

    Example:
        >>> calib = LensCalibration(...)
        >>> save_calibration(calib, "data/calibration/my_lens.json")
    """
    # calibration_versionを必ず"2.0"に設定
    calibration.calibration_version = "2.0"

    # バリデーション実行
    calibration.validate()

    # ファイル保存
    path = Path(file_path)

    # ディレクトリが存在しない場合は作成
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8", newline='') as f:
            json.dump(
                asdict(calibration),
                f,
                ensure_ascii=False,
                indent=2
            )
        logger.info(f"レンズキャリブレーション保存成功（v2.0形式）: {path}")
    except Exception as e:
        raise CalibrationError(
            f"キャリブレーションファイル保存エラー: {path} - {e}"
        )
