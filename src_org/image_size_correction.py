"""画像サイズ変換検出とキャリブレーション補正モジュール

キャリブレーション画像と実際の動画フレーム画像のサイズが異なる場合の
変換タイプ（トリミング/リサイズ）を自動検出し、適切な補正を適用します。

主要機能:
- サイズ比較によるトリミング/リサイズ判定（アスペクト比不変→リサイズ、変化→トリミング）
- トリミングオフセット補正
- リサイズスケール補正
- 変換タイプのログ出力

Author: CameraCarSim Development Team
Date: 2025-11-25
"""

from typing import Tuple, Literal
from dataclasses import dataclass
import logging

# 変換タイプの型定義
TransformationType = Literal["none", "trim", "resize", "unknown"]


@dataclass
class CameraCorrection:
    """カメラパラメータ補正結果
    
    Attributes:
        f: 焦点距離（ピクセル）
        cx: 主点X座標（ピクセル）
        cy: 主点Y座標（ピクセル）
        transformation_type: 変換タイプ（"none", "trim", "resize", "unknown"）
        scale_x: X方向スケール比（リサイズ時のみ）
        scale_y: Y方向スケール比（リサイズ時のみ）
        trim_offset_x: X方向トリミングオフセット（トリミング時のみ）
        trim_offset_y: Y方向トリミングオフセット（トリミング時のみ）
    """
    f: float
    cx: float
    cy: float
    transformation_type: TransformationType
    scale_x: float = 1.0
    scale_y: float = 1.0
    trim_offset_x: float = 0.0
    trim_offset_y: float = 0.0


def detect_transformation_type(
    calib_width: int,
    calib_height: int,
    frame_width: int,
    frame_height: int,
    aspect_ratio_tolerance: float = 0.01
) -> TransformationType:
    """画像サイズ変換タイプを検出
    
    アスペクト比の変化からトリミング/リサイズを自動判定します。
    
    判定基準:
    - アスペクト比不変 → リサイズ（等倍/非等倍スケール）
    - アスペクト比変化 → トリミング（片方または両方向）
    
    Args:
        calib_width: キャリブレーション画像幅（ピクセル）
        calib_height: キャリブレーション画像高さ（ピクセル）
        frame_width: フレーム画像幅（ピクセル）
        frame_height: フレーム画像高さ（ピクセル）
        aspect_ratio_tolerance: アスペクト比許容誤差（デフォルト: 0.01 = 1%）
    
    Returns:
        変換タイプ: "none", "trim", "resize", "unknown"
        
    Examples:
        >>> detect_transformation_type(1920, 1080, 1920, 1080)
        'none'
        >>> detect_transformation_type(1920, 1080, 1220, 1080)  # 左右トリミング
        'trim'
        >>> detect_transformation_type(1920, 1080, 960, 540)  # 等倍リサイズ
        'resize'
    """
    # サイズ一致チェック
    if calib_width == frame_width and calib_height == frame_height:
        return "none"
    
    # アスペクト比計算
    aspect_ratio_calib = calib_width / calib_height
    aspect_ratio_frame = frame_width / frame_height
    
    # アスペクト比の差分
    aspect_ratio_diff = abs(aspect_ratio_calib - aspect_ratio_frame)
    
    # 判定
    if aspect_ratio_diff < aspect_ratio_tolerance:
        # アスペクト比不変 → リサイズ
        return "resize"
    else:
        # アスペクト比変化 → トリミング
        return "trim"


def apply_camera_correction(
    fx: float,
    cx: float,
    cy: float,
    calib_width: int,
    calib_height: int,
    frame_width: int,
    frame_height: int,
    aspect_ratio_tolerance: float = 0.01,
    logger: logging.Logger = None
) -> CameraCorrection:
    """カメラパラメータ補正を適用
    
    変換タイプを自動検出し、適切な補正を適用します。
    
    Args:
        fx: キャリブレーション焦点距離（ピクセル）
        cx: キャリブレーション主点X座標（ピクセル）
        cy: キャリブレーション主点Y座標（ピクセル）
        calib_width: キャリブレーション画像幅（ピクセル）
        calib_height: キャリブレーション画像高さ（ピクセル）
        frame_width: フレーム画像幅（ピクセル）
        frame_height: フレーム画像高さ（ピクセル）
        aspect_ratio_tolerance: アスペクト比許容誤差（デフォルト: 0.01）
        logger: ロガー（オプション）
    
    Returns:
        補正結果（CameraCorrection）
        
    Examples:
        >>> result = apply_camera_correction(
        ...     fx=986.76, cx=986.76, cy=540.0,
        ...     calib_width=1920, calib_height=1080,
        ...     frame_width=1220, frame_height=1080
        ... )
        >>> result.transformation_type
        'trim'
        >>> round(result.cx, 2)
        636.76
    """
    # 変換タイプ検出
    transformation_type = detect_transformation_type(
        calib_width, calib_height,
        frame_width, frame_height,
        aspect_ratio_tolerance
    )
    
    # 補正適用
    if transformation_type == "none":
        # 補正不要
        result = CameraCorrection(
            f=fx,
            cx=cx,
            cy=cy,
            transformation_type="none"
        )
        
        if logger:
            logger.info(
                f"カメラモデル初期化: v2.0形式（直接値使用）\n"
                f"  f={result.f:.2f}px, cx={result.cx:.1f}, cy={result.cy:.1f}\n"
                f"  frame_size=({frame_width}x{frame_height})"
            )
    
    elif transformation_type == "trim":
        # トリミング補正
        trim_offset_x = (calib_width - frame_width) / 2.0
        trim_offset_y = (calib_height - frame_height) / 2.0
        
        result = CameraCorrection(
            f=fx,  # 焦点距離は不変
            cx=cx - trim_offset_x,
            cy=cy - trim_offset_y,
            transformation_type="trim",
            trim_offset_x=trim_offset_x,
            trim_offset_y=trim_offset_y
        )
        
        if logger:
            logger.info(
                f"カメラモデル初期化: v2.0形式（トリミング補正適用）\n"
                f"  calib_size=({calib_width}x{calib_height}) → "
                f"frame_size=({frame_width}x{frame_height})\n"
                f"  trim_offset=({trim_offset_x:.1f}px, {trim_offset_y:.1f}px)\n"
                f"  f={result.f:.2f}px, cx={result.cx:.1f}, cy={result.cy:.1f}"
            )
    
    elif transformation_type == "resize":
        # リサイズ補正
        scale_x = frame_width / calib_width
        scale_y = frame_height / calib_height
        
        result = CameraCorrection(
            f=fx * (scale_x + scale_y) / 2.0,
            cx=cx * scale_x,
            cy=cy * scale_y,
            transformation_type="resize",
            scale_x=scale_x,
            scale_y=scale_y
        )
        
        if logger:
            logger.info(
                f"カメラモデル初期化: v2.0形式（スケール補正適用）\n"
                f"  calib_size=({calib_width}x{calib_height}) → "
                f"frame_size=({frame_width}x{frame_height})\n"
                f"  scale=({scale_x:.3f}, {scale_y:.3f})\n"
                f"  f={result.f:.2f}px, cx={result.cx:.1f}, cy={result.cy:.1f}"
            )
    
    else:
        # 未知の変換タイプ（エラー）
        raise ValueError(f"Unknown transformation type: {transformation_type}")
    
    return result
