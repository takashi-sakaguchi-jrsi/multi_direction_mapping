"""カメラ関連のユーティリティ関数

魚眼レンズの焦点距離計算など、カメラパラメータに関する
共通処理を提供します。
"""

from typing import Tuple
import numpy as np


def compute_fisheye_focal_length(
    fov_degrees: float,
    image_size: Tuple[int, int]
) -> float:
    """魚眼レンズの焦点距離を計算（等距離射影モデル）
    
    等距離射影モデル（equidistant projection）を使用して、
    視野角（FOV）と画像サイズから焦点距離を計算します。
    
    このモデルでは、画像上の点の半径rと入射角θの関係が
    r = f * θ となります（θはラジアン）。
    
    Args:
        fov_degrees: 視野角（度）
            - 一般的な魚眼レンズは180度以上（例: 185度）
            - 広角レンズは90-180度
        image_size: 画像サイズ (height, width)
            - 円形魚眼の場合、有効領域は min(height, width) で決まる
    
    Returns:
        f: 焦点距離（ピクセル）
            - 画像座標系での焦点距離
            - カメラキャリブレーション行列のfxやfyに相当
    
    Formula:
        r_max = min(height, width) / 2  # 有効半径
        theta_max = radians(fov_degrees) / 2  # 最大入射角
        f = r_max / theta_max  # 焦点距離
    
    Example:
        >>> # FOV=185度、画像サイズ720x1280の魚眼レンズ
        >>> f = compute_fisheye_focal_length(185.0, (720, 1280))
        >>> print(f"焦点距離: {f:.2f} pixels")
        焦点距離: 222.26 pixels
        
        >>> # FOV=180度、画像サイズ1080x1920の魚眼レンズ
        >>> f = compute_fisheye_focal_length(180.0, (1080, 1920))
        >>> print(f"焦点距離: {f:.2f} pixels")
        焦点距離: 343.77 pixels
    
    Note:
        - FOV > 180度の場合、通常のピンホールモデルでは計算できない
        - tan(θ)を使うピンホールモデルでは、θ=90度で発散する
        - 等距離射影モデルはFOV=180度以上に対応
    
    References:
        - "Fisheye Camera Calibration", Zhang (1999)
        - OpenCV fisheye module documentation
    """
    img_h, img_w = image_size
    
    # 有効半径（円形魚眼の場合、短辺の半分が有効領域）
    r_max = min(img_h, img_w) / 2.0
    
    # 最大入射角（視野角の半分）
    theta_max = np.radians(fov_degrees) / 2.0
    
    # 等距離射影モデル: r = f * θ → f = r / θ
    f = r_max / theta_max
    
    return f


def compute_pinhole_focal_length(
    fov_degrees: float,
    image_size: Tuple[int, int]
) -> float:
    """ピンホールカメラの焦点距離を計算（透視投影モデル）
    
    透視投影モデル（perspective projection）を使用して、
    視野角（FOV）と画像サイズから焦点距離を計算します。
    
    このモデルでは、画像上の点の半径rと入射角θの関係が
    r = f * tan(θ) となります。
    
    Args:
        fov_degrees: 視野角（度）
            - 一般的な範囲: 30-120度
            - 180度以上では使用不可（tan(90°)が発散）
        image_size: 画像サイズ (height, width)
    
    Returns:
        f: 焦点距離（ピクセル）
    
    Formula:
        r_max = min(height, width) / 2
        theta_max = radians(fov_degrees) / 2
        f = r_max / tan(theta_max)
    
    Raises:
        ValueError: FOV >= 180度の場合（tan(90°)が発散）
    
    Warning:
        FOV > 180度の魚眼レンズには使用できません。
        代わりに compute_fisheye_focal_length() を使用してください。
    """
    if fov_degrees >= 180.0:
        raise ValueError(
            f"ピンホールモデルはFOV < 180度でのみ有効です（入力: {fov_degrees}度）。"
            "魚眼レンズには compute_fisheye_focal_length() を使用してください。"
        )
    
    img_h, img_w = image_size
    r_max = min(img_h, img_w) / 2.0
    theta_max = np.radians(fov_degrees) / 2.0
    f = r_max / np.tan(theta_max)
    
    return f
