"""座標変換モジュール

管内カメラカーシミュレーション用の座標変換機能を提供します。
カメラ座標系、画像座標系、ワールド座標系（円筒面）、カラーマップ座標系の
相互変換を実装しています。

主な機能:
- カメラモデル（魚眼/ピンホール）による画像投影
- ピクセル座標と3D座標の相互変換
- 円筒面との交点計算
- 回転行列による座標系変換
- カラーマップ座標への変換

座標系の定義:
- ワールド座標系: z軸が管路延長方向、x-y平面が管路断面、原点は管路中心軸上
- カメラ座標系: z軸が光軸（前方）、x軸が右、y軸が下
- 画像座標系: u軸が右、v軸が下、原点は画像左上
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


# ============================================================================
# カスタム例外クラス
# ============================================================================

class CoordinateTransformError(Exception):
    """座標変換関連のベース例外"""
    pass


class InvalidCameraModelError(CoordinateTransformError):
    """無効なカメラモデル"""
    pass


class CylinderIntersectionError(CoordinateTransformError):
    """円筒交点計算エラー"""
    pass


# ============================================================================
# カメラモデル抽象基底クラス
# ============================================================================

@dataclass
class CameraModel(ABC):
    """カメラモデルの抽象基底クラス
    
    Attributes:
        f: 焦点距離（ピクセル）
        cx: 画像中心x座標（ピクセル）
        cy: 画像中心y座標（ピクセル）
        image_width: 画像幅（ピクセル）
        image_height: 画像高さ（ピクセル）
    """
    f: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    
    @abstractmethod
    def cam_to_img(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """カメラ座標系から画像座標系への変換
        
        Args:
            x: カメラ座標x（右方向）
            y: カメラ座標y（下方向）
            z: カメラ座標z（奥行き方向、光軸）
            
        Returns:
            (u, v): 画像座標（ピクセル）
        """
        pass
    
    @abstractmethod
    def pixel_to_angles_cam(
        self,
        pts_px: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """画像座標からカメラ座標系の角度へ変換
        
        Args:
            pts_px: 画像座標 (N, 2) [u, v]
            
        Returns:
            (yaw_cam, pitch_cam): カメラ座標系でのヨー角、ピッチ角（ラジアン）
        """
        pass

    
    @abstractmethod
    def undistort_points(
        self,
        distorted_points: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> np.ndarray:
        """歪み補正: 歪んだ画像座標を補正された座標に変換

        Args:
            distorted_points: 歪んだ画像座標 (N, 2) [u, v] (ピクセル)
            dist_coeffs: 歪み係数配列
                - Fisheye: [k1, k2, k3, k4]
                - Pinhole: [k1, k2, p1, p2, k3]

        Returns:
            undistorted_points: 補正された画像座標 (N, 2) [u, v] (ピクセル)

        Note:
            - 歪み係数が全て0の場合、入力をそのまま返す
            - OpenCVのアンディストート関数を使用
        """
        pass

    @abstractmethod
    def distort_points(
        self,
        undistorted_points: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> np.ndarray:
        """順歪み変換: 理想座標から歪んだ画像座標への変換

        Args:
            undistorted_points: 理想画像座標 (N, 2) [u, v] (ピクセル)
            dist_coeffs: 歪み係数配列
                - Fisheye: [k1, k2, k3, k4]
                - Pinhole: [k1, k2, p1, p2, k3]

        Returns:
            distorted_points: 歪んだ画像座標 (N, 2) [u, v] (ピクセル)

        Note:
            - 歪み係数が全て0の場合、入力をそのまま返す
            - 解析的な順歪みモデルを使用（OpenCVのundistortの逆変換）
        """
        pass


# ============================================================================
# 魚眼カメラモデル
# ============================================================================

class FisheyeCamera(CameraModel):
    """魚眼カメラモデル（等距離射影）
    
    射影式: r = f * θ
    ここで θ = arccos(z / ||(x, y, z)||)
    
    等距離射影モデルでは、3D空間の点とレンズ中心を結ぶ直線が光軸となす角θに比例して
    画像上の動径距離rが決まります。
    """
    
    def cam_to_img(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """魚眼カメラの等距離射影を実装
        
        Args:
            x: カメラ座標x（右方向）
            y: カメラ座標y（下方向）
            z: カメラ座標z（奥行き方向、光軸）
            
        Returns:
            (u, v): 画像座標（ピクセル）
        
        数式:
            norm = sqrt(x^2 + y^2 + z^2)
            theta = arccos(z / norm)  # 光軸からの角度
            r = f * theta             # 動径距離
            u = cx + r * (x / sqrt(x^2 + y^2))
            v = cy - r * (y / sqrt(x^2 + y^2))
        """
        # ノルムを計算（ゼロ除算対策）
        norm = np.linalg.norm([x, y, z], axis=0) + 1e-12
        
        # 光軸からの角度（0 <= theta <= pi）
        theta = np.arccos(np.clip(z / norm, -1.0, 1.0))
        
        # 動径距離
        r = self.f * theta
        
        # xy平面上の距離（ゼロ除算対策）
        rho_xy = np.hypot(x, y) + 1e-12
        
        # 画像座標（y軸は下向きなので符号反転）
        u = self.cx + r * (x / rho_xy)
        v = self.cy - r * (y / rho_xy)
        
        return u, v
    
    def pixel_to_angles_cam(
        self,
        pts_px: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """魚眼画像から角度を計算
        
        Args:
            pts_px: 画像座標 (N, 2) [u, v]
            
        Returns:
            (yaw_cam, pitch_cam): カメラ座標系でのヨー角、ピッチ角（ラジアン）
        
        処理の流れ:
            1. 画像座標から画像中心への相対位置を計算
            2. 等距離射影の逆変換で角度gammaを計算
            3. 3D方向ベクトルに変換
            4. ヨー角・ピッチ角を計算
        """
        # 画像中心からの相対位置
        dx = pts_px[:, 0] - self.cx
        dy = -(pts_px[:, 1] - self.cy)  # カメラ座標系では上が正
        
        # 動径距離
        r = np.hypot(dx, dy) + 1e-12
        
        # 等距離射影の逆変換（光軸からの角度）
        gamma = r / self.f
        
        # xy平面上の角度
        phi_fish = np.arctan2(dy, dx)
        
        # 3D方向ベクトル
        vx = np.sin(gamma) * np.cos(phi_fish)
        vy = np.sin(gamma) * np.sin(phi_fish)
        vz = np.cos(gamma)
        
        # カメラ座標系のヨー角（水平回転）、ピッチ角（垂直回転）
        yaw_cam = np.arctan2(vx, vz)
        pitch_cam = np.arctan2(vy, np.hypot(vx, vz))
        
        return yaw_cam, pitch_cam

    
    def undistort_points(
        self,
        distorted_points: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> np.ndarray:
        """魚眼レンズの歪み補正
        
        cv2.fisheye.undistortPoints()を使用します。
        
        Args:
            distorted_points: 歪んだ画像座標 (N, 2) [u, v]
            dist_coeffs: 魚眼歪み係数 [k1, k2, k3, k4]
        
        Returns:
            undistorted_points: 補正された画像座標 (N, 2) [u, v]
        """
        import cv2
        
        # 歪み係数が全て0の場合はスキップ
        if np.allclose(dist_coeffs, 0.0):
            return distorted_points.copy()
        
        # カメラ行列（3x3）
        K = np.array([
            [self.f, 0, self.cx],
            [0, self.f, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 歪み係数をreshape（OpenCVの要求形式）
        D = dist_coeffs.reshape(-1, 1).astype(np.float64)
        
        # 入力点の形状を(N, 1, 2)に変換
        pts_input = distorted_points.reshape(-1, 1, 2).astype(np.float64)
        
        # 歪み補正（出力形状: (N, 1, 2)）
        pts_undist = cv2.fisheye.undistortPoints(
            pts_input, K, D, P=K  # P=Kで画像座標系に戻す
        )
        
        # (N, 2)形状に戻す
        return pts_undist.reshape(-1, 2)

    def distort_points(
        self,
        undistorted_points: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> np.ndarray:
        """魚眼レンズの順歪み変換

        等距離射影モデルの順歪み:
            theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)

        Args:
            undistorted_points: 理想画像座標 (N, 2) [u, v]
            dist_coeffs: 魚眼歪み係数 [k1, k2, k3, k4]

        Returns:
            distorted_points: 歪んだ画像座標 (N, 2) [u, v]
        """
        if np.allclose(dist_coeffs, 0.0):
            return undistorted_points.copy()

        k1, k2, k3, k4 = dist_coeffs[:4]
        dx = undistorted_points[:, 0] - self.cx
        dy = undistorted_points[:, 1] - self.cy
        r = np.hypot(dx, dy)

        # 等距離射影: r = f * theta → theta = r / f
        theta = r / self.f
        theta2 = theta * theta

        # 順歪みモデル: theta_d = theta * (1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8)
        scale = 1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
        scale = np.where(r < 1e-8, 1.0, scale)

        return np.column_stack([
            self.cx + dx * scale,
            self.cy + dy * scale
        ])


# ============================================================================
# ピンホールカメラモデル
# ============================================================================

class PinholeCamera(CameraModel):
    """ピンホールカメラモデル（透視投影）

    射影式: u = cx + f * (x / z), v = cy - f * (y / z)
    
    透視投影モデルでは、3D空間の点を光軸方向に投影した位置に対応する
    画像座標が決まります。
    """
    
    def cam_to_img(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ピンホールカメラの透視投影を実装
        
        Args:
            x: カメラ座標x（右方向）
            y: カメラ座標y（下方向）
            z: カメラ座標z（奥行き方向、光軸）
            
        Returns:
            (u, v): 画像座標（ピクセル）
        
        数式:
            u = cx + f * (x / z)
            v = cy - f * (y / z)
        
        Note:
            z < 0（カメラ背後）の場合は無効な値となります
        """
        # ゼロ除算対策
        z_safe = np.where(np.abs(z) < 1e-10, 1e-10, z)
        
        # 透視投影
        u = self.cx + self.f * (x / z_safe)
        v = self.cy - self.f * (y / z_safe)
        
        return u, v
    
    def pixel_to_angles_cam(
        self,
        pts_px: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ピンホール画像から角度を計算
        
        Args:
            pts_px: 画像座標 (N, 2) [u, v]
            
        Returns:
            (yaw_cam, pitch_cam): カメラ座標系でのヨー角、ピッチ角（ラジアン）
        """
        # 画像中心からの相対位置
        dx = pts_px[:, 0] - self.cx
        dy = self.cy - pts_px[:, 1]  # カメラ座標系では上が正
        
        # 正規化平面上の座標（z=fの平面）
        x_cam = dx
        y_cam = dy
        z_cam = self.f
        
        # ヨー角（水平回転）、ピッチ角（垂直回転）
        yaw_cam = np.arctan2(x_cam, z_cam)
        pitch_cam = np.arctan2(y_cam, np.sqrt(z_cam**2 + x_cam**2))
        
        return yaw_cam, pitch_cam

    
    def undistort_points(
        self,
        distorted_points: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> np.ndarray:
        """ピンホールカメラの歪み補正
        
        cv2.undistortPoints()を使用します。
        
        Args:
            distorted_points: 歪んだ画像座標 (N, 2) [u, v]
            dist_coeffs: 歪み係数 [k1, k2, p1, p2, k3]
        
        Returns:
            undistorted_points: 補正された画像座標 (N, 2) [u, v]
        """
        import cv2
        
        # 歪み係数が全て0の場合はスキップ
        if np.allclose(dist_coeffs, 0.0):
            return distorted_points.copy()
        
        # カメラ行列（3x3）
        K = np.array([
            [self.f, 0, self.cx],
            [0, self.f, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 歪み係数（OpenCVの要求形式）
        D = dist_coeffs.astype(np.float64)
        
        # 入力点の形状を(N, 1, 2)に変換
        pts_input = distorted_points.reshape(-1, 1, 2).astype(np.float64)
        
        # 歪み補正（出力形状: (N, 1, 2)）
        pts_undist = cv2.undistortPoints(
            pts_input, K, D, P=K  # P=Kで画像座標系に戻す
        )
        
        # (N, 2)形状に戻す
        return pts_undist.reshape(-1, 2)

    def distort_points(
        self,
        undistorted_points: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> np.ndarray:
        """ピンホールカメラの順歪み変換

        Brown-Conradyモデルの順歪み:
            x_d = x * radial + tangential_x
            y_d = y * radial + tangential_y

        Args:
            undistorted_points: 理想画像座標 (N, 2) [u, v]
            dist_coeffs: 歪み係数 [k1, k2, p1, p2, k3]

        Returns:
            distorted_points: 歪んだ画像座標 (N, 2) [u, v]
        """
        if np.allclose(dist_coeffs, 0.0):
            return undistorted_points.copy()

        k1, k2, p1, p2, k3 = dist_coeffs[:5]

        # 正規化座標系に変換
        x = (undistorted_points[:, 0] - self.cx) / self.f
        y = (undistorted_points[:, 1] - self.cy) / self.f

        r2 = x * x + y * y

        # 放射歪み
        radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3

        # 接線歪み
        x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        return np.column_stack([
            self.f * x_d + self.cx,
            self.f * y_d + self.cy
        ])


# ============================================================================
# 回転行列関数
# ============================================================================

def R_roll(roll: float) -> np.ndarray:
    """ロール回転行列（z軸回りの回転）
    
    Args:
        roll: ロール角（ラジアン、時計回り正）
        
    Returns:
        3x3回転行列
    
    数式:
        | cos(r)  sin(r)  0 |
        |-sin(r)  cos(r)  0 |
        |   0       0     1 |
    """
    c, s = np.cos(roll), np.sin(roll)
    return np.array([
        [ c,  s, 0],
        [-s,  c, 0],
        [ 0,  0, 1]
    ])


def R_pitch(pitch: float) -> np.ndarray:
    """ピッチ回転行列（y軸回りの回転）
    
    Args:
        pitch: ピッチ角（ラジアン、上向き正）
        
    Returns:
        3x3回転行列
    
    数式:
        | 1    0        0    |
        | 0  cos(p)  -sin(p) |
        | 0  sin(p)   cos(p) |
    """
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])


def R_yaw(yaw: float) -> np.ndarray:
    """ヨー回転行列（x軸回りの回転）
    
    Args:
        yaw: ヨー角（ラジアン、右向き正）
        
    Returns:
        3x3回転行列
    
    数式:
        | cos(y)  0  -sin(y) |
        |   0     1     0    |
        | sin(y)  0   cos(y) |
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [ c, 0, -s],
        [ 0, 1,  0],
        [ s, 0,  c]
    ])


def R_c2w(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """カメラ座標系からワールド座標系への回転行列
    
    Args:
        roll: ロール角（ラジアン）
        yaw: ヨー角（ラジアン）
        pitch: ピッチ角（ラジアン）
        
    Returns:
        3x3回転行列
    
    Note:
        回転の順序: roll → pitch → yaw
        R_c2w = R_roll @ R_pitch @ R_yaw
    """
    return R_roll(roll) @ R_pitch(pitch) @ R_yaw(yaw)


def R_w2c(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ワールド座標系からカメラ座標系への回転行列
    
    Args:
        roll: ロール角（ラジアン）
        yaw: ヨー角（ラジアン）
        pitch: ピッチ角（ラジアン）
        
    Returns:
        3x3回転行列
    
    Note:
        R_w2c = R_c2w^T（回転行列の転置は逆行列と等しい）
    """
    return R_c2w(roll, pitch, yaw).T


# ============================================================================
# 座標変換ヘルパー関数（レガシー互換の細かい関数分割）
# ============================================================================

def apply_roll_correction(
    x_cam: np.ndarray,
    y_cam: np.ndarray,
    roll: float
) -> Tuple[np.ndarray, np.ndarray]:
    """カメラ平面座標にロール補正を適用

    Args:
        x_cam: カメラ平面x座標（右が正）
        y_cam: カメラ平面y座標（上が正）
        roll: ロール角（ラジアン、時計回り正）

    Returns:
        (x_u, y_u): ロール補正後の座標

    Note:
        画像を-roll回転することでカメラの傾きを補正します。
    """
    c, s = np.cos(-roll), np.sin(-roll)
    x_u = c * x_cam + s * y_cam
    y_u = -s * x_cam + c * y_cam
    return x_u, y_u


def pixel_to_angles_fisheye(
    pts_px: np.ndarray,
    f: float,
    center: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """ピクセル座標から魚眼カメラの角度パラメータへ変換

    Args:
        pts_px: 画像座標 (N, 2) [u, v]
        f: 焦点距離（ピクセル）
        center: 画像中心座標 (cx, cy)

    Returns:
        (gamma, phi_fish): 光軸からの角度、xy平面上の角度（ラジアン）

    Note:
        等距離射影の逆変換: gamma = r / f
    """
    dx = pts_px[:, 0] - center[0]
    dy = pts_px[:, 1] - center[1]
    x_cam = dx
    y_cam = -dy  # カメラ座標系では上が+

    r = np.hypot(x_cam, y_cam)
    gamma = r / f
    phi_fish = np.arctan2(y_cam, x_cam)

    return gamma, phi_fish


def fisheye_to_cam_angles(
    gamma: np.ndarray,
    phi_fish: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """魚眼カメラの角度パラメータからカメラ座標系の角度へ変換

    Args:
        gamma: 光軸からの角度（ラジアン）
        phi_fish: xy平面上の角度（ラジアン）

    Returns:
        (theta_cam, phi_cam): カメラ座標系のヨー角、ピッチ角（ラジアン）

    処理の流れ:
        1. 角度パラメータから3D方向ベクトルを計算
        2. 方向ベクトルからヨー・ピッチ角を計算
    """
    vx = np.sin(gamma) * np.cos(phi_fish)
    vy = np.sin(gamma) * np.sin(phi_fish)
    vz = np.cos(gamma)

    theta_cam = np.arctan2(vx, vz)
    phi_cam = np.arctan2(vy, np.hypot(vx, vz))

    return theta_cam, phi_cam


def pinhole_to_cam_angles(
    pts_px: np.ndarray,
    f: float,
    center: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """ピンホールカメラでピクセル座標からカメラ座標系の角度へ変換

    Args:
        pts_px: 画像座標 (N, 2) [u, v]
        f: 焦点距離（ピクセル）
        center: 画像中心座標 (cx, cy)

    Returns:
        (theta_cam, phi_cam): カメラ座標系のヨー角、ピッチ角（ラジアン）
    """
    dx = pts_px[:, 0] - center[0]
    dy = center[1] - pts_px[:, 1]  # カメラ座標系では上が+

    theta_cam = np.arctan2(dx, f)
    phi_cam = np.arctan2(dy, np.sqrt(f**2 + dx**2))

    return theta_cam, phi_cam


def convert_pixel_to_pose(
    vp_x: float,
    vp_y: float,
    camera: CameraModel,
    roll: float
) -> Tuple[float, float]:
    """ピクセル座標（消失点）から姿勢角を計算

    Args:
        vp_x: x座標（ピクセル）
        vp_y: y座標（ピクセル）
        camera: カメラモデル
        roll: ロール角（ラジアン）

    Returns:
        (yaw, pitch): ヨー角、ピッチ角（ラジアン）

    処理の流れ:
        1. ピクセル座標→カメラ平面座標
        2. ロール補正
        3. カメラモデルに応じた角度計算

    Note:
        この関数はレガシーの座標変換ロジックを統合した共通関数です。
        estimate_yaw_pitch_from_dark_region()でも同じロジックを使用します。
    """
    # 1. ピクセル座標→カメラ平面座標
    dx = vp_x - camera.cx
    dy = vp_y - camera.cy
    x_cam = dx
    y_cam = -dy  # カメラ座標系では上が+

    # 2. ロール補正
    x_u, y_u = apply_roll_correction(x_cam, y_cam, roll)

    # 3. カメラモデルに応じた角度計算
    if isinstance(camera, FisheyeCamera):
        # 魚眼カメラ：等距離射影モデル
        r = np.hypot(x_u, y_u) + 1e-12
        gamma = r / camera.f
        phi_fish = np.arctan2(y_u, x_u)
        theta_cam, phi_cam = fisheye_to_cam_angles(gamma, phi_fish)
    elif isinstance(camera, PinholeCamera):
        # ピンホールカメラ：透視投影モデル
        vx = x_u
        vy = y_u
        vz = camera.f
        norm = np.linalg.norm([vx, vy, vz]) + 1e-12
        vx, vy, vz = vx / norm, vy / norm, vz / norm
        theta_cam = np.arctan2(vx, vz)
        phi_cam = np.arctan2(vy, np.hypot(vx, vz))
    else:
        raise InvalidCameraModelError(f"Unknown camera model: {type(camera)}")

    # カメラ座標系の角度→姿勢角（符号調整）
    yaw = -theta_cam
    pitch = -phi_cam

    return yaw, pitch


# ============================================================================
# CoordinateTransformer クラス
# ============================================================================

class CoordinateTransformer:
    """座標変換ユーティリティクラス
    
    Attributes:
        camera: カメラモデル（FisheyeCamera または PinholeCamera）
        pipe_radius: 管路半径（mm）
    """
    
    def __init__(self, camera: CameraModel, pipe_radius: float,
                 calibration: Optional['LensCalibration'] = None):
        """初期化

        Args:
            camera: カメラモデル
            pipe_radius: 管路半径（mm）
            calibration: レンズキャリブレーション情報（歪み補正用、オプション）
        """
        self.camera = camera
        self.pipe_radius = pipe_radius
        self.calibration = calibration

    
    def _get_dist_coeffs(self, calibration: 'LensCalibration') -> np.ndarray:
        """キャリブレーション情報から歪み係数を抽出
        
        Args:
            calibration: LensCalibration オブジェクト
        
        Returns:
            歪み係数配列
            - Fisheye: [k1, k2, k3, k4]
            - Pinhole: [k1, k2, p1, p2, k3]
        """
        if calibration.camera_model == "fisheye":
            return np.array([
                calibration.radial_distortion_k1,
                calibration.radial_distortion_k2,
                calibration.radial_distortion_k3,
                calibration.radial_distortion_k4,
            ])
        else:  # pinhole
            return np.array([
                calibration.radial_distortion_k1,
                calibration.radial_distortion_k2,
                calibration.tangential_distortion_p1,
                calibration.tangential_distortion_p2,
                calibration.radial_distortion_k3,
            ])
    
    def cam_to_world_angles(
        self,
        yaw_cam: np.ndarray,
        pitch_cam: np.ndarray,
        roll: float,
        yaw: float,
        pitch: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """カメラ座標系の角度からワールド座標系の角度へ変換
        
        Args:
            yaw_cam: カメラ座標系ヨー角（ラジアン）
            pitch_cam: カメラ座標系ピッチ角（ラジアン）
            roll: カメラのロール角（ラジアン）
            yaw: カメラのヨー角（ラジアン）
            pitch: カメラのピッチ角（ラジアン）
            
        Returns:
            (yaw_world, pitch_world): ワールド座標系の角度（ラジアン）
        
        処理の流れ:
            1. カメラ座標系の角度から3D方向ベクトルを計算
            2. 回転行列でワールド座標系に変換
            3. ワールド座標系の角度を計算
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
    
    def pixel_to_world(
        self,
        pts_px: np.ndarray,
        cam_pos: np.ndarray,
        roll: float,
        yaw: float,
        pitch: float,
        calibration: Optional['LensCalibration'] = None
    ) -> np.ndarray:
        """画像座標からワールド座標（円筒面上）へ変換
        
        Args:
            pts_px: 画像座標 (N, 2) [u, v]
            cam_pos: カメラ位置 [x, y, z] (mm)
            roll: カメラのロール角（ラジアン）
            yaw: カメラのヨー角（ラジアン）
            pitch: カメラのピッチ角（ラジアン）
            calibration: レンズキャリブレーション情報（オプション）
                指定時、歪み補正を適用します
            
        Returns:
            points_3d: ワールド座標 (N, 3) [x, y, z] (mm)
        
        Raises:
            CylinderIntersectionError: 円筒との交点が見つからない場合
        
        処理の流れ:
            1. 歪み補正（calibrationが指定されている場合）
            2. ピクセル座標からカメラ座標系の角度を計算
            3. カメラ座標系の角度をワールド座標系の角度に変換
            4. ワールド座標系の角度から視線ベクトルを計算
            5. 視線と円筒の交点を計算
        """
        # 1. 歪み補正（calibrationが指定されている場合）
        pts_corrected = pts_px
        if calibration is not None:
            dist_coeffs = self._get_dist_coeffs(calibration)
            if not np.allclose(dist_coeffs, 0.0):
                pts_corrected = self.camera.undistort_points(pts_px, dist_coeffs)
        
        # 2. ピクセル → カメラ座標系の角度
        yaw_cam, pitch_cam = self.camera.pixel_to_angles_cam(pts_corrected)
        
        # カメラ座標系 → ワールド座標系の角度
        yaw_world, pitch_world = self.cam_to_world_angles(
            yaw_cam, pitch_cam, roll, yaw, pitch
        )
        
        # ワールド座標系の視線ベクトル（正規化済み）
        dx_v = -np.cos(pitch_world) * np.sin(yaw_world)  # 右側が+
        dy_v = np.sin(pitch_world)
        dz_v = np.cos(pitch_world) * np.cos(yaw_world)
        ray_direction = np.stack((dx_v, dy_v, dz_v), axis=-1)
        
        # レイの原点（カメラ位置）
        N = len(pts_px)
        ray_origin = np.tile(cam_pos, (N, 1))
        
        # 円筒との交点を計算
        intersection = self.compute_cylinder_intersection(ray_origin, ray_direction)
        
        return intersection

    def pixel_to_world_R_c2w(
        self,
        pts_px: np.ndarray,
        cam_pos: np.ndarray,
        roll: float,
        yaw: float,
        pitch: float,
    ) -> np.ndarray:
        """画像座標 → 円筒面。world_to_pixel の逆（視線は v_cam @ R_c2w）。

        pixel_to_world は正面カメラ用にワールド角からレイを組み直すため、
        真横（pitch=90°）では画像 x が反転する。こちらは回転行列と一致する。
        """
        yaw_cam, pitch_cam = self.camera.pixel_to_angles_cam(pts_px)
        vx = np.cos(pitch_cam) * np.sin(yaw_cam)
        vy = np.sin(pitch_cam)
        vz = np.cos(pitch_cam) * np.cos(yaw_cam)
        v_cam = np.stack((vx, vy, vz), axis=-1)
        ray_direction = v_cam @ R_c2w(roll, pitch, yaw)
        ray_direction = ray_direction / (
            np.linalg.norm(ray_direction, axis=1, keepdims=True) + 1e-12
        )
        ray_origin = np.tile(np.asarray(cam_pos, dtype=float).reshape(3), (len(pts_px), 1))
        return self.compute_cylinder_intersection(ray_origin, ray_direction)

    def world_to_pixel(
        self,
        points_3d: np.ndarray,
        cam_pos: np.ndarray,
        roll: float,
        yaw: float,
        pitch: float,
        apply_distortion: bool = False
    ) -> np.ndarray:
        """ワールド座標から画像座標へ変換

        Args:
            points_3d: ワールド座標 (N, 3) [x, y, z] (mm)
            cam_pos: カメラ位置 [x, y, z] (mm)
            roll: カメラのロール角（ラジアン）
            yaw: カメラのヨー角（ラジアン）
            pitch: カメラのピッチ角（ラジアン）
            apply_distortion: 順歪み変換を適用するか（デフォルト: False）

        Returns:
            pts_px: 画像座標 (N, 2) [u, v]

        処理の流れ:
            1. ワールド座標をカメラ位置を原点とする相対座標に変換
            2. 回転行列でカメラ座標系に変換
            3. カメラモデルで画像座標に投影
            4. 順歪み適用（apply_distortion=True かつ calibration有効時）
        """
        # ワールド座標 → カメラ中心からの相対座標
        pts_relative = points_3d - cam_pos

        # ワールド座標系 → カメラ座標系
        R = R_w2c(roll, pitch, yaw)
        pts_cam = pts_relative @ R
        x, y, z = pts_cam.T

        # カメラ座標系 → 画像座標系
        u, v = self.camera.cam_to_img(x, y, z)
        pts_px = np.column_stack((u, v))

        # 順歪み適用
        if apply_distortion and self.calibration is not None:
            dist_coeffs = self._get_dist_coeffs(self.calibration)
            if not np.allclose(dist_coeffs, 0.0):
                pts_px = self.camera.distort_points(pts_px, dist_coeffs)

        return pts_px
    
    def compute_cylinder_intersection(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray
    ) -> np.ndarray:
        """レイと円筒の交点を計算
        
        円筒の式: x^2 + y^2 = R^2
        レイの式: P = O + t * D
        
        二次方程式 A*t^2 + B*t + C = 0 を解く
        A = dx^2 + dy^2
        B = 2(x_o*dx + y_o*dy)
        C = x_o^2 + y_o^2 - R^2
        
        Args:
            ray_origin: レイの原点 (N, 3)
            ray_direction: レイの方向（正規化済み） (N, 3)
            
        Returns:
            intersection: 交点座標 (N, 3)
            
        Raises:
            CylinderIntersectionError: 交点が存在しない場合
        
        Note:
            二つの交点がある場合、カメラから遠い方（t が大きい方）を選択します。
        """
        x_o = ray_origin[:, 0]
        y_o = ray_origin[:, 1]
        z_o = ray_origin[:, 2]
        
        dx = ray_direction[:, 0]
        dy = ray_direction[:, 1]
        dz = ray_direction[:, 2]
        
        # 二次方程式の係数
        A = dx**2 + dy**2
        B = 2 * (x_o * dx + y_o * dy)
        C = x_o**2 + y_o**2 - self.pipe_radius**2

        # レイが円筒軸にほぼ平行な場合（A ≈ 0）をチェック
        # 許容閾値は浮動小数点の精度と実用性を考慮
        A_threshold = 1e-20
        if np.any(np.abs(A) < A_threshold):
            n_invalid = np.sum(np.abs(A) < A_threshold)
            raise CylinderIntersectionError(
                f"レイが円筒軸にほぼ平行（A≈0）：交点計算不可（{n_invalid}/{len(A)} 点）"
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
        # t1 = (-B - sqrt(D)) / (2*A)  # 近い方
        # t2 = (-B + sqrt(D)) / (2*A)  # 遠い方
        sqrt_D = np.sqrt(D)
        t = (-B + sqrt_D) / (2 * A)

        # カメラ前方の交点のみ有効（t > 0）
        if np.any(t <= 0):
            n_invalid = np.sum(t <= 0)
            raise CylinderIntersectionError(
                f"交点がレイの原点より後ろ側（t≤0）：無効（{n_invalid}/{len(t)} 点）"
            )
        
        # 交点座標
        x = x_o + t * dx
        y = y_o + t * dy
        z = z_o + t * dz
        
        return np.stack((x, y, z), axis=-1)
    
    def map_to_colormap(
        self,
        points_3d: np.ndarray,
        pixels_per_mm: float,
        z_min: float,
        z_max: float,
        colormap_height: int
    ) -> np.ndarray:
        """ワールド座標からカラーマップ座標へ変換
        
        Args:
            points_3d: ワールド座標 (N, 3) [x, y, z] (mm)
            pixels_per_mm: カラーマップの解像度（ピクセル/mm）
            z_min: z座標の最小値（mm）
            z_max: z座標の最大値（mm）
            colormap_height: カラーマップの高さ（ピクセル）
            
        Returns:
            colormap_coords: カラーマップ座標 (N, 2) [x_map, y_map]
        
        カラーマップの定義:
            - 横軸（x_map）: z座標（管路延長方向）
            - 縦軸（y_map）: 円周角η（0 <= η < 2π）
            - 原点: 左上（z=z_min, η=0）
        
        数式:
            η = atan2(y, x) + π/2  # 上を0として時計回りに増加
            x_map = (z - z_min) * pixels_per_mm
            y_map = (η / (2π)) * colormap_height
        """
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]
        
        # 円周角η（atan2の結果は-π～πなので、0～2πに正規化）
        eta = np.arctan2(y, x) + np.pi / 2
        eta = np.where(eta < 0, eta + 2 * np.pi, eta)
        eta = np.where(eta >= 2 * np.pi, eta - 2 * np.pi, eta)
        
        # カラーマップ座標
        x_map = (z - z_min) * pixels_per_mm
        y_map = (eta / (2 * np.pi)) * colormap_height
        
        return np.column_stack((x_map, y_map))


# ============================================================================
# ヘルパー関数
# ============================================================================

def estimate_yaw_pitch_from_dark_region(
    frame: np.ndarray,
    camera: CameraModel,
    roll: float,
    threshold: int = 30,
    radius_limit: Optional[float] = None
) -> Tuple[Optional[float], Optional[float]]:
    """暗部重心からヨー・ピッチを推定
    
    Args:
        frame: 入力画像（BGR、OpenCV形式）
        camera: カメラモデル
        roll: カメラのロール角（ラジアン）
        threshold: 暗部判定の閾値（0-255）
        radius_limit: 暗部検出対象の半径制限（ピクセル）
        
    Returns:
        (yaw, pitch): 推定されたヨー角、ピッチ角（ラジアン）
                      検出失敗時は (None, None)
    
    処理の流れ:
        1. グレースケール化
        2. 閾値処理で暗部抽出
        3. 最大輪郭の重心を計算
        4. 重心座標からヨー・ピッチを計算
    
    Note:
        管路内カメラでは、奥行き方向（管路の先）が暗部となるため、
        暗部の重心がカメラの向きを表します。
    """
    import cv2
    
    # グレースケール化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 暗部抽出
    _, dark_mask = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY_INV)
    
    # 半径制限がある場合
    if radius_limit is not None:
        h, w = frame.shape[:2]
        Y, X = np.ogrid[:h, :w]
        dist_sq = (X - camera.cx)**2 + (Y - camera.cy)**2
        mask_radius = (dist_sq <= radius_limit**2).astype(np.uint8)
        dark_mask = dark_mask & mask_radius
    
    # 輪郭検出
    contours, _ = cv2.findContours(
        dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None, None
    
    # 最大輪郭
    largest = max(contours, key=cv2.contourArea)
    
    # 重心計算
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None
    
    gx = M["m10"] / M["m00"]
    gy = M["m01"] / M["m00"]

    # 共通関数を使用して消失点座標から姿勢角を計算
    yaw, pitch = convert_pixel_to_pose(gx, gy, camera, roll)

    return yaw, pitch
