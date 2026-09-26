"""合成円筒テクスチャと魚眼レンダラ（実動画が無い間の検証用）"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.coordinate_transform import CoordinateTransformer
from src.validation.geometry import pose_R_c2w, world_to_eta
from src.validation.sideview_projection_mapper import SideviewProjectionMapper


def cylinder_texture(eta: np.ndarray, z: np.ndarray) -> np.ndarray:
    """位置が一意に分かる RGB テクスチャ"""
    r = (np.sin(eta * 3.0) * 0.5 + 0.5) * 255
    g = (np.cos(z / 40.0) * 0.5 + 0.5) * 255
    b = ((eta / (2 * np.pi) + (z / 400.0)) % 1.0) * 255
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def render_fisheye_frame(
    transformer: CoordinateTransformer,
    mapper: SideviewProjectionMapper,
    position: np.ndarray,
    orientation: np.ndarray,
    width: int = 320,
    height: int = 240,
) -> np.ndarray:
    """各画素を円筒へ飛ばしてテクスチャをサンプリングする"""
    camera = transformer.camera
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.column_stack((us.ravel().astype(float), vs.ravel().astype(float)))
    yaw_cam, pitch_cam = camera.pixel_to_angles_cam(pts)
    roll, yaw, pitch = orientation
    vx = np.cos(pitch_cam) * np.sin(yaw_cam)
    vy = np.sin(pitch_cam)
    vz = np.cos(pitch_cam) * np.cos(yaw_cam)
    v_cam = np.stack((vx, vy, vz), axis=-1)
    R = pose_R_c2w(roll, yaw, pitch)
    direction = v_cam @ R
    direction = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12)
    origin = np.repeat(position.reshape(1, 3), len(pts), axis=0)
    p_wall, _, hit = mapper._intersect_forward_wall(origin, direction)
    eta = world_to_eta(p_wall[:, 0], p_wall[:, 1])
    colors = cylinder_texture(np.nan_to_num(eta), np.nan_to_num(p_wall[:, 2]))
    colors[~hit] = 0
    bgr = colors[:, ::-1].reshape(height, width, 3)
    return bgr


def make_sequence(
    transformer: CoordinateTransformer,
    mapper: SideviewProjectionMapper,
    orientation: np.ndarray,
    n: int = 5,
    z0: float = 0.0,
    dz: float = 8.0,
    width: int = 240,
    height: int = 180,
) -> Tuple[list, list]:
    frames = []
    poses = []
    for i in range(n):
        pos = np.array([0.0, 0.0, z0 + i * dz])
        frames.append(render_fisheye_frame(transformer, mapper, pos, orientation, width, height))
        poses.append((pos, orientation.copy()))
    return frames, poses
