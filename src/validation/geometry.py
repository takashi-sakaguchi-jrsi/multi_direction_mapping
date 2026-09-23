"""角度規約・run_reference・実効視野の共通幾何"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from src.coordinate_transform import R_c2w


SUFFIX_TO_PHYSICAL_ROLL_DEG = {
    "U": 0.0,
    "R": 120.0,
    "L": 240.0,
}

PHYSICAL_ROLL_TO_RUN_ID = {
    0.0: "U",
    120.0: "R",
    240.0: "L",
    -120.0: "L",
}

# カメラカー初期方位 (pitch_car, roll_car) → プログラム姿勢 (roll, yaw, pitch) [deg]
# カメラカーの roll/pitch はプログラム Euler とは別定義。回転順は変換に使わない。
# プログラム: CameraState = [roll, yaw, pitch]、R_c2w = R_roll @ R_pitch @ R_yaw。
# 投影は v_world = v_cam @ R_c2w。生成側 Demo は v_world = R_demo @ v_cam なので
# R_c2w.T が R_demo と一致する組を使う（光軸だけでなくカメラ上下も合わせる）。
# 120° の光軸は水平より 30° 下。光軸周りの 180° 不定性は Demo の「上」に合わせ、
# R のプログラム roll は -90、L は +90。
_CAPTURE_TO_PROGRAM_RPY_DEG = {
    (90.0, 0.0): (0.0, 0.0, 90.0),
    (90.0, 120.0): (-90.0, 90.0, -30.0),
    (90.0, -120.0): (90.0, -90.0, -30.0),
}


def wrap_angle_deg(angle_deg: float) -> float:
    """角度[deg]を [-180, 180) に正規化する"""
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def wrap_angle_rad(angle_rad: float) -> float:
    """角度[rad]を [-pi, pi) に正規化する"""
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """a - b を wrap した差分[deg]"""
    return wrap_angle_deg(a_deg - b_deg)


def angle_diff_rad(a_rad: float, b_rad: float) -> float:
    """a - b を wrap した差分[rad]"""
    return wrap_angle_rad(a_rad - b_rad)


def physical_roll_to_internal_deg(physical_roll_deg: float) -> float:
    """カメラカー表記 0/120/240° を 0/120/-120° へ変換する"""
    return wrap_angle_deg(physical_roll_deg)


def run_id_from_physical_roll(physical_roll_deg: float) -> str:
    """カメラカー roll から U/R/L を返す"""
    internal = physical_roll_to_internal_deg(physical_roll_deg)
    if abs(internal) < 1e-6:
        return "U"
    if abs(internal - 120.0) < 1e-6:
        return "R"
    if abs(internal + 120.0) < 1e-6:
        return "L"
    raise ValueError(f"未対応の physical_roll_deg: {physical_roll_deg}")


def physical_roll_from_suffix(suffix: str) -> float:
    """ファイル名識別子 _U/_R/_L からカメラカー roll[deg] を返す"""
    key = suffix.strip().upper().lstrip("_")
    if key not in SUFFIX_TO_PHYSICAL_ROLL_DEG:
        raise ValueError(f"未知の方向 suffix: {suffix}")
    return SUFFIX_TO_PHYSICAL_ROLL_DEG[key]


def infer_suffix_from_path(video_path: str) -> Optional[str]:
    """パス末尾の _U/_R/_L を検出する。無ければ None"""
    name = video_path.replace("\\", "/").split("/")[-1]
    stem = name.rsplit(".", 1)[0]
    for suffix in ("U", "R", "L"):
        if stem.endswith(f"_{suffix}"):
            return suffix
    return None


def capture_to_program_orientation_deg(
    physical_pitch_deg: float,
    physical_roll_deg: float,
) -> Tuple[float, float, float]:
    """カメラカー初期方位 → プログラム姿勢 (roll, yaw, pitch) [deg]

    カメラカーの roll は回転軸の向き、pitch はその軸周りの回転で、
    プログラムの yaw/pitch/roll とは定義が異なる。回転順は問わず、
    プログラム角へは次の対応で変換する。

        (90, 0)    → yaw=0,   pitch=90,  roll=0    → (roll,yaw,pitch)=(0, 0, 90)
        (90, 120)  → yaw=90,  pitch=-30, roll=-90  → (-90, 90, -30)
        (90, -120) → yaw=-90, pitch=-30, roll=90   → (90, -90, -30)
    """
    pitch_c = float(physical_pitch_deg)
    roll_c = physical_roll_to_internal_deg(physical_roll_deg)
    key = (round(pitch_c, 6), round(roll_c, 6))
    if key not in _CAPTURE_TO_PROGRAM_RPY_DEG:
        raise ValueError(
            f"未対応のカメラカー初期方位: pitch={physical_pitch_deg}, "
            f"roll={physical_roll_deg}（内部 roll={roll_c}）"
        )
    return _CAPTURE_TO_PROGRAM_RPY_DEG[key]


def pose_R_c2w(roll_rad: float, yaw_rad: float, pitch_rad: float) -> np.ndarray:
    """CameraState [roll, yaw, pitch] から現行 R_c2w を作る"""
    return R_c2w(roll_rad, pitch_rad, yaw_rad)


def build_run_reference(
    physical_roll_deg: float,
    physical_pitch_deg: float = 90.0,
    yaw_ref_deg: float = 0.0,
    x_ref_mm: float = 0.0,
    y_ref_mm: float = 0.0,
) -> Dict[str, float]:
    """run_reference。*_ref はプログラム姿勢。physical_* はカメラカー初期方位。

    yaw_ref_deg 引数は後方互換のため残すが、プログラム yaw は変換結果を使う。
    """
    del yaw_ref_deg
    roll_car = physical_roll_to_internal_deg(physical_roll_deg)
    roll_p, yaw_p, pitch_p = capture_to_program_orientation_deg(
        physical_pitch_deg, physical_roll_deg
    )
    return {
        "x_ref_mm": float(x_ref_mm),
        "y_ref_mm": float(y_ref_mm),
        "physical_pitch_deg": float(physical_pitch_deg),
        "physical_roll_deg": float(physical_roll_deg),
        "physical_roll_internal_deg": float(roll_car),
        "roll_ref_deg": float(roll_p),
        "yaw_ref_deg": float(yaw_p),
        "pitch_ref_deg": float(pitch_p),
        "roll_ref_rad": float(np.radians(roll_p)),
        "yaw_ref_rad": float(np.radians(yaw_p)),
        "pitch_ref_rad": float(np.radians(pitch_p)),
        "run_id": run_id_from_physical_roll(physical_roll_deg),
    }


def derive_usable_half_fov_deg(
    fx_px: float,
    image_height_px: float,
    outer_radius_ratio: float,
) -> float:
    """等距離射影 r=fγ から usable half FOV[deg] を導出する"""
    r_image = image_height_px / 2.0
    r_usable = r_image * outer_radius_ratio
    gamma_rad = r_usable / fx_px
    return float(np.degrees(gamma_rad))


def optical_axis_world(roll_rad: float, yaw_rad: float, pitch_rad: float) -> np.ndarray:
    """カメラ光軸 [0,0,1] のワールド方向。引数順は CameraState [roll, yaw, pitch]"""
    v_cam = np.array([0.0, 0.0, 1.0])
    R = pose_R_c2w(roll_rad, yaw_rad, pitch_rad)
    v_w = v_cam @ R
    return v_w / (np.linalg.norm(v_w) + 1e-12)


def world_to_eta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """現行規約: η = atan2(y, x) + π/2、[0, 2π)"""
    eta = np.arctan2(y, x) + np.pi / 2.0
    eta = np.where(eta < 0.0, eta + 2.0 * np.pi, eta)
    eta = np.where(eta >= 2.0 * np.pi, eta - 2.0 * np.pi, eta)
    return eta


# カメラカー周方向（U=0°, R=120°, L=-120°）とプログラム η の関係:
# η_deg = (180 - car_deg) mod 360。重なり中央は 60° / -60° / 180°。
OVERLAP_CENTER_CAMERA_CAR_DEG = {
    ("U", "R"): 60.0,
    ("U", "L"): -60.0,
    ("R", "L"): 180.0,
}

SECTOR_CAMERA_CAR_DEG = {
    "U": (-60.0, 60.0),
    "R": (60.0, 180.0),
    "L": (180.0, 300.0),
}


def camera_car_deg_to_eta(car_deg) -> np.ndarray:
    """カメラカー周方向角[deg] → プログラム η[rad]。U=0° が η=π。"""
    deg = np.asarray(car_deg, dtype=float)
    return np.radians((180.0 - deg) % 360.0)


def eta_to_camera_car_deg(eta) -> np.ndarray:
    """プログラム η[rad] → カメラカー周方向角[deg]（[0, 360)）。"""
    eta = np.asarray(eta, dtype=float)
    return (180.0 - np.degrees(eta)) % 360.0


def optical_axis_eta(roll_rad: float, yaw_rad: float, pitch_rad: float) -> float:
    """光軸のプログラム η[rad]。"""
    axis = optical_axis_world(roll_rad, yaw_rad, pitch_rad)
    return float(np.asarray(world_to_eta(axis[0], axis[1])).reshape(-1)[0])


def expected_edge_etas(
    roll_rad: float,
    yaw_rad: float,
    pitch_rad: float,
    half_fov_rad: float,
) -> Tuple[float, float, float]:
    """初期方位から見たエッジ角。戻り値は (中心η, エッジ0, エッジ1) [rad]。"""
    center = optical_axis_eta(roll_rad, yaw_rad, pitch_rad)
    lo = float((center - half_fov_rad) % (2.0 * np.pi))
    hi = float((center + half_fov_rad) % (2.0 * np.pi))
    return center, lo, hi


def mode_a_estimate_yaw_pitch(run_id: str) -> Tuple[bool, bool]:
    """Mode A で推定する角度 (estimate_yaw, estimate_pitch)。

    U: フレーム残差は (dz, dyaw) を同一予測。yaw=0 付近では dy→dz・dx→dyaw。
       pitch=90°・roll=0・x=y=0 固定。yaw は 0° 求心。
       pitch は進行面のチルトなので横残差から推定しない（z 復元を壊す）。
    R/L: 画像x=pitch, 画像y=z。roll=r0, yaw=y0, x=y=0 固定。pitch は p0 求心。
    """
    key = str(run_id or "").strip().upper()
    if key == "A":
        key = "U"
    if key == "B":
        key = "R"
    if key == "U":
        return True, False
    if key in ("R", "L"):
        return False, True
    return True, False


def mode_a_use_frame_xy_residual(run_id: str) -> bool:
    """Mode A の U はフレーム座標 (dx, dy) 残差。R/L は円筒 3D 残差。"""
    key = str(run_id or "").strip().upper()
    if key == "A":
        key = "U"
    return key == "U"


def theta_in_camera_car_sector(eta, run_id: str) -> np.ndarray:
    """η が run の 120° 担当帯（重なり中央で分割）に入るか。"""
    key = run_id.strip().upper()
    if key in ("A",):
        key = "U"
    if key in ("B",):
        key = "R"
    if key not in SECTOR_CAMERA_CAR_DEG:
        raise ValueError(f"未知の run_id: {run_id}")
    lo, hi = SECTOR_CAMERA_CAR_DEG[key]
    car = eta_to_camera_car_deg(eta)
    lo = lo % 360.0
    hi = hi % 360.0
    if lo < hi:
        return (car >= lo) & (car < hi)
    return (car >= lo) | (car < hi)
