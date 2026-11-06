"""角度規約・run_reference・実効視野の共通幾何"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
# 投影は v_world = v_cam @ R_c2w。生成側 Demo は v_world = R_demo @ v_cam。
# U/R/L とも R_c2w.T = R_demo（光軸とカメラ上下）。
# 120° の光軸は水平より 30° 下。R のプログラム roll は -90、L は +90。
# U は (0,0,90) だと pitch=90° 特異点で車体 roll（XY 振り）を書けず、ねじれ
# （roll+yaw）に漏れる。(-90,90,90) は (0,0,90) と同一のカメラ姿勢で、
# dpitch だけが周方向＝車体 roll になる。(90,90,90) は roll 符号が逆で
# 光軸まわり 180° ずれる。
_CAPTURE_TO_PROGRAM_RPY_DEG = {
    (90.0, 0.0): (-90.0, 90.0, 90.0),
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


def _run_id_or_canonical(physical_roll_deg: float) -> str:
    """U/R/L なら従来 ID、それ以外は 3 桁 run_id。"""
    try:
        return run_id_from_physical_roll(physical_roll_deg)
    except ValueError:
        return canonical_run_id(physical_roll_deg)


def canonical_run_id(physical_roll_deg: float) -> str:
    """カメラカー roll[deg] を 3 桁 run_id（000…359）にする。45° には量子化しない。"""
    wrapped = float(physical_roll_deg) % 360.0
    if wrapped < 0.0:
        wrapped += 360.0
    return f"{int(round(wrapped)) % 360:03d}"


def physical_roll_from_run_id(run_id: str) -> float:
    """U/R/L/A/B/C または 3 桁 run_id からカメラカー roll[deg] を返す。"""
    key = str(run_id or "").strip().upper().lstrip("_")
    if key == "A":
        key = "U"
    elif key == "B":
        key = "R"
    elif key == "C":
        key = "L"
    if key in SUFFIX_TO_PHYSICAL_ROLL_DEG:
        return SUFFIX_TO_PHYSICAL_ROLL_DEG[key]
    if key.isdigit():
        return float(int(key) % 360)
    raise ValueError(f"未知の run_id: {run_id}")


def physical_roll_from_suffix(suffix: str) -> float:
    """ファイル名識別子 _U/_R/_L または _000… からカメラカー roll[deg] を返す"""
    return physical_roll_from_run_id(suffix)


def infer_suffix_from_path(video_path: str) -> Optional[str]:
    """パス末尾の _000/_U/_R/_L、またはファイル名そのものが U/R/L/3桁 のとき検出する。"""
    name = video_path.replace("\\", "/").split("/")[-1]
    stem = name.rsplit(".", 1)[0]
    upper = stem.upper()
    if upper in ("U", "R", "L"):
        return upper
    if upper.isdigit() and len(upper) <= 3:
        return f"{int(upper):03d}"
    matched = re.search(r"_(\d{3})$", stem)
    if matched:
        return f"{int(matched.group(1)):03d}"
    for suffix in ("U", "R", "L"):
        if stem.endswith(f"_{suffix}"):
            return suffix
    return None


def capture_to_program_orientation_deg(
    physical_pitch_deg: float,
    physical_roll_deg: float,
) -> Tuple[float, float, float]:
    """カメラカー初期方位 → プログラム姿勢 (roll, yaw, pitch) [deg]

    壁面正対（physical_pitch=90°）を仮定する。光軸はカメラカー roll θ の壁
    ``(sin θ, cos θ, 0)`` を向き、現行 U と同じくカメラ +Y（画像下）は −Z。
    0° は ``(-90, 90, 90)``。0°/180° は pitch 特異点のため、内部 roll の
    符号で Euler 分岐する（θ≥0 は roll=-90,yaw=90、θ<0 は roll=90,yaw=-90）。

        (90, 0)    → (roll,yaw,pitch)=(-90, 90, 90)
        (90, 120)  → (-90, 90, -30)
        (90, -120) → (90, -90, -30)
    """
    pitch_c = float(physical_pitch_deg)
    if abs(pitch_c - 90.0) > 1e-3:
        raise ValueError(
            f"未対応のカメラカー pitch: {physical_pitch_deg}（壁面正対 90° のみ）"
        )
    th = physical_roll_to_internal_deg(physical_roll_deg)
    if th >= 0.0:
        return (-90.0, 90.0, 90.0 - th)
    return (90.0, -90.0, 90.0 + th)


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
        "run_id": _run_id_or_canonical(physical_roll_deg),
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


@dataclass(frozen=True)
class DirectionLayout:
    """n 方向の担当帯・カット・隣接接合。3 方向 U/R/L 定数とは別物。"""

    n_directions: int
    fov_deg: float
    optical_axis_spacing_deg: float
    rolls_deg: Tuple[float, ...]
    run_ids: Tuple[str, ...]
    assigned_width_deg: float
    cut_camera_car_deg: Tuple[float, ...]
    sectors: Dict[str, Tuple[float, float]]
    seam_pairs: Tuple[Tuple[str, str], ...]
    overlap_center_deg: Dict[Tuple[str, str], float]
    boundary_partners: Dict[str, Tuple[str, str]]


def build_direction_layout(
    n_directions: int = 8,
    optical_axis_spacing_deg: float = 45.0,
    fov_deg: float = 60.0,
) -> DirectionLayout:
    """光軸間隔と台数から担当帯・カット線・隣接 SEAM を作る。"""
    n = int(n_directions)
    if n < 1:
        raise ValueError(f"n_directions は 1 以上: {n_directions}")
    spacing = float(optical_axis_spacing_deg)
    if spacing <= 0.0:
        raise ValueError(f"optical_axis_spacing_deg は正: {optical_axis_spacing_deg}")
    fov = float(fov_deg)
    assigned = 360.0 / float(n)
    rolls = tuple((i * spacing) % 360.0 for i in range(n))
    run_ids = tuple(canonical_run_id(r) for r in rolls)
    half = assigned / 2.0
    cuts = tuple((rolls[i] + half) % 360.0 for i in range(n))
    sectors = {
        run_ids[i]: (rolls[i] - half, rolls[i] + half) for i in range(n)
    }
    pairs: List[Tuple[str, str]] = []
    overlap_center: Dict[Tuple[str, str], float] = {}
    partners: Dict[str, Tuple[str, str]] = {}
    for i in range(n):
        a = run_ids[i]
        nxt = run_ids[(i + 1) % n]
        prv = run_ids[(i - 1) % n]
        pairs.append((a, nxt))
        overlap_center[(a, nxt)] = float(cuts[i])
        partners[a] = (prv, nxt)
    return DirectionLayout(
        n_directions=n,
        fov_deg=fov,
        optical_axis_spacing_deg=spacing,
        rolls_deg=rolls,
        run_ids=run_ids,
        assigned_width_deg=assigned,
        cut_camera_car_deg=cuts,
        sectors=sectors,
        seam_pairs=tuple(pairs),
        overlap_center_deg=overlap_center,
        boundary_partners=partners,
    )


def _legacy_run_id_key(run_id: str) -> str:
    key = str(run_id or "").strip().upper()
    if key == "A":
        return "U"
    if key == "B":
        return "R"
    if key == "C":
        return "L"
    return key


def sector_camera_car_bounds(
    run_id: str,
    layout: Optional[DirectionLayout] = None,
) -> Tuple[float, float]:
    """run の担当帯 (lo, hi) [カメラカー deg]。U/R/L は 3 方向定数、3 桁は layout。"""
    key = _legacy_run_id_key(run_id)
    if layout is not None:
        if key in layout.sectors:
            return layout.sectors[key]
        try:
            cid = canonical_run_id(physical_roll_from_run_id(key))
        except ValueError:
            cid = ""
        if cid and cid in layout.sectors:
            return layout.sectors[cid]
    if key in SECTOR_CAMERA_CAR_DEG:
        return SECTOR_CAMERA_CAR_DEG[key]
    try:
        roll = physical_roll_from_run_id(key)
    except ValueError as exc:
        raise ValueError(f"未知の run_id: {run_id}") from exc
    default = build_direction_layout(8, 45.0, 60.0)
    cid = canonical_run_id(roll)
    if cid in default.sectors:
        return default.sectors[cid]
    half = default.assigned_width_deg / 2.0
    return (roll - half, roll + half)


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


def optical_axis_near_pipe_vertical(
    run_id: str,
    threshold_deg: float = 20.0,
) -> bool:
    """光軸が ±Y 近傍（カメラカー 0° または 180°）か。"""
    try:
        roll = physical_roll_from_run_id(run_id)
    except ValueError:
        return _legacy_run_id_key(run_id) == "U"
    internal = abs(physical_roll_to_internal_deg(roll))
    return internal <= float(threshold_deg) or abs(internal - 180.0) <= float(threshold_deg)


def mode_a_estimate_yaw_pitch(run_id: str) -> Tuple[bool, bool]:
    """Mode A で推定する角度 (estimate_yaw, estimate_pitch)。

    U: 初期 (roll,yaw,pitch)=(-90,90,90)。(0,0,90) と同一姿勢。
       車体 roll はプログラム pitch。1段階の自由変数は dz+dpitch のみ
       （yaw を同時に入れると pitch に相殺が漏れ積算する）。
       2段階 yaw（`mode_a_yaw_second_stage`）は実験用で既定オフ。
       画像 x→pitch、画像 y→z。pitch は 90° 求心。前後のうなずきは dz に載せる。
    R/L および 8 方向: 画像x=pitch, 画像y=z。roll=r0, yaw=y0, x=y=0 固定。
    """
    key = _legacy_run_id_key(run_id)
    if key in ("U", "R", "L"):
        return False, True
    try:
        physical_roll_from_run_id(key)
    except ValueError:
        return True, False
    return False, True


def mode_a_use_frame_xy_residual(run_id: str) -> bool:
    """Mode A の ±Y 光軸（0°/180°）はフレーム座標残差。その他は円筒 3D 残差。"""
    return optical_axis_near_pipe_vertical(run_id)


def direction_layout_from_config(td) -> DirectionLayout:
    """two_direction 設定から担当帯 layout を作る。"""
    return build_direction_layout(
        int(getattr(td, "n_directions", 8) or 8),
        float(getattr(td, "optical_axis_spacing_deg", 45.0) or 45.0),
        float(getattr(td, "fov_deg", 60.0) or 60.0),
    )


def theta_in_camera_car_sector(
    eta,
    run_id: str,
    layout: Optional[DirectionLayout] = None,
) -> np.ndarray:
    """η が run の担当帯（重なり中央で分割）に入るか。"""
    lo, hi = sector_camera_car_bounds(run_id, layout=layout)
    car = eta_to_camera_car_deg(eta)
    lo = lo % 360.0
    hi = hi % 360.0
    if lo < hi:
        return (car >= lo) & (car < hi)
    return (car >= lo) | (car < hi)


def assigned_sector_contains_eta0(
    run_id: str,
    layout: Optional[DirectionLayout] = None,
) -> bool:
    """担当帯がプログラム η=0（カメラカー 180°）を含むか。周期パッド対象。"""
    eta0 = np.array([0.0])
    mask = theta_in_camera_car_sector(eta0, run_id, layout=layout)
    return bool(np.asarray(mask).reshape(-1)[0])
