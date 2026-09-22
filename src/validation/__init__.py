"""2方向（真横）動画合成カラーマップ検証パッケージ"""

from src.validation.fisheye_sideview_renderer import FisheyeSideviewRenderer
from src.validation.geometry import (
    wrap_angle_deg,
    wrap_angle_rad,
    physical_roll_to_internal_deg,
    run_id_from_physical_roll,
    physical_roll_from_suffix,
    capture_to_program_orientation_deg,
    build_run_reference,
    derive_usable_half_fov_deg,
    optical_axis_world,
    pose_R_c2w,
)

__all__ = [
    "wrap_angle_deg",
    "wrap_angle_rad",
    "physical_roll_to_internal_deg",
    "run_id_from_physical_roll",
    "physical_roll_from_suffix",
    "capture_to_program_orientation_deg",
    "build_run_reference",
    "derive_usable_half_fov_deg",
    "optical_axis_world",
    "pose_R_c2w",
    "FisheyeSideviewRenderer",
]
