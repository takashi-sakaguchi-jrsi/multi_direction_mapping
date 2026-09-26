"""T1: カメラカー初期方位とプログラム姿勢の対応"""

import numpy as np

from src.validation.geometry import (
    angle_diff_deg,
    build_run_reference,
    capture_to_program_orientation_deg,
    optical_axis_world,
    physical_roll_to_internal_deg,
    pose_R_c2w,
    wrap_angle_deg,
)


def test_physical_240_is_internal_minus_120():
    assert abs(physical_roll_to_internal_deg(240.0) + 120.0) < 1e-9
    assert abs(wrap_angle_deg(240.0) + 120.0) < 1e-9
    assert abs(angle_diff_deg(240.0, -120.0)) < 1e-9


def test_capture_to_program_orientation():
    # (program_roll, program_yaw, program_pitch)
    # R/L の roll 符号は Demo のカメラ上下に合わせる（光軸は同じ右下/左下）
    assert capture_to_program_orientation_deg(90.0, 0.0) == (-90.0, 90.0, 90.0)
    assert capture_to_program_orientation_deg(90.0, 120.0) == (-90.0, 90.0, -30.0)
    assert capture_to_program_orientation_deg(90.0, -120.0) == (90.0, -90.0, -30.0)
    assert capture_to_program_orientation_deg(90.0, 240.0) == (90.0, -90.0, -30.0)


def test_run_reference_ids_and_program_angles():
    u = build_run_reference(0.0)
    r = build_run_reference(120.0)
    l = build_run_reference(240.0)
    assert u["run_id"] == "U"
    assert r["run_id"] == "R"
    assert l["run_id"] == "L"
    assert abs(l["physical_roll_internal_deg"] + 120.0) < 1e-9
    assert (u["roll_ref_deg"], u["yaw_ref_deg"], u["pitch_ref_deg"]) == (-90.0, 90.0, 90.0)
    assert (r["roll_ref_deg"], r["yaw_ref_deg"], r["pitch_ref_deg"]) == (-90.0, 90.0, -30.0)
    assert (l["roll_ref_deg"], l["yaw_ref_deg"], l["pitch_ref_deg"]) == (90.0, -90.0, -30.0)


def test_u_minus90_90_90_is_same_pose_as_0_0_90():
    """U の (-90,90,90) は旧 (0,0,90) と同一のカメラ姿勢（roll 符号を合わせた組）。"""
    r_old = pose_R_c2w(*np.radians((0.0, 0.0, 90.0)))
    r_new = pose_R_c2w(*np.radians((-90.0, 90.0, 90.0)))
    r_wrong = pose_R_c2w(*np.radians((90.0, 90.0, 90.0)))
    np.testing.assert_allclose(r_new, r_old, atol=1e-6)
    assert np.linalg.norm(r_wrong - r_old) > 1.0


def test_program_optical_axes_from_converted_orientation():
    """変換後のプログラム姿勢が向く壁方向。

    U は真上 (+Y)。R/L は水平より 30° 下の右下/左下。
    """
    expected = {
        0.0: np.array([0.0, 1.0, 0.0]),
        120.0: np.array([np.sqrt(3.0) / 2.0, -0.5, 0.0]),
        -120.0: np.array([-np.sqrt(3.0) / 2.0, -0.5, 0.0]),
    }
    for physical_roll, axis_exp in expected.items():
        ref = build_run_reference(physical_roll)
        ori = np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]])
        v = optical_axis_world(*ori)
        assert abs(v[2]) < 0.08, f"光軸が管軸方向に残っている: {v} roll={physical_roll}"
        assert float(v @ axis_exp) > 0.95, f"{physical_roll}: {v} vs {axis_exp}"


def test_program_rotation_matches_demo_full_frame():
    """光軸だけでなく、Demo 生成時のカメラ上下左右も一致すること。

    投影は v_world = v_cam @ R_c2w、生成は v_world = R_demo @ v_cam。
    """
    from src.validation.fisheye_sideview_renderer import FisheyeSideviewRenderer
    from src.validation.geometry import pose_R_c2w

    for physical_roll in (0.0, 120.0, -120.0):
        R_demo = FisheyeSideviewRenderer.demo_rotation(physical_roll, 90.0)
        ref = build_run_reference(physical_roll)
        R_prog = pose_R_c2w(ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"])
        np.testing.assert_allclose(
            R_prog.T, R_demo, atol=1e-6, err_msg=f"roll={physical_roll}"
        )
