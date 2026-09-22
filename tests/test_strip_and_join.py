"""エッジ中心の θ 補正と 60/-60/180 接合"""

import numpy as np

from src.validation.geometry import (
    camera_car_deg_to_eta,
    eta_to_camera_car_deg,
    expected_edge_etas,
    optical_axis_eta,
    theta_in_camera_car_sector,
)
from src.validation.geometry import build_run_reference
from src.validation.seam_join import join_at_overlap_centers
from src.validation.strip_correction import warp_z_to_ocr, warp_theta_to_edge_center


def test_overlap_centers_map_to_expected_eta():
    np.testing.assert_allclose(eta_to_camera_car_deg(camera_car_deg_to_eta(60.0)) % 360.0, 60.0)
    np.testing.assert_allclose(eta_to_camera_car_deg(camera_car_deg_to_eta(-60.0)) % 360.0, 300.0)
    np.testing.assert_allclose(eta_to_camera_car_deg(camera_car_deg_to_eta(180.0)) % 360.0, 180.0)


def test_u_r_l_sectors_meet_at_overlap_centers():
    eta60 = float(camera_car_deg_to_eta(60.0))
    eta_m60 = float(camera_car_deg_to_eta(-60.0))
    eta180 = float(camera_car_deg_to_eta(180.0))
    # 60° は R 側（境界は後半 run）
    assert not bool(theta_in_camera_car_sector(eta60, "U"))
    assert bool(theta_in_camera_car_sector(eta60, "R"))
    assert bool(theta_in_camera_car_sector(eta_m60, "U"))
    assert not bool(theta_in_camera_car_sector(eta_m60, "L"))
    assert bool(theta_in_camera_car_sector(eta180, "L"))
    assert not bool(theta_in_camera_car_sector(eta180, "R"))


def test_u_optical_axis_is_inside_u_sector():
    ref = build_run_reference(0.0)
    eta = optical_axis_eta(ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"])
    assert bool(theta_in_camera_car_sector(eta, "U"))
    np.testing.assert_allclose(eta_to_camera_car_deg(eta) % 360.0, 0.0, atol=1e-5)


def test_join_cuts_at_sixty_and_keeps_one_run():
    h, w = 36, 40
    ppm = 1.0
    rgb_u = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_r = np.zeros((h, w, 3), dtype=np.uint8)
    filled = np.ones((h, w), dtype=bool)
    rgb_u[:] = (255, 0, 0)
    rgb_r[:] = (0, 255, 0)
    out = join_at_overlap_centers(
        [
            {"rgb": rgb_u, "filled": filled, "z_min": 0.0, "z_max": 40.0, "run_id": "U"},
            {"rgb": rgb_r, "filled": filled, "z_min": 0.0, "z_max": 40.0, "run_id": "R"},
        ],
        pixels_per_mm=ppm,
        z_pad_mm=0.0,
    )
    eta = np.linspace(0.0, 2.0 * np.pi, h)
    car = eta_to_camera_car_deg(eta)
    u_rows = out["rgb"][:, 10, 0] > 0
    r_rows = out["rgb"][:, 10, 1] > 0
    assert np.any(u_rows)
    assert np.any(r_rows)
    # 同じ行に両色が乗らない
    assert not np.any(u_rows & r_rows)
    # 光軸付近（カメラカー 0° / 120°）はそれぞれの run
    row_u = int(np.argmin(np.abs((car - 0.0 + 180) % 360 - 180)))
    row_r = int(np.argmin(np.abs((car - 120.0 + 180) % 360 - 180)))
    assert u_rows[row_u]
    assert r_rows[row_r]


def test_z_warp_moves_columns_toward_ocr():
    h, w = 8, 40
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, 10, :] = 200
    filled = np.ones((h, w), dtype=bool)
    z_est = np.array([0.0, 10.0, 20.0, 30.0])
    z_ocr = np.array([0.0, 12.0, 24.0, 36.0])
    out, z0, z1, _ctrl, msg = warp_z_to_ocr(
        rgb, filled, z_min=0.0, pixels_per_mm=1.0,
        z_est_mm=z_est, z_ocr_mm=z_ocr, control_spacing_mm=5.0,
    )
    assert "z補正" in msg
    assert z0 == 0.0
    # 元 x=10 が OCR 側へ寄る
    col = np.argmax(out["rgb"][0, :, 0])
    assert col > 10


def test_z_warp_ocr_span_width_matches_za_max():
    """za_max までの出力幅は OCR で決まり、U/R の z_est 終端差の影響を受けない。"""
    h, w = 6, 80
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    filled = np.ones((h, w), dtype=bool)
    z_ocr = np.array([0.0, 20.0, 40.0, 60.0])
    ppm = 1.0
    out_a, z0a, _z1a, _, _ = warp_z_to_ocr(
        rgb, filled, z_min=0.0, pixels_per_mm=ppm,
        z_est_mm=np.array([0.0, 18.0, 36.0, 50.0]), z_ocr_mm=z_ocr,
        control_spacing_mm=15.0,
    )
    out_b, z0b, _z1b, _, _ = warp_z_to_ocr(
        rgb, filled, z_min=0.0, pixels_per_mm=ppm,
        z_est_mm=np.array([0.0, 19.0, 38.0, 55.0]), z_ocr_mm=z_ocr,
        control_spacing_mm=15.0,
    )
    assert z0a == 0.0 and z0b == 0.0
    za_max = 60.0
    col_a = int(round(za_max * ppm))
    col_b = int(round(za_max * ppm))
    assert col_a == col_b
    assert out_a["rgb"].shape[1] >= col_a
    assert out_b["rgb"].shape[1] >= col_b


def test_theta_warp_shifts_strip_toward_expected_center():
    h, w = 120, 30
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    filled = np.zeros((h, w), dtype=bool)
    # 中心を 20 行ずらした帯
    filled[30:80, :] = True
    rgb[30:80, :] = 180
    ref = build_run_reference(0.0)
    ori = np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]])
    half = np.radians(72.0)
    out, dtheta, msg = warp_theta_to_edge_center(
        rgb, filled, z_min=0.0, pixels_per_mm=1.0,
        orientation_ref=ori, half_fov_rad=half, control_spacing_mm=5.0,
    )
    assert dtheta is not None
    ys = np.where(out["filled"][:, 10])[0]
    mid = 0.5 * (ys.min() + ys.max())
    center, _, _ = expected_edge_etas(ori[0], ori[1], ori[2], half)
    expected_row = center / (2.0 * np.pi) * (h - 1)
    assert abs(mid - expected_row) < abs(55.0 - expected_row)


def test_theta_warp_does_not_flare_at_leading_edge():
    """開始列だけ被覆が欠けていても、左端で大きくせん断しない。"""
    h, w = 120, 80
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    filled = np.zeros((h, w), dtype=bool)
    filled[40:90, 20:] = True
    rgb[40:90, 20:] = 180
    filled[5:20, :20] = True
    rgb[5:20, :20] = 180
    ref = build_run_reference(0.0)
    ori = np.array([ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]])
    half = np.radians(72.0)
    out, dtheta, msg = warp_theta_to_edge_center(
        rgb, filled, z_min=0.0, pixels_per_mm=1.0,
        orientation_ref=ori, half_fov_rad=half, control_spacing_mm=8.0,
    )
    assert dtheta is not None
    # 欠け列（x=5）と安定列（x=40）の θ 補正が同程度
    assert abs(dtheta[5] - dtheta[40]) < np.radians(8.0)
    ys_l = np.where(out["filled"][:, 25])[0]
    ys_m = np.where(out["filled"][:, 50])[0]
    assert ys_l.size > 0 and ys_m.size > 0
    mid_l = 0.5 * (ys_l.min() + ys_l.max())
    mid_m = 0.5 * (ys_m.min() + ys_m.max())
    assert abs(mid_l - mid_m) < 12.0


def test_z_warp_dest_origin_is_zero():
    """出力 z 原点は 0。za_max より左は OCR 軸、右は見越しの貼り合わせ。"""
    h, w = 6, 50
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, 25, :] = 200
    filled = np.ones((h, w), dtype=bool)
    z_est = np.array([20.0, 40.0, 60.0])
    z_ocr = np.array([20.0, 40.0, 60.0])
    out, z0, z1, _ctrl, msg = warp_z_to_ocr(
        rgb, filled, z_min=0.0, pixels_per_mm=1.0,
        z_est_mm=z_est, z_ocr_mm=z_ocr, control_spacing_mm=15.0,
    )
    assert "z補正" in msg
    assert z0 == 0.0
    col = int(np.argmax(out["rgb"][0, :, 0]))
    assert 15 <= col <= 35


def test_z_warp_keeps_source_right_tail():
    """最後のカメラ z より先の列（見越し）が補正後も残ること。"""
    h, w = 6, 80
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, -8:, :] = (0, 200, 255)
    filled = np.ones((h, w), dtype=bool)
    z_est = np.array([0.0, 20.0, 40.0])
    z_ocr = np.array([10.0, 30.0, 50.0])
    out, z0, z1, _ctrl, msg = warp_z_to_ocr(
        rgb, filled, z_min=0.0, pixels_per_mm=1.0,
        z_est_mm=z_est, z_ocr_mm=z_ocr, control_spacing_mm=15.0,
    )
    assert "z補正" in msg
    cyan = (out["rgb"][..., 0] < 40) & (out["rgb"][..., 1] > 150) & (out["rgb"][..., 2] > 200)
    assert np.any(cyan)
    xs = np.where(cyan.any(axis=0))[0]
    assert xs.size >= 6
    assert int(xs.max()) >= out["rgb"].shape[1] - 12
