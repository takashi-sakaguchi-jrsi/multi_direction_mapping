"""pitch 移動平均トレンドの θ 補正"""

import numpy as np

from src.validation.geometry import build_run_reference, optical_axis_eta, wrap_angle_rad
from src.validation.strip_correction import (
    StripCorrectionResult,
    apply_pitch_trend_warp,
    pitch_trend_dtheta_rad,
)


def test_constant_pitch_gives_near_zero_dtheta():
    ref = build_run_reference(0.0)
    z = np.linspace(0.0, 4000.0, 400)
    pitch = np.full(z.shape, ref["pitch_ref_rad"])
    dth = pitch_trend_dtheta_rad(
        z, pitch,
        ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"],
        window_mm=800.0,
    )
    assert float(np.max(np.abs(np.degrees(dth)))) < 0.05


def test_slow_pitch_ramp_is_undone_at_the_end():
    ref = build_run_reference(0.0)
    z = np.linspace(0.0, 5000.0, 500)
    pitch = ref["pitch_ref_rad"] + np.linspace(0.0, np.radians(-8.0), z.size)
    dth = pitch_trend_dtheta_rad(
        z, pitch,
        ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"],
        window_mm=800.0,
    )
    eta_ref = optical_axis_eta(
        ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"],
    )
    eta_end = optical_axis_eta(
        ref["roll_ref_rad"], ref["yaw_ref_rad"], float(pitch[-1]),
    )
    expected = wrap_angle_rad(eta_ref - eta_end)
    assert abs(np.degrees(expected)) > 1.0
    assert np.sign(dth[-1]) == np.sign(expected) or abs(dth[-1]) < 1e-9
    assert abs(dth[-1]) > 0.5 * abs(expected)
    assert abs(dth[0]) < 0.25 * abs(dth[-1])


def test_fast_pitch_oscillation_is_mostly_kept():
    ref = build_run_reference(0.0)
    z = np.linspace(0.0, 4000.0, 400)
    pitch = ref["pitch_ref_rad"] + np.radians(1.0) * np.sin(2.0 * np.pi * z / 80.0)
    dth = pitch_trend_dtheta_rad(
        z, pitch,
        ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"],
        window_mm=800.0,
    )
    assert float(np.max(np.abs(np.degrees(dth)))) < 0.25


def test_apply_pitch_trend_shifts_a_horizontal_line():
    ref = build_run_reference(0.0)
    h, w = 80, 120
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    filled = np.ones((h, w), dtype=bool)
    row0 = 40
    rgb[row0, :, :] = 255
    z = np.linspace(0.0, 120.0, 40)
    pitch = np.full(z.shape, ref["pitch_ref_rad"])
    src = StripCorrectionResult(
        rgb=rgb, filled=filled, z_min=0.0, z_max=120.0,
        dtheta_rad=None, z_ctrl=None, message="z ok",
    )
    out = apply_pitch_trend_warp(
        src, z, pitch,
        ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"],
        pixels_per_mm=1.0, window_mm=40.0,
    )
    assert "pitch θトレンド" in out.message
    bright = np.argmax(out.rgb[:, w // 2, 0])
    assert abs(int(bright) - row0) <= 1
