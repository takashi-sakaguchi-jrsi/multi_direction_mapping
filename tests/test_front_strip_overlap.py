"""展開後の帯重なり特徴点誤差と広域変換"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.config import CameraConfig, Config, FrontConfig
from src.front_strip_overlap import (
    StripOverlapResult,
    allow_bin_worse_for_pose,
    leftover_to_pose_delta,
    match_unwrapped_strips,
    pose_correction_helps,
    pose_correction_verdict,
    pose_dc_from_inliers,
    shift_strip_eta,
    warp_strip_affine,
)
from src.product_runtime import apply_front_profile


def _textured(h: int, w: int, seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    yy, xx = np.indices((h, w))
    cell = 12
    g = np.where(((yy // cell) + (xx // cell)) % 2 == 0, 40, 220).astype(np.int16)
    g = np.clip(g + rng.randint(-12, 13, g.shape), 0, 255).astype(np.uint8)
    return np.stack([g, np.roll(g, 3, 0), np.roll(g, 5, 1)], axis=-1)


def test_match_recovers_strip_translation():
    h, w = 180, 100
    base = _textured(h, w)
    painted = np.zeros((h, w + 40, 3), dtype=np.uint8)
    painted[:, 20:20 + w] = base
    strip = np.roll(np.roll(base, 7, axis=0), 4, axis=1)
    mask = np.ones((h, w), dtype=bool)
    result = match_unwrapped_strips(painted, strip, mask, start_col=20)
    assert result.ok
    assert result.n_inlier >= 8
    assert result.med_drow_px == pytest.approx(7.0, abs=1.5)
    assert result.med_dcol_px == pytest.approx(4.0, abs=1.5)
    assert result.affine is not None
    # 帯→既描画の並進は誤差の逆向き
    assert result.affine[0, 2] == pytest.approx(-4.0, abs=2.0)
    assert result.affine[1, 2] == pytest.approx(-7.0, abs=2.0)
    assert result.affine[0, 0] == pytest.approx(1.0)
    assert result.affine[0, 1] == pytest.approx(0.0)
    assert result.affine[1, 0] == pytest.approx(0.0)
    assert result.affine[1, 1] == pytest.approx(1.0)


def test_warp_moves_featureless_pixels():
    h, w = 80, 120
    colors = np.zeros((h, w, 3), dtype=np.uint8)
    colors[20:50, 90:115] = (200, 10, 10)
    mask = np.zeros((h, w), dtype=bool)
    mask[20:50, 90:115] = True
    radius = np.zeros((h, w), dtype=np.float32)
    radius[20:50, 90:115] = 40.0
    affine = np.array([[1.0, 0.0, -25.0], [0.0, 1.0, 8.0]], dtype=np.float64)
    warped, mask_w, radius_w = warp_strip_affine(colors, mask, radius, affine)
    # (35, 100) が (43, 75) へ。特徴点のない画素も同じ変換。
    assert warped[43, 75, 0] > 150
    assert colors[43, 75, 0] == 0
    assert mask_w[43, 75]
    assert not mask_w[35, 100]
    assert radius_w[43, 75] > 20.0


def test_paste_shift_aligns_without_resampling():
    """η 周期シフトと start_col ずらしで、取り直しなしに既描画へ重なる。"""
    h, w = 180, 100
    base = _textured(h, w)
    painted = np.zeros((h, w + 40, 3), dtype=np.uint8)
    painted[:, 20:20 + w] = base
    strip = np.roll(np.roll(base, 7, axis=0), 4, axis=1)
    mask = np.ones((h, w), dtype=bool)
    result = match_unwrapped_strips(painted, strip, mask, start_col=20)
    assert result.ok
    shifted, mask2, _ = shift_strip_eta(strip, mask, None, result.med_drow_px)
    start2 = 20 - int(round(result.med_dcol_px))
    after = match_unwrapped_strips(painted, shifted, mask2, start_col=start2)
    assert after.ok
    assert after.med_abs_px < result.med_abs_px * 0.5


def test_rolling_strip_by_minus_median_reduces_error():
    """取り出し並進と同じ符号: 帯を -d だけ回すと既描画に重なる。"""
    h, w = 180, 100
    base = _textured(h, w)
    painted = np.zeros((h, w + 40, 3), dtype=np.uint8)
    painted[:, 20:20 + w] = base
    strip = np.roll(np.roll(base, 7, axis=0), 4, axis=1)
    mask = np.ones((h, w), dtype=bool)
    result = match_unwrapped_strips(painted, strip, mask, start_col=20)
    assert result.ok
    aligned = np.roll(
        np.roll(strip, -int(round(result.med_drow_px)), axis=0),
        -int(round(result.med_dcol_px)),
        axis=1,
    )
    after = match_unwrapped_strips(painted, aligned, mask, start_col=20)
    assert after.ok
    assert after.med_abs_px < result.med_abs_px * 0.5


def test_warp_wraps_eta_period():
    h, w = 40, 20
    colors = np.zeros((h, w, 3), dtype=np.uint8)
    colors[0:5, 5:15] = (200, 10, 10)
    mask = np.zeros((h, w), dtype=bool)
    mask[0:5, 5:15] = True
    affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -3.0]], dtype=np.float64)
    warped, mask_w, _ = warp_strip_affine(colors, mask, None, affine)
    # 行 0 が (0-3) mod 40 = 37 へ。行 3 は 0 へ回る。
    assert warped[37, 10, 0] > 150
    assert mask_w[37, 10]
    assert mask_w[0, 10]
    assert not mask_w[3, 10]


def test_strip_overlap_match_disables_phase_corr_align():
    cfg = Config()
    cfg.pipeline = "front"
    apply_front_profile(cfg)
    cfg.front.strip_overlap_match = True
    cfg.front.overlap_paste_align = True
    from src.colormap_generator import ColorMapGenerator

    gen = ColorMapGenerator(cfg, MagicMock())
    assert gen.strip_overlap_match is True
    assert gen.overlap_paste_align is False
    assert gen.strip_overlap_correct is False

    cfg.front.strip_overlap_correct = True
    gen2 = ColorMapGenerator(cfg, MagicMock())
    assert gen2.strip_overlap_correct is True


def test_sideview_does_not_enable_strip_overlap_match():
    cfg = Config()
    cfg.pipeline = "sideview"
    cfg.front.strip_overlap_match = True
    from src.colormap_generator import ColorMapGenerator

    gen = ColorMapGenerator(cfg, MagicMock())
    assert gen.strip_overlap_match is False


def test_transformer_calibration_can_drop_kannala():
    cam = CameraConfig(apply_lens_calibration=False)
    cam._lens_calibration = object()
    assert cam.transformer_calibration() is None
    cam.apply_lens_calibration = True
    assert cam.transformer_calibration() is not None


def _overlap_result(painted_xy: np.ndarray, strip_xy: np.ndarray) -> StripOverlapResult:
    dcol = strip_xy[:, 0] - painted_xy[:, 0]
    drow = strip_xy[:, 1] - painted_xy[:, 1]
    mag = np.hypot(dcol, drow)
    n = len(painted_xy)
    return StripOverlapResult(
        n_raw=n,
        n_kept=n,
        n_inlier=n,
        med_drow_px=float(np.median(drow)),
        med_dcol_px=float(np.median(dcol)),
        rms_px=float(np.sqrt(np.mean(mag ** 2))),
        med_abs_px=float(np.median(mag)),
        affine=None,
        pts_painted=painted_xy.astype(float),
        pts_strip=strip_xy.astype(float),
        inlier_mask=np.ones(n, dtype=bool),
        src_c0=0,
        dest_c0=0,
        ok=True,
    )


def test_leftover_to_pose_delta_units():
    dz, droll = leftover_to_pose_delta(10.0, 6.0, eta_resolution=360, pixels_per_mm=3.0)
    assert dz == pytest.approx(2.0)
    assert np.degrees(droll) == pytest.approx(10.0)
    # 帯が +η/+z なら leftover は正。カメラは roll0-droll, z0-dz で戻す。


def test_pose_dc_ignores_left_wall_bias():
    rng = np.random.RandomState(0)
    h = 160
    painted = []
    strip = []
    for b in range(8):
        y0 = b * (h / 8.0) + 4.0
        n = 200 if b == 2 else 12
        drow = 20.0 if b == 2 else 3.0
        xs = rng.uniform(50.0, 90.0, n)
        ys = rng.uniform(y0, y0 + h / 8.0 - 4.0, n)
        for x, y in zip(xs, ys):
            painted.append([x, y])
            strip.append([x + 4.0, y + drow])
    result = _overlap_result(np.array(painted), np.array(strip))
    dc = pose_dc_from_inliers(result, h)
    assert dc is not None
    dc_drow, dc_dcol, n_bins = dc
    assert n_bins >= 4
    assert dc_dcol == pytest.approx(4.0, abs=0.5)
    assert dc_drow == pytest.approx(3.0, abs=2.5)
    assert result.med_drow_px > 8.0


def test_pose_dc_prefers_high_z_seam():
    painted = []
    strip = []
    for i in range(40):
        y = 20.0 + (i % 8) * 15.0
        painted.append([12.0, y])
        strip.append([12.0 + 4.0, y + 28.0])
    for i in range(40):
        y = 20.0 + (i % 8) * 15.0
        painted.append([70.0, y])
        strip.append([70.0 + 4.0, y + 2.0])
    result = _overlap_result(np.array(painted), np.array(strip))
    dc = pose_dc_from_inliers(result, eta_height=160, seam_frac=0.4)
    assert dc is not None
    dc_drow, dc_dcol, _ = dc
    assert dc_dcol == pytest.approx(4.0, abs=0.5)
    assert dc_drow == pytest.approx(2.0, abs=2.0)


def test_pose_dc_falls_back_when_seam_has_few_eta_bins():
    painted = []
    strip = []
    for i in range(40):
        painted.append([80.0, 45.0 + (i % 5)])
        strip.append([84.0, 65.0 + (i % 5)])
    for b in range(8):
        y0 = b * 20.0 + 6.0
        for j in range(6):
            painted.append([20.0 + j, y0 + j])
            strip.append([24.0 + j, y0 + j + 3.0])
    result = _overlap_result(np.array(painted), np.array(strip))
    dc = pose_dc_from_inliers(result, eta_height=160)
    assert dc is not None
    dc_drow, dc_dcol, n_bins = dc
    assert n_bins >= 3
    assert dc_dcol == pytest.approx(4.0, abs=0.5)
    assert abs(dc_drow) < 8.0


def test_pose_correction_helps_rejects_one_wall_worse():
    h = 160
    painted = []
    strip_b = []
    strip_a = []
    for b in range(8):
        y0 = b * (h / 8.0) + 6.0
        for j in range(8):
            x = 40.0 + j
            y = y0 + j
            painted.append([x, y])
            strip_b.append([x + 4.0, y + 4.0])
            if b < 4:
                strip_a.append([x + 1.0, y + 1.0])
            else:
                strip_a.append([x + 12.0, y + 12.0])
    before = _overlap_result(np.array(painted), np.array(strip_b))
    after = _overlap_result(np.array(painted), np.array(strip_a))
    assert not pose_correction_helps(before, after, h)


def test_pose_correction_verdict_reports_bin_worse_when_rms_drops():
    h = 160
    painted = []
    strip_b = []
    strip_a = []
    for b in range(8):
        y0 = b * (h / 8.0) + 6.0
        for j in range(8):
            x = 40.0 + j
            y = y0 + j
            painted.append([x, y])
            strip_b.append([x + 8.0, y])
            if b < 4:
                strip_a.append([x + 1.0, y])
            else:
                strip_a.append([x + 10.0, y])
    before = _overlap_result(np.array(painted), np.array(strip_b))
    after = _overlap_result(np.array(painted), np.array(strip_a))
    assert after.rms_px < before.rms_px
    ok, why = pose_correction_verdict(before, after, h)
    assert ok is False
    assert "ηビン悪化" in why
    ok_allow, why_allow = pose_correction_verdict(
        before, after, h, allow_bin_worse=True
    )
    assert ok_allow is True
    assert "ηビン悪化許容" in why_allow
    assert pose_correction_helps(before, after, h, allow_bin_worse=True)


def test_z_only_allows_bin_worse_even_when_flag_off():
    assert allow_bin_worse_for_pose(False, 0.0) is True
    assert allow_bin_worse_for_pose(False, 0.01) is False
    assert allow_bin_worse_for_pose(True, 0.01) is True


def test_pose_correction_verdict_reports_rms_and_match_fail():
    h = 160
    painted = []
    strip_b = []
    strip_worse = []
    for b in range(8):
        y0 = b * (h / 8.0) + 6.0
        for j in range(8):
            x = 40.0 + j
            y = y0 + j
            painted.append([x, y])
            strip_b.append([x + 2.0, y + 2.0])
            strip_worse.append([x + 8.0, y + 8.0])
    before = _overlap_result(np.array(painted), np.array(strip_b))
    after_rms = _overlap_result(np.array(painted), np.array(strip_worse))
    ok, why = pose_correction_verdict(before, after_rms, h)
    assert ok is False
    assert "全体RMS非改善" in why
    ok_allow, _ = pose_correction_verdict(
        before, after_rms, h, allow_bin_worse=True
    )
    assert ok_allow is False
    after_fail = _overlap_result(np.array(painted), np.array(strip_b))
    after_fail.ok = False
    after_fail.n_inlier = 0
    ok, why = pose_correction_verdict(before, after_fail, h)
    assert ok is False
    assert "取り直しマッチ失敗" in why


def test_pose_correction_helps_accepts_uniform_drop():
    h = 160
    painted = []
    strip_b = []
    strip_a = []
    for b in range(8):
        y0 = b * (h / 8.0) + 6.0
        for j in range(8):
            x = 40.0 + j
            y = y0 + j
            painted.append([x, y])
            strip_b.append([x + 5.0, y + 5.0])
            strip_a.append([x + 1.0, y + 1.0])
    before = _overlap_result(np.array(painted), np.array(strip_b))
    after = _overlap_result(np.array(painted), np.array(strip_a))
    assert pose_correction_helps(before, after, h)


def test_front_defaults_keep_experiment_off():
    front = FrontConfig()
    assert front.strip_overlap_match is False
    assert front.strip_overlap_correct is False
    assert front.strip_overlap_allow_bin_worse is False
    assert front.deroll_colormap_after_phase2 is False
    assert front.residual_paste_warp is True
    assert front.physical_attitude_source == "match"
    cam = CameraConfig()
    assert cam.apply_lens_calibration is True
