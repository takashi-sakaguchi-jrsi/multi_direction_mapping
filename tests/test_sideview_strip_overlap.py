"""工程5: 側視の帯重なり leftover は (z, pitch)。正面モジュールは import しない。"""

import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import Config, ConfigValidationError
from src.validation import sideview_strip_overlap as svo
from src.validation.sideview_strip_overlap import (
    StripOverlapResult,
    allow_bin_worse_for_pose,
    inherit_running_pose,
    leftover_to_pose_delta,
    match_unwrapped_strips,
    maybe_correct_strip,
    painted_row_span,
    pose_correction_verdict,
    pose_dc_from_inliers,
)


def _textured(h: int, w: int, seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    yy, xx = np.indices((h, w))
    cell = 12
    g = np.where(((yy // cell) + (xx // cell)) % 2 == 0, 40, 220).astype(np.int16)
    g = np.clip(g + rng.randint(-12, 13, g.shape), 0, 255).astype(np.uint8)
    return np.stack([g, np.roll(g, 3, 0), np.roll(g, 5, 1)], axis=-1)


def _overlap_result(painted_pts, strip_pts) -> StripOverlapResult:
    n = len(painted_pts)
    return StripOverlapResult(
        n_raw=n, n_kept=n, n_inlier=n,
        med_drow_px=0.0, med_dcol_px=0.0, rms_px=0.0, med_abs_px=0.0,
        pts_painted=np.asarray(painted_pts, dtype=float),
        pts_strip=np.asarray(strip_pts, dtype=float),
        inlier_mask=np.ones(n, dtype=bool),
        src_c0=0, dest_c0=0, ok=True,
    )


def test_sideview_module_does_not_import_front():
    tree = ast.parse(inspect.getsource(svo))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    joined = " ".join(imported)
    assert "front_strip" not in joined
    assert "colormap_generator" not in joined


def test_leftover_to_pose_delta_units_and_sign():
    d_z, d_pitch = leftover_to_pose_delta(10.0, 6.0, eta_resolution=360, pixels_per_mm=3.0)
    assert d_z == pytest.approx(2.0)
    assert np.degrees(d_pitch) == pytest.approx(10.0)
    z0, pitch0 = 100.0, 0.20
    assert (z0 - d_z) < z0
    assert (pitch0 - d_pitch) < pitch0


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


def test_pose_dc_from_inliers_uses_eta_bins():
    rng = np.random.RandomState(0)
    h = 160
    painted = []
    strip = []
    for b in range(8):
        y0 = b * (h / 8.0) + 4.0
        n = 12
        xs = rng.uniform(50.0, 90.0, n)
        ys = rng.uniform(y0, y0 + h / 8.0 - 4.0, n)
        for x, y in zip(xs, ys):
            painted.append([x, y])
            strip.append([x + 4.0, y + 3.0])
    result = _overlap_result(np.array(painted), np.array(strip))
    dc = pose_dc_from_inliers(result, h)
    assert dc is not None
    dc_drow, dc_dcol, n_bins = dc
    assert n_bins >= 3
    assert dc_dcol == pytest.approx(4.0, abs=0.5)
    assert dc_drow == pytest.approx(3.0, abs=0.5)


def test_painted_row_span_uses_mask_not_canvas():
    mask = np.zeros((2357, 40), dtype=bool)
    mask[880:1480] = True
    row0, row1 = painted_row_span(mask)
    assert row0 == 880
    assert row1 == 1480


def test_pose_dc_bins_painted_60deg_not_full_canvas():
    """60° 帯を 6 ビンにすると drow が残り、全周 8 ビンだと 2 ビンフォールバックで 0 になる。"""
    canvas_h = 2357
    row0, row1 = 880, 1480
    painted = []
    strip = []
    for y in np.linspace(890.0, 1470.0, 6):
        for j in range(5):
            painted.append([60.0 + j, y])
            strip.append([64.0 + j, y + 3.0])
    result = _overlap_result(np.array(painted), np.array(strip))
    dc_full = pose_dc_from_inliers(
        result, canvas_h, n_bins=8, min_bin_points=3,
    )
    assert dc_full is not None
    assert dc_full[0] == pytest.approx(0.0)
    assert dc_full[2] == 2
    dc_band = pose_dc_from_inliers(
        result, canvas_h, row0=row0, row1=row1, n_bins=6, min_bin_points=2,
    )
    assert dc_band is not None
    dc_drow, dc_dcol, n_used = dc_band
    assert n_used >= 3
    assert dc_dcol == pytest.approx(4.0, abs=0.5)
    assert dc_drow == pytest.approx(3.0, abs=0.5)


def test_allow_bin_worse_for_z_only():
    assert allow_bin_worse_for_pose(False, d_pitch=0.0) is True
    assert allow_bin_worse_for_pose(False, d_pitch=0.01) is False
    assert allow_bin_worse_for_pose(True, d_pitch=0.01) is True


def test_pose_correction_verdict_requires_rms_improve():
    h = 160
    before_pts = np.array([[10.0, 20.0 + 20 * i] for i in range(8)])
    before = _overlap_result(before_pts, before_pts + np.array([4.0, 3.0]))
    drow = before.pts_strip[:, 1] - before.pts_painted[:, 1]
    dcol = before.pts_strip[:, 0] - before.pts_painted[:, 0]
    before.rms_px = float(np.sqrt(np.mean(dcol ** 2 + drow ** 2)))
    after = _overlap_result(before_pts, before_pts + np.array([0.4, 0.3]))
    drow_a = after.pts_strip[:, 1] - after.pts_painted[:, 1]
    dcol_a = after.pts_strip[:, 0] - after.pts_painted[:, 0]
    after.rms_px = float(np.sqrt(np.mean(dcol_a ** 2 + drow_a ** 2)))
    ok, why = pose_correction_verdict(before, after, h)
    assert ok
    assert "改善" in why
    worse = _overlap_result(before_pts, before_pts + np.array([8.0, 6.0]))
    drow_w = worse.pts_strip[:, 1] - worse.pts_painted[:, 1]
    dcol_w = worse.pts_strip[:, 0] - worse.pts_painted[:, 0]
    worse.rms_px = float(np.sqrt(np.mean(dcol_w ** 2 + drow_w ** 2)))
    ok2, _ = pose_correction_verdict(before, worse, h)
    assert ok2 is False


def test_maybe_correct_off_is_identity():
    cfg = Config.from_defaults()
    rec = SimpleNamespace(
        position=np.array([0.0, 0.0, 10.0]),
        orientation=np.array([0.0, 0.0, 0.1]),
        frame_num=0,
    )
    strip = {
        "colors_2d": np.zeros((8, 8, 3), np.uint8),
        "mask_2d": np.ones((8, 8), dtype=bool),
        "z_min_strip": 10.0,
    }
    out = maybe_correct_strip(None, None, strip, rec, None, "U", cfg, 40.0, 1.0, 1.0)
    assert out is strip
    assert rec.position[2] == pytest.approx(10.0)
    assert rec.orientation[2] == pytest.approx(0.1)


def test_maybe_correct_subtracts_z_and_pitch_when_rerender_improves():
    h, w = 180, 100
    ppm = 1.0
    base = _textured(h, w)
    painted = np.zeros((h, w + 40, 3), dtype=np.uint8)
    painted[:, 20:20 + w] = base
    shifted = np.roll(np.roll(base, 7, axis=0), 4, axis=1)
    mask = np.ones((h, w), dtype=bool)
    acc = SimpleNamespace(
        buf=SimpleNamespace(
            color=painted, z_min=0.0, pixels_per_mm=ppm, theta_bins=h,
        ),
        theta_min=0.0,
        theta_max=2.0 * np.pi,
    )
    aligned = {
        "colors_2d": base.copy(),
        "mask_2d": mask,
        "z_min_strip": 20.0,
    }
    mapper = SimpleNamespace()
    calls = []

    def _sample(frame, pos, ori, **kwargs):
        calls.append((float(pos[2]), float(ori[2])))
        return aligned

    mapper.sample_strip = _sample
    cfg = Config.from_defaults()
    cfg.two_direction.strip_overlap_match = True
    cfg.two_direction.strip_overlap_correct = True
    rec = SimpleNamespace(
        position=np.array([0.0, 0.0, 20.0], dtype=float),
        orientation=np.array([0.0, 0.0, 0.20], dtype=float),
        frame_num=3,
    )
    strip = {
        "colors_2d": shifted,
        "mask_2d": mask,
        "z_min_strip": 20.0,
    }
    out = maybe_correct_strip(
        mapper, acc, strip, rec, np.zeros((8, 8, 3), np.uint8),
        "U", cfg, 40.0, 1.0, ppm,
    )
    assert calls, "取り直し sample_strip が呼ばれる"
    assert rec.position[2] < 20.0
    assert rec.orientation[2] < 0.20
    assert out is aligned


def test_correct_requires_match_flag():
    cfg = Config.from_defaults()
    cfg.two_direction.strip_overlap_match = False
    cfg.two_direction.strip_overlap_correct = True
    with pytest.raises(ConfigValidationError, match="strip_overlap_match"):
        cfg.validate_two_direction_config()


def test_residual_bin_keys_are_sideview_only():
    cfg = Config.from_defaults()
    assert cfg.two_direction.residual_theta_bins == 6
    assert cfg.two_direction.residual_min_bin_points == 2
    assert cfg.front.residual_theta_bins == 12
    assert cfg.front.residual_min_bin_points == 2
    cfg.two_direction.residual_theta_bins = 1
    with pytest.raises(ConfigValidationError, match="residual_theta_bins"):
        cfg.validate_two_direction_config()


def test_inherit_running_pose_first_frame_is_analysis():
    z, pitch = inherit_running_pose(100.0, 0.20, None, None, None, None)
    assert z == pytest.approx(100.0)
    assert pitch == pytest.approx(0.20)


def test_inherit_running_pose_dz_only_keeps_corrected_pitch():
    """dz_only は解析 pitch が増えないので、直した pitch が次の初期値。"""
    z, pitch = inherit_running_pose(
        z_ana=104.0, pitch_ana=1.5708,
        z_ana_prev=100.0, pitch_ana_prev=1.5708,
        z_run=99.2, pitch_run=1.5808,
    )
    assert z == pytest.approx(103.2)
    assert pitch == pytest.approx(1.5808)


def test_inherit_running_pose_adds_match_pitch_increment():
    z, pitch = inherit_running_pose(
        z_ana=108.0, pitch_ana=1.5900,
        z_ana_prev=104.0, pitch_ana_prev=1.5750,
        z_run=103.5, pitch_run=1.5820,
    )
    assert z == pytest.approx(107.5)
    assert pitch == pytest.approx(1.5970)
