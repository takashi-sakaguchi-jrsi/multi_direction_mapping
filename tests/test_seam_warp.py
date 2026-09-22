"""接合面基準の 2 次元 1/2 相互補正"""

import numpy as np
import pytest

from src.config import SeamWarpConfig
from src.validation.seam_warp import (
    SeamMismatch,
    _seam_row,
    boundary_displacement,
    eval_mismatch,
    sector_blend_weight,
    warp_strips_to_seams,
)


def _pattern(h, w, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = ((xx * 7 + yy * 3) % 220 + 20).astype(np.uint8)
    img[..., 1] = ((xx * 5 + 40) % 220 + 20).astype(np.uint8)
    img[..., 2] = ((yy * 11 + xx) % 220 + 20).astype(np.uint8)
    img = np.bitwise_xor(img, rng.integers(0, 18, img.shape, dtype=np.uint8))
    return img


def test_sector_blend_weight_u_edges_and_center():
    w_lo = float(sector_blend_weight(-60.0, "U"))
    w_mid = float(sector_blend_weight(0.0, "U"))
    w_hi = float(sector_blend_weight(60.0, "U"))
    np.testing.assert_allclose(w_lo, 0.0, atol=1e-6)
    np.testing.assert_allclose(w_mid, 0.5, atol=1e-6)
    np.testing.assert_allclose(w_hi, 1.0, atol=1e-6)
    # 帯の直前は 0（360° 回り込みしない）
    assert float(sector_blend_weight(50.0, "R")) == pytest.approx(0.0, abs=1e-6)


def test_three_way_half_mismatch_interpolates_on_u():
    z = np.array([0.0, 80.0, 160.0])
    ur = SeamMismatch(("U", "R"), z, np.full(3, 4.0), np.zeros(3), 12, 12, "ok")
    ul = SeamMismatch(("U", "L"), z, np.full(3, -2.0), np.zeros(3), 12, 12, "ok")
    ulo_z, _, uhi_z, _ = boundary_displacement("U", z, [ur, ul])
    # L 側: m=-2 → U は -m/2 = +1。R 側: m=4 → U は -2
    np.testing.assert_allclose(ulo_z, 1.0)
    np.testing.assert_allclose(uhi_z, -2.0)
    wt = float(sector_blend_weight(0.0, "U"))
    u_mid = (1.0 - wt) * ulo_z[1] + wt * uhi_z[1]
    np.testing.assert_allclose(u_mid, -0.5)


def test_mismatch_at_ends_is_not_pinned_to_zero():
    z = np.array([0.0, 50.0, 100.0])
    mm = SeamMismatch(
        ("U", "R"), z, np.array([6.0, 4.0, 2.0]), np.zeros(3), 9, 9, "ok"
    )
    mz, _ = eval_mismatch(mm, np.array([0.0, 100.0]))
    np.testing.assert_allclose(mz, [6.0, 2.0])
    _ulo, _, uhi, _ = boundary_displacement("U", np.array([0.0, 100.0]), [mm])
    np.testing.assert_allclose(uhi, [-3.0, -1.0])


def test_ur_shift_meets_at_seam_and_free_edge_stays():
    h, w = 180, 200
    ppm = 1.0
    base = _pattern(h, w)
    dx, dy = 8, 2
    rgb_u = base.copy()
    rgb_r = np.roll(np.roll(base, dx, axis=1), dy, axis=0)
    filled = np.ones((h, w), dtype=bool)
    cfg = SeamWarpConfig(
        enabled=True,
        half_band_deg=10.0,
        control_spacing_mm=20.0,
        max_dz_mm=12.0,
        max_dtheta_deg=6.0,
        min_matches=4,
        ncc_window_mm=32.0,
        ncc_step_mm=10.0,
    )
    result = warp_strips_to_seams(
        [
            {"rgb": rgb_u, "filled": filled, "z_min": 0.0, "z_max": float(w), "run_id": "U"},
            {"rgb": rgb_r, "filled": filled, "z_min": 0.0, "z_max": float(w), "run_id": "R"},
        ],
        pixels_per_mm=ppm,
        config=cfg,
    )
    assert result.success
    assert result.mismatches and result.mismatches[0].n_used >= 4
    u_out = result.strips[0]["rgb"]
    r_out = result.strips[1]["rgb"]
    seam_row = int(round(_seam_row(h, 60.0, 0.0, 2.0 * np.pi)))
    free_row = int(round(_seam_row(h, -60.0, 0.0, 2.0 * np.pi)))
    band = 6
    y0, y1 = max(0, seam_row - band), min(h, seam_row + band + 1)
    a = u_out[y0:y1, 40:160, 1].astype(np.float32)
    b = r_out[y0:y1, 40:160, 1].astype(np.float32)
    (shift_x, shift_y), _ = __import__("cv2").phaseCorrelate(a, b)
    # 接合線では半分ずつ戻すので残差は元の 8px / 2px より小さい
    assert abs(shift_x) < abs(dx) - 2
    assert abs(shift_y) < abs(dy) + 1.5

    yf0, yf1 = max(0, free_row - band), min(h, free_row + band + 1)
    orig = rgb_u[yf0:yf1, 40:160, 1].astype(np.float32)
    warped = u_out[yf0:yf1, 40:160, 1].astype(np.float32)
    (free_x, _free_y), _ = __import__("cv2").phaseCorrelate(orig, warped)
    # L 側（相手なし）は u=0 なのでほとんど動かない
    assert abs(free_x) < 2.0
