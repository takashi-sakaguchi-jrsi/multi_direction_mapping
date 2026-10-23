"""既描画との重ね合わせ並進推定"""

import numpy as np
import pytest

from src.config import Config, FrontConfig
from src.front_overlap_align import (
    apply_overlap_shift_to_grids,
    estimate_overlap_shift_px,
)
from src.product_runtime import apply_front_profile


def _pattern(h: int, w: int, row0: int = 0, col0: int = 0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    val = 128.0 + 70.0 * np.sin((yy + row0) / 4.5) + 35.0 * np.cos((xx + col0) / 6.0)
    rgb = np.clip(val, 0, 255).astype(np.uint8)
    return np.stack([rgb, rgb, rgb], axis=-1)


def test_overlap_shift_recovers_eta_and_z_offset():
    h, w = 160, 80
    base = _pattern(h, w)
    painted = np.zeros((h, w + 10, 3), dtype=np.uint8)
    painted[:, 10:10 + w] = base
    new = np.roll(np.roll(base, 6, axis=0), -3, axis=1)
    mask = np.ones((h, w), dtype=bool)
    out = estimate_overlap_shift_px(
        painted, new, mask, start_col=10, min_pixels=200, min_response=0.02
    )
    assert out is not None
    d_row, d_col, resp = out
    assert resp > 0.02
    assert d_row == pytest.approx(6.0, abs=1.0)
    assert d_col == pytest.approx(-3.0, abs=1.0)


def test_overlap_shift_skips_tiny_overlap():
    painted = np.zeros((32, 32, 3), dtype=np.uint8)
    painted[:4, :4] = 200
    new = np.full((32, 16, 3), 180, dtype=np.uint8)
    mask = np.zeros((32, 16), dtype=bool)
    mask[:2, :2] = True
    assert estimate_overlap_shift_px(painted, new, mask, start_col=0) is None


def test_apply_overlap_shift_to_grids_units():
    h = 360
    eta = np.zeros((h, 4))
    z = np.zeros((h, 4))
    eta_s, z_s = apply_overlap_shift_to_grids(eta, z, d_row=h / 360.0, d_col=6.0, eta_height=h, pixels_per_mm=3.0)
    assert float(np.median(eta_s)) == pytest.approx(np.radians(1.0), abs=1e-6)
    assert float(np.median(z_s)) == pytest.approx(2.0, abs=1e-6)


def test_front_overlap_align_default_enabled():
    assert FrontConfig().overlap_paste_align is True


def test_colormap_generator_overlap_only_on_front_pipeline():
    from unittest.mock import MagicMock

    from src.colormap_generator import ColorMapGenerator

    cfg_sv = Config()
    cfg_sv.pipeline = "sideview"
    gen_sv = ColorMapGenerator(cfg_sv, MagicMock())
    assert gen_sv.overlap_paste_align is False

    cfg_f = Config()
    cfg_f.pipeline = "front"
    apply_front_profile(cfg_f)
    gen_f = ColorMapGenerator(cfg_f, MagicMock())
    assert gen_f.overlap_paste_align is True
