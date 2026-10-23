"""塗済み画素を後フレームで上書きしない（後戻りスクラッチ防止）"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.colormap_generator import ColorMapGenerator
from src.config import Config
from src.product_runtime import apply_front_profile


def _generator(keep: bool) -> ColorMapGenerator:
    cfg = Config()
    cfg.pipe.diameter_mm = 10.0
    cfg.colormap.pixels_per_mm = 1.0
    cfg.colormap.keep_painted_pixels = keep
    return ColorMapGenerator(cfg, MagicMock())


def _patch(h: int, w: int, rgb) -> tuple:
    colors = np.zeros((h, w, 3), dtype=np.uint8)
    colors[:] = np.asarray(rgb, dtype=np.uint8)
    mask = np.ones((h, w), dtype=bool)
    return colors, mask


def test_overwrite_replaces_painted_pixels():
    gen = _generator(keep=False)
    red, mask = _patch(8, 4, (200, 10, 10))
    img = gen.update_colormap(None, red, mask, start_col=0)
    blue, mask = _patch(8, 4, (10, 10, 200))
    img = gen.update_colormap(img, blue, mask, start_col=0)
    assert np.all(img[:8, :4] == (10, 10, 200))


def test_keep_preserves_first_write_and_fills_empty():
    gen = _generator(keep=True)
    red, mask = _patch(8, 4, (200, 10, 10))
    img = gen.update_colormap(None, red, mask, start_col=0)
    blue, mask = _patch(8, 6, (10, 10, 200))
    img = gen.update_colormap(img, blue, mask, start_col=0)
    assert np.all(img[:8, :4] == (200, 10, 10))
    assert np.all(img[:8, 4:6] == (10, 10, 200))


def test_keep_blend_mode_alias_sets_flag():
    cfg = Config.from_dict({"colormap": {"blend_mode": "keep"}})
    gen = ColorMapGenerator(cfg, MagicMock())
    assert gen.keep_painted_pixels is True


def test_front_profile_enables_prefer_peripheral_not_keep():
    config = Config()
    assert config.colormap.keep_painted_pixels is False
    assert config.colormap.prefer_peripheral_pixels is False
    assert config.colormap.reject_z_behind_camera is False
    apply_front_profile(config)
    assert config.colormap.keep_painted_pixels is False
    assert config.colormap.prefer_peripheral_pixels is True
    assert config.colormap.reject_z_behind_camera is True


def test_front_config_can_disable_keep():
    cfg = Config.from_dict({"front": {"keep_painted_colormap": False}})
    assert cfg.front.keep_painted_colormap is False
    apply_front_profile(cfg)
    assert cfg.colormap.keep_painted_pixels is False
    assert Config().front.keep_painted_colormap is False


def test_prefer_peripheral_outer_replaces_inner():
    cfg = Config()
    cfg.pipe.diameter_mm = 10.0
    cfg.colormap.pixels_per_mm = 1.0
    cfg.colormap.prefer_peripheral_pixels = True
    gen = ColorMapGenerator(cfg, MagicMock())
    inner, mask = _patch(8, 4, (200, 10, 10))
    r_inner = np.full((8, 4), 20.0, dtype=np.float32)
    img = gen.update_colormap(None, inner, mask, start_col=0, sample_radius=r_inner)
    outer, mask = _patch(8, 4, (10, 10, 200))
    r_outer = np.full((8, 4), 40.0, dtype=np.float32)
    img = gen.update_colormap(img, outer, mask, start_col=0, sample_radius=r_outer)
    assert np.all(img[:8, :4] == (10, 10, 200))


def test_prefer_peripheral_inner_does_not_replace_outer():
    cfg = Config()
    cfg.pipe.diameter_mm = 10.0
    cfg.colormap.pixels_per_mm = 1.0
    cfg.colormap.prefer_peripheral_pixels = True
    gen = ColorMapGenerator(cfg, MagicMock())
    outer, mask = _patch(8, 4, (10, 200, 10))
    r_outer = np.full((8, 4), 40.0, dtype=np.float32)
    img = gen.update_colormap(None, outer, mask, start_col=0, sample_radius=r_outer)
    inner, mask = _patch(8, 6, (200, 10, 10))
    r_inner = np.full((8, 6), 20.0, dtype=np.float32)
    img = gen.update_colormap(img, inner, mask, start_col=0, sample_radius=r_inner)
    assert np.all(img[:8, :4] == (10, 200, 10))
    assert np.all(img[:8, 4:6] == (200, 10, 10))


def test_reject_z_behind_camera_masks_back_wall():
    cfg = Config()
    cfg.pipe.diameter_mm = 10.0
    cfg.colormap.pixels_per_mm = 1.0
    cfg.colormap.inner_radius_ratio = 0.4
    cfg.colormap.outer_radius_ratio = 0.8
    cfg.colormap.reject_z_behind_camera = True
    xf = MagicMock()

    def _w2p(pts, *args, **kwargs):
        n = len(pts)
        return np.repeat(np.array([[74.0, 50.0]]), n, axis=0)

    xf.world_to_pixel.side_effect = _w2p
    xf.distort_ideal_pixels.side_effect = lambda p: p
    xf.kannala_rising_branch_mask.side_effect = lambda p: np.ones(len(p), dtype=bool)
    gen = ColorMapGenerator(cfg, xf)
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    eta = np.zeros((3, 5), dtype=float)
    z = np.tile(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), (3, 1))
    camera_state = {
        "position": np.array([0.0, 0.0, 35.0]),
        "orientation": [0.0, 0.0, 0.0],
    }
    camera_params = {"center": (50.0, 50.0), "radius": 40.0}
    _colors, mask, _radius = gen.sample_colors_from_frame(
        frame, eta, z, camera_state, camera_params
    )
    assert not np.any(mask[:, :3])
    assert np.all(mask[:, 3:])


def test_front_config_can_disable_reject_z_behind_camera():
    cfg = Config.from_dict({"front": {"reject_z_behind_camera": False}})
    apply_front_profile(cfg)
    assert cfg.colormap.reject_z_behind_camera is False
    assert Config().front.reject_z_behind_camera is True
