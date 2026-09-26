"""テスト共通フィクスチャ"""

import numpy as np
import pytest

from src.config import Config
from src.coordinate_transform import CoordinateTransformer, FisheyeCamera
from src.validation.sideview_projection_mapper import SideviewProjectionMapper


FX = 343.9361312858878
WIDTH = 192
HEIGHT = 108


@pytest.fixture
def config():
    cfg = Config.from_defaults()
    cfg.pipe.diameter_mm = 250.0
    cfg.colormap.outer_radius_ratio = 0.8
    cfg.colormap.pixels_per_mm = 1.0
    cfg.two_direction.capture.image_width_px = WIDTH
    cfg.two_direction.capture.image_height_px = HEIGHT
    cfg.two_direction.capture.usable_outer_radius_ratio = 0.8
    cfg.two_direction.legacy_vp.enabled = False
    cfg.estimation.use_vanishing_point_constraints = False
    cfg.estimation.use_ocr_z_constraints = True
    cfg.estimation.feature_matching.use_cylindrical_matching = False
    return cfg


@pytest.fixture
def transformer(config):
    cam = FisheyeCamera(
        f=FX, cx=WIDTH / 2.0, cy=HEIGHT / 2.0,
        image_width=WIDTH, image_height=HEIGHT,
    )
    return CoordinateTransformer(camera=cam, pipe_radius=config.pipe.diameter_mm / 2.0)


@pytest.fixture
def mapper(config, transformer):
    return SideviewProjectionMapper(config, transformer)
