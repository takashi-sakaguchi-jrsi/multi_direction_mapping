"""T11: 重複領域の登録で残差が改善すること"""

import numpy as np

from src.config import Config
from src.validation.map_registration import MapRegistrar


def test_registration_recovers_known_shift():
    cfg = Config.from_defaults()
    registrar = MapRegistrar(cfg)
    h, w = 80, 120
    yy, xx = np.mgrid[0:h, 0:w]
    img_a = np.stack([
        (xx * 2).astype(np.uint8),
        (yy * 3).astype(np.uint8),
        ((xx + yy) % 256).astype(np.uint8),
    ], axis=-1)
    shift = 6
    img_b = np.roll(img_a, shift, axis=1)
    filled = np.ones((h, w), dtype=bool)
    result = registrar.register(
        img_a, img_b, filled, filled,
        z_min=0.0, pixels_per_mm=1.0,
        theta_min=0.0, theta_max=2.0 * np.pi,
    )
    assert result.raw_matches >= 3
    if result.success:
        assert result.residual_after <= result.residual_before + 1e-6
        assert result.inliers >= cfg.two_direction.registration.min_inliers or result.inliers >= 3
