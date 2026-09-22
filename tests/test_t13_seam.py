"""同一セルは後から来た色だけが残る（座標上書き）"""

import numpy as np

from src.config import Config
from src.validation.best_view_accumulator import BestViewAccumulator


def test_seam_does_not_keep_both_colors(config=None):
    cfg = config or Config.from_defaults()
    acc = BestViewAccumulator(cfg, z_min=0.0, z_max=30.0, theta_bins=16)
    theta = 1.0
    acc.add_projection(
        {
            "z": np.array([5.0]),
            "theta": np.array([theta]),
            "d_wall": np.array([120.0]),
            "colors": np.array([[0, 0, 255]], dtype=np.uint8),
            "u": np.array([2.0]),
            "v": np.array([2.0]),
            "gamma": np.array([0.2]),
            "valid": np.array([True]),
            "frame_num": 0,
        },
        "A",
    )
    acc.add_projection(
        {
            "z": np.array([5.0]),
            "theta": np.array([theta]),
            "d_wall": np.array([140.0]),
            "colors": np.array([[255, 0, 0]], dtype=np.uint8),
            "u": np.array([1.0]),
            "v": np.array([1.0]),
            "gamma": np.array([0.2]),
            "valid": np.array([True]),
            "frame_num": 1,
        },
        "A",
    )
    rgb = acc.colormap_rgb()
    filled_colors = rgb[acc.buf.filled]
    unique = np.unique(filled_colors.reshape(-1, 3), axis=0)
    assert len(unique) == 1
    np.testing.assert_array_equal(unique[0], [255, 0, 0])
