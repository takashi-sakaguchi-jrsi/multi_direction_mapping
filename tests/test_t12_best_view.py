"""座標上書き蓄積（後から来た帯が残る）"""

import numpy as np

from src.validation.best_view_accumulator import BestViewAccumulator


def test_later_pixel_overwrites(config):
    acc = BestViewAccumulator(config, z_min=0.0, z_max=50.0, theta_bins=32)

    def proj(z, theta, d, color, frame):
        return {
            "z": np.array([z]),
            "theta": np.array([theta]),
            "d_wall": np.array([d]),
            "colors": np.array([color], dtype=np.uint8),
            "u": np.array([10.0]),
            "v": np.array([20.0]),
            "gamma": np.array([0.1]),
            "valid": np.array([True]),
            "frame_num": frame,
        }

    theta = np.pi / 2
    acc.add_projection(proj(10.0, theta, 110.0, [0, 255, 0], 1), "A")
    acc.add_projection(proj(10.0, theta, 130.0, [255, 0, 0], 2), "B")
    rgb = acc.colormap_rgb()
    filled = acc.buf.filled
    assert np.any(filled)
    # 後から書いた赤が残る（d_wall が長くても上書き）
    assert np.any(np.all(rgb == np.array([255, 0, 0]), axis=2))
    assert not np.any(np.all(rgb == np.array([0, 255, 0]), axis=2))
    assert acc.unfilled_ratio() > 0.0


def test_strip_overwrites_by_coordinate(config):
    acc = BestViewAccumulator(config, z_min=0.0, z_max=40.0, theta_bins=16)
    h, w = 16, 10
    first = np.zeros((h, w, 3), dtype=np.uint8)
    first[:] = (0, 255, 0)
    mask = np.ones((h, w), dtype=bool)
    acc.add_strip(first, mask, z_min_mm=0.0, run_id="A", frame_num=0)
    second = np.zeros((h, w, 3), dtype=np.uint8)
    second[:] = (255, 0, 0)
    acc.add_strip(second, mask, z_min_mm=0.0, run_id="A", frame_num=1)
    rgb = acc.colormap_rgb()
    filled = acc.buf.filled[:, :w]
    assert np.all(rgb[:, :w][filled] == np.array([255, 0, 0]))
