"""WSL など GUI 無し OpenCV でも DebugVisualizer.close が落ちないこと"""

from unittest.mock import patch

import cv2
import numpy as np

from src.debug_visualizer import DebugConfig, DebugVisualizer


def test_close_ignores_headless_destroy_all_windows():
    vis = DebugVisualizer(DebugConfig(enabled=False))
    err = cv2.error("The function is not implemented. cvDestroyAllWindows")
    with patch("src.debug_visualizer.cv2.destroyAllWindows", side_effect=err):
        vis.close()


def test_draw_feature_vectors_paints_arrows():
    vis = DebugVisualizer(DebugConfig(enabled=True, show_feature_points=True))
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    prev = np.array([[10.0, 10.0], [40.0, 40.0]])
    curr = np.array([[20.0, 12.0], [41.0, 40.5]])
    dz = np.array([12.0, 0.2])
    out = vis.draw_feature_vectors(frame, prev, curr, dZ_mm=dz)
    assert out.shape == frame.shape
    assert int(np.count_nonzero(out)) > 0


def test_draw_feature_vectors_skips_when_disabled():
    vis = DebugVisualizer(DebugConfig(enabled=True, show_feature_points=False))
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    prev = np.array([[5.0, 5.0]])
    curr = np.array([[15.0, 8.0]])
    out = vis.draw_feature_vectors(frame, prev, curr)
    assert int(np.count_nonzero(out)) == 0
