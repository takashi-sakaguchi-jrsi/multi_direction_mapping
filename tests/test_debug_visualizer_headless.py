"""WSL など GUI 無し OpenCV でも DebugVisualizer.close が落ちないこと"""

from unittest.mock import patch

import cv2

from src.debug_visualizer import DebugConfig, DebugVisualizer


def test_close_ignores_headless_destroy_all_windows():
    vis = DebugVisualizer(DebugConfig(enabled=False))
    err = cv2.error("The function is not implemented. cvDestroyAllWindows")
    with patch("src.debug_visualizer.cv2.destroyAllWindows", side_effect=err):
        vis.close()
