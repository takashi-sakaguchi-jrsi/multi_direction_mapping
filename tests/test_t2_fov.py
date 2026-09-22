"""T2: usable half FOV"""

from src.validation.geometry import derive_usable_half_fov_deg


def test_usable_half_fov_from_calibration_values():
    fx = 343.9361312858878
    half = derive_usable_half_fov_deg(fx, 1080.0, 0.8)
    assert abs(half - 71.97) < 0.05
    assert abs(2.0 * half - 143.93) < 0.1
