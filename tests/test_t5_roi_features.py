"""T5: レンズ中央矩形での抽出と逆走ベクトル除外"""

import numpy as np

from src.config import Config
from src.validation.sideview_feature_matcher import SideviewFeatureMatcher


def test_roi_mask_excludes_outside_points():
    cfg = Config.from_defaults()
    matcher = SideviewFeatureMatcher(cfg)
    mask = matcher.build_roi_mask((100, 120), [10, 20, 50, 70], None)
    assert mask[30, 30] == 255
    assert mask[0, 0] == 0
    assert mask[90, 110] == 0


def test_default_mask_is_center_rect_not_full_frame():
    cfg = Config.from_defaults()
    matcher = SideviewFeatureMatcher(cfg)
    mask = matcher.build_roi_mask((1080, 1920), None, None, center=(960.0, 540.0))
    assert mask[540, 960] == 255
    assert mask[0, 0] == 0
    assert mask[1079, 1919] == 0
    rect = matcher.center_rect((1080, 1920), (960.0, 540.0))
    half = 0.4 * (1080 / 2.0)
    assert abs((rect[2] - rect[0]) / 2.0 - half) < 2.0
    assert abs((rect[3] - rect[1]) / 2.0 - half) < 2.0


def test_trim_circle_applied_when_requested():
    cfg = Config.from_defaults()
    matcher = SideviewFeatureMatcher(cfg)
    mask = matcher.build_roi_mask(
        (100, 100), [0, 0, 100, 100], None, center=(50, 50), max_radius_px=20
    )
    assert mask[50, 50] == 255
    assert mask[0, 0] == 0


def test_downward_vector_is_rejected_as_reverse():
    cfg = Config.from_defaults()
    matcher = SideviewFeatureMatcher(cfg)
    pts1 = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    pts2 = np.array([[10.0, 18.0], [20.0, 12.0]], dtype=np.float32)
    a, b = matcher.filter_motion_vectors(pts1, pts2)
    assert len(a) == 1
    np.testing.assert_allclose(a[0], [20.0, 20.0])


def test_downward_vector_is_kept_when_forward_dy_is_positive():
    cfg = Config.from_defaults()
    matcher = SideviewFeatureMatcher(cfg)
    pts1 = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    pts2 = np.array([[10.0, 18.0], [20.0, 12.0]], dtype=np.float32)
    a, b = matcher.filter_motion_vectors(pts1, pts2, forward_dy_sign=1)
    assert len(a) == 1
    np.testing.assert_allclose(a[0], [10.0, 10.0])


def test_upward_vector_is_kept():
    cfg = Config.from_defaults()
    matcher = SideviewFeatureMatcher(cfg)
    pts1 = np.array([[40.0, 50.0]], dtype=np.float32)
    pts2 = np.array([[40.0, 40.0]], dtype=np.float32)
    a, b = matcher.filter_motion_vectors(pts1, pts2)
    assert len(a) == 1


def _cluster_up(n=12, length=10.0, x0=20.0):
    pts1 = np.column_stack([
        np.linspace(x0, x0 + 40.0, n),
        np.full(n, 80.0),
    ]).astype(np.float32)
    pts2 = pts1.copy()
    pts2[:, 1] -= length
    return pts1, pts2


def test_length_outlier_is_removed():
    cfg = Config.from_defaults()
    matcher = SideviewFeatureMatcher(cfg)
    pts1, pts2 = _cluster_up(12, 10.0)
    pts2 = pts2.copy()
    pts2[0, 1] = pts1[0, 1] - 40.0
    a, b = matcher.filter_motion_vectors(pts1, pts2)
    dlen = np.linalg.norm(b - a, axis=1)
    assert len(a) == 11
    assert np.all(dlen < 15.0)


def test_direction_outlier_is_removed():
    cfg = Config.from_defaults()
    matcher = SideviewFeatureMatcher(cfg)
    pts1, pts2 = _cluster_up(12, 10.0)
    pts2 = pts2.copy()
    pts2[-1, 0] = pts1[-1, 0] + 10.0
    pts2[-1, 1] = pts1[-1, 1]
    a, b = matcher.filter_motion_vectors(pts1, pts2)
    d = b - a
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    assert len(a) == 11
    assert np.all(ang < -60.0)
