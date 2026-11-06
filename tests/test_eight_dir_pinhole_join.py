"""工程6: ピンホール 8 方向の仮想動画による接合の基本確認"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Config
from src.main_two_direction_validation import run_two_direction_validation
from src.product_runtime import apply_sideview_named_input, resolve_named_run_videos
from src.validation.geometry import build_direction_layout
from src.validation.known_z import infer_known_z_metadata_path

PINHOLE_DIR = Path("data/input/videos")
PINHOLE_NAME = "phi250_pinhole_side"
PINHOLE_IDS = ("000", "045", "090", "135", "180", "225", "270", "315")


def _pinhole_videos():
    return [PINHOLE_DIR / f"{PINHOLE_NAME}_{rid}.mp4" for rid in PINHOLE_IDS]


def _pinhole_ready() -> bool:
    if not all(p.is_file() for p in _pinhole_videos()):
        return False
    for vid in _pinhole_videos():
        if infer_known_z_metadata_path(str(vid)) is None:
            return False
    return True


@pytest.mark.skipif(not _pinhole_ready(), reason="phi250_pinhole_side の 8 本または metadata が無い")
def test_pinhole_eight_videos_resolve_and_metadata():
    found = resolve_named_run_videos(PINHOLE_DIR, PINHOLE_NAME)
    assert set(found) == set(PINHOLE_IDS)
    for rid, path in found.items():
        meta = infer_known_z_metadata_path(str(path))
        assert meta is not None
        assert path.stem in meta.stem or meta.stem.endswith("pinhole_side")


@pytest.mark.skipif(not _pinhole_ready(), reason="phi250_pinhole_side の 8 本または metadata が無い")
def test_pinhole_eight_dir_short_known_z(tmp_path):
    cfg = Config.from_json("data/config/two_direction_config.json")
    cfg.two_direction.z_source = "known"
    cfg.two_direction.known_z_metadata_path = None
    cfg.known_z_metadata_path = None
    cfg.two_direction.modes = ["A"]
    cfg.two_direction.max_frames = 4
    cfg.two_direction.output_dir = str(tmp_path / "two_direction")
    cfg.two_direction.n_directions = 8
    cfg.two_direction.optical_axis_spacing_deg = 45.0
    cfg.two_direction.fov_deg = 60.0
    cfg.two_direction.match_source_pixels_per_mm = False
    cfg.two_direction.projection.sample_stride = 6
    cfg.two_direction.registration.min_inliers = 3
    cfg.estimation.use_offset_moving_average = False
    cfg.estimation.use_vanishing_point_constraints = False
    apply_sideview_named_input(cfg, str(PINHOLE_DIR), PINHOLE_NAME)
    assert [r.run_id for r in cfg.two_direction.runs] == list(PINHOLE_IDS)

    with patch("src.ocr_utils.extract_distance_from_frame") as ocr:
        result = run_two_direction_validation(cfg, modes=["A"])
        ocr.assert_not_called()

    assert result["z_source"] == "known"
    assert result["n_runs"] == 8
    mode = result["modes"]["A"]
    assert mode["run_ids"] == list(PINHOLE_IDS)
    layout = build_direction_layout(8, 45.0, 60.0)
    assert set(mode["run_ids"]) == set(layout.run_ids)
    out = Path(cfg.two_direction.output_dir) / "mode_A"
    assert (out / "final.png").is_file()
    for rid in PINHOLE_IDS:
        assert (out / f"partial_{rid}.png").is_file()
        assert f"run_{rid}" in mode
        assert mode[f"run_{rid}"]["n_frames"] == 4
    pairs = {tuple(name.split("_")) for name in mode["seam_warp"]["seams"]}
    assert ("000", "045") in pairs
    assert ("315", "000") in pairs
    assert ("000", "090") not in pairs
