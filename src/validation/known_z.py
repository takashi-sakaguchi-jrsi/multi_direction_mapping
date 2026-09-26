"""第1段階: 生成 metadata の既知 z を OCR 読み取りの代用にする"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from src.validation.fisheye_sideview_renderer import pixels_per_mm_from_equirect
from src.validation.ocr_simulation import ocr_z_from_metadata


def infer_known_z_metadata_path(video_path: str) -> Optional[Path]:
    """`<name>_U.mp4` から `<name>_metadata.json` を探す"""
    path = Path(video_path)
    stem = path.stem
    candidates = []
    for suffix in ("_U", "_R", "_L"):
        if stem.endswith(suffix):
            candidates.append(path.with_name(stem[: -len(suffix)] + "_metadata.json"))
            break
    candidates.append(path.with_name(stem + "_metadata.json"))
    candidates.append(path.with_name(path.name + "_metadata.json"))
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def png_hw(path: Path) -> tuple:
    """PNG IHDR から (width, height) を読む。巨大画像でも全体は読まない。"""
    with path.open("rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"PNG ではありません: {path}")
        f.read(4)
        if f.read(4) != b"IHDR":
            raise ValueError(f"IHDR がありません: {path}")
        width = int.from_bytes(f.read(4), "big")
        height = int.from_bytes(f.read(4), "big")
    return width, height


def source_pixels_per_mm_from_metadata(meta: Dict[str, Any]) -> float:
    """生成 metadata または元展開図サイズから pix/mm を得る。"""
    if meta.get("source_pixels_per_mm"):
        return float(meta["source_pixels_per_mm"])
    radius = float(meta.get("radius_mm") or 0.0)
    if radius <= 0.0 and meta.get("pipe_diameter_mm"):
        radius = float(meta["pipe_diameter_mm"]) / 2.0
    height = meta.get("colormap_height_px")
    if height:
        return pixels_per_mm_from_equirect(int(height), radius)
    cmap = meta.get("colormap")
    if cmap:
        path = Path(str(cmap))
        if path.is_file():
            _w, h = png_hw(path)
            if radius <= 0.0:
                raise ValueError("radius_mm が metadata にありません")
            return pixels_per_mm_from_equirect(h, radius)
    raise ValueError("元展開図の pix/mm を求められません")


def load_generation_metadata(metadata_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"既知z metadata が見つかりません: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def known_z_mm_from_metadata(meta: Dict[str, Any]) -> np.ndarray:
    """OCR 代用値。ocr_z_mm（10mm遅れ）があればそれを使い、なければ真値 z。"""
    z = ocr_z_from_metadata(meta, fallback_true_z=True)
    if z.size == 0:
        raise ValueError("OCR/z 列が空です")
    return z


def true_z_mm_from_metadata(meta: Dict[str, Any]) -> np.ndarray:
    if "z_values_mm" not in meta:
        raise ValueError("metadata に z_values_mm がありません")
    z = np.asarray(meta["z_values_mm"], dtype=float).reshape(-1)
    if z.size == 0:
        raise ValueError("z_values_mm が空です")
    return z


def load_known_z_mm(metadata_path: Union[str, Path]) -> np.ndarray:
    return known_z_mm_from_metadata(load_generation_metadata(metadata_path))


def apply_generation_metadata_to_config(config, meta: Dict[str, Any]) -> None:
    """生成時の fx / 画角 / 管径を Config に反映する。

    仮想動画は歪みなし等距離魚眼（主点=画像中心、k=0）で作っているため、
    本番キャリブレーション（Kannala-Brandt + 主点オフセット）は外す。
    """
    config.camera.lens_calibration_file = None
    config.camera._lens_calibration = None
    config.camera.center_offset_x = 0
    config.camera.center_offset_y = 0
    if "f_px" in meta and meta["f_px"]:
        fx = float(meta["f_px"])
        config.camera.fx = fx
        config.camera.fy = fx
    if "fov_deg" in meta and meta["fov_deg"]:
        config.camera.fov_degrees = float(meta["fov_deg"])
    if "width" in meta and meta["width"]:
        w = int(meta["width"])
        config.camera.image_width = w
        config.camera.cx = w / 2.0
        config.two_direction.capture.image_width_px = w
    if "height" in meta and meta["height"]:
        h = int(meta["height"])
        config.camera.image_height = h
        config.camera.cy = h / 2.0
        config.two_direction.capture.image_height_px = h
    if "pipe_diameter_mm" in meta and meta["pipe_diameter_mm"]:
        config.pipe.diameter_mm = float(meta["pipe_diameter_mm"])
    elif "radius_mm" in meta and meta["radius_mm"]:
        config.pipe.diameter_mm = float(meta["radius_mm"]) * 2.0


def resolve_known_z_mm(config) -> Optional[np.ndarray]:
    """z_source=known のとき metadata から既知 z を読む。ocr なら None。"""
    td = config.two_direction
    if str(td.z_source).lower() != "known":
        return None
    path = td.known_z_metadata_path
    if path:
        return load_known_z_mm(path)
    for _tag, run in td.active_run_slots():
        if run.video_path:
            inferred = infer_known_z_metadata_path(run.video_path)
            if inferred is not None:
                td.known_z_metadata_path = str(inferred)
                return load_known_z_mm(inferred)
    raise FileNotFoundError(
        "z_source=known ですが known_z_metadata_path も "
        "動画横の *_metadata.json も見つかりません"
    )
