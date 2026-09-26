"""2/3方向（真横）動画合成カラーマップ検証エントリ

V0 設定 → V1 各Run解析 → V2 側方投影 → V3 仮部分図（変形なし）
→ V4 登録・共同Z補正 → V6 元フレーム再投影
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.camera_estimation import CameraEstimator
from src.calibration import load_calibration
from src.config import Config, load_config
from src.coordinate_transform import CoordinateTransformer, FisheyeCamera
from src.validation.best_view_accumulator import BestViewAccumulator
from src.validation.frame_analyzer import FrameAnalyzer, resolve_frame_window, zip_records_with_frames
from src.validation.geometry import (
    build_run_reference,
    derive_usable_half_fov_deg,
    infer_suffix_from_path,
    physical_roll_from_suffix,
)
from src.validation.known_z import (
    apply_generation_metadata_to_config,
    load_generation_metadata,
    resolve_known_z_mm,
    source_pixels_per_mm_from_metadata,
)
from src.validation.report import (
    summarize_records,
    write_frame_motion_csv,
    write_validation_report,
)
from src.validation.seam_join import join_at_overlap_centers
from src.validation.seam_warp import normalize_run_id, warp_strips_to_seams
from src.validation.sideview_projection_mapper import SideviewProjectionMapper
from src.validation.strip_correction import correct_strip


logger = logging.getLogger(__name__)


GENERATION_F_PX = 341.87536947032544


def build_transformer(config: Config) -> CoordinateTransformer:
    calib = None
    calib_path = config.camera.lens_calibration_file
    if calib_path:
        try:
            calib = load_calibration(calib_path)
        except Exception as exc:
            logger.warning(f"キャリブレーション読み込み失敗: {exc}")
            calib = None
    else:
        config.camera._lens_calibration = None
    w = config.two_direction.capture.image_width_px or config.camera.image_width
    h = config.two_direction.capture.image_height_px or config.camera.image_height
    if calib is not None and getattr(calib, "fx", 0) > 0:
        fx = float(calib.fx)
        cx = float(getattr(calib, "cx", w / 2.0))
        cy = float(getattr(calib, "cy", h / 2.0))
    else:
        calib = None
        fx = float(config.camera.fx) if config.camera.fx else GENERATION_F_PX
        cx = float(config.camera.cx) if config.camera.cx else w / 2.0
        cy = float(config.camera.cy) if config.camera.cy else h / 2.0
    camera = FisheyeCamera(
        f=fx, cx=cx, cy=cy, image_width=int(w), image_height=int(h)
    )
    radius = config.pipe.diameter_mm / 2.0
    logger.info(
        f"レンズ: calib={'on' if calib is not None else 'off'} "
        f"f={fx:.6f} cx={cx:.3f} cy={cy:.3f} size={int(w)}x{int(h)}"
    )
    return CoordinateTransformer(camera=camera, pipe_radius=radius, calibration=calib)


def _apply_run_cli(run_cfg, path: Path) -> None:
    run_cfg.video_path = str(path)
    suffix = infer_suffix_from_path(str(path))
    if suffix:
        run_cfg.physical_roll_deg = physical_roll_from_suffix(suffix)
        run_cfg.run_id = suffix


def resolve_run_roll(run_cfg) -> float:
    if run_cfg.video_path:
        suffix = infer_suffix_from_path(run_cfg.video_path)
        if suffix and run_cfg.physical_roll_deg in (None,):
            return physical_roll_from_suffix(suffix)
    return float(run_cfg.physical_roll_deg)


def run_two_direction_validation(
    config: Config,
    frames_a: Optional[List[np.ndarray]] = None,
    frames_b: Optional[List[np.ndarray]] = None,
    frames_c: Optional[List[np.ndarray]] = None,
    modes: Optional[List[str]] = None,
    known_z_mm: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    td = config.two_direction
    modes = modes or list(td.modes)
    frames_by_tag = {"A": frames_a, "B": frames_b, "C": frames_c}
    jobs = td.active_run_slots(frames_by_tag)
    if len(jobs) < 2:
        raise ValueError("接合には 2 本以上の run が必要です")
    logger.info(
        "active runs: "
        + ", ".join(
            f"{tag}={run_cfg.run_id or tag}" for tag, run_cfg in jobs
        )
    )
    if known_z_mm is None and str(td.z_source).lower() == "known":
        known_z_mm = resolve_known_z_mm(config)
        if td.known_z_metadata_path:
            meta = load_generation_metadata(td.known_z_metadata_path)
            apply_generation_metadata_to_config(config, meta)
            if td.match_source_pixels_per_mm:
                ppm = source_pixels_per_mm_from_metadata(meta)
                config.colormap.pixels_per_mm = ppm
                logger.info(f"生成カラーマップ pix/mm を元展開図に合わせる: {ppm:.6f}")
        logger.info(
            f"第1段階: 既知zをOCR代用 ({len(known_z_mm)} 点, "
            f"metadata={td.known_z_metadata_path})"
        )
    transformer = build_transformer(config)
    estimator = CameraEstimator(config.estimation, transformer)
    mapper = SideviewProjectionMapper(config, transformer)

    fx = transformer.camera.f
    derived_fov = derive_usable_half_fov_deg(
        fx, td.capture.image_height_px, td.capture.usable_outer_radius_ratio
    )
    logger.info(
        f"usable half FOV={derived_fov:.2f} deg "
        f"(config={td.capture.derived_usable_half_fov_deg}), "
        f"pixels_per_mm={config.colormap.pixels_per_mm:.6f}"
    )
    win_start, win_end = resolve_frame_window(td)
    logger.info(
        f"frame window [{win_start}, {win_end if win_end is not None else 'end'}) "
        f"(start_frame={td.start_frame}, end_frame={td.end_frame}, "
        f"max_frames={td.max_frames})"
    )

    output_dir = Path(td.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "derived_usable_half_fov_deg": derived_fov,
        "z_source": "known" if known_z_mm is not None else "ocr",
        "pixels_per_mm": float(config.colormap.pixels_per_mm),
        "n_runs": len(jobs),
        "run_tags": [tag for tag, _ in jobs],
        "modes": {},
    }

    for mode in modes:
        analyzer = FrameAnalyzer(config, estimator, mode=mode)
        analyzed: List[Dict[str, Any]] = []
        for tag, run_cfg in jobs:
            rec, extra = analyzer.analyze_run(
                run_cfg, frames=frames_by_tag.get(tag), known_z_mm=known_z_mm
            )
            analyzed.append({"tag": tag, "run_cfg": run_cfg, "rec": rec, "extra": extra})

        zs_all = [
            r.position[2]
            for item in analyzed
            for r in item["rec"]
        ]
        z_lo = min(zs_all) - 50.0
        z_hi = max(zs_all) + 50.0
        logger.info(
            f"accumulate z span=[{z_lo:.1f}, {z_hi:.1f}] mm "
            f"(full trip ~{(z_hi - z_lo) * config.colormap.pixels_per_mm:.0f} px; "
            f"canvas grows from a small seed, same as ColorMapGenerator)"
        )
        ppm = config.colormap.pixels_per_mm
        half_fov = np.radians(derived_fov)
        spacing = float(getattr(td.registration, "control_point_spacing_mm", 50.0) or 50.0)
        seam_cfg = td.seam_warp
        apply_edge = bool(getattr(seam_cfg, "apply_edge_theta", False))
        if not bool(getattr(seam_cfg, "enabled", True)):
            apply_edge = True
        mode_dir = output_dir / f"mode_{mode}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        theta_min = 0.0
        theta_max = 2.0 * np.pi
        for item in analyzed:
            rid = _run_id(item["rec"], item["tag"])
            logger.info(f"{rid}: 帯の蓄積を開始")
            item["acc"] = _accumulate_run(
                mapper, item["rec"], item["extra"],
                item["extra"]["run_reference"], config,
                z_min=z_lo, z_max=z_hi,
            )
            item["unfilled"] = item["acc"].unfilled_ratio()
            theta_min = item["acc"].theta_min
            theta_max = item["acc"].theta_max
            _save_rgb_png(mode_dir / f"partial_{rid}.png", item["acc"].colormap_rgb())
            item["corr"] = _correct_accumulated(
                item["acc"], item["rec"], item["extra"],
                ppm, half_fov, spacing, warp_theta=apply_edge,
            )
            item["acc"] = None
            gc.collect()
        seam = warp_strips_to_seams(
            [
                {
                    "rgb": item["corr"].rgb, "filled": item["corr"].filled,
                    "z_min": item["corr"].z_min, "z_max": item["corr"].z_max,
                    "run_id": _run_id(item["rec"], item["tag"]),
                }
                for item in analyzed
            ],
            pixels_per_mm=ppm,
            theta_min=theta_min,
            theta_max=theta_max,
            config=seam_cfg,
        )
        logger.info(seam.message)
        joined = join_at_overlap_centers(
            seam.strips,
            pixels_per_mm=ppm,
        )

        motion_csv: Dict[str, str] = {}
        for item in analyzed:
            rid = _run_id(item["rec"], item["tag"])
            csv_path = write_frame_motion_csv(
                mode_dir / f"motion_{rid}.csv", item["rec"]
            )
            motion_csv[rid] = str(csv_path)
            logger.info(f"移動量CSV: {csv_path}")
        for i, item in enumerate(analyzed):
            rid = _run_id(item["rec"], item["tag"])
            _save_rgb_png(mode_dir / f"partial_{rid}_corrected.png", seam.strips[i]["rgb"])
        _save_rgb_png(mode_dir / "final.png", joined["rgb"])
        if joined["rgb"].size >= 8_000_000:
            np.save(mode_dir / "provenance_final_color.npy", joined["rgb"])
            np.save(mode_dir / "provenance_final_filled.npy", joined["filled"])
            np.save(mode_dir / "provenance_final_source_run.npy", joined["source_run"])
            sidecar = {
                "run_ids": [str(x) for x in np.asarray(joined["run_ids"]).tolist()],
                "z_min": float(joined["z_min"]),
                "z_max": float(joined["z_max"]),
                "cut_camera_car_deg": float(joined.get("cut_camera_car_deg", 0.0)),
                "unwrap_row_shift": int(joined.get("unwrap_row_shift", 0)),
            }
            (mode_dir / "provenance_final.json").write_text(
                json.dumps(sidecar, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            np.savez_compressed(
                mode_dir / "provenance_final.npz",
                color=joined["rgb"],
                filled=joined["filled"],
                source_run=joined["source_run"],
                run_ids=joined["run_ids"],
                z_min=joined["z_min"],
                z_max=joined["z_max"],
                cut_camera_car_deg=joined.get("cut_camera_car_deg", 0.0),
                unwrap_row_shift=joined.get("unwrap_row_shift", 0),
            )

        mode_payload: Dict[str, Any] = {
            "n_runs": len(analyzed),
            "run_ids": [_run_id(item["rec"], item["tag"]) for item in analyzed],
            "seam_warp": seam.to_report(),
            "correction": {
                _run_id(item["rec"], item["tag"]): item["corr"].message
                for item in analyzed
            },
            "coverage": {
                "final_unfilled": float(1.0 - np.mean(joined["filled"])),
                "run_adoption": joined["adoption"],
            },
            "motion_csv": motion_csv,
        }
        mode_payload["correction"]["seam"] = seam.message
        for item in analyzed:
            rid = _run_id(item["rec"], item["tag"])
            mode_payload[f"run_{rid}"] = summarize_records(item["rec"])
            mode_payload["coverage"][f"{rid}_unfilled"] = float(item["unfilled"])
        results["modes"][mode] = mode_payload

    report_path = write_validation_report(output_dir, results)
    results["report_path"] = str(report_path)
    return results


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag)
    crc = zlib.crc32(data, crc) & 0xffffffff
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


# OpenCV/libpng と同じく約 8KiB IDAT。
_PNG_IDAT_SIZE = 8192
# Windows フォトビューアー / WIC の実用上限（GPU テクスチャ 16384）。
_PHOTOVIEWER_MAX_WIDTH = 16384


def _save_rgb_preview(
    path: Path,
    rgb: np.ndarray,
    max_width: int = 4096,
    max_height: int = 800,
) -> None:
    """巨大展開図をビューアで開ける縮小 PNG にする。"""
    arr = np.asarray(rgb)
    h = int(arr.shape[0])
    w = int(arr.shape[1])
    if w <= max_width and h <= max_height:
        _save_rgb_png(path, arr, write_preview=False, write_photoviewer=False)
        return
    step_x = max(1, int(np.ceil(w / float(max_width))))
    step_y = max(1, int(np.ceil(h / float(max_height))))
    _save_rgb_png(
        path, arr[::step_y, ::step_x], write_preview=False, write_photoviewer=False
    )


def _png_filtered_row(row: np.ndarray) -> bytes:
    """OpenCV と同じ Sub フィルタ（type=1）。None(0) より展開が速い。"""
    raw = np.ascontiguousarray(row)
    if raw.ndim == 1:
        flat = raw
        bpp = 1
    else:
        flat = raw.reshape(-1)
        bpp = int(raw.shape[-1])
    if flat.size <= bpp:
        return b"\x00" + flat.tobytes()
    filt = np.empty_like(flat)
    filt[:bpp] = flat[:bpp]
    filt[bpp:] = flat[bpp:] - flat[:-bpp]
    return b"\x01" + filt.tobytes()


def _save_rgb_png_zlib(path: Path, arr: np.ndarray) -> None:
    """OpenCV の巨大 PNG に近い形式（Sub + zlib fastest + 8KiB IDAT）。"""
    color_type = 0 if arr.ndim == 2 else 2
    h, w = int(arr.shape[0]), int(arr.shape[1])
    compressor = zlib.compressobj(level=1)
    pending = bytearray()
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
        f.write(_png_chunk(b"IHDR", ihdr))
        for y in range(h):
            pending.extend(compressor.compress(_png_filtered_row(arr[y])))
            while len(pending) >= _PNG_IDAT_SIZE:
                f.write(_png_chunk(b"IDAT", bytes(pending[:_PNG_IDAT_SIZE])))
                del pending[:_PNG_IDAT_SIZE]
        pending.extend(compressor.flush())
        data = bytes(pending)
        if not data:
            data = zlib.compress(b"", 1)
        off = 0
        while off < len(data):
            nxt = min(len(data), off + _PNG_IDAT_SIZE)
            f.write(_png_chunk(b"IDAT", data[off:nxt]))
            off = nxt
        f.write(_png_chunk(b"IEND", b""))


def _save_photoviewer_png(path: Path, rgb: np.ndarray) -> None:
    """フォトビューア用に z 方向だけ間引く（θ=高さは実寸のまま）。"""
    arr = np.asarray(rgb)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if w <= _PHOTOVIEWER_MAX_WIDTH:
        return
    step = max(1, int(np.ceil(w / float(_PHOTOVIEWER_MAX_WIDTH))))
    small = np.ascontiguousarray(arr[:, ::step])
    _save_rgb_png(path, small, write_preview=False, write_photoviewer=False)
    logger.info(f"フォトビューア用: {path} {small.shape[1]}x{small.shape[0]}")


def _save_rgb_png(
    path: Path,
    rgb: np.ndarray,
    write_preview: bool = True,
    write_photoviewer: bool = True,
) -> None:
    """RGB uint8 を PNG に書く。巨大画像は zlib、縮小コピーを別保存する。"""
    try:
        arr = np.asarray(rgb)
        if arr.ndim != 2:
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            elif arr.shape[-1] != 3:
                raise ValueError("RGB uint8 画像が必要です")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        used_pil = False
        if w <= _PHOTOVIEWER_MAX_WIDTH and h <= _PHOTOVIEWER_MAX_WIDTH:
            try:
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = None
                mode = "L" if arr.ndim == 2 else "RGB"
                Image.fromarray(np.ascontiguousarray(arr), mode=mode).save(
                    path, format="PNG", compress_level=1
                )
                used_pil = True
            except Exception as pil_exc:
                logger.info(f"Pillow PNG を使わず zlib に切替: {pil_exc}")
        if not used_pil:
            _save_rgb_png_zlib(path, arr)
        if write_preview and (w > 4096 or h > 1200):
            preview = path.with_name(f"{path.stem}_preview.png")
            _save_rgb_preview(preview, arr)
            logger.info(f"縮小プレビュー: {preview}")
        if write_photoviewer and w > _PHOTOVIEWER_MAX_WIDTH:
            pv = path.with_name(f"{path.stem}_photoviewer.png")
            _save_photoviewer_png(pv, arr)
    except Exception as exc:
        logger.warning(f"画像保存失敗: {path}: {exc}")


def _run_id(records, fallback: str) -> str:
    raw = str(records[0].run_id or fallback) if records else fallback
    return normalize_run_id(raw)


def _correct_accumulated(acc, records, extra, ppm, half_fov_rad, spacing_mm, warp_theta=True):
    z_est = np.array([r.position[2] for r in records], dtype=float)
    z_ocr = np.asarray(extra.get("ocr_dist"), dtype=float)
    if z_ocr.size != z_est.size:
        z_hp = extra.get("z_positions")
        if z_hp is not None and len(z_hp) == z_est.size and np.isfinite(z_ocr).any():
            z_ocr = float(z_ocr[np.isfinite(z_ocr)][0]) + np.asarray(z_hp, dtype=float)
        else:
            z_ocr = z_est.copy()
    ref = extra["run_reference"]
    ori = np.array([
        ref["roll_ref_rad"], ref["yaw_ref_rad"], ref["pitch_ref_rad"]
    ], dtype=float)
    return correct_strip(
        acc.colormap_rgb(), acc.buf.filled, acc.buf.z_min, ppm,
        z_est, z_ocr, ori, half_fov_rad,
        theta_min=acc.theta_min, theta_max=acc.theta_max,
        control_spacing_mm=spacing_mm,
        warp_theta=warp_theta,
    )


def _accumulate_run(
    mapper, records, extra, reference, config, z_min=None, z_max=None
) -> BestViewAccumulator:
    zs = [r.position[2] for r in records]
    z_min = min(zs) - 50.0
    z_max_init = z_min + 80.0
    ppm = float(config.colormap.pixels_per_mm)
    if z_max is not None and (float(z_max) - float(z_min)) * ppm <= 4096.0:
        z_max_init = float(z_max)
    if (z_max_init - z_min) * ppm > 4096.0:
        z_max_init = z_min + 4096.0 / max(ppm, 1e-9)
    z_max = z_max_init
    acc = BestViewAccumulator(config, z_min=z_min, z_max=z_max)
    run_id = records[0].run_id if records else "A"
    max_dz = float(config.estimation.motion.max_dz)
    alpha = float(config.colormap.alpha)
    n = len(records)
    for i, (rec, frame) in enumerate(zip_records_with_frames(records, extra)):
        if rec.status not in ("OK", "INIT"):
            continue
        strip = mapper.sample_strip(
            frame,
            rec.position,
            rec.orientation,
            theta_bins=acc.buf.theta_bins,
            pixels_per_mm=ppm,
            max_dz=max_dz,
            alpha=alpha,
            theta_min=acc.theta_min,
            theta_max=acc.theta_max,
            run_id=run_id,
            frame_num=rec.frame_num,
        )
        acc.add_strip(
            strip["colors_2d"], strip["mask_2d"], float(strip["z_min_strip"]),
            run_id, rec.frame_num,
        )
        if (i + 1) % 50 == 0 or (i + 1) == n:
            logger.info(f"{run_id}: accumulate {i + 1}/{n}")
    h, w = acc.buf.color.shape[:2]
    logger.info(f"{run_id}: canvas shape=({h}, {w})")
    return acc


def _reproject_run(mapper, records, frames, reference, acc, reg, run_tag):
    run_id = records[0].run_id if records else run_tag
    max_dz = float(acc.config.estimation.motion.max_dz)
    alpha = float(acc.config.colormap.alpha)
    ppm = float(acc.buf.pixels_per_mm)
    for rec, frame in zip(records, frames):
        if rec.status not in ("OK", "INIT"):
            continue
        strip = mapper.sample_strip(
            frame,
            rec.position,
            rec.orientation,
            theta_bins=acc.buf.theta_bins,
            pixels_per_mm=ppm,
            max_dz=max_dz,
            alpha=alpha,
            theta_min=acc.theta_min,
            theta_max=acc.theta_max,
            run_id=run_id,
            frame_num=rec.frame_num,
        )
        z0 = float(reg.apply_z(np.array([float(strip["z_min_strip"])]), run_tag)[0])
        acc.add_strip(
            strip["colors_2d"], strip["mask_2d"], z0, run_id, rec.frame_num,
        )


def _finite(arr):
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    mask = np.isfinite(a)
    if not np.any(mask):
        return None
    return a[mask]


def parse_args():
    p = argparse.ArgumentParser(description="2/3方向動画合成カラーマップ検証")
    p.add_argument("--config", type=Path, default=Path("data/config/two_direction_config.json"))
    p.add_argument("--run-a", type=Path, default=None)
    p.add_argument("--run-b", type=Path, default=None)
    p.add_argument("--run-c", type=Path, default=None, help="第3走行（L）。未指定時は config の run_C")
    p.add_argument("--mode", choices=["A", "C", "AC"], default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--z-source",
        choices=["ocr", "known"],
        default=None,
        help="ocr=Tesseract、known=生成 metadata の z を OCR 代用（第1段階）",
    )
    p.add_argument(
        "--known-z-metadata",
        type=Path,
        default=None,
        help="z_values_mm を含む生成 metadata.json（--z-source known と併用）",
    )
    p.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="処理開始フレーム（0始まり、含む）",
    )
    p.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="処理終了フレーム（含まない。[start, end)）",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="start-frame から使う最大枚数。end-frame より狭いときに効く",
    )
    p.add_argument(
        "--ppm",
        type=float,
        default=None,
        help="生成カラーマップの解像度（pixels/mm）。未指定時は設定値（既定 1.0）",
    )
    p.add_argument(
        "--match-source-ppm",
        action="store_true",
        help="元展開図（生成 metadata）の pix/mm に生成カラーマップを揃える",
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(str(args.config) if args.config else None)
    if args.run_a:
        _apply_run_cli(config.two_direction.run_A, args.run_a)
    if args.run_b:
        _apply_run_cli(config.two_direction.run_B, args.run_b)
    if args.run_c:
        _apply_run_cli(config.two_direction.run_C, args.run_c)
    if args.output_dir:
        config.two_direction.output_dir = str(args.output_dir)
    if args.mode == "A":
        config.two_direction.modes = ["A"]
    elif args.mode == "C":
        config.two_direction.modes = ["C"]
    elif args.mode == "AC":
        config.two_direction.modes = ["A", "C"]
    if args.known_z_metadata:
        config.two_direction.known_z_metadata_path = str(args.known_z_metadata)
        config.two_direction.z_source = "known"
    if args.z_source:
        config.two_direction.z_source = args.z_source
    if args.start_frame is not None:
        config.two_direction.start_frame = int(args.start_frame)
    if args.end_frame is not None:
        config.two_direction.end_frame = int(args.end_frame)
    if args.max_frames is not None:
        config.two_direction.max_frames = int(args.max_frames)
    if args.match_source_ppm:
        config.two_direction.match_source_pixels_per_mm = True
    if args.ppm is not None:
        config.colormap.pixels_per_mm = float(args.ppm)
        config.two_direction.match_source_pixels_per_mm = False
    if args.debug:
        config.debug.enabled = True
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    try:
        result = run_two_direction_validation(config)
        print(f"report: {result.get('report_path')}")
        return 0
    except FileNotFoundError as exc:
        logger.error(str(exc))
        logger.error("動画未配置です。data/input/videos/README.md を参照してください。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
