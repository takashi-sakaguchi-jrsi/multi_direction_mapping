"""2方向（真横）動画合成カラーマップ検証エントリ

V0 設定 → V1 各Run解析 → V2 側方投影 → V3 仮部分図（変形なし）
→ V4 登録・共同Z補正 → V6 元フレーム再投影
"""

from __future__ import annotations

import argparse
import logging
import sys
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
from src.validation.frame_analyzer import FrameAnalyzer, resolve_frame_window
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
from src.validation.seam_warp import warp_strips_to_seams
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
    modes: Optional[List[str]] = None,
    known_z_mm: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    td = config.two_direction
    modes = modes or list(td.modes)
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
        "modes": {},
    }

    for mode in modes:
        analyzer = FrameAnalyzer(config, estimator, mode=mode)
        rec_a, extra_a = analyzer.analyze_run(
            td.run_A, frames=frames_a, known_z_mm=known_z_mm
        )
        rec_b, extra_b = analyzer.analyze_run(
            td.run_B, frames=frames_b, known_z_mm=known_z_mm
        )
        frames_a_use = extra_a["frames"]
        frames_b_use = extra_b["frames"]

        zs_all = [r.position[2] for r in rec_a] + [r.position[2] for r in rec_b]
        z_lo = min(zs_all) - 50.0
        z_hi = max(zs_all) + 50.0
        acc_a = _accumulate_run(
            mapper, rec_a, frames_a_use, extra_a["run_reference"], config,
            z_min=z_lo, z_max=z_hi,
        )
        acc_b = _accumulate_run(
            mapper, rec_b, frames_b_use, extra_b["run_reference"], config,
            z_min=z_lo, z_max=z_hi,
        )

        ppm = config.colormap.pixels_per_mm
        half_fov = np.radians(derived_fov)
        spacing = float(getattr(td.registration, "control_point_spacing_mm", 50.0) or 50.0)
        seam_cfg = td.seam_warp
        apply_edge = bool(getattr(seam_cfg, "apply_edge_theta", False))
        if not bool(getattr(seam_cfg, "enabled", True)):
            apply_edge = True
        corr_a = _correct_accumulated(
            acc_a, rec_a, extra_a, ppm, half_fov, spacing, warp_theta=apply_edge
        )
        corr_b = _correct_accumulated(
            acc_b, rec_b, extra_b, ppm, half_fov, spacing, warp_theta=apply_edge
        )
        seam = warp_strips_to_seams(
            [
                {
                    "rgb": corr_a.rgb, "filled": corr_a.filled,
                    "z_min": corr_a.z_min, "z_max": corr_a.z_max,
                    "run_id": rec_a[0].run_id if rec_a else "U",
                },
                {
                    "rgb": corr_b.rgb, "filled": corr_b.filled,
                    "z_min": corr_b.z_min, "z_max": corr_b.z_max,
                    "run_id": rec_b[0].run_id if rec_b else "R",
                },
            ],
            pixels_per_mm=ppm,
            theta_min=acc_a.theta_min,
            theta_max=acc_a.theta_max,
            config=seam_cfg,
        )
        logger.info(seam.message)
        joined = join_at_overlap_centers(
            seam.strips,
            pixels_per_mm=ppm,
        )

        mode_dir = output_dir / f"mode_{mode}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        csv_a = write_frame_motion_csv(
            mode_dir / f"motion_{_run_id(rec_a, 'A')}.csv", rec_a
        )
        csv_b = write_frame_motion_csv(
            mode_dir / f"motion_{_run_id(rec_b, 'B')}.csv", rec_b
        )
        logger.info(f"移動量CSV: {csv_a}")
        logger.info(f"移動量CSV: {csv_b}")
        try:
            import matplotlib.pyplot as plt
            plt.imsave(mode_dir / "partial_A.png", acc_a.colormap_rgb())
            plt.imsave(mode_dir / "partial_B.png", acc_b.colormap_rgb())
            plt.imsave(mode_dir / "partial_A_corrected.png", seam.strips[0]["rgb"])
            plt.imsave(mode_dir / "partial_B_corrected.png", seam.strips[1]["rgb"])
            plt.imsave(mode_dir / "final.png", joined["rgb"])
        except Exception as exc:
            logger.warning(f"画像保存失敗: {exc}")
        np.savez_compressed(
            mode_dir / "provenance_final.npz",
            color=joined["rgb"],
            filled=joined["filled"],
            source_run=joined["source_run"],
            run_ids=joined["run_ids"],
            z_min=joined["z_min"],
            z_max=joined["z_max"],
        )

        results["modes"][mode] = {
            "run_A": summarize_records(rec_a),
            "run_B": summarize_records(rec_b),
            "seam_warp": seam.to_report(),
            "correction": {
                "A": corr_a.message,
                "B": corr_b.message,
                "seam": seam.message,
            },
            "coverage": {
                "A_unfilled": acc_a.unfilled_ratio(),
                "B_unfilled": acc_b.unfilled_ratio(),
                "final_unfilled": float(1.0 - np.mean(joined["filled"])),
                "run_adoption": joined["adoption"],
            },
            "motion_csv": {
                "A": str(csv_a),
                "B": str(csv_b),
            },
        }

    report_path = write_validation_report(output_dir, results)
    results["report_path"] = str(report_path)
    return results


def _run_id(records, fallback: str) -> str:
    if records:
        return str(records[0].run_id or fallback)
    return fallback


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
    mapper, records, frames, reference, config, z_min=None, z_max=None
) -> BestViewAccumulator:
    zs = [r.position[2] for r in records]
    if z_min is None:
        z_min = min(zs) - 50.0
    if z_max is None:
        z_max = max(zs) + 50.0
    acc = BestViewAccumulator(config, z_min=z_min, z_max=z_max)
    run_id = records[0].run_id if records else "A"
    max_dz = float(config.estimation.motion.max_dz)
    alpha = float(config.colormap.alpha)
    ppm = float(config.colormap.pixels_per_mm)
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
        acc.add_strip(
            strip["colors_2d"], strip["mask_2d"], float(strip["z_min_strip"]),
            run_id, rec.frame_num,
        )
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
    p = argparse.ArgumentParser(description="2方向動画合成カラーマップ検証")
    p.add_argument("--config", type=Path, default=Path("data/config/two_direction_config.json"))
    p.add_argument("--run-a", type=Path, default=None)
    p.add_argument("--run-b", type=Path, default=None)
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
        config.two_direction.run_A.video_path = str(args.run_a)
        suffix = infer_suffix_from_path(str(args.run_a))
        if suffix:
            config.two_direction.run_A.physical_roll_deg = physical_roll_from_suffix(suffix)
            config.two_direction.run_A.run_id = suffix
    if args.run_b:
        config.two_direction.run_B.video_path = str(args.run_b)
        suffix = infer_suffix_from_path(str(args.run_b))
        if suffix:
            config.two_direction.run_B.physical_roll_deg = physical_roll_from_suffix(suffix)
            config.two_direction.run_B.run_id = suffix
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
