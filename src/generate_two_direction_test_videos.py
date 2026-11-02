"""既存展開図から2方向固定走行の魚眼テスト動画を生成する。

CameraCarDemo の円筒サンプリングを使い、カメラモデルだけ等距離魚眼に差し替える。
正本は original_colormap/phi250tenkaizu.png（管路直径 250mm）。
距離 overlay と Phase1（OCR / 特徴点姿勢）は行わない。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from src.calibration import fisheye_intrinsics_for_size, load_calibration
from src.validation.fisheye_sideview_renderer import (
    DEFAULT_COLORMAP_RELATIVE,
    DEFAULT_FOV_DEG,
    DEFAULT_N_FRAMES,
    DEFAULT_PITCH_DEG,
    DEFAULT_RADIUS_MM,
    DEFAULT_Z_START_MM,
    N_FRAMES_HINT_MAX,
    N_FRAMES_HINT_MIN,
    FisheyeSideviewRenderer,
    crop_colormap_z_window,
    open_mp4_writer,
    pixels_per_mm_from_equirect,
    resolve_colormap_path,
    z_values_for_segment,
)
from src.validation.ocr_simulation import (
    DEFAULT_Z_STEP_MM,
    OCR_STEP_MM,
    ocr_metadata_fields,
)
from src.validation.pose_jitter import (
    PoseJitterConfig,
    build_motion_series,
    independent_run_seed,
    independent_z_step_mm,
    motion_metadata_fields,
)

_CAMERA_META_KEYS = (
    "colormap",
    "camera_model",
    "fov_deg",
    "n_directions",
    "optical_axis_spacing_deg",
    "f_px",
    "fx",
    "fy",
    "cx",
    "cy",
    "distortion_k",
    "width",
    "height",
    "radius_mm",
    "pipe_diameter_mm",
    "colormap_width_px",
    "colormap_height_px",
    "source_pixels_per_mm",
    "physical_pitch_deg",
    "fps",
    "distance_overlay",
    "phase1",
    "lens_calibration_file",
)
from src.validation.geometry import (
    SUFFIX_TO_PHYSICAL_ROLL_DEG,
    build_direction_layout,
    build_run_reference,
    canonical_run_id,
    physical_roll_from_run_id,
    physical_roll_to_internal_deg,
)


logger = logging.getLogger(__name__)


def parse_runs(text: str) -> List[str]:
    runs = [p.strip().upper().lstrip("_") for p in text.split(",") if p.strip()]
    if not runs:
        raise ValueError("runs が空です")
    out: List[str] = []
    for run in runs:
        roll = physical_roll_from_run_id(run)
        if run in SUFFIX_TO_PHYSICAL_ROLL_DEG:
            out.append(run)
        else:
            out.append(canonical_run_id(roll))
    return out


def default_pinhole_run_ids(
    n_directions: int = 8,
    optical_axis_spacing_deg: float = 45.0,
    fov_deg: float = 60.0,
) -> List[str]:
    layout = build_direction_layout(
        n_directions, optical_axis_spacing_deg, fov_deg
    )
    return list(layout.run_ids)


def generate_two_direction_videos(
    colormap: Path,
    output_dir: Path,
    runs: Sequence[str],
    width: int,
    height: int,
    radius_mm: float,
    fov_deg: float,
    pitch_deg: float,
    n_frames: int,
    fps: float,
    name: str,
    z_start_mm: float,
    z_step_mm: float,
    z_margin_mm: float,
    shading: bool,
    reconstruct: bool,
    save_png: bool,
    jitter: bool = False,
    jitter_seed: int = 0,
    calibration_path: Optional[str] = None,
    camera_model: str = "fisheye",
    n_directions: Optional[int] = None,
    optical_axis_spacing_deg: float = 45.0,
) -> Dict:
    model = str(camera_model or "fisheye").lower()
    if model not in ("fisheye", "pinhole"):
        raise ValueError(f"未知の camera_model: {camera_model}")
    if model == "pinhole" and calibration_path:
        raise ValueError(
            "ピンホール経路ではレンズ校正 JSON は使えません。"
            " --calibration を外してください。"
        )
    calib = None
    dist_coeffs = None
    f_px = None
    cx = None
    cy = None
    fy = None
    if calibration_path:
        calib = load_calibration(str(calibration_path))
        f_px, cx, cy, dist_coeffs, calib_fov = fisheye_intrinsics_for_size(
            calib, int(width), int(height)
        )
        if calib_fov > 0.0:
            fov_deg = float(calib_fov)
        fy = float(getattr(calib, "fy", 0.0) or 0.0)
        if fy > 0.0 and int(getattr(calib, "image_width", 0) or 0) > 0:
            fy = fy * float(width) / float(calib.image_width)
        logger.info(
            f"校正レンズ: {calibration_path} f={f_px:.4f} cx={cx:.3f} cy={cy:.3f} "
            f"k={dist_coeffs} size={int(width)}x{int(height)}"
        )
    renderer = FisheyeSideviewRenderer(
        colormap=colormap,
        output_size=(width, height),
        radius_mm=radius_mm,
        fov_deg=fov_deg,
        apply_distance_shading=shading,
        keep=radius_mm if shading else 0.0,
        f_px=f_px,
        cx=cx,
        cy=cy,
        dist_coeffs=dist_coeffs,
        camera_model=model,
    )
    if n_frames < N_FRAMES_HINT_MIN or n_frames > N_FRAMES_HINT_MAX:
        logger.warning(
            f"目視確認には {N_FRAMES_HINT_MIN}～{N_FRAMES_HINT_MAX} フレーム程度を推奨"
            f"（指定: {n_frames}）"
        )
    z_max_mm = float(renderer.z_extent_mm) - float(z_margin_mm)
    program_rpy_by_run = {}
    shared_zs = None
    shared_ocr_fields = None
    if not jitter:
        shared_zs = z_values_for_segment(
            renderer,
            z_start_mm=z_start_mm,
            n_frames=n_frames,
            z_step_mm=z_step_mm,
            z_margin_mm=z_margin_mm,
        )
        shared_ocr_fields = ocr_metadata_fields(
            shared_zs, z_step_mm=float(z_step_mm), ocr_step_mm=OCR_STEP_MM
        )
        if len(shared_zs) < n_frames:
            logger.warning(
                f"展開図終端のため {n_frames} フレームから {len(shared_zs)} に短縮 "
                f"(z_start={shared_zs[0]:.1f} mm, z_end={shared_zs[-1]:.1f} mm, "
                f"extent={renderer.z_extent_mm:.1f} mm)"
            )
        logger.info(
            f"生成区間 z={shared_zs[0]:.1f}～{shared_zs[-1]:.1f} mm, "
            f"{len(shared_zs)} frames, step={z_step_mm} mm"
        )
    logger.info(
        f"フレームは逐次エンコードします（RAM に全枚保持しない）: "
        f"最大 {n_frames} frames × {len(runs)} runs, {width}x{height}"
        + ("（方向ごとに独立走行）" if jitter else "")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir
    result: Dict = {
        "colormap": str(colormap),
        "camera_model": (
            "pinhole" if model == "pinhole"
            else ("fisheye_calibrated" if calib is not None else "fisheye_equidistant")
        ),
        "fov_deg": float(fov_deg),
        "n_directions": int(n_directions) if n_directions is not None else len(list(runs)),
        "optical_axis_spacing_deg": float(optical_axis_spacing_deg),
        "f_px": float(renderer.f),
        "fx": float(renderer.f),
        "fy": float(fy) if fy else float(renderer.f),
        "cx": float(renderer.cx),
        "cy": float(renderer.cy),
        "distortion_k": (
            [0.0, 0.0, 0.0, 0.0] if model == "pinhole"
            else (
                [float(v) for v in renderer.dist_coeffs]
                if renderer.dist_coeffs is not None
                else [0.0, 0.0, 0.0, 0.0]
            )
        ),
        "width": int(width),
        "height": int(height),
        "radius_mm": float(radius_mm),
        "pipe_diameter_mm": float(radius_mm) * 2.0,
        "colormap_width_px": int(renderer.eq_w),
        "colormap_height_px": int(renderer.eq_h),
        "source_pixels_per_mm": float(
            pixels_per_mm_from_equirect(renderer.eq_h, radius_mm)
        ),
        "physical_pitch_deg": float(pitch_deg),
        "fps": float(fps),
        "distance_overlay": False,
        "phase1": False,
        "motion_jitter": bool(jitter),
        "motion_jitter_independent_runs": bool(jitter),
        "jitter_seed": int(jitter_seed) if jitter else None,
        "runs": {},
    }
    if model != "pinhole":
        result["lens_calibration_file"] = (
            str(calibration_path) if calib is not None else None
        )
    rec_canvas = None
    rec_filled = None
    rec_dmin = None
    if reconstruct:
        rec_canvas = np.zeros_like(renderer.equirect_img)
        rec_filled = np.zeros((renderer.eq_h, renderer.eq_w), dtype=bool)
        rec_dmin = np.full((renderer.eq_h, renderer.eq_w), np.inf, dtype=np.float32)
    n_runs = len(runs)
    z_lo_all = None
    z_hi_all = None
    for run_i, run in enumerate(runs):
        roll = physical_roll_from_run_id(run)
        run_key = canonical_run_id(roll) if model == "pinhole" else run
        rpy_list = None
        series = None
        run_seed = None
        if jitter:
            run_seed = independent_run_seed(int(jitter_seed), run_i)
            z_step_run = independent_z_step_mm(float(z_step_mm), run_seed)
            jitter_cfg = PoseJitterConfig(seed=run_seed)
            series = build_motion_series(
                n_frames=n_frames,
                z_start_mm=z_start_mm,
                z_step_mm=z_step_run,
                z_max_mm=z_max_mm,
                config=jitter_cfg,
            )
            zs = series["z_mm"]
            ocr_fields = motion_metadata_fields(
                series, z_step_mm=float(z_step_run), config=jitter_cfg
            )
            logger.info(
                f"{run_key}: 独立振動走行 seed={run_seed} "
                f"z={zs[0]:.1f}～{zs[-1]:.1f} mm, {len(zs)} frames, "
                f"dz_mean={ocr_fields['dz_mean_mm']:.3f} mm"
            )
            ref = build_run_reference(roll, physical_pitch_deg=pitch_deg)
            yaw_off = series["yaw_offset_deg"]
            pitch_off = series["pitch_offset_deg"]
            rpy_list = [
                (
                    float(ref["roll_ref_rad"]),
                    float(np.radians(ref["yaw_ref_deg"] + yaw_off[i])),
                    float(np.radians(ref["pitch_ref_deg"] + pitch_off[i])),
                )
                for i in range(len(zs))
            ]
            program_rpy_by_run[run_key] = rpy_list
        else:
            zs = shared_zs
            ocr_fields = dict(shared_ocr_fields)
        if z_lo_all is None:
            z_lo_all = float(zs[0])
            z_hi_all = float(zs[-1])
        else:
            z_lo_all = min(z_lo_all, float(zs[0]))
            z_hi_all = max(z_hi_all, float(zs[-1]))
        if run_i == 0:
            result["z_start_mm"] = float(zs[0])
            result["z_end_mm"] = float(zs[-1])
            result.update(ocr_fields)
        video_path = videos_dir / f"{name}_{run_key}.mp4"
        writer = open_mp4_writer(
            video_path, (renderer.output_width, renderer.output_height), fps
        )
        frame_dir = None
        if save_png:
            frame_dir = videos_dir / f"{name}_{run_key}_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        progress_every = 50 if len(zs) >= 100 else max(1, len(zs))
        try:
            for i, (frame, z) in enumerate(
                renderer.iter_rendered_frames(
                    zs, roll_deg=roll, pitch_deg=pitch_deg, program_rpy_rad=rpy_list
                )
            ):
                writer.write(frame)
                if frame_dir is not None:
                    cv2.imwrite(str(frame_dir / f"{i:06d}.png"), frame)
                if reconstruct:
                    ori = (
                        program_rpy_by_run[run_key][i]
                        if jitter
                        else (float(roll), float(pitch_deg))
                    )
                    renderer.accumulate_reconstruct(
                        rec_canvas,
                        rec_filled,
                        rec_dmin,
                        frame,
                        (0.0, 0.0, float(z)),
                        ori,
                    )
                n_written += 1
                if (i + 1) % progress_every == 0 or (i + 1) == len(zs):
                    logger.info(
                        f"{run}: {i + 1}/{len(zs)} frames "
                        f"(run {run_i + 1}/{n_runs})"
                    )
        finally:
            writer.release()
        run_entry = {
            "run_id": run_key,
            "physical_roll_deg": float(roll),
            "physical_roll_internal_deg": float(physical_roll_to_internal_deg(roll)),
            "video_path": str(video_path),
            "n_frames": n_written,
            "z_start_mm": float(zs[0]),
            "z_end_mm": float(zs[-1]),
            **ocr_fields,
        }
        if jitter:
            run_entry["motion_jitter_seed"] = int(run_seed)
        run_meta = {
            key: result[key]
            for key in _CAMERA_META_KEYS
            if key in result
        }
        run_meta.update(ocr_fields)
        run_meta["run_id"] = run_key
        run_meta["physical_roll_deg"] = float(roll)
        run_meta["z_start_mm"] = float(zs[0])
        run_meta["z_end_mm"] = float(zs[-1])
        run_meta["motion_jitter"] = bool(jitter)
        run_meta["n_frames"] = n_written
        if jitter:
            run_meta["motion_jitter_seed"] = int(run_seed)
        run_meta_path = videos_dir / f"{name}_{run_key}_metadata.json"
        run_meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        run_entry["metadata_path"] = str(run_meta_path)
        result["runs"][run_key] = run_entry
        logger.info(f"{run_key}: {video_path} ({n_written} frames, roll={roll})")
    if z_lo_all is not None:
        result["z_start_mm"] = float(z_lo_all)
        result["z_end_mm"] = float(z_hi_all)

    if reconstruct:
        rec_dir = output_dir / "reconstruct_skip_phase1"
        rec_dir.mkdir(parents=True, exist_ok=True)
        rec_canvas[~rec_filled] = 0
        canvas, filled = rec_canvas, rec_filled
        crop_lo = float(z_lo_all if z_lo_all is not None else zs[0])
        crop_hi = float(z_hi_all if z_hi_all is not None else zs[-1])
        preview = crop_colormap_z_window(canvas, renderer, crop_lo, crop_hi)
        source_preview = crop_colormap_z_window(
            renderer.equirect_img, renderer, crop_lo, crop_hi
        )
        mask_preview = crop_colormap_z_window(
            (filled.astype(np.uint8) * 255), renderer, crop_lo, crop_hi
        )
        rec_path = rec_dir / f"{name}_reconstruct.png"
        src_path = rec_dir / f"{name}_source.png"
        mask_path = rec_dir / f"{name}_filled.png"
        cv2.imwrite(str(rec_path), preview)
        cv2.imwrite(str(src_path), source_preview)
        cv2.imwrite(str(mask_path), mask_preview)
        if np.any(filled):
            src = renderer.equirect_img
            mae = float(np.mean(np.abs(canvas[filled].astype(np.int16) - src[filled].astype(np.int16))))
        else:
            mae = None
        result["reconstruct"] = {
            "path": str(rec_path),
            "filled_ratio": float(np.mean(filled)),
            "mae": mae,
        }
        logger.info(
            f"Phase1省略の再投影: filled={np.mean(filled):.3f}, mae={mae}"
        )

    meta_path = output_dir / f"{name}_metadata.json"
    meta_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["metadata_path"] = str(meta_path)
    return result


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="既存展開図から2方向魚眼テスト動画を生成（距離overlay/Phase1なし）"
    )
    p.add_argument(
        "--colormap",
        type=Path,
        default=DEFAULT_COLORMAP_RELATIVE,
        help="入力展開図 PNG（既定: original_colormap/phi250tenkaizu.png）",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/input/videos"),
        help="mp4 の出力先",
    )
    p.add_argument("--name", type=str, default="phi250_fisheye_side")
    p.add_argument(
        "--runs",
        type=str,
        default=None,
        help="U,R,L または 000,045,...。未指定時は魚眼 U,R、ピンホールは 8 方向",
    )
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--radius-mm", type=float, default=DEFAULT_RADIUS_MM)
    p.add_argument("--fov", type=float, default=None)
    p.add_argument(
        "--camera-model",
        type=str,
        default="fisheye",
        choices=["fisheye", "pinhole"],
        help="生成レンズ。pinhole は 60°・8 方向が既定",
    )
    p.add_argument("--n-directions", type=int, default=None)
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="レンズ校正 JSON（魚眼のみ。ピンホール経路では拒否）",
    )
    p.add_argument("--pitch", type=float, default=DEFAULT_PITCH_DEG)
    p.add_argument(
        "--z-start-mm",
        type=float,
        default=DEFAULT_Z_START_MM,
        help="開始点の管路距離[mm]。変えると別の特徴区間で試せる",
    )
    p.add_argument(
        "--z-step-mm",
        type=float,
        default=DEFAULT_Z_STEP_MM,
        help="フレーム間の前進量[mm]。OCR 10mm 刻みと同期しないよう既定 4.5",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_N_FRAMES,
        help=(
            f"生成フレーム数（目視確認は {N_FRAMES_HINT_MIN}～{N_FRAMES_HINT_MAX} 程度。"
            "全長でも1フレームずつ書き出す）"
        ),
    )
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument(
        "--z-margin-mm",
        type=float,
        default=10.0,
        help="展開図端からの余白[mm]",
    )
    p.add_argument("--shading", action="store_true", help="Demo 相当の距離減衰を入れる")
    p.add_argument(
        "--jitter",
        action="store_true",
        help="dz/dpitch/dyaw の合成振動を方向ごとに独立して入れる（速度も非同期）",
    )
    p.add_argument("--jitter-seed", type=int, default=0)
    p.add_argument(
        "--reconstruct",
        action="store_true",
        help="既知姿勢で展開図へ戻す（Phase1省略の往復確認）",
    )
    p.add_argument("--save-png", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        colormap = resolve_colormap_path(args.colormap)
        camera_model = str(args.camera_model or "fisheye").lower()
        n_directions = args.n_directions
        if n_directions is None and camera_model == "pinhole":
            n_directions = 8
        fov_deg = args.fov
        if fov_deg is None:
            fov_deg = 60.0 if camera_model == "pinhole" else DEFAULT_FOV_DEG
        if args.runs:
            runs = parse_runs(args.runs)
        elif camera_model == "pinhole":
            runs = default_pinhole_run_ids(
                int(n_directions or 8), 45.0, float(fov_deg)
            )
        else:
            runs = parse_runs("U,R")
        calib_path = str(args.calibration) if args.calibration else None
        width = args.width
        height = args.height
        if calib_path:
            calib = load_calibration(calib_path)
            if width is None and int(getattr(calib, "image_width", 0) or 0) > 0:
                width = int(calib.image_width)
            if height is None and int(getattr(calib, "image_height", 0) or 0) > 0:
                height = int(calib.image_height)
        if width is None:
            width = 1920
        if height is None:
            height = 1080
        result = generate_two_direction_videos(
            colormap=colormap,
            output_dir=args.output_dir,
            runs=runs,
            width=width,
            height=height,
            radius_mm=args.radius_mm,
            fov_deg=fov_deg,
            pitch_deg=args.pitch,
            n_frames=args.frames,
            fps=args.fps,
            name=args.name,
            z_start_mm=args.z_start_mm,
            z_step_mm=args.z_step_mm,
            z_margin_mm=args.z_margin_mm,
            shading=args.shading,
            reconstruct=args.reconstruct,
            save_png=args.save_png,
            jitter=bool(args.jitter),
            jitter_seed=int(args.jitter_seed),
            calibration_path=calib_path,
            camera_model=camera_model,
            n_directions=n_directions,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(str(exc))
        return 1
    print(f"metadata: {result.get('metadata_path')}")
    print(
        f"z={result.get('z_start_mm'):.1f}～{result.get('z_end_mm'):.1f} mm "
        f"(全 run の範囲)"
    )
    for run, info in result.get("runs", {}).items():
        zs = info.get("z_values_mm") or []
        n = int(info.get("n_frames") or len(zs))
        if zs:
            print(
                f"{run}: {info['video_path']} "
                f"z={zs[0]:.1f}～{zs[-1]:.1f} mm n={n}"
            )
        else:
            print(f"{run}: {info['video_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
