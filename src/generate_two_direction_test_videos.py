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
from typing import Dict, List, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

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
    pixels_per_mm_from_equirect,
    resolve_colormap_path,
    write_mp4,
    z_values_for_segment,
)
from src.validation.ocr_simulation import (
    DEFAULT_Z_STEP_MM,
    OCR_STEP_MM,
    ocr_metadata_fields,
)
from src.validation.geometry import (
    SUFFIX_TO_PHYSICAL_ROLL_DEG,
    physical_roll_to_internal_deg,
    run_id_from_physical_roll,
)


logger = logging.getLogger(__name__)


def parse_runs(text: str) -> List[str]:
    runs = [p.strip().upper() for p in text.split(",") if p.strip()]
    if not runs:
        raise ValueError("runs が空です")
    for run in runs:
        if run not in SUFFIX_TO_PHYSICAL_ROLL_DEG:
            raise ValueError(f"未知の run: {run}（U/R/L）")
    return runs


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
) -> Dict:
    renderer = FisheyeSideviewRenderer(
        colormap=colormap,
        output_size=(width, height),
        radius_mm=radius_mm,
        fov_deg=fov_deg,
        apply_distance_shading=shading,
        keep=radius_mm if shading else 0.0,
    )
    if n_frames < N_FRAMES_HINT_MIN or n_frames > N_FRAMES_HINT_MAX:
        logger.warning(
            f"目視確認には {N_FRAMES_HINT_MIN}～{N_FRAMES_HINT_MAX} フレーム程度を推奨"
            f"（指定: {n_frames}）"
        )
    zs = z_values_for_segment(
        renderer,
        z_start_mm=z_start_mm,
        n_frames=n_frames,
        z_step_mm=z_step_mm,
        z_margin_mm=z_margin_mm,
    )
    if len(zs) < n_frames:
        logger.warning(
            f"展開図終端のため {n_frames} フレームから {len(zs)} に短縮 "
            f"(z_start={zs[0]:.1f} mm, z_end={zs[-1]:.1f} mm, extent={renderer.z_extent_mm:.1f} mm)"
        )
    logger.info(
        f"生成区間 z={zs[0]:.1f}～{zs[-1]:.1f} mm, {len(zs)} frames, step={z_step_mm} mm"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir
    result: Dict = {
        "colormap": str(colormap),
        "camera_model": "fisheye_equidistant",
        "fov_deg": float(fov_deg),
        "f_px": float(renderer.f),
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
        "z_start_mm": float(zs[0]),
        "z_end_mm": float(zs[-1]),
        **ocr_metadata_fields(zs, z_step_mm=float(z_step_mm), ocr_step_mm=OCR_STEP_MM),
        "fps": float(fps),
        "distance_overlay": False,
        "phase1": False,
        "runs": {},
    }
    all_frames: Dict[str, List[np.ndarray]] = {}
    for run in runs:
        roll = SUFFIX_TO_PHYSICAL_ROLL_DEG[run]
        frames, _ = renderer.render_sequence(zs, roll_deg=roll, pitch_deg=pitch_deg)
        all_frames[run] = frames
        video_path = videos_dir / f"{name}_{run}.mp4"
        write_mp4(video_path, frames, fps)
        if save_png:
            frame_dir = videos_dir / f"{name}_{run}_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                cv2.imwrite(str(frame_dir / f"{i:04d}.png"), frame)
        result["runs"][run] = {
            "run_id": run_id_from_physical_roll(roll),
            "physical_roll_deg": float(roll),
            "physical_roll_internal_deg": float(physical_roll_to_internal_deg(roll)),
            "video_path": str(video_path),
            "n_frames": len(frames),
        }
        logger.info(f"{run}: {video_path} ({len(frames)} frames, roll={roll})")

    if reconstruct:
        rec_dir = output_dir / "reconstruct_skip_phase1"
        rec_dir.mkdir(parents=True, exist_ok=True)
        frames_all = []
        positions = []
        orientations = []
        for run in runs:
            roll = SUFFIX_TO_PHYSICAL_ROLL_DEG[run]
            for frame, z in zip(all_frames[run], zs):
                frames_all.append(frame)
                positions.append((0.0, 0.0, float(z)))
                orientations.append((float(roll), float(pitch_deg)))
        canvas, filled = renderer.reconstruct_from_hits(frames_all, positions, orientations)
        preview = crop_colormap_z_window(canvas, renderer, float(zs[0]), float(zs[-1]))
        source_preview = crop_colormap_z_window(
            renderer.equirect_img, renderer, float(zs[0]), float(zs[-1])
        )
        mask_preview = crop_colormap_z_window(
            (filled.astype(np.uint8) * 255), renderer, float(zs[0]), float(zs[-1])
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
    p.add_argument("--runs", type=str, default="U,R")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--radius-mm", type=float, default=DEFAULT_RADIUS_MM)
    p.add_argument("--fov", type=float, default=DEFAULT_FOV_DEG)
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
        help=f"生成フレーム数（目視確認は {N_FRAMES_HINT_MIN}～{N_FRAMES_HINT_MAX} 程度）",
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
        runs = parse_runs(args.runs)
        result = generate_two_direction_videos(
            colormap=colormap,
            output_dir=args.output_dir,
            runs=runs,
            width=args.width,
            height=args.height,
            radius_mm=args.radius_mm,
            fov_deg=args.fov,
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
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(str(exc))
        return 1
    print(f"metadata: {result.get('metadata_path')}")
    print(
        f"z={result.get('z_start_mm'):.1f}～{result.get('z_end_mm'):.1f} mm, "
        f"frames={len(result.get('z_values_mm', []))}"
    )
    for run, info in result.get("runs", {}).items():
        print(f"{run}: {info['video_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
