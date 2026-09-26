"""既存の 3 方向最終図から、ビューアで開ける縮小プレビューを書き出す。"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.main_two_direction_validation import (
    _save_photoviewer_png,
    _save_rgb_png,
    _save_rgb_png_zlib,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite-final", action="store_true")
    args = parser.parse_args()

    mode_a = _ROOT / "data" / "output" / "three_direction" / "mode_A"
    color_path = mode_a / "provenance_final_color.npy"
    src_path = mode_a / "provenance_final_source_run.npy"
    if not color_path.is_file():
        raise SystemExit(f"missing {color_path}")

    color = np.load(color_path, mmap_mode="r")
    if args.rewrite_final:
        dest = mode_a / "final.png"
        tmp = mode_a / "final.rewrite.png"
        print("rewriting", dest, "from npy", color.shape)
        _save_rgb_png_zlib(tmp, color)
        tmp.replace(dest)
        print("rewrote", dest, dest.stat().st_size)

    _save_photoviewer_png(mode_a / "final_photoviewer.png", color)
    print("wrote photoviewer")

    preview = np.ascontiguousarray(color[::5, ::20])
    _save_rgb_png(
        mode_a / "final_preview.png", preview,
        write_preview=False, write_photoviewer=False,
    )
    print("wrote", mode_a / "final_preview.png", preview.shape, preview.mean(axis=(0, 1)))

    h, w = color.shape[:2]
    mid = w // 2
    crops = {
        "final_preview_left.png": color[:, :4000][::4, ::2],
        "final_preview_mid.png": color[:, mid - 2000 : mid + 2000][::4, ::2],
        "final_preview_right.png": color[:, -4000:][::4, ::2],
    }
    del color
    gc.collect()
    for name, arr in crops.items():
        arr_c = np.ascontiguousarray(arr)
        _save_rgb_png(
            mode_a / name, arr_c, write_preview=False, write_photoviewer=False,
        )
        print("wrote", name, arr_c.shape)

    if src_path.is_file():
        src = np.load(src_path, mmap_mode="r")
        vis = np.zeros(src.shape + (3,), dtype=np.uint8)
        vis[src == 0] = (255, 0, 0)
        vis[src == 1] = (0, 255, 0)
        vis[src == 2] = (0, 0, 255)
        small = vis[::5, ::20]
        _save_rgb_png(
            mode_a / "final_source_preview.png", small,
            write_preview=False, write_photoviewer=False,
        )
        print("wrote final_source_preview.png", small.shape)
        del src, vis, small
        gc.collect()

    print("ok")


if __name__ == "__main__":
    main()
