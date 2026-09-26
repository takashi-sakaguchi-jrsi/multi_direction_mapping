"""実機相当の OCR 距離シミュレーション。

実 OCR は 10mm 単位。N mm の刻みラインを超えたフレームでカウントアップする。
ジャスト N mm ではタイムラグのためまだ上がらない（z==N では N-10 のまま）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

OCR_STEP_MM = 10.0
DEFAULT_Z_STEP_MM = 4.5


def simulate_ocr_distance_mm(
    z_true_mm,
    step_mm: float = OCR_STEP_MM,
) -> np.ndarray:
    """真の走行距離から、10mm 刻み・遅れ付きの OCR 読みを作る。

    初期フレームがちょうど刻みに乗っているときは、その値を初期表示とする。
    以降は ``z > 現在値 + step`` になったフレームでのみカウントアップする。
    """
    z = np.asarray(z_true_mm, dtype=float).reshape(-1)
    step = float(step_mm)
    if step <= 0.0:
        raise ValueError(f"OCR step は正である必要があります: {step}")
    if z.size == 0:
        return z.copy()

    ocr = np.empty(z.size, dtype=float)
    z0 = float(z[0])
    n0 = z0 / step
    if abs(n0 - round(n0)) < 1e-9:
        current = float(round(n0) * step)
    else:
        current = float(np.floor(n0) * step)
    ocr[0] = current
    for i in range(1, z.size):
        zi = float(z[i])
        while zi > current + step:
            current += step
        ocr[i] = current
    return ocr


def ocr_metadata_fields(
    z_true_mm,
    z_step_mm: float,
    ocr_step_mm: float = OCR_STEP_MM,
) -> dict:
    """生成 metadata に載せる OCR 関連フィールド。"""
    z = np.asarray(z_true_mm, dtype=float).reshape(-1)
    ocr = simulate_ocr_distance_mm(z, step_mm=ocr_step_mm)
    return {
        "z_step_mm": float(z_step_mm),
        "ocr_step_mm": float(ocr_step_mm),
        "ocr_count_up": "strict_greater_than_tick",
        "ocr_note": (
            f"{ocr_step_mm:g}mm 刻み。ラインを超えたフレームでカウントアップ。"
            f"ジャスト刻みではタイムラグのため上がらない。"
            f"走行 {z_step_mm:g}mm/frame（{ocr_step_mm:g}mm と同期しない）。"
        ),
        "ocr_z_mm": [float(v) for v in ocr],
        "z_values_mm": [float(v) for v in z],
    }


def ocr_z_from_metadata(meta: dict, fallback_true_z: Optional[bool] = True) -> np.ndarray:
    """metadata の OCR 列。無ければ真値 z（後方互換）。"""
    if meta.get("ocr_z_mm") is not None:
        z = np.asarray(meta["ocr_z_mm"], dtype=float).reshape(-1)
        if z.size == 0:
            raise ValueError("ocr_z_mm が空です")
        return z
    if fallback_true_z and meta.get("z_values_mm") is not None:
        return np.asarray(meta["z_values_mm"], dtype=float).reshape(-1)
    raise ValueError("metadata に ocr_z_mm / z_values_mm がありません")
