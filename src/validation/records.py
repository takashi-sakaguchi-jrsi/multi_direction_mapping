"""検証パイプラインのレコード定義"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class FrameAnalysisRecord:
    run_id: str
    frame_num: int
    timestamp: float
    position: np.ndarray
    orientation: np.ndarray
    reference: np.ndarray
    z_ocr: Optional[float]
    motion_raw: np.ndarray
    motion_final: np.ndarray
    match_count: int
    residual_stats: dict
    prior_norm: float
    bound_hit: dict
    status: str
    mode: str = "A"


@dataclass
class FrameProjectionSummary:
    run_id: str
    frame_num: int
    z_min: float
    z_max: float
    theta_min: float
    theta_max: float
    valid_ratio: float
    d_wall_mean: float
    d_wall_min: float
    gamma_mean: float
