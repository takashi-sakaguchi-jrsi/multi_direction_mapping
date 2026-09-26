"""OCR-only + 側方特徴点によるフレーム解析（V1）"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from src.camera_estimation import CameraEstimator, compute_distance_constraints_only
from src.validation.geometry import (
    build_run_reference,
    infer_suffix_from_path,
    mode_a_estimate_yaw_pitch,
    physical_roll_from_suffix,
)
from src.validation.records import FrameAnalysisRecord
from src.validation.sideview_feature_matcher import SideviewFeatureMatcher

logger = logging.getLogger(__name__)


def resolve_frame_window(td, n_total: Optional[int] = None) -> Tuple[int, Optional[int]]:
    """[start, end) 。end が None なら末尾まで。max_frames は start からの枚数上限。"""
    start = int(getattr(td, "start_frame", 0) or 0)
    if start < 0:
        raise ValueError(f"start_frame は 0 以上: {start}")
    end = getattr(td, "end_frame", None)
    end = int(end) if end is not None else None
    max_frames = getattr(td, "max_frames", None)
    if max_frames is not None:
        cap = start + int(max_frames)
        end = cap if end is None else min(end, cap)
    if end is not None and end <= start:
        raise ValueError(f"空のフレーム範囲: start={start}, end={end}")
    if n_total is not None:
        n_total = int(n_total)
        start = min(start, n_total)
        if end is None:
            end = n_total
        else:
            end = min(end, n_total)
        if start >= n_total or (end is not None and end <= start):
            raise ValueError(
                f"フレーム範囲が動画長を外れます: start={start}, end={end}, n={n_total}"
            )
    return start, end


def video_meta(video_path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画が見つかりません: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return fps, n_total


def iter_bgr_frames(
    video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """指定範囲のフレームを1枚ずつ返す。全枚は RAM に持たない。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画が見つかりません: {video_path}")
    start = int(start_frame or 0)
    try:
        idx = 0
        while idx < start:
            if not cap.grab():
                raise ValueError(
                    f"start_frame={start} が動画長を超えています: {video_path}"
                )
            idx += 1
        while end_frame is None or idx < int(end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def zip_records_with_frames(records: List[FrameAnalysisRecord], extra: Dict):
    """解析結果とフレームを対応付ける。動画なら再オープンして逐次読む。"""
    frames = extra.get("frames")
    if frames is not None:
        for rec, frame in zip(records, frames):
            yield rec, frame
        return
    path = extra.get("video_path")
    if not path:
        raise ValueError("frames も video_path もありません")
    start = int(extra.get("start_frame") or 0)
    end = extra.get("end_frame")
    end = int(end) if end is not None else None
    for rec, (_idx, frame) in zip(records, iter_bgr_frames(path, start, end)):
        yield rec, frame


class FrameAnalyzer:
    def __init__(self, config, estimator: CameraEstimator, mode: str = "A"):
        self.config = config
        self.estimator = estimator
        self.mode = mode.upper()
        self.matcher = SideviewFeatureMatcher(config)
        td = config.two_direction
        self.roi_rect = td.capture.roi_rect
        cap_h = td.capture.image_height_px
        cap_w = td.capture.image_width_px
        self.max_radius_px = min(cap_w, cap_h) / 2.0 * td.capture.usable_outer_radius_ratio

    def analyze_run(
        self,
        run_cfg,
        frames: Optional[List[np.ndarray]] = None,
        fps: float = 30.0,
        known_z_mm: Optional[np.ndarray] = None,
    ) -> Tuple[List[FrameAnalysisRecord], Dict]:
        physical_roll = run_cfg.physical_roll_deg
        if run_cfg.video_path:
            suffix = infer_suffix_from_path(run_cfg.video_path)
            if suffix is not None and run_cfg.physical_roll_deg in (0.0, 120.0, 240.0, -120.0):
                # 明示指定を優先。未設定時のみ suffix 補完。
                pass
        reference = build_run_reference(
            physical_roll_deg=physical_roll,
            physical_pitch_deg=run_cfg.physical_pitch_deg,
            yaw_ref_deg=run_cfg.yaw_ref_deg,
            x_ref_mm=run_cfg.x_ref_mm,
            y_ref_mm=run_cfg.y_ref_mm,
        )
        run_id = run_cfg.run_id or reference["run_id"]
        td = self.config.two_direction
        keep_frames = frames
        video_path = str(run_cfg.video_path) if run_cfg.video_path else ""

        if keep_frames is None:
            if not video_path:
                raise ValueError("video_path または frames が必要です")
            video_fps, n_total = video_meta(video_path)
            if video_fps and video_fps > 0:
                fps = float(video_fps)
            n_hint = n_total if n_total > 0 else None
            if n_hint is None and known_z_mm is not None:
                n_hint = int(np.asarray(known_z_mm).reshape(-1).size)
            start, end = resolve_frame_window(td, n_total=n_hint)
        else:
            start, end = resolve_frame_window(td, n_total=len(keep_frames))
            keep_frames = keep_frames[start:end]
        if fps <= 0:
            fps = 30.0
        if known_z_mm is not None:
            known_z_mm = np.asarray(known_z_mm, dtype=float).reshape(-1)
            start_z, end_z = resolve_frame_window(td, n_total=known_z_mm.size)
            known_z_mm = known_z_mm[start_z:end_z]
            n_expect = len(keep_frames) if keep_frames is not None else (
                (end - start) if end is not None else None
            )
            if n_expect is not None and known_z_mm.size < n_expect:
                raise ValueError(
                    f"known_z_mm の長さ ({known_z_mm.size}) がフレーム数 "
                    f"({n_expect}) より短いです"
                )
            if n_expect is not None:
                known_z_mm = known_z_mm[:n_expect]

        constraints, z_positions, ocr_dist, success = compute_distance_constraints_only(
            video_path=Path(video_path) if video_path and keep_frames is None else None,
            start_frame=start,
            end_frame=end,
            frames=None if known_z_mm is not None else keep_frames,
            estimator=self.estimator,
            ocr_roi_ratio=tuple(self.config.camera.ocr_roi_ratio),
            config=self.config.estimation,
            ocr_tesseract_config=self.config.ocr.tesseract_config,
            ocr_preprocessing_enabled=self.config.ocr.preprocessing_enabled,
            known_z_mm=known_z_mm,
        )

        n_target = int(success.size)
        records: List[FrameAnalysisRecord] = []
        position = np.array([
            reference["x_ref_mm"], reference["y_ref_mm"],
            float(z_positions[0]) if z_positions is not None else 0.0,
        ], dtype=float)
        orientation = np.array([
            reference["roll_ref_rad"],
            reference["yaw_ref_rad"],
            reference["pitch_ref_rad"],
        ], dtype=float)

        mapper_roi = self.roi_rect
        prev = None
        center = (0.0, 0.0)
        camera_params: Dict = {}
        if keep_frames is not None:
            frame_iter = enumerate(keep_frames)
        else:
            frame_iter = (
                (i, fr) for i, (_abs, fr) in enumerate(
                    iter_bgr_frames(video_path, start, end)
                )
            )

        for local_i, curr in frame_iter:
            if local_i >= n_target:
                break
            if prev is None:
                h, w = curr.shape[:2]
                center = (w / 2.0, h / 2.0)
                camera_params = {
                    "center": center,
                    "radius": min(w, h) / 2.0,
                    "f": self.estimator.transformer.camera.f,
                    "model": "fisheye",
                }
                records.append(self._make_record(
                    run_id, 0, start, start / fps, position, orientation, reference,
                    ocr_dist, success, z_positions, np.zeros(6), np.zeros(6),
                    0, {}, 0.0, {}, "INIT",
                ))
                prev = curr
                continue

            orig = start + local_i
            motion_raw, motion_final, match_count, residual_stats, prior_norm, bound_hit, status = (
                self._motion_between(
                    prev, curr, local_i, run_id, position, orientation, reference,
                    camera_params, mapper_roi, constraints,
                )
            )
            if status == "OK":
                position = position + np.array([
                    motion_final[0], motion_final[1], motion_final[2]
                ])
                orientation = orientation + np.array([
                    motion_final[3], motion_final[4], motion_final[5]
                ])
                if self.mode == "A":
                    position[0] = reference["x_ref_mm"]
                    position[1] = reference["y_ref_mm"]
                    orientation[0] = reference["roll_ref_rad"]
                    est_yaw, est_pitch = mode_a_estimate_yaw_pitch(run_id)
                    if not est_yaw:
                        orientation[1] = reference["yaw_ref_rad"]
                    if not est_pitch:
                        orientation[2] = reference["pitch_ref_rad"]
                prev = curr
            records.append(self._make_record(
                run_id, local_i, orig, orig / fps, position.copy(), orientation.copy(), reference,
                ocr_dist, success, z_positions, motion_raw, motion_final,
                match_count, residual_stats, prior_norm, bound_hit, status,
            ))
            if (local_i + 1) % 50 == 0 or (local_i + 1) == n_target:
                logger.info(f"{run_id}: analyze {local_i + 1}/{n_target}")
            if (local_i + 1) % 100 == 0:
                gc.collect()

        if not records:
            raise ValueError(
                f"指定範囲のフレームが空です: {video_path or 'frames'} "
                f"[{start}, {end})"
            )

        extra = {
            "z_positions": z_positions,
            "ocr_dist": ocr_dist[: len(records)],
            "ocr_success": success[: len(records)],
            "run_reference": reference,
            "frames": keep_frames,
            "video_path": video_path,
            "z_source": "known" if known_z_mm is not None else "ocr",
            "start_frame": start,
            "end_frame": start + len(records),
            "fps": fps,
        }
        return records, extra

    def _motion_between(
        self,
        prev,
        curr,
        local_i,
        run_id,
        position,
        orientation,
        reference,
        camera_params,
        mapper_roi,
        constraints,
    ):
        status = "OK"
        match_count = 0
        motion_raw = np.zeros(6)
        motion_final = np.zeros(6)
        residual_stats = {}
        prior_norm = 0.0
        bound_hit = {}
        try:
            pts1, pts2 = self.matcher.detect_and_match(
                prev, curr,
                roi_rect=mapper_roi,
                center=camera_params["center"],
                forward_dy_sign=self._forward_image_dy_sign(
                    run_id, position, orientation
                ),
            )
            if pts1 is None:
                status = "FAILED"
            else:
                match_count = len(pts1)
                camera_state = {"position": position.copy(), "orientation": orientation.copy()}
                motion = self.estimator.estimate_motion_flexible(
                    pts1, pts2, camera_state, camera_params,
                    constraints=constraints.get(local_i),
                    run_reference=reference,
                    center_prior=self.config.two_direction.center_prior,
                    estimation_mode=self.mode,
                    hard_bounds=self.config.two_direction.hard_bounds,
                )
                motion_raw = np.array([
                    motion.get("dx", 0.0), motion.get("dy", 0.0), motion.get("dz", 0.0),
                    motion.get("droll", 0.0), motion.get("dyaw", 0.0), motion.get("dpitch", 0.0),
                ])
                constrained = self.estimator._constrain_camera_pose(
                    motion, position, orientation, local_i,
                    run_reference=reference,
                    hard_bounds=self.config.two_direction.hard_bounds,
                )
                motion_final = np.array([
                    constrained.get("dx", 0.0), constrained.get("dy", 0.0),
                    constrained.get("dz", 0.0), constrained.get("droll", 0.0),
                    constrained.get("dyaw", 0.0), constrained.get("dpitch", 0.0),
                ])
                if self.mode == "A":
                    motion_final[0] = 0.0
                    motion_final[1] = 0.0
                    motion_final[3] = 0.0
                    est_yaw, est_pitch = mode_a_estimate_yaw_pitch(run_id)
                    if not est_yaw:
                        motion_final[4] = 0.0
                    if not est_pitch:
                        motion_final[5] = 0.0
                residual_stats = {
                    "residual_rms": float(motion.get("residual_rms", 0.0)),
                }
                prior_norm = float(constrained.get("prior_norm", motion.get("prior_norm", 0.0)))
                bound_hit = constrained.get("bound_hit", {})
        except Exception as exc:
            status = "FAILED"
            residual_stats = {"error": str(exc)}
        return motion_raw, motion_final, match_count, residual_stats, prior_norm, bound_hit, status

    @staticmethod
    def _load_frames(
        video_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"動画が見つかりません: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        start = int(start_frame or 0)
        idx = 0
        while idx < start:
            ok = cap.grab()
            if not ok:
                cap.release()
                raise ValueError(
                    f"start_frame={start} が動画長を超えています: {video_path}"
                )
            idx += 1
        frames = []
        while end_frame is None or idx < int(end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            idx += 1
        cap.release()
        if not frames:
            raise ValueError(
                f"指定範囲のフレームが空です: {video_path} "
                f"[{start_frame}, {end_frame})"
            )
        return frames, fps

    def _forward_image_dy_sign(
        self,
        run_id: str,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> int:
        """+z 前進で画像 y が増えるなら +1、減るなら -1。

        U の (-90,90,90) は Demo / 旧 (0,0,90) とカメラ上下が同じなので R/L と同じ -1。
        """
        return -1

    def _make_record(
        self, run_id, local_i, frame_num, timestamp, position, orientation, reference,
        ocr_dist, success, z_positions, motion_raw, motion_final,
        match_count, residual_stats, prior_norm, bound_hit, status,
    ) -> FrameAnalysisRecord:
        z_ocr = None
        if local_i < len(success) and success[local_i]:
            z_ocr = float(ocr_dist[local_i])
        ref_vec = np.array([
            reference["x_ref_mm"], reference["y_ref_mm"],
            reference["roll_ref_rad"], reference["yaw_ref_rad"], reference["pitch_ref_rad"],
        ])
        return FrameAnalysisRecord(
            run_id=run_id,
            frame_num=frame_num,
            timestamp=timestamp,
            position=np.asarray(position, dtype=float),
            orientation=np.asarray(orientation, dtype=float),
            reference=ref_vec,
            z_ocr=z_ocr,
            motion_raw=np.asarray(motion_raw, dtype=float),
            motion_final=np.asarray(motion_final, dtype=float),
            match_count=int(match_count),
            residual_stats=residual_stats,
            prior_norm=float(prior_norm),
            bound_hit=bound_hit or {},
            status=status,
            mode=self.mode,
        )
