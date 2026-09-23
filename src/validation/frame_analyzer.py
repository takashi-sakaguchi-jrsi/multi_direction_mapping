"""OCR-only + 側方特徴点によるフレーム解析（V1）"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

        loaded = frames
        td = self.config.two_direction
        if loaded is None:
            start, end = resolve_frame_window(td)
            loaded, video_fps = self._load_frames(
                run_cfg.video_path,
                start_frame=start,
                end_frame=end,
            )
            if video_fps and video_fps > 0:
                fps = float(video_fps)
        else:
            start, end = resolve_frame_window(td, n_total=len(loaded))
            loaded = loaded[start:end]
        if fps <= 0:
            fps = 30.0
        if known_z_mm is not None:
            known_z_mm = np.asarray(known_z_mm, dtype=float).reshape(-1)
            start_z, end_z = resolve_frame_window(td, n_total=known_z_mm.size)
            known_z_mm = known_z_mm[start_z:end_z]
            if known_z_mm.size < len(loaded):
                raise ValueError(
                    f"known_z_mm の長さ ({known_z_mm.size}) がフレーム数 "
                    f"({len(loaded)}) より短いです"
                )
            known_z_mm = known_z_mm[: len(loaded)]

        constraints, z_positions, ocr_dist, success = compute_distance_constraints_only(
            frames=loaded,
            estimator=self.estimator,
            ocr_roi_ratio=tuple(self.config.camera.ocr_roi_ratio),
            config=self.config.estimation,
            ocr_tesseract_config=self.config.ocr.tesseract_config,
            ocr_preprocessing_enabled=self.config.ocr.preprocessing_enabled,
            known_z_mm=known_z_mm,
        )

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

        h, w = loaded[0].shape[:2]
        center = (w / 2.0, h / 2.0)
        camera_params = {
            "center": center,
            "radius": min(w, h) / 2.0,
            "f": self.estimator.transformer.camera.f,
            "model": "fisheye",
        }
        mapper_roi = self.roi_rect

        prev = loaded[0]
        records.append(self._make_record(
            run_id, 0, start, start / fps, position, orientation, reference,
            ocr_dist, success, z_positions, np.zeros(6), np.zeros(6),
            0, {}, 0.0, {}, "INIT",
        ))

        for i in range(1, len(loaded)):
            curr = loaded[i]
            status = "OK"
            match_count = 0
            motion_raw = np.zeros(6)
            motion_final = np.zeros(6)
            residual_stats = {}
            prior_norm = 0.0
            bound_hit = {}
            orig = start + i
            try:
                pts1, pts2 = self.matcher.detect_and_match(
                    prev, curr,
                    roi_rect=mapper_roi,
                    center=center,
                )
                if pts1 is None:
                    status = "FAILED"
                else:
                    match_count = len(pts1)
                    camera_state = {"position": position.copy(), "orientation": orientation.copy()}
                    motion = self.estimator.estimate_motion_flexible(
                        pts1, pts2, camera_state, camera_params,
                        constraints=constraints.get(i),
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
                        motion, position, orientation, i,
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
                    residual_stats = {
                        "residual_rms": float(motion.get("residual_rms", 0.0)),
                    }
                    prior_norm = float(constrained.get("prior_norm", motion.get("prior_norm", 0.0)))
                    bound_hit = constrained.get("bound_hit", {})
            except Exception as exc:
                status = "FAILED"
                residual_stats = {"error": str(exc)}

            records.append(self._make_record(
                run_id, i, orig, orig / fps, position.copy(), orientation.copy(), reference,
                ocr_dist, success, z_positions, motion_raw, motion_final,
                match_count, residual_stats, prior_norm, bound_hit, status,
            ))
            if status == "OK":
                prev = curr

        extra = {
            "z_positions": z_positions,
            "ocr_dist": ocr_dist,
            "ocr_success": success,
            "run_reference": reference,
            "frames": loaded,
            "z_source": "known" if known_z_mm is not None else "ocr",
            "start_frame": start,
            "end_frame": start + len(loaded),
            "fps": fps,
        }
        return records, extra

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
