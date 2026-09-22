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
        if loaded is None:
            loaded = self._load_frames(
                run_cfg.video_path,
                max_frames=getattr(self.config.two_direction, "max_frames", None),
            )
        else:
            max_frames = getattr(self.config.two_direction, "max_frames", None)
            if max_frames is not None:
                loaded = loaded[: int(max_frames)]
        if known_z_mm is not None:
            known_z_mm = np.asarray(known_z_mm, dtype=float).reshape(-1)
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
            run_id, 0, 0.0, position, orientation, reference,
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

            z_ocr = float(ocr_dist[i]) if success[i] else None
            records.append(self._make_record(
                run_id, i, i / fps, position.copy(), orientation.copy(), reference,
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
        }
        return records, extra

    @staticmethod
    def _load_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"動画が見つかりません: {video_path}")
        frames = []
        limit = int(max_frames) if max_frames is not None else None
        while True:
            if limit is not None and len(frames) >= limit:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"フレームが空です: {video_path}")
        return frames

    def _make_record(
        self, run_id, frame_num, timestamp, position, orientation, reference,
        ocr_dist, success, z_positions, motion_raw, motion_final,
        match_count, residual_stats, prior_norm, bound_hit, status,
    ) -> FrameAnalysisRecord:
        z_ocr = None
        if frame_num < len(success) and success[frame_num]:
            z_ocr = float(ocr_dist[frame_num])
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
