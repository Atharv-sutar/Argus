"""Evaluation Harness for Multi-Camera Person Tracking and Re-Identification."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.benchmark.types import (
    EvaluationMetrics,
    FrameEvaluationResult,
    GroundTruthDataset,
    GroundTruthObservation,
)
from src.camera.capture import OpenCVCamera
from src.core.config import AppConfig
from src.core.interfaces import BaseDetector, BaseReID, BaseTracker
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.core.types import BoundingBox, TargetState
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline

logger = logging.getLogger("argus.benchmark")


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    if box_a is None or box_b is None:
        return 0.0

    xA = max(box_a.x1, box_b.x1)
    yA = max(box_a.y1, box_b.y1)
    xB = min(box_a.x2, box_b.x2)
    yB = min(box_a.y2, box_b.y2)

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a.x2 - box_a.x1) * max(0.0, box_a.y2 - box_a.y1)
    area_b = max(0.0, box_b.x2 - box_b.x1) * max(0.0, box_b.y2 - box_b.y1)

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


class ReIDEvaluationHarness:
    """
    Rigorously evaluates person ReID accuracy, lock-in persistence, false-positive/scoop rates,
    and cross-camera target handoffs on real synchronized surveillance footage against ground truth.
    """

    def __init__(
        self,
        graph: CameraGraph,
        config: Optional[AppConfig] = None,
        camera_sources: Optional[Dict[str, Union[str, int]]] = None,
        dataset: Optional[GroundTruthDataset] = None,
        target_person_id: str = "person_1",
        reid_extractor: Optional[BaseReID] = None,
        shared_detector: Optional[BaseDetector] = None,
        tracker_factory: Optional[Callable[[], BaseTracker]] = None,
        iou_match_threshold: float = 0.40,
    ) -> None:
        self.graph = graph
        self.config = config or AppConfig()
        self.camera_sources = camera_sources or {}
        self.dataset = dataset or GroundTruthDataset(target_person_id=target_person_id)
        self.target_person_id = target_person_id
        self.reid_extractor = reid_extractor
        self.shared_detector = shared_detector
        self.tracker_factory = tracker_factory
        self.iou_match_threshold = iou_match_threshold

        self._pipeline: Optional[MultiCameraPipeline] = None
        self._frame_results: List[FrameEvaluationResult] = []

    def _create_camera_factory(self) -> Callable[[CameraNodeConfig], Any]:
        sources = dict(self.camera_sources)

        def factory(node_cfg: CameraNodeConfig) -> Any:
            src = sources.get(node_cfg.camera_id, node_cfg.source)
            # Use non-threaded synchronous camera for deterministic frame-by-frame evaluation
            return OpenCVCamera(source=src, use_thread=False)

        return factory

    def run(
        self,
        seed_camera_id: str = "cam_0",
        seed_frame_id: int = 0,
        seed_box: Optional[BoundingBox] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> EvaluationMetrics:
        """
        Executes evaluation over the synchronized camera feeds.

        Args:
            seed_camera_id: Camera where the target is initially selected.
            seed_frame_id: Frame number on seed_camera_id to register target seed.
            seed_box: Target bounding box on seed frame (if None, pulled from dataset).
            max_frames: Optional frame limit.
            progress_callback: Optional callback(current_frame, total_frames).

        Returns:
            EvaluationMetrics object containing precision, recall, FP rate, and state metrics.
        """
        camera_factory = self._create_camera_factory()

        self._pipeline = MultiCameraPipeline(
            graph=self.graph,
            config=self.config,
            reid_extractor=self.reid_extractor,
            shared_detector=self.shared_detector,
            tracker_factory=self.tracker_factory,
            camera_factory=camera_factory,
        )

        metrics = EvaluationMetrics()
        self._frame_results.clear()

        last_confirmed_person_id: Optional[str] = None
        was_previously_lost: bool = True
        frame_idx = 0
        target_seeded = False

        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            # 1. Step pipeline synchronously across all cameras
            step_success = self._pipeline.step()
            if not step_success:
                logger.info(f"End of footage reached at frame {frame_idx}.")
                break

            # 2. Seed target at designated seed frame
            if not target_seeded and frame_idx >= seed_frame_id:
                # Find ground-truth or specified seed box
                sbox = seed_box
                if sbox is None:
                    gt_target = self.dataset.get_target_for_frame(seed_camera_id, frame_idx)
                    if gt_target is not None:
                        sbox = gt_target.box

                if sbox is not None:
                    # Select target center
                    cx = (sbox.x1 + sbox.x2) / 2.0
                    cy = (sbox.y1 + sbox.y2) / 2.0
                    selected = self._pipeline.select_target_on_camera(seed_camera_id, cx, cy)
                    if selected:
                        target_seeded = True
                        logger.info(f"Target '{self.target_person_id}' successfully seeded on '{seed_camera_id}' at frame {frame_idx}.")

            # 3. Evaluate frame state against Ground Truth
            active_cam = self._pipeline.active_camera_id or seed_camera_id
            target = self._pipeline.target_manager.target
            target_state = target.state

            # Ground truth persons in active camera
            gt_persons = self.dataset.get_for_frame(active_cam, frame_idx)
            gt_target = self.dataset.get_target_for_frame(active_cam, frame_idx)
            target_present = gt_target is not None

            metrics.total_frames += 1
            if target_present:
                metrics.target_present_frames += 1

            # Update state distribution
            st_name = target_state.name
            metrics.state_frames[st_name] = metrics.state_frames.get(st_name, 0) + 1

            # Match system's target box against ground truth persons
            system_has_target = target_state in (TargetState.TRACKING, TargetState.CONFIRMED, TargetState.LOCKED)
            sys_box = target.last_known_box if system_has_target else None

            best_iou = 0.0
            matched_person: Optional[str] = None

            if sys_box is not None and gt_persons:
                for p_obs in gt_persons:
                    iou = compute_iou(sys_box, p_obs.box)
                    if iou > best_iou:
                        best_iou = iou
                        matched_person = p_obs.person_id

            iou_with_target = compute_iou(sys_box, gt_target.box) if (sys_box and gt_target) else 0.0

            # Classification
            is_tp = False
            is_fp = False
            is_fn = False
            is_tn = False
            is_switch = False

            if system_has_target:
                if matched_person == self.target_person_id and best_iou >= self.iou_match_threshold:
                    # True Positive: System claims target is locked and box matches ground truth target
                    is_tp = True
                    metrics.true_positives += 1

                    if was_previously_lost:
                        metrics.recovery_events += 1
                        was_previously_lost = False

                    if last_confirmed_person_id is not None and last_confirmed_person_id != self.target_person_id:
                        is_switch = True
                        metrics.id_switches += 1

                    last_confirmed_person_id = self.target_person_id
                else:
                    # False Positive / Scoop: System locked onto a bystander or phantom box
                    is_fp = True
                    metrics.false_positives += 1

                    if last_confirmed_person_id is not None and last_confirmed_person_id == self.target_person_id:
                        is_switch = True
                        metrics.id_switches += 1

                    if matched_person is not None:
                        last_confirmed_person_id = matched_person
            else:
                was_previously_lost = True
                if target_present:
                    # False Negative: Target was visible in ground truth but system lost/uncertain
                    is_fn = True
                    metrics.false_negatives += 1
                else:
                    # True Negative: Target not present, system correctly does not lock
                    is_tn = True
                    metrics.true_negatives += 1

            # Collect ReID candidate similarity telemetry
            scores = self._pipeline.last_candidate_scores
            t_sim = None
            b_sims = []

            for tid, sim in scores.items():
                if target.track_id == tid and is_tp:
                    t_sim = sim
                    metrics.target_similarities.append(sim)
                else:
                    b_sims.append(sim)
                    metrics.bystander_similarities.append(sim)

            best_b_sim = max(b_sims) if b_sims else None

            # Record frame result
            frame_res = FrameEvaluationResult(
                frame_id=frame_idx,
                camera_id=active_cam,
                target_present=target_present,
                system_has_target=system_has_target,
                system_target_state=target_state,
                system_active_cam=self._pipeline.active_camera_id,
                system_track_id=target.track_id if system_has_target else None,
                system_box=sys_box,
                matched_person_id=matched_person,
                iou_with_target=iou_with_target,
                is_true_positive=is_tp,
                is_false_positive=is_fp,
                is_false_negative=is_fn,
                is_true_negative=is_tn,
                is_id_switch=is_switch,
                candidate_scores=dict(scores),
                target_sim=t_sim,
                best_bystander_sim=best_b_sim,
            )
            self._frame_results.append(frame_res)

            frame_idx += 1
            if progress_callback:
                progress_callback(frame_idx, max_frames or 0)

        # Teardown
        if self._pipeline:
            self._pipeline.stop()

        return metrics

    def get_frame_results(self) -> List[FrameEvaluationResult]:
        """Returns detailed frame-by-frame evaluation results."""
        return list(self._frame_results)
