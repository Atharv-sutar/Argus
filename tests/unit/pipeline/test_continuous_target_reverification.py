"""Unit tests for continuous target re-verification and hysteresis lock switching (Issue 1)."""

import numpy as np
import pytest

from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig
from src.core.multi_camera_types import CameraEdgeConfig, CameraNodeConfig, EdgeType, SourceType
from src.core.types import BoundingBox, Detection, DetectionResult, Embedding, TargetState
from src.detection.yolo_detector import BaseDetector
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.tracking.byte_tracker import ByteTracker


class ControllableMockDetector(BaseDetector):
    def __init__(self):
        self.detections = []

    def set_persons(self, boxes):
        """boxes is a list of (x1, y1, x2, y2, conf)"""
        self.detections = [
            Detection(box=BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], confidence=b[4]), class_id=0, confidence=b[4])
            for b in boxes
        ]

    def detect(self, frame, timestamp_ms=0.0):
        return DetectionResult(detections=self.detections, frame_id=0, timestamp_ms=timestamp_ms)

    def detect_batch(self, frames, frame_ids=None, timestamps_ms=None):
        frame_ids = frame_ids or [0]*len(frames)
        timestamps_ms = timestamps_ms or [0.0]*len(frames)
        return [DetectionResult(detections=self.detections, frame_id=fid, timestamp_ms=ts) 
                for fid, ts in zip(frame_ids, timestamps_ms)]


def test_lock_switches_when_bystander_hijacks_track_and_real_target_is_present():
    """Verify Issue 1: If track 1 is hijacked by a bystander, and real target is track 2, lock switches to track 2."""
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_A", name="Lobby", source="synthetic", source_type=SourceType.SYNTHETIC))

    config = AppConfig()
    config.reid.match_threshold = 0.80
    config.reid.lock_switch_margin = 0.05
    config.reid.extract_interval_frames = 1

    det = ControllableMockDetector()

    # Define target vector [1, 0, 0, 0] and bystander vector [0, 1, 0, 0] (similarity = 0.0)
    target_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    bystander_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    class DynamicMockReID:
        def extract(self, crop):
            h, w = crop.shape[:2]
            if w > 80:
                return Embedding(vector=target_vec)
            return Embedding(vector=bystander_vec)

        def extract_batch(self, crops):
            return [self.extract(c) for c in crops]

    mock_reid = DynamicMockReID()

    pipe = MultiCameraPipeline(
        graph=graph,
        config=config,
        reid_extractor=mock_reid,
        camera_factory=lambda node_cfg: SyntheticCamera(width=640, height=480, fps=30),
        tracker_factory=lambda: ByteTracker(track_thresh=0.4, match_thresh=0.5),
        shared_detector=det,
    )

    # 1. Lock Target on Track 1 (wide box: width=100 -> returns target_vec)
    det.set_persons([(50.0, 50.0, 150.0, 200.0, 0.95)])
    worker = pipe._get_or_create_worker("cam_A")

    pipe.step()
    pipe.select_target_on_camera("cam_A", 100.0, 100.0)
    pipe.step()

    assert pipe.target_manager.target.state == TargetState.TRACKING
    locked_id = pipe.target_manager.target.track_id
    assert locked_id is not None
    assert pipe.identity.size >= 1

    # 2. Frame now has TWO people:
    # Person 1 (locked track ID) is now a bystander (narrow box: width=50 -> returns bystander_vec with sim=0.0)
    # Person 2 (new track ID) is the REAL TARGET (wide box: width=100 -> returns target_vec with sim=1.0)
    det.set_persons([
        (50.0, 50.0, 100.0, 200.0, 0.95),   # Track 1 (bystander)
        (250.0, 50.0, 350.0, 200.0, 0.95),  # Track 2 (real target)
    ])

    pipe.step()

    # The system must have detected that Track 1 failed ReID (sim=0.0) and Track 2 matched (sim=1.0),
    # so it must have SWITCHED the lock to Track 2!
    new_locked_id = pipe.target_manager.target.track_id
    assert new_locked_id is not None
    assert new_locked_id != locked_id
    assert pipe.target_manager.target.state == TargetState.TRACKING
