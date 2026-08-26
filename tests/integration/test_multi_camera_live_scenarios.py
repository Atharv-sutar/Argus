"""Comprehensive integration tests for live multi-camera production scenarios."""

import numpy as np
import pytest

from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig
from src.core.interfaces import BaseDetector, BaseReID
from src.core.multi_camera_types import (
    CameraEdgeConfig,
    CameraNodeConfig,
    CameraStatus,
    EdgeType,
    SourceType,
)
from src.core.types import BoundingBox, Detection, DetectionResult, Embedding, TargetState
from src.identity.manager import IdentityManager
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.tracking.byte_tracker import ByteTracker


class ControllableMockDetector(BaseDetector):
    def __init__(self) -> None:
        self.detections_to_return: list[Detection] = []

    def set_person(self, x1: float, y1: float, x2: float, y2: float, conf: float = 0.95):
        box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
        self.detections_to_return = [
            Detection(box=box, class_id=0, class_name="person", confidence=conf)
        ]

    def set_people(self, boxes: list[tuple[float, float, float, float]], conf: float = 0.95):
        self.detections_to_return = [
            Detection(box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf), class_id=0, class_name="person", confidence=conf)
            for x1, y1, x2, y2 in boxes
        ]

    def clear(self):
        self.detections_to_return = []

    def detect(self, frame: np.ndarray, frame_id: int = 0, timestamp_ms: float = 0.0) -> DetectionResult:
        return DetectionResult(
            detections=list(self.detections_to_return),
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
        )


class DeterministicMockReID(BaseReID):
    def __init__(self, target_vector: np.ndarray, bystander_vector: np.ndarray) -> None:
        self.target_vector = target_vector
        self.bystander_vector = bystander_vector
        self.is_target_mode = True

    def extract(self, crop: np.ndarray) -> Embedding:
        if crop is None or crop.size == 0:
            return Embedding(vector=np.zeros(4, dtype=np.float32))
        vec = self.target_vector if self.is_target_mode else self.bystander_vector
        return Embedding(vector=vec)

    def extract_batch(self, crops: list[np.ndarray]) -> list[Embedding]:
        return [self.extract(c) for c in crops]


def _build_test_setup():
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_A", name="Lobby", source="synthetic", source_type=SourceType.SYNTHETIC))
    graph.add_node(CameraNodeConfig(camera_id="cam_B", name="Corridor", source="synthetic", source_type=SourceType.SYNTHETIC))
    graph.add_edge(CameraEdgeConfig(source_camera_id="cam_A", target_camera_id="cam_B", edge_type=EdgeType.ADJACENT))

    config = AppConfig()
    config.multi_camera.search.confirmation_frames = 2
    config.multi_camera.search.per_radius_timeout_s = 5.0
    config.reid.similarity_threshold = 0.80

    target_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    bystander_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    mock_reid = DeterministicMockReID(target_vec, bystander_vec)

    identity_mgr = IdentityManager(
        reid_extractor=mock_reid,
        similarity_threshold=config.reid.similarity_threshold,
        min_margin=0.05,
    )

    det_a = ControllableMockDetector()
    det_b = ControllableMockDetector()

    def camera_factory(node_cfg: CameraNodeConfig):
        return SyntheticCamera(width=320, height=240, fps=30)

    def tracker_factory():
        return ByteTracker(track_thresh=0.4, match_thresh=0.5)

    pipe = MultiCameraPipeline(
        graph=graph,
        config=config,
        reid_extractor=mock_reid,
        identity_manager=identity_mgr,
        camera_factory=camera_factory,
        tracker_factory=tracker_factory,
    )
    p_a = pipe._get_or_create_pipeline("cam_A")
    p_b = pipe._get_or_create_pipeline("cam_B")
    p_a.detector = det_a
    p_b.detector = det_b

    return pipe, det_a, det_b, mock_reid


def test_target_absent_everywhere_bystander_never_adopted():
    """Verify Failure 2: When target leaves all cameras, visible bystander is NEVER adopted."""
    pipe, det_a, det_b, mock_reid = _build_test_setup()

    # 1. Target selected on Cam A
    det_a.set_person(50.0, 50.0, 150.0, 180.0)
    det_b.clear()
    pipe.step()
    pipe.select_target_on_camera("cam_A", 100.0, 100.0)

    for _ in range(3):
        pipe.step()
    assert pipe.active_camera_id == "cam_A"
    assert pipe.get_camera_status("cam_A") == CameraStatus.ACTIVE_TARGET

    # 2. Target leaves everywhere
    det_a.clear()
    det_b.clear()
    pipe.step()
    assert pipe.search_manager.is_searching

    # 3. Bystander appears on Cam A while search is running
    det_a.set_person(50.0, 50.0, 150.0, 180.0)
    mock_reid.is_target_mode = False  # Pure bystander

    # Run for 20 frames
    for _ in range(20):
        pipe.step()

    # The bystander must NEVER be adopted as the target!
    pipe_a = pipe._pipelines["cam_A"]
    assert pipe_a.current_target.state in (TargetState.LOST, TargetState.UNSELECTED)
    assert pipe_a.current_target.track_id is None or pipe_a.current_target.state == TargetState.LOST


def test_cross_camera_handoff_and_reacquisition():
    """Verify Failure 1 & 4: Legitimate target moving Cam A -> Cam B is cleanly recovered."""
    pipe, det_a, det_b, mock_reid = _build_test_setup()

    # 1. Target on Cam A
    det_a.set_person(50.0, 50.0, 150.0, 180.0)
    det_b.clear()
    pipe.step()
    pipe.select_target_on_camera("cam_A", 100.0, 100.0)
    for _ in range(3):
        pipe.step()

    # 2. Target leaves Cam A
    det_a.clear()
    pipe.step()
    assert pipe.search_manager.is_searching

    # 3. Target appears on Cam B
    det_b.set_person(60.0, 60.0, 160.0, 190.0)
    mock_reid.is_target_mode = True

    for _ in range(4):
        pipe.step()

    # Handoff to Cam B must succeed
    assert pipe.active_camera_id == "cam_B"
    pipe_b = pipe._pipelines["cam_B"]
    assert pipe_b.current_target.state == TargetState.TRACKING
