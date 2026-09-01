"""Integration tests for MultiCameraPipeline handoff and search orchestration."""

import numpy as np
import pytest

from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig, SearchConfig
from src.core.interfaces import BaseDetector, BaseReID
from src.core.multi_camera_types import (
    CameraEdgeConfig,
    CameraNodeConfig,
    CameraStatus,
    EdgeDirection,
    EdgeType,
    SourceType,
)
from src.core.types import BoundingBox, Detection, DetectionResult, Embedding, TargetState
from src.identity.manager import IdentityManager
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.tracking.byte_tracker import ByteTracker


class ControllableMockDetector(BaseDetector):
    """Detector whose detections can be updated dynamically per test step."""

    def __init__(self) -> None:
        self.detections_to_return: list[Detection] = []

    def set_person(self, x1: float, y1: float, x2: float, y2: float, conf: float = 0.95):
        box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
        self.detections_to_return = [
            Detection(box=box, class_id=0, class_name="person", confidence=conf)
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
    """ReID extractor that yields fixed feature vectors for target vs bystander."""

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


def test_multi_camera_selection_and_cross_camera_handoff():
    # 1. Setup Camera Graph: cam_A <--> cam_B
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_A", name="Lobby", source="synthetic", source_type=SourceType.SYNTHETIC))
    graph.add_node(CameraNodeConfig(camera_id="cam_B", name="Corridor", source="synthetic", source_type=SourceType.SYNTHETIC))
    graph.add_edge(CameraEdgeConfig(source_camera_id="cam_A", target_camera_id="cam_B", edge_type=EdgeType.ADJACENT))

    # 2. Config & Mock Models
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

    # 3. Create MultiCameraPipeline with custom factories
    pipe = MultiCameraPipeline(
        graph=graph,
        config=config,
        reid_extractor=mock_reid,
        identity_manager=identity_mgr,
        camera_factory=camera_factory,
        tracker_factory=tracker_factory,
    )
    # Wire detector lookup per camera
    p_a = pipe._get_or_create_pipeline("cam_A")
    p_b = pipe._get_or_create_pipeline("cam_B")
    p_a.detector = det_a
    p_b.detector = det_b

    # Step 1: Place person on cam_A
    det_a.set_person(50.0, 50.0, 150.0, 180.0)
    det_b.clear()

    # Initial frame step on cam_A
    pipe.step()

    # Select target on cam_A at (100, 100)
    selected_id = pipe.select_target_on_camera("cam_A", 100.0, 100.0)
    assert selected_id is not None
    assert pipe.active_camera_id == "cam_A"
    assert pipe.get_camera_status("cam_A") == CameraStatus.ACTIVE_TARGET

    # Step 2: Target is actively tracked on cam_A
    res = pipe.step()
    _, _, target_a = res["cam_A"]
    assert target_a is not None
    assert target_a.state == TargetState.TRACKING
    assert not pipe.search_manager.is_searching

    # Step 3: Person leaves cam_A (target LOST on cam_A)
    det_a.clear()
    res = pipe.step()
    _, _, target_a_lost = res["cam_A"]
    assert target_a_lost.state == TargetState.LOST

    # Verify search manager initiated search on neighbor cam_B
    assert pipe.search_manager.is_searching
    assert "cam_B" in pipe.search_manager.get_progress().active_cameras

    # Step 4: Person appears on cam_B
    det_b.set_person(60.0, 60.0, 160.0, 190.0)
    mock_reid.is_target_mode = True  # Verified target appearance

    # Step 4+: Person appears on cam_B -> Multi-frame reacquisition confirms target
    for _ in range(4):
        pipe.step()

    # Verify Handoff succeeded
    assert pipe.active_camera_id == "cam_B"
    assert pipe.get_camera_status("cam_B") == CameraStatus.ACTIVE_TARGET
    assert not pipe.search_manager.is_searching

    pipe.stop()
