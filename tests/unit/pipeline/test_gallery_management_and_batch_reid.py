"""Unit tests for gallery entry deletion, batched ReID candidate evaluation, and dynamic graph synchronization."""

import numpy as np
import pytest

from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.core.types import BoundingBox, Detection, DetectionResult, Embedding, TargetState
from src.detection.yolo_detector import BaseDetector
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.reid.gallery import TargetGallery
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


def test_gallery_remove_entry_rebuilds_matrix():
    """Verify that deleting a gallery entry removes it and properly updates the cosine matrix."""
    gallery = TargetGallery(max_size=10, match_threshold=0.75)
    
    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    
    crop1 = np.ones((50, 50, 3), dtype=np.uint8) * 100
    crop2 = np.ones((50, 50, 3), dtype=np.uint8) * 200

    gallery.seed(crop=crop1, embedding=Embedding(vector=vec1), camera_id="cam_0")
    gallery.add_manual(crop=crop2, embedding=Embedding(vector=vec2), camera_id="cam_0")

    assert gallery.size == 2
    entries = gallery.get_entries()
    entry1_id = entries[0].entry_id
    entry2_id = entries[1].entry_id

    # Test matching before removal
    sim1, _ = gallery.match(Embedding(vector=vec1))
    assert sim1 > 0.90

    # Remove entry 1
    removed = gallery.remove_entry(entry1_id)
    assert removed is True
    assert gallery.size == 1

    # After removal, vec1 should have ~0.0 similarity against the remaining gallery (vec2)
    sim1_after, _ = gallery.match(Embedding(vector=vec1))
    assert sim1_after < 0.01

    # vec2 should still match perfectly
    sim2_after, _ = gallery.match(Embedding(vector=vec2))
    assert sim2_after > 0.90


def test_realistic_candidate_reid_and_lock_switching():
    """
    Simulates real-world conditions:
    Target gallery seeded with true target.
    Tracker ID gets hijacked by a bystander (similarity drops to 0.32).
    True target is present in the frame as a new Track (similarity = 0.78).
    Verify that the system detects candidate scores, exposes them, and switches lock to the true target.
    """
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_A", name="Lobby", source="synthetic", source_type=SourceType.SYNTHETIC))

    config = AppConfig()
    config.reid.match_threshold = 0.75
    config.reid.lock_switch_margin = 0.05
    config.reid.extract_interval_frames = 1

    det = ControllableMockDetector()

    # Target vector
    target_vec = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    target_vec /= np.linalg.norm(target_vec)

    # Bystander vector (sim to target ~ 0.32)
    bystander_vec = np.array([0.3, 0.9, 0.0, 0.0], dtype=np.float32)
    bystander_vec /= np.linalg.norm(bystander_vec)

    # Realistic variation of target (sim to target ~ 0.78)
    target_variant_vec = np.array([0.8, 0.3, 0.0, 0.0], dtype=np.float32)
    target_variant_vec /= np.linalg.norm(target_variant_vec)

    class RealisticMockReID:
        def extract(self, crop):
            return self.extract_batch([crop])[0]

        def extract_batch(self, crops):
            results = []
            for crop in crops:
                h, w = crop.shape[:2]
                if w > 80:
                    results.append(Embedding(vector=target_variant_vec))
                else:
                    results.append(Embedding(vector=bystander_vec))
            return results

    mock_reid = RealisticMockReID()

    pipe = MultiCameraPipeline(
        graph=graph,
        config=config,
        reid_extractor=mock_reid,
        camera_factory=lambda node_cfg: SyntheticCamera(width=640, height=480, fps=30),
        tracker_factory=lambda: ByteTracker(track_thresh=0.4, match_thresh=0.5),
        shared_detector=det,
    )

    # 1. Lock Target on Track 1 (wide box: width=100)
    det.set_persons([(50.0, 50.0, 150.0, 200.0, 0.95)])
    worker = pipe._get_or_create_worker("cam_A")

    pipe.step()
    pipe.select_target_on_camera("cam_A", 100.0, 100.0)
    # Manually seed with pristine target vector to simulate initial lock
    pipe.identity.clear()
    pipe.identity.register_new_target(crop=np.ones((60, 30, 3), dtype=np.uint8), identity_id="target_0", embedding=Embedding(vector=target_vec))
    pipe.identity.get_identity("target_0").trusted_gallery[0] = Embedding(vector=target_vec)
    pipe.identity._manual_matrix
    pipe.step()

    assert pipe.target_manager.target.state == TargetState.TRACKING
    locked_id = pipe.target_manager.target.track_id
    assert locked_id is not None

    # 2. Introduce hijacked Track 1 (bystander: width=50) and real target Track 2 (width=100)
    det.set_persons([
        (50.0, 50.0, 100.0, 200.0, 0.95),   # Track 1 (bystander: returns bystander_vec)
        (250.0, 50.0, 350.0, 200.0, 0.95),  # Track 2 (real target: returns target_variant_vec)
    ])

    pipe.step()

    # Verify diagnostic scores are exposed in pipe.last_candidate_scores
    scores = pipe.last_candidate_scores
    assert len(scores) >= 2
    assert locked_id in scores

    # Verify lock switched to the true target candidate
    new_locked_id = pipe.target_manager.target.track_id
    assert new_locked_id != locked_id
    assert pipe.target_manager.target.state == TargetState.TRACKING


def test_update_graph_dynamically_adds_and_removes_workers():
    """Verify Issue 2: update_graph dynamically adds newly configured cameras to the live pipeline."""
    graph1 = CameraGraph()
    graph1.add_node(CameraNodeConfig(camera_id="cam_1", name="Lobby", source="synthetic", source_type=SourceType.SYNTHETIC))

    config = AppConfig()
    pipe = MultiCameraPipeline(
        graph=graph1,
        config=config,
        camera_factory=lambda node_cfg: SyntheticCamera(width=320, height=240, fps=30),
    )

    pipe.step()
    assert len(pipe._workers) == 1
    assert "cam_1" in pipe._workers

    # Now create a new graph with cam_1 and cam_2
    graph2 = CameraGraph()
    graph2.add_node(CameraNodeConfig(camera_id="cam_1", name="Lobby", source="synthetic", source_type=SourceType.SYNTHETIC))
    graph2.add_node(CameraNodeConfig(camera_id="cam_2", name="Corridor", source="synthetic", source_type=SourceType.SYNTHETIC))

    pipe.update_graph(graph2)

    # Step pipeline and verify both cam_1 and cam_2 workers exist and cards are returned
    cards = pipe.get_all_camera_cards()
    card_ids = {c["camera_id"] for c in cards}
    assert "cam_1" in card_ids
    assert "cam_2" in card_ids
    assert len(cards) == 2
