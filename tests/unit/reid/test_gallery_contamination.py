"""Unit tests for gallery contamination resistance, manual-anchored scoring, and lock-switch rollback."""

import numpy as np
import pytest

from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.core.types import BoundingBox, Detection, DetectionResult, Embedding, GalleryEntry, TargetState
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


def test_gallery_contamination_resistance_via_manual_anchoring():
    """
    Verify that an imposter embedding in the auto gallery CANNOT hijack the lock.
    Target A is the manual seed ([1, 0, 0, 0]).
    Imposter B is injected as an auto entry ([0, 1, 0, 0]).
    A candidate presenting Imposter B vector must NOT get a high score because it fails manual anchor.
    """
    gallery = TargetGallery(match_threshold=0.75, auto_add_threshold=0.85)

    target_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    imposter_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    crop = np.ones((60, 30, 3), dtype=np.uint8)

    # 1. Seed with Target A
    gallery.seed(crop=crop, embedding=Embedding(vector=target_vec), camera_id="cam_0")
    assert gallery.manual_count == 1
    assert gallery.auto_count == 0

    # 2. Artificially insert Imposter B as an auto-enrolled entry (simulating a past contamination bug)
    imposter_entry = GalleryEntry(
        entry_id="auto_imposter_1",
        embedding=Embedding(vector=imposter_vec),
        crop=crop,
        is_manual=False,
        confidence=0.95,
        track_id=999,
    )
    gallery._entries.append(imposter_entry)
    gallery._rebuild_matrix()

    assert gallery.size == 2
    assert gallery.manual_count == 1
    assert gallery.auto_count == 1

    # 3. Present Target A candidate and Imposter B candidate
    cand_target = Embedding(vector=target_vec)
    cand_imposter = Embedding(vector=imposter_vec)

    results = gallery.match_batch_details([cand_target, cand_imposter])
    target_eff, target_man, target_auto, _ = results[0]
    imposter_eff, imposter_man, imposter_auto, _ = results[1]

    # Target A matches manual perfectly (man=1.0, eff=1.0)
    assert pytest.approx(target_man, rel=1e-3) == 1.0
    assert pytest.approx(target_eff, rel=1e-3) == 1.0

    # Imposter B matches the rogue auto-entry (auto=1.0), BUT fails manual seed (man=0.0)
    # Effective score must be bounded by man + max_auto_boost (0.0 + 0.05 = 0.05), NOT 1.0!
    assert pytest.approx(imposter_man, rel=1e-3) == 0.0
    assert pytest.approx(imposter_auto, rel=1e-3) == 1.0
    assert imposter_eff <= 0.06  # Bounded, cannot score 1.0!


def test_lock_switch_rolls_back_contaminated_entries():
    """
    Verify that when a lock switch occurs, auto-enrolled entries from the deposed track
    are purged from the gallery while protected manual entries remain intact.
    """
    gallery = TargetGallery(max_size=20, match_threshold=0.75, auto_add_threshold=0.85)

    target_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    crop = np.ones((60, 30, 3), dtype=np.uint8)

    # Seed target (manual protected)
    gallery.seed(crop=crop, embedding=Embedding(vector=target_vec), camera_id="cam_0")
    assert gallery.manual_count == 1

    # Add 2 auto-entries for Track 10
    gallery._entries.append(GalleryEntry(
        entry_id="auto_1",
        embedding=Embedding(vector=np.array([0.95, 0.1, 0.0, 0.0], dtype=np.float32)),
        crop=crop,
        is_manual=False,
        track_id=10,
    ))
    gallery._entries.append(GalleryEntry(
        entry_id="auto_2",
        embedding=Embedding(vector=np.array([0.92, 0.15, 0.0, 0.0], dtype=np.float32)),
        crop=crop,
        is_manual=False,
        track_id=10,
    ))
    gallery._rebuild_matrix()

    assert gallery.size == 3
    assert gallery.auto_count == 2

    # Purge Track 10 auto-entries (simulating lock switch away from Track 10)
    purged = gallery.rollback_auto_entries(for_track_id=10)
    assert purged == 2
    assert gallery.size == 1
    assert gallery.manual_count == 1
    assert gallery.auto_count == 0


def test_add_auto_rejects_candidate_failing_manual_anchor():
    """
    Verify that add_auto enforces the ground-truth manual anchor check and rejects
    any candidate that does not match manual entries with at least match_threshold.
    """
    gallery = TargetGallery(match_threshold=0.75, auto_add_threshold=0.85, auto_add_min_consecutive=1)

    target_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    crop = np.ones((60, 30, 3), dtype=np.uint8)

    gallery.seed(crop=crop, embedding=Embedding(vector=target_vec), camera_id="cam_0")

    # Candidate with low similarity to manual seed (0.3)
    drifted_vec = np.array([0.3, 0.95, 0.0, 0.0], dtype=np.float32)
    drifted_emb = Embedding(vector=drifted_vec)

    # Even if candidate_similarity argument is claimed to be 0.90, add_auto must check manual anchor and reject
    success = gallery.add_auto(
        crop=crop,
        embedding=drifted_emb,
        candidate_similarity=0.90,
        camera_id="cam_0",
        track_id=5,
    )
    assert success is False
    assert gallery.auto_count == 0


def test_continuity_does_not_trigger_auto_enrollment():
    """
    Verify that a track maintaining continuity with similarity 0.78 (>= match_threshold 0.75,
    but < auto_add_threshold 0.85) keeps tracking without adding entries to the gallery.
    """
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_A", name="Lobby", source="synthetic", source_type=SourceType.SYNTHETIC))

    config = AppConfig()
    config.reid.match_threshold = 0.75
    config.reid.auto_add_threshold = 0.85
    config.reid.extract_interval_frames = 1

    det = ControllableMockDetector()

    # Target vector
    target_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Candidate vector with similarity = 0.78
    sim78_vec = np.array([0.78, np.sqrt(1 - 0.78**2), 0.0, 0.0], dtype=np.float32)

    class FixedMockReID:
        def extract(self, crop):
            return self.extract_batch([crop])[0]

        def extract_batch(self, crops):
            return [Embedding(vector=sim78_vec) for _ in crops]

    mock_reid = FixedMockReID()

    pipe = MultiCameraPipeline(
        graph=graph,
        config=config,
        reid_extractor=mock_reid,
        camera_factory=lambda node_cfg: SyntheticCamera(width=640, height=480, fps=30),
        tracker_factory=lambda: ByteTracker(track_thresh=0.4, match_thresh=0.5),
    )

    # 1. Lock Target
    det.set_persons([(50.0, 50.0, 150.0, 200.0, 0.95)])
    worker = pipe._get_or_create_worker("cam_A")
    worker.detector = det

    pipe.step()
    pipe.select_target_on_camera("cam_A", 100.0, 100.0)

    # Override manual seed with pristine target_vec
    pipe.gallery.clear()
    pipe.gallery.seed(
        crop=np.ones((60, 30, 3), dtype=np.uint8),
        embedding=Embedding(vector=target_vec),
        camera_id="cam_A",
    )

    initial_gallery_size = pipe.gallery.size
    assert initial_gallery_size == 1

    # 2. Step 10 times with similarity 0.78
    for _ in range(10):
        pipe.step()

    # Target must remain in TRACKING state
    assert pipe.target_manager.target.state == TargetState.TRACKING
    # Gallery size must NOT have increased because 0.78 < auto_add_threshold (0.85)
    assert pipe.gallery.size == initial_gallery_size
    assert pipe.gallery.auto_count == 0
