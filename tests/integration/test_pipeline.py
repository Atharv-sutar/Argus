"""Integration tests for the single-camera pipeline with target tracking and ReID."""

from typing import Optional
import numpy as np
import pytest

from src.camera.capture import SyntheticCamera
from src.core.interfaces import BaseDetector, BaseReID
from src.core.types import BoundingBox, Detection, DetectionResult, Embedding, TargetState
from src.identity.manager import IdentityManager
from src.pipeline.single_camera import SingleCameraPipeline
from src.target.manager import TargetManager
from src.tracking.byte_tracker import ByteTracker
from src.visualization.annotator import FrameAnnotator


class MockDetector(BaseDetector):
    """Simple deterministic detector for integration testing."""

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0
    ) -> DetectionResult:
        cx = int(100 + (frame_id * 2) % 200)
        box = BoundingBox(
            x1=float(cx - 30),
            y1=100.0,
            x2=float(cx + 30),
            y2=220.0,
            confidence=0.92
        )
        det = Detection(box=box, class_id=0, class_name="person", confidence=0.92)
        return DetectionResult(detections=[det], frame_id=frame_id, timestamp_ms=timestamp_ms)


class MockReID(BaseReID):
    """Deterministic mock ReID for integration testing."""

    def extract(self, crop: np.ndarray) -> Embedding:
        if crop is None or crop.size == 0:
            return Embedding(vector=np.zeros(4, dtype=np.float32))
        mean_val = float(np.mean(crop))
        angle = (mean_val / 255.0) * (np.pi / 2.0)
        return Embedding(vector=np.array([np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=np.float32))

    def extract_batch(self, crops: list[np.ndarray]) -> list[Embedding]:
        return [self.extract(c) for c in crops]


def test_single_camera_pipeline_with_target():
    camera = SyntheticCamera(width=320, height=240, fps=30, max_frames=5)
    detector = MockDetector()
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.5)
    target_manager = TargetManager()
    annotator = FrameAnnotator()

    pipeline = SingleCameraPipeline(
        camera=camera,
        detector=detector,
        tracker=tracker,
        target_manager=target_manager,
        annotator=annotator,
        camera_id="test_cam"
    )

    # Process first frame
    success, frame, ts = camera.read()
    det_res, track_res, target, annotated = pipeline.process_frame(frame, ts)
    assert track_res.count == 1
    track_id = track_res.tracks[0].track_id

    # Select target by point click
    selected_id = pipeline.select_target_by_point(100.0, 150.0)
    assert selected_id == track_id
    assert pipeline.current_target.state == TargetState.LOCKED

    # Process remaining frames and verify target stays in TRACKING state
    for annotated_frame, track_result, target in pipeline.stream():
        assert target.state == TargetState.TRACKING
        assert target.track_id == track_id

    pipeline.stop()


def test_single_camera_pipeline_with_reid_recovery():
    camera = SyntheticCamera(width=320, height=240, fps=30, max_frames=5)
    detector = MockDetector()
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.5)
    target_manager = TargetManager()
    reid = MockReID()
    identity_manager = IdentityManager(reid_extractor=reid, similarity_threshold=0.8)

    pipeline = SingleCameraPipeline(
        camera=camera,
        detector=detector,
        tracker=tracker,
        target_manager=target_manager,
        identity_manager=identity_manager,
        reid_interval=1,
    )

    # Frame 1: track target
    success, frame, ts = camera.read()
    det_res, track_res, target, annotated = pipeline.process_frame(frame, ts)
    track_id = track_res.tracks[0].track_id
    pipeline.select_target_by_point(100.0, 150.0)

    # Immediate registration should have captured reference appearance
    ident = identity_manager.get_identity("target_0")
    assert ident is not None
    assert ident.reference_embedding is not None
    assert len(ident.embeddings) > 0

    # Frame 2: process frame and verify target continues TRACKING
    success, frame, ts = camera.read()
    _, _, target, _ = pipeline.process_frame(frame, ts)
    assert target.state == TargetState.TRACKING

    pipeline.stop()


def test_target_leaves_frame_b_rejected_a_recovered():
    """
    Direct verification of the manual failure scenario:
    1. Frame 1: Person A and Person B are both visible. User selects Person A.
    2. Frame 2: Person A leaves frame, only Person B remains.
       Result: Target must be LOST. Person B must NEVER become target.
    3. Frame 3: Person A returns (both A and B visible).
       Result: Person A is recognized and recovered as target.
    """
    class TwoPersonDetector(BaseDetector):
        def __init__(self):
            self.mode = "BOTH_VISIBLE"

        def detect(self, frame, frame_id=0, timestamp_ms=0.0):
            dets = []
            if self.mode in ("BOTH_VISIBLE", "ONLY_A"):
                # Person A on left: (50, 50, 100, 200)
                box_a = BoundingBox(50.0, 50.0, 100.0, 200.0, confidence=0.95)
                dets.append(Detection(box=box_a, class_id=0, class_name="person", confidence=0.95))
            if self.mode in ("BOTH_VISIBLE", "ONLY_B"):
                # Person B on right: (200, 50, 250, 200)
                box_b = BoundingBox(200.0, 50.0, 250.0, 200.0, confidence=0.95)
                dets.append(Detection(box=box_b, class_id=0, class_name="person", confidence=0.95))
            return DetectionResult(detections=dets, frame_id=frame_id, timestamp_ms=timestamp_ms)

    class ColorDiscriminativeReID(BaseReID):
        def extract(self, crop: np.ndarray) -> Embedding:
            # Person A (left region) has distinct blue/green tone; Person B (right region) has dark tone
            # Mean color of crop determines embedding
            mean_c = np.mean(crop, axis=(0, 1)) if (crop is not None and crop.size > 0) else np.zeros(3)
            # Create normalized vector where blue/green is distinct from dark/red
            vec = np.array([mean_c[0], mean_c[1], mean_c[2], 100.0], dtype=np.float32)
            return Embedding(vector=vec)

        def extract_batch(self, crops: list[np.ndarray]) -> list[Embedding]:
            return [self.extract(c) for c in crops]

    camera = SyntheticCamera(width=320, height=240, fps=30, max_frames=10)
    detector = TwoPersonDetector()
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.5)
    reid = ColorDiscriminativeReID()
    identity_manager = IdentityManager(reid_extractor=reid, similarity_threshold=0.85, min_margin=0.08)

    pipeline = SingleCameraPipeline(
        camera=camera,
        detector=detector,
        tracker=tracker,
        identity_manager=identity_manager,
        reid_interval=1,
    )

    # Frame 1: Person A and B visible. Paint Person A blue and Person B red on frame
    frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
    frame1[50:200, 50:100] = [255, 50, 50]   # Person A: Blue
    frame1[50:200, 200:250] = [50, 50, 255]  # Person B: Red

    det_res, track_res, target, _ = pipeline.process_frame(frame1, timestamp_ms=100.0)
    assert track_res.count == 2
    # Select Person A (x=75, y=100)
    track_id_a = pipeline.select_target_by_point(75.0, 100.0)
    assert track_id_a is not None
    assert pipeline.current_target.state == TargetState.LOCKED

    # Frame 2: Person A leaves. Only Person B is visible
    detector.mode = "ONLY_B"
    frame2 = np.zeros((240, 320, 3), dtype=np.uint8)
    frame2[50:200, 200:250] = [50, 50, 255]  # Person B: Red

    _, track_res2, target2, _ = pipeline.process_frame(frame2, timestamp_ms=133.3)
    # Target MUST be LOST. Person B must NEVER be accepted!
    assert target2.state == TargetState.LOST
    assert target2.track_id == track_id_a  # Logical target still refers to A

    # Frame 3+: Person A enters again. Both A and B are visible
    detector.mode = "BOTH_VISIBLE"
    frame3 = np.zeros((240, 320, 3), dtype=np.uint8)
    frame3[50:200, 50:100] = [255, 50, 50]   # Person A: Blue
    frame3[50:200, 200:250] = [50, 50, 255]  # Person B: Red

    target3 = None
    for i in range(4):
        _, track_res3, target3, _ = pipeline.process_frame(frame3, timestamp_ms=166.6 + i * 33.3)

    # Person A MUST be recovered after temporal consensus!
    assert target3 is not None
    assert target3.state == TargetState.TRACKING
    assert target3.last_known_box.x1 == pytest.approx(50.0, abs=5.0)

    pipeline.stop()


def test_id_swap_recovery():
    """
    Verifies that if a tracker accidentally swaps the track ID onto Person B (track scoop),
    the pipeline detects the ReID mismatch and reassociates to Person A after temporal consensus.
    """
    class SwappedTracker:
        """Mock tracker to simulate an ID swap between Person A and Person B."""
        def __init__(self):
            self.swapped = False

        def update(self, detection_result, frame=None):
            from src.core.types import Track, TrackResult
            tracks = []
            if not self.swapped:
                # Normal: Track 1 = Person A (left), Track 2 = Person B (right)
                tracks.append(Track(track_id=1, box=BoundingBox(50.0, 50.0, 100.0, 200.0, confidence=0.95), confidence=0.95))
                tracks.append(Track(track_id=2, box=BoundingBox(200.0, 50.0, 250.0, 200.0, confidence=0.95), confidence=0.95))
            else:
                # Swapped: Track 1 = Person B (right), Track 2 = Person A (left)
                tracks.append(Track(track_id=1, box=BoundingBox(200.0, 50.0, 250.0, 200.0, confidence=0.95), confidence=0.95))
                tracks.append(Track(track_id=2, box=BoundingBox(50.0, 50.0, 100.0, 200.0, confidence=0.95), confidence=0.95))
            return TrackResult(tracks=tracks, frame_id=detection_result.frame_id, timestamp_ms=detection_result.timestamp_ms)

    class DummyDetector(BaseDetector):
        def detect(self, frame, frame_id=0, timestamp_ms=0.0):
            return DetectionResult(detections=[], frame_id=frame_id, timestamp_ms=timestamp_ms)

    class ColorDiscriminativeReID(BaseReID):
        def extract(self, crop: np.ndarray) -> Embedding:
            mean_c = np.mean(crop, axis=(0, 1)) if (crop is not None and crop.size > 0) else np.zeros(3)
            vec = np.array([mean_c[0], mean_c[1], mean_c[2], 100.0], dtype=np.float32)
            return Embedding(vector=vec)

        def extract_batch(self, crops: list[np.ndarray]) -> list[Embedding]:
            return [self.extract(c) for c in crops]

    camera = SyntheticCamera(width=320, height=240, fps=30, max_frames=10)
    detector = DummyDetector()
    tracker = SwappedTracker()
    reid = ColorDiscriminativeReID()
    identity_manager = IdentityManager(reid_extractor=reid, similarity_threshold=0.85, reacquisition_threshold=0.85, min_margin=0.05)

    pipeline = SingleCameraPipeline(
        camera=camera,
        detector=detector,
        tracker=tracker,
        identity_manager=identity_manager,
        reid_interval=1,
    )

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[50:200, 50:100] = [255, 50, 50]   # Person A: Blue
    frame[50:200, 200:250] = [50, 50, 255]  # Person B: Red

    # Frame 1: Select Person A (Track ID 1)
    _, track_res1, target1, _ = pipeline.process_frame(frame, timestamp_ms=100.0)
    pipeline.select_target_by_id(1)
    assert pipeline.current_target.track_id == 1
    assert pipeline.current_target.state == TargetState.LOCKED

    # Frame 2+: Tracker swaps! Track 1 is now on Person B (Red), Track 2 is on Person A (Blue)
    tracker.swapped = True
    target2 = None
    for i in range(4):
        _, track_res2, target2, _ = pipeline.process_frame(frame, timestamp_ms=133.3 + i * 33.3)

    # Verification: The pipeline detects that Track 1 is not the target, sweeps candidates,
    # finds Track 2 matches Person A, and reassociates to Track 2!
    assert target2 is not None
    assert target2.state == TargetState.TRACKING
    assert target2.track_id == 2
    assert target2.last_known_box.x1 == pytest.approx(50.0, abs=5.0)

    pipeline.stop()

