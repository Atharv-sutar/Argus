"""Integration tests for the single-camera pipeline with target tracking."""

from typing import Optional
import numpy as np

from src.camera.capture import SyntheticCamera
from src.core.interfaces import BaseDetector
from src.core.types import BoundingBox, Detection, DetectionResult, TargetState
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
        # Detect a simulated person box in the frame
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
