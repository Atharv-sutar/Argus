"""Integration tests for the single-camera pipeline."""

from typing import Optional
import numpy as np

from src.camera.capture import SyntheticCamera
from src.core.interfaces import BaseDetector
from src.core.types import BoundingBox, Detection, DetectionResult
from src.pipeline.single_camera import SingleCameraPipeline
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


def test_single_camera_pipeline_execution():
    camera = SyntheticCamera(width=320, height=240, fps=30, max_frames=5)
    detector = MockDetector()
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.5)
    annotator = FrameAnnotator()

    pipeline = SingleCameraPipeline(
        camera=camera,
        detector=detector,
        tracker=tracker,
        annotator=annotator,
        camera_id="test_cam"
    )

    processed_frames = 0
    for annotated_frame, track_result in pipeline.stream():
        processed_frames += 1
        assert annotated_frame is not None
        assert annotated_frame.shape == (240, 320, 3)
        assert track_result.count >= 1
        # Check that track ID is maintained across frames
        assert track_result.tracks[0].track_id == 1

    assert processed_frames == 5
    pipeline.stop()
