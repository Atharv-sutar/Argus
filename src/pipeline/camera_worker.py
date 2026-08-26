"""Per-camera acquisition, detection, and tracking worker."""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple
import numpy as np

from src.core.interfaces import BaseCamera, BaseDetector, BaseTracker
from src.core.types import BoundingBox, DetectionResult, Target, TargetState, TrackResult
from src.visualization.annotator import FrameAnnotator

logger = logging.getLogger(__name__)


class CameraWorker:
    """
    Manages frame acquisition, object detection, and motion tracking for a single camera node.
    Decoupled from global target identity matching and multi-camera orchestration.
    """

    def __init__(
        self,
        camera: BaseCamera,
        detector: BaseDetector,
        tracker: BaseTracker,
        annotator: Optional[FrameAnnotator] = None,
        camera_id: str = "camera_0",
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.annotator = annotator or FrameAnnotator()
        self.camera_id = camera_id

        self.target_evaluation_enabled: bool = True
        self.current_target: Target = Target(state=TargetState.UNSELECTED)
        self._fps: float = 0.0
        self._last_frame_time: float = 0.0
        self._last_frame: Optional[np.ndarray] = None
        self._last_track_result: Optional[TrackResult] = None

    @property
    def fps(self) -> float:
        return self._fps

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Reads a raw frame from the underlying camera stream."""
        if not self.camera.is_opened():
            return False, None, 0.0
        return self.camera.read()

    def process_frame(
        self, frame: np.ndarray, timestamp_ms: float
    ) -> Tuple[DetectionResult, TrackResult]:
        """
        Executes YOLO detection and ByteTrack motion tracking on a single frame.
        """
        # Calculate instantaneous FPS
        now = time.time()
        if self._last_frame_time > 0:
            dt = now - self._last_frame_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps = 0.9 * self._fps + 0.1 * inst_fps if self._fps > 0 else inst_fps
        self._last_frame_time = now

        # 1. Detection
        det_result = self.detector.detect(frame, timestamp_ms=timestamp_ms)

        # 2. Tracking
        track_result = self.tracker.update(det_result, frame)

        self._last_frame = frame
        self._last_track_result = track_result
        return det_result, track_result

    def annotate(
        self,
        frame: np.ndarray,
        track_result: TrackResult,
        target: Optional[Target] = None,
        candidate_similarities: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Draws bounding boxes, IDs, target overlays, and real-time similarity metrics on the frame."""
        if target is not None:
            self.current_target = target
        else:
            self.current_target = Target(state=TargetState.UNSELECTED)
        return self.annotator.annotate(
            frame=frame,
            track_result=track_result,
            target=target,
            fps=self._fps,
            camera_id=self.camera_id,
            candidate_similarities=candidate_similarities,
        )

    def extract_crop(self, frame: np.ndarray, box: BoundingBox) -> Optional[np.ndarray]:
        """Safely extracts a bounded person crop from the frame."""
        if frame is None or box is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box.as_xyxy())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def stop(self) -> None:
        """Stops camera stream and releases resources."""
        if self.camera is not None:
            self.camera.release()
            logger.info(f"[CAMERA_WORKER] Camera '{self.camera_id}' released.")
