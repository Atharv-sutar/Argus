"""Single-camera real-time processing pipeline."""

from __future__ import annotations

import logging
import time
from typing import Generator, Optional, Tuple
import numpy as np

from src.core.config import AppConfig
from src.core.interfaces import BaseCamera, BaseDetector, BaseTracker
from src.core.types import DetectionResult, TrackResult
from src.visualization.annotator import FrameAnnotator

logger = logging.getLogger(__name__)


class SingleCameraPipeline:
    """
    Orchestrates real-time single-camera processing:
    Camera -> Detection -> Tracking -> Visualization
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

        self._frame_id = 0
        self._fps = 0.0
        self._last_time = time.time()
        self._fps_smoothing = 0.9

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float
    ) -> Tuple[DetectionResult, TrackResult, np.ndarray]:
        """
        Processes a single video frame through detection, tracking, and visualization.

        Args:
            frame: Input BGR image array.
            timestamp_ms: Capture timestamp in milliseconds.

        Returns:
            Tuple[DetectionResult, TrackResult, np.ndarray]:
                - Detection output
                - Tracking output
                - Annotated output frame
        """
        self._frame_id += 1

        # 1. Detection
        det_result = self.detector.detect(
            frame=frame,
            frame_id=self._frame_id,
            timestamp_ms=timestamp_ms
        )

        # 2. Tracking
        track_result = self.tracker.update(
            detection_result=det_result,
            frame=frame
        )

        # 3. Calculate dynamic FPS
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        current_fps = (1.0 / dt) if dt > 0 else 0.0
        self._fps = self._fps * self._fps_smoothing + current_fps * (1.0 - self._fps_smoothing)

        # 4. Visualization Annotation
        annotated_frame = self.annotator.annotate(
            frame=frame,
            track_result=track_result,
            fps=self._fps,
            camera_id=self.camera_id
        )

        return det_result, track_result, annotated_frame

    def stream(self) -> Generator[Tuple[np.ndarray, TrackResult], None, None]:
        """
        Generator yielding (annotated_frame, track_result) continuously from the camera.
        """
        while self.camera.is_opened():
            success, frame, timestamp_ms = self.camera.read()
            if not success or frame is None:
                logger.info("Camera stream ended or read failed.")
                break

            _, track_result, annotated_frame = self.process_frame(frame, timestamp_ms)
            yield annotated_frame, track_result

    def stop(self) -> None:
        """Stops the pipeline and releases resources."""
        self.camera.release()
        logger.info("SingleCameraPipeline stopped and camera released.")
