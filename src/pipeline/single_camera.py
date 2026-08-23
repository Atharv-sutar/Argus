"""Single-camera real-time processing pipeline with target selection and tracking."""

from __future__ import annotations

import logging
import time
from typing import Generator, Optional, Tuple
import numpy as np

from src.core.interfaces import BaseCamera, BaseDetector, BaseTracker
from src.core.types import DetectionResult, Target, TrackResult
from src.target.manager import TargetManager
from src.visualization.annotator import FrameAnnotator

logger = logging.getLogger(__name__)


class SingleCameraPipeline:
    """
    Orchestrates real-time single-camera processing:
    Camera -> Detection -> Tracking -> Target Management -> Visualization
    """

    def __init__(
        self,
        camera: BaseCamera,
        detector: BaseDetector,
        tracker: BaseTracker,
        target_manager: Optional[TargetManager] = None,
        annotator: Optional[FrameAnnotator] = None,
        camera_id: str = "camera_0",
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.target_manager = target_manager or TargetManager()
        self.annotator = annotator or FrameAnnotator()
        self.camera_id = camera_id

        self._frame_id = 0
        self._fps = 0.0
        self._last_time = time.time()
        self._fps_smoothing = 0.9
        self._last_track_result: Optional[TrackResult] = None

    @property
    def current_target(self) -> Target:
        """Returns the current target state."""
        return self.target_manager.target

    def select_target_by_point(self, x: float, y: float) -> Optional[int]:
        """Manually select a target at pixel coordinates (x, y)."""
        if self._last_track_result is not None:
            return self.target_manager.select_by_point(x, y, self._last_track_result)
        return None

    def select_target_by_id(self, track_id: int) -> bool:
        """Manually select and lock onto a specific track ID."""
        return self.target_manager.select_by_track_id(track_id, self._last_track_result)

    def clear_target(self) -> None:
        """Deselect the current target."""
        self.target_manager.clear()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float
    ) -> Tuple[DetectionResult, TrackResult, Target, np.ndarray]:
        """
        Processes a single video frame through detection, tracking, target management, and visualization.

        Args:
            frame: Input BGR image array.
            timestamp_ms: Capture timestamp in milliseconds.

        Returns:
            Tuple[DetectionResult, TrackResult, Target, np.ndarray]:
                - Detection output
                - Tracking output
                - Target state
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
        self._last_track_result = track_result

        # 3. Target Management
        target = self.target_manager.update(track_result)

        # 4. Calculate dynamic FPS
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        current_fps = (1.0 / dt) if dt > 0 else 0.0
        self._fps = self._fps * self._fps_smoothing + current_fps * (1.0 - self._fps_smoothing)

        # 5. Visualization Annotation
        annotated_frame = self.annotator.annotate(
            frame=frame,
            track_result=track_result,
            target=target,
            fps=self._fps,
            camera_id=self.camera_id
        )

        return det_result, track_result, target, annotated_frame

    def stream(self) -> Generator[Tuple[np.ndarray, TrackResult, Target], None, None]:
        """
        Generator yielding (annotated_frame, track_result, target) continuously from the camera.
        """
        while self.camera.is_opened():
            success, frame, timestamp_ms = self.camera.read()
            if not success or frame is None:
                logger.info("Camera stream ended or read failed.")
                break

            _, track_result, target, annotated_frame = self.process_frame(frame, timestamp_ms)
            yield annotated_frame, track_result, target

    def stop(self) -> None:
        """Stops the pipeline and releases resources."""
        self.camera.release()
        logger.info("SingleCameraPipeline stopped and camera released.")
