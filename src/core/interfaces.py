"""Core interfaces and contracts for Argus subsystems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

from src.core.types import DetectionResult, TrackResult


class BaseCamera(ABC):
    """Abstract interface for video/camera acquisition sources."""

    @abstractmethod
    def is_opened(self) -> bool:
        """Return True if the camera stream is currently active/open."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Acquire the next frame from the camera stream.

        Returns:
            Tuple[bool, Optional[np.ndarray], float]:
                - success: True if a frame was read, False on error or EOF.
                - frame: BGR image array or None on failure.
                - timestamp_ms: Frame acquisition timestamp in milliseconds.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release any hardware or stream resources."""
        pass


class BaseDetector(ABC):
    """Abstract interface for object/person detection models."""

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0
    ) -> DetectionResult:
        """
        Run detection on a single frame.

        Args:
            frame: Input image array (BGR, uint8).
            frame_id: Monotonically increasing frame index.
            timestamp_ms: Capture timestamp in milliseconds.

        Returns:
            DetectionResult: Detected bounding boxes and metadata.
        """
        pass


class BaseTracker(ABC):
    """Abstract interface for multi-object tracking algorithms."""

    @abstractmethod
    def update(
        self,
        detection_result: DetectionResult,
        frame: Optional[np.ndarray] = None
    ) -> TrackResult:
        """
        Update tracking state with new detections for the current frame.

        Args:
            detection_result: Detections output from the detector.
            frame: Optional image array (for trackers requiring visual cues).

        Returns:
            TrackResult: Active persistent tracks with stable IDs.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracker state (e.g. on stream restart)."""
        pass
