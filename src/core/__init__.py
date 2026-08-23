"""Core data types and interfaces for Argus surveillance system."""

from src.core.types import (
    BoundingBox,
    Detection,
    DetectionResult,
    FrameData,
    Target,
    TargetState,
    Track,
    TrackResult,
    TrackState,
)
from src.core.interfaces import (
    BaseCamera,
    BaseDetector,
    BaseTracker,
)

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "FrameData",
    "Target",
    "TargetState",
    "Track",
    "TrackResult",
    "TrackState",
    "BaseCamera",
    "BaseDetector",
    "BaseTracker",
]
