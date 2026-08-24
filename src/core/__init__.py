"""Core data types and interfaces for Argus surveillance system."""

from src.core.types import (
    BoundingBox,
    Detection,
    DetectionResult,
    Embedding,
    FrameData,
    Identity,
    Target,
    TargetState,
    Track,
    TrackResult,
    TrackState,
)
from src.core.interfaces import (
    BaseCamera,
    BaseDetector,
    BaseReID,
    BaseTracker,
    BaseVectorStore,
)

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "Embedding",
    "FrameData",
    "Identity",
    "Target",
    "TargetState",
    "Track",
    "TrackResult",
    "TrackState",
    "BaseCamera",
    "BaseDetector",
    "BaseReID",
    "BaseTracker",
    "BaseVectorStore",
]
