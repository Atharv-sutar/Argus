"""Detection subsystem."""

from src.detection.yolo_detector import (
    DETECTOR_REGISTRY,
    YOLODetector,
    create_detector,
    register_detector_backend,
)

__all__ = [
    "DETECTOR_REGISTRY",
    "YOLODetector",
    "create_detector",
    "register_detector_backend",
]