"""Unit tests for YOLO detector."""

import numpy as np
import pytest
from src.core.types import DetectionResult
from src.detection.yolo_detector import YOLODetector


def test_yolo_detector_initialization():
    detector = YOLODetector(
        model_name="yolov8n.pt",
        confidence_threshold=0.4,
        iou_threshold=0.45,
        device="cpu"
    )
    assert detector.confidence_threshold == 0.4
    assert detector.target_classes == [0]
    assert detector.device == "cpu"


def test_yolo_detector_empty_frame():
    detector = YOLODetector(device="cpu")
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    res = detector.detect(empty_frame, frame_id=1, timestamp_ms=0.0)
    assert isinstance(res, DetectionResult)
    assert res.count == 0


def test_yolo_detector_synthetic_frame():
    detector = YOLODetector(device="cpu")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy_frame, frame_id=1, timestamp_ms=33.3)
    assert isinstance(res, DetectionResult)
    assert res.frame_id == 1
