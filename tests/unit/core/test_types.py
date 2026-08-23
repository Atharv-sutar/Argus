"""Unit tests for core data types."""

import pytest
from src.core.types import BoundingBox, Detection, DetectionResult, Track, TrackResult, TrackState


def test_bounding_box_validations():
    box = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0, confidence=0.95)
    assert box.width == 40.0
    assert box.height == 60.0
    assert box.area == 2400.0
    assert box.center == (30.0, 50.0)
    assert box.as_xyxy() == (10.0, 20.0, 50.0, 80.0)
    assert box.as_xywh() == (10.0, 20.0, 40.0, 60.0)

    # Invalid coordinates
    with pytest.raises(ValueError):
        BoundingBox(x1=50.0, y1=20.0, x2=10.0, y2=80.0)

    with pytest.raises(ValueError):
        BoundingBox(x1=10.0, y1=80.0, x2=50.0, y2=20.0)

    # Invalid confidence
    with pytest.raises(ValueError):
        BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0, confidence=1.5)


def test_bounding_box_iou():
    box1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    box2 = BoundingBox(x1=5.0, y1=0.0, x2=15.0, y2=10.0)

    # Intersection: [5, 0] to [10, 10] -> area 50
    # Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 1/3
    assert pytest.approx(box1.iou(box2), 0.001) == 1.0 / 3.0

    # Non-overlapping boxes
    box3 = BoundingBox(x1=20.0, y1=20.0, x2=30.0, y2=30.0)
    assert box1.iou(box3) == 0.0


def test_detection_and_result():
    box = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0, confidence=0.85)
    det = Detection(box=box, class_id=0, class_name="person", confidence=0.85)
    result = DetectionResult(detections=[det], frame_id=1, timestamp_ms=33.3)

    assert result.count == 1
    assert result.detections[0].class_name == "person"
    assert result.frame_id == 1


def test_track_and_result():
    box = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0, confidence=0.9)
    track = Track(
        track_id=1,
        box=box,
        class_id=0,
        confidence=0.9,
        state=TrackState.TRACKED,
        age=5,
        hits=5,
    )
    result = TrackResult(tracks=[track], frame_id=5, timestamp_ms=166.5)

    assert result.count == 1
    assert result.tracks[0].track_id == 1
    assert result.tracks[0].state == TrackState.TRACKED
