"""Unit tests for tracking module."""

from src.core.types import BoundingBox, Detection, DetectionResult, TrackState
from src.tracking.byte_tracker import ByteTracker


def test_byte_tracker_lifecycle():
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.5, track_buffer=5)

    # Frame 1: Detection at (100, 100, 150, 200)
    det1 = Detection(
        box=BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=200.0, confidence=0.9),
        class_id=0,
        confidence=0.9
    )
    res1 = tracker.update(DetectionResult(detections=[det1], frame_id=1, timestamp_ms=33.3))
    assert res1.count == 1
    track_id_1 = res1.tracks[0].track_id
    assert track_id_1 == 1

    # Frame 2: Same target moved slightly to (105, 102, 155, 202)
    det2 = Detection(
        box=BoundingBox(x1=105.0, y1=102.0, x2=155.0, y2=202.0, confidence=0.88),
        class_id=0,
        confidence=0.88
    )
    res2 = tracker.update(DetectionResult(detections=[det2], frame_id=2, timestamp_ms=66.6))
    assert res2.count == 1
    # Track ID should be persistently maintained
    assert res2.tracks[0].track_id == track_id_1
    assert res2.tracks[0].hits == 2
    assert res2.tracks[0].state == TrackState.TRACKED


def test_byte_tracker_reset():
    tracker = ByteTracker()
    det = Detection(
        box=BoundingBox(x1=50.0, y1=50.0, x2=100.0, y2=150.0, confidence=0.8),
        class_id=0,
        confidence=0.8
    )
    tracker.update(DetectionResult(detections=[det], frame_id=1))
    tracker.reset()
    assert tracker._frame_id == 0
    assert len(tracker._tracks) == 0
