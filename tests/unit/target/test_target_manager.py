"""Unit tests for TargetManager."""

import pytest
from src.core.types import BoundingBox, TargetState, Track, TrackResult, TrackState
from src.target.manager import TargetManager


def test_target_selection_by_track_id():
    tm = TargetManager()
    assert tm.target.state == TargetState.UNSELECTED

    box = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
    track = Track(track_id=1, box=box, state=TrackState.TRACKED)
    res = TrackResult(tracks=[track], frame_id=1, timestamp_ms=33.3)

    tm.select_by_track_id(1, res)
    assert tm.target.track_id == 1
    assert tm.target.state == TargetState.LOCKED
    assert tm.target.last_known_box == box


def test_target_selection_by_point():
    tm = TargetManager()
    box1 = BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=250.0)
    box2 = BoundingBox(x1=300.0, y1=100.0, x2=400.0, y2=300.0)
    t1 = Track(track_id=10, box=box1)
    t2 = Track(track_id=20, box=box2)
    res = TrackResult(tracks=[t1, t2], frame_id=1, timestamp_ms=33.3)

    # Click inside box 1
    selected_id = tm.select_by_point(100.0, 150.0, res)
    assert selected_id == 10
    assert tm.target.track_id == 10
    assert tm.target.state == TargetState.LOCKED

    # Click empty space
    tm.clear()
    assert tm.target.state == TargetState.UNSELECTED
    empty_click = tm.select_by_point(500.0, 500.0, res)
    assert empty_click is None
    assert tm.target.state == TargetState.UNSELECTED


def test_target_lifecycle_and_loss():
    tm = TargetManager(lost_timeout_ms=500.0)
    box = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
    t1 = Track(track_id=5, box=box)
    res1 = TrackResult(tracks=[t1], frame_id=1, timestamp_ms=100.0)

    # Select target
    tm.select_by_track_id(5, res1)
    assert tm.target.state == TargetState.LOCKED

    # Frame 2: Target visible -> TRACKING
    t1_moved = Track(track_id=5, box=BoundingBox(x1=105.0, y1=100.0, x2=205.0, y2=300.0))
    res2 = TrackResult(tracks=[t1_moved], frame_id=2, timestamp_ms=133.3)
    tm.update(res2)
    assert tm.target.state == TargetState.TRACKING
    assert tm.target.lost_duration_ms == 0.0

    # Frame 3: Target lost temporarily
    res3 = TrackResult(tracks=[], frame_id=3, timestamp_ms=200.0)
    tm.update(res3)
    assert tm.target.state == TargetState.LOST
    assert tm.target.lost_duration_ms == pytest.approx(66.7, 0.01)

    # Frame 4: Target exceeds lost timeout
    res4 = TrackResult(tracks=[], frame_id=4, timestamp_ms=700.0)
    tm.update(res4)
    assert tm.target.state == TargetState.LOST
    assert tm.target.lost_duration_ms >= 500.0


def test_target_spatial_reassociation():
    tm = TargetManager(reassociation_iou_thresh=0.3)
    box = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
    t1 = Track(track_id=1, box=box)
    res1 = TrackResult(tracks=[t1], frame_id=1, timestamp_ms=100.0)
    tm.select_by_track_id(1, res1)
    tm.update(res1)

    # Frame 2: Track ID 1 dropped, but new Track ID 2 appears with overlapping box
    box_new = BoundingBox(x1=102.0, y1=101.0, x2=202.0, y2=301.0)
    t2 = Track(track_id=2, box=box_new)
    res2 = TrackResult(tracks=[t2], frame_id=2, timestamp_ms=133.3)

    tm.update(res2)
    # Should re-associate target to ID 2
    assert tm.target.track_id == 2
    assert tm.target.state == TargetState.TRACKING
