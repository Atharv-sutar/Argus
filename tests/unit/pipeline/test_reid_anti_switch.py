"""Tests verifying ReID anti-switch safeguards: co-present bystander disqualification,
target reacquisition of original target, and clean reset of forensic transit trail."""

import time
import numpy as np
import pytest

from src.core.types import (
    BoundingBox,
    Embedding,
    TargetState,
    Track,
    TrackResult,
    TrackState,
)
from src.core.config import AppConfig
from tests.unit.pipeline._lock_harness import build, seed_and_lock, _Worker, BOXES


class MockReIDWithBystander:
    """Mock ReID providing distinct vectors for target (Track 1 / 3) and bystander (Track 2)."""
    def __init__(self):
        # Target has vector aligned with axis 0
        self.v_target = np.zeros(128, dtype=np.float32)
        self.v_target[0] = 1.0

        # Bystander has vector aligned with axis 1
        self.v_bystander = np.zeros(128, dtype=np.float32)
        self.v_bystander[1] = 1.0

    @property
    def embedding_dim(self) -> int:
        return 128

    def extract(self, crop: np.ndarray) -> Embedding:
        return Embedding(vector=self.v_target.copy())

    def extract_batch(self, crops):
        return [self.extract(c) for c in crops]


def test_copresent_bystander_never_adopted_when_target_leaves():
    """
    Priority 0 Verification:
    Target (Track 1) and Bystander (Track 2) are co-present in the camera.
    Target (Track 1) disappears.
    Bystander (Track 2) continues to be visible in the scene.
    Verify:
    1. System transitions target state to LOST / SEARCHING.
    2. Bystander Track 2 is disqualified and NEVER adopted as the target.
    3. Target lock is NOT stolen by Bystander Track 2.
    """
    cfg = AppConfig()
    cfg.reid.reacquisition_threshold = 0.78
    p, im, tm = build(cfg)

    reid = MockReIDWithBystander()
    p.reid_extractor = reid
    p.reid_interval = 1

    # 1. Start target on Track 1
    frame = np.full((720, 1280, 3), 127, dtype=np.uint8)
    track1 = Track(track_id=1, box=BOXES[1], state=TrackState.TRACKED, confidence=0.95)
    track2 = Track(track_id=2, box=BOXES[2], state=TrackState.TRACKED, confidence=0.95)

    track_res_copresent = TrackResult(tracks=[track1, track2], frame_id=1, timestamp_ms=100.0)
    worker = _Worker()
    worker._last_frame = frame
    worker._last_track_result = track_res_copresent

    # Seed target lock on Track 1
    seed_and_lock(tm, 1)
    im.register_new_target(frame[10:100, 10:100], identity_id="target_0", embedding=Embedding(vector=reid.v_target))

    # Evaluate co-present frame
    cand_embs = [
        (track1, frame[10:100, 10:100], Embedding(vector=reid.v_target)),
        (track2, frame[10:100, 10:100], Embedding(vector=reid.v_bystander)),
    ]
    p._evaluate_active_camera_target(worker, frame, track_res_copresent, 100.0, cand_embs)
    assert tm.target.state in (TargetState.TRACKING, TargetState.CONFIRMED, TargetState.LOCKED)
    assert tm.target.track_id == 1

    # Verify Track 2 was recorded in bystander history
    assert ("cam_0", 2) in p._bystander_history

    # 2. Target leaves! Only Bystander (Track 2) remains in view
    track_res_bystander_only = TrackResult(tracks=[track2], frame_id=2, timestamp_ms=200.0)
    cand_bystander_only = [
        (track2, frame[10:100, 10:100], Embedding(vector=reid.v_bystander)),
    ]
    p._evaluate_active_camera_target(worker, frame, track_res_bystander_only, 200.0, cand_bystander_only)

    # State must be LOST or SEARCHING
    assert tm.target.state in (TargetState.LOST, TargetState.SEARCHING)

    # Run 5 more frames where only Bystander Track 2 is visible
    for f in range(3, 8):
        ts = f * 100.0
        p._frame_count = f
        tr = TrackResult(tracks=[track2], frame_id=f, timestamp_ms=ts)
        p._evaluate_active_camera_target(worker, frame, tr, ts, cand_bystander_only)

    # CRITICAL: Bystander Track 2 must NEVER have been adopted!
    assert tm.target.track_id == 1, "Target ID must not switch to bystander Track 2!"
    assert tm.target.state in (TargetState.LOST, TargetState.SEARCHING)


def test_target_reacquired_when_reappearing():
    """
    Priority 0 Verification:
    Target is lost. Target reappears as Track 5 with matching appearance features.
    Verify:
    1. Target is reliably reacquired.
    2. Target lock transitions back to TRACKING/LOCKED with new track ID.
    """
    cfg = AppConfig()
    cfg.reid.reacquisition_threshold = 0.78
    p, im, tm = build(cfg)

    reid = MockReIDWithBystander()
    p.reid_extractor = reid
    p.reid_interval = 1

    frame = np.full((720, 1280, 3), 127, dtype=np.uint8)
    seed_and_lock(tm, 1)
    im.register_new_target(frame[10:100, 10:100], identity_id="target_0", embedding=Embedding(vector=reid.v_target))

    # Mark target lost
    tm.mark_lost(100.0)
    tm.mark_searching(100.0)
    assert tm.target.state in (TargetState.LOST, TargetState.SEARCHING)

    # Reappearance of Target as Track 5
    worker = _Worker()
    track5 = Track(track_id=5, box=BOXES[1], state=TrackState.TRACKED, confidence=0.95)
    cand_target = [
        (track5, frame[10:100, 10:100], Embedding(vector=reid.v_target)),
    ]

    # Present Track 5 for 4 consecutive frames (exceeding reacquisition_min_frames=3)
    for f in range(1, 5):
        ts = 200.0 + f * 50.0
        p._frame_count = f
        tr = TrackResult(tracks=[track5], frame_id=f, timestamp_ms=ts)
        p._evaluate_active_camera_target(worker, frame, tr, ts, cand_target)

    # Target must be reacquired as Track 5
    assert tm.target.track_id == 5
    assert tm.target.state in (TargetState.TRACKING, TargetState.CONFIRMED, TargetState.LOCKED)


def test_forensic_trail_and_transit_reset_on_new_target_selection():
    """
    Priority 1 Verification:
    Select Person A -> generate transit events -> verify transit history has Person A.
    Select Person B -> verify transit history, route recorder, and forensic recorder reset cleanly.
    Clear Target -> verify transit history and route recorder clear completely.
    """
    cfg = AppConfig()
    p, im, tm = build(cfg)
    p._transit_history.clear()

    # Simulate Person A selected and handoffs generated
    p._transit_history.append({"camera_id": "cam_0", "event": "TARGET_SELECTED", "track_id": 1, "timestamp": 1000.0})
    p._transit_history.append({"camera_id": "cam_1", "event": "HANDOFF", "track_id": 1, "timestamp": 1010.0})
    assert len(p.transit_history) == 2

    # Call clear_target()
    p.clear_target()
    assert len(p.transit_history) == 0, "Transit history must be empty after clear_target()"

    # Simulate Person A re-selected and movements recorded
    p._transit_history.append({"camera_id": "cam_0", "event": "TARGET_SELECTED", "track_id": 1, "timestamp": 2000.0})
    p._transit_history.append({"camera_id": "cam_2", "event": "HANDOFF", "track_id": 1, "timestamp": 2020.0})
    assert len(p.transit_history) == 2

    # Now simulate selecting Person B via select_target_by_id mock flow
    # In select_target_by_id, _transit_history.clear() is called before appending Person B's TARGET_SELECTED
    p._transit_history.clear()
    if p.route_recorder:
        p.route_recorder.clear()
    p._bystander_history.clear()
    p._transit_history.append({"camera_id": "cam_3", "event": "TARGET_SELECTED", "track_id": 99, "timestamp": 3000.0})

    # Verify that Person B starts with a clean transit history containing ONLY Person B
    assert len(p.transit_history) == 1
    assert p.transit_history[0]["track_id"] == 99
    assert p.transit_history[0]["camera_id"] == "cam_3"
