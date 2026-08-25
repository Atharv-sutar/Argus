"""Unit tests for TargetManager — comprehensive identity preservation & verification tests."""

import numpy as np
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

    selected_id = tm.select_by_point(100.0, 150.0, res)
    assert selected_id == 10
    assert tm.target.track_id == 10
    assert tm.target.state == TargetState.LOCKED

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

    tm.select_by_track_id(5, res1)
    assert tm.target.state == TargetState.LOCKED

    t1_moved = Track(track_id=5, box=BoundingBox(x1=105.0, y1=100.0, x2=205.0, y2=300.0))
    res2 = TrackResult(tracks=[t1_moved], frame_id=2, timestamp_ms=133.3)
    tm.update(res2)
    assert tm.target.state == TargetState.TRACKING
    assert tm.target.lost_duration_ms == 0.0

    res3 = TrackResult(tracks=[], frame_id=3, timestamp_ms=200.0)
    tm.update(res3)
    assert tm.target.state == TargetState.LOST
    assert tm.target.lost_duration_ms == pytest.approx(66.7, 0.01)

    res4 = TrackResult(tracks=[], frame_id=4, timestamp_ms=700.0)
    tm.update(res4)
    assert tm.target.state == TargetState.LOST
    assert tm.target.lost_duration_ms >= 500.0


# ═══════════════════════════════════════════════════════════════════
# Test A: Same tracker ID but wrong person inside (mismatch detected)
# ═══════════════════════════════════════════════════════════════════

def test_same_tracker_id_wrong_person_rejected():
    """
    Tracker 52 continues to exist, but Person B walked in front and occupied Tracker 52.
    ReID verification must detect mismatch and transition target to LOST (not track Person B).
    """
    tm = TargetManager(min_margin=0.05)
    box = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
    t1 = Track(track_id=52, box=box)
    res1 = TrackResult(tracks=[t1], frame_id=1, timestamp_ms=100.0)
    tm.select_by_track_id(52, res1)

    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # Verifier reports Tracker 52 occupant has low similarity (Person B)
    def verify_wrong(crop: np.ndarray):
        return (False, 0.28)

    tm.update(res1, frame=frame, verify_fn=verify_wrong)

    # Must NOT remain TRACKING on wrong person
    assert tm.target.state == TargetState.LOST
    assert tm.target.track_id == 52


# ═══════════════════════════════════════════════════════════════════
# Test B: Wrong person nearby is rejected
# ═══════════════════════════════════════════════════════════════════

def test_nearby_wrong_person_rejected():
    """
    Person B is nearby with high IoU but poor ReID similarity.
    The system must NOT switch to Person B.
    """
    tm = TargetManager(reassociation_iou_thresh=0.3, min_margin=0.05)

    box_a = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
    t_a = Track(track_id=1, box=box_a)
    res1 = TrackResult(tracks=[t_a], frame_id=1, timestamp_ms=100.0)
    tm.select_by_track_id(1, res1)

    # Person B appears nearby — track 1 is gone, track 2 is visible
    box_b = BoundingBox(x1=102.0, y1=101.0, x2=202.0, y2=301.0)
    t_b = Track(track_id=2, box=box_b)
    res2 = TrackResult(tracks=[t_b], frame_id=2, timestamp_ms=133.3)

    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    def reject_wrong_person(crop: np.ndarray):
        return (False, 0.25)

    tm.update(res2, frame=frame, verify_fn=reject_wrong_person)

    assert tm.target.track_id == 1  # Must NOT switch to 2
    assert tm.target.state == TargetState.LOST


# ═══════════════════════════════════════════════════════════════════
# Test C: Target found in another tracker (52 becomes B, 53 is A)
# ═══════════════════════════════════════════════════════════════════

def test_target_found_in_another_tracker():
    """
    Tracker 52 has become Person B (mismatch).
    Tracker 53 is Person A.
    System must reject 52 and switch to 53!
    """
    tm = TargetManager(min_margin=0.05)
    box_52 = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
    box_53 = BoundingBox(x1=250.0, y1=100.0, x2=350.0, y2=300.0)
    t52 = Track(track_id=52, box=box_52)
    t53 = Track(track_id=53, box=box_53)
    res = TrackResult(tracks=[t52, t53], frame_id=2, timestamp_ms=133.3)

    tm.select_by_track_id(52, res)

    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    def verify_by_crop_location(crop: np.ndarray):
        # Tracker 53 is Person A, 52 is Person B
        if crop.shape[1] > 0 and crop.shape[0] > 0:
            # Distinguish based on dummy crop check or simulated mock
            pass
        return (True, 0.92)

    # Let verify_fn distinguish: 52 is False, 53 is True (0.92)
    call_records = []
    def selective_verify(crop: np.ndarray):
        # We can identify which track is being queried
        # First query is 52 (current track) -> returns False
        # Second query is 53 (candidate) -> returns True (0.92)
        if len(call_records) == 0:
            call_records.append(52)
            return (False, 0.30)
        else:
            call_records.append(53)
            return (True, 0.92)

    tm.update(res, frame=frame, verify_fn=selective_verify)

    # Invariant: TargetManager alone never silently switches ID; it transitions to LOST
    assert tm.target.state == TargetState.LOST

    # Explicit authorized reassociation via token
    from src.core.types import MatchDecisionState, VerifiedIdentityDecision
    token = VerifiedIdentityDecision(
        target_identity_id="target_0",
        authorized_track_id=53,
        decision_state=MatchDecisionState.MATCH,
        confidence=0.92,
        margin=0.62,
        timestamp_ms=133.3,
        reason="Verified candidate 53",
    )
    ok = tm.reassociate_target(t53, frame_id=2, timestamp_ms=133.3, decision=token)
    assert ok is True
    assert tm.target.track_id == 53
    assert tm.target.state == TargetState.TRACKING


# ═══════════════════════════════════════════════════════════════════
# Test D: Two plausible candidates with clear margin -> token reassociation
# ═══════════════════════════════════════════════════════════════════

def test_margin_based_candidate_selection():
    """
    Candidate 24 -> 0.70
    Candidate 25 -> 0.91 (Margin = 0.21 >= 0.05)
    Reassociation requires authorized verification decision.
    """
    from src.core.types import MatchDecisionState, VerifiedIdentityDecision

    tm = TargetManager(min_margin=0.05)
    t24 = Track(track_id=24, box=BoundingBox(10, 10, 50, 100))
    t25 = Track(track_id=25, box=BoundingBox(60, 10, 100, 100))
    res = TrackResult(tracks=[t24, t25], frame_id=2, timestamp_ms=133.3)

    # Target 52 is missing
    tm.select_by_track_id(52)

    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    calls = [0]
    def verify_two(crop: np.ndarray):
        calls[0] += 1
        if calls[0] == 1:
            return (True, 0.70)  # track 24
        return (True, 0.91)      # track 25

    tm.update(res, frame=frame, verify_fn=verify_two)
    assert tm.target.state == TargetState.LOST

    # Reassociate with token
    token = VerifiedIdentityDecision(
        target_identity_id="target_0",
        authorized_track_id=25,
        decision_state=MatchDecisionState.MATCH,
        confidence=0.91,
        margin=0.21,
        timestamp_ms=133.3,
        reason="Winner with margin",
    )
    ok = tm.reassociate_target(t25, frame_id=2, timestamp_ms=133.3, decision=token)
    assert ok is True
    assert tm.target.track_id == 25
    assert tm.target.state == TargetState.TRACKING


# ═══════════════════════════════════════════════════════════════════
# Test E: Ambiguous candidates (margin < min_margin) -> no switch
# ═══════════════════════════════════════════════════════════════════

def test_ambiguous_candidates_rejected():
    """
    Candidate 24 -> 0.73
    Candidate 25 -> 0.71 (Margin = 0.02 < 0.05)
    Ambiguous -> must NOT switch, must transition to LOST!
    """
    tm = TargetManager(min_margin=0.05)
    t24 = Track(track_id=24, box=BoundingBox(10, 10, 50, 100))
    t25 = Track(track_id=25, box=BoundingBox(60, 10, 100, 100))
    res = TrackResult(tracks=[t24, t25], frame_id=2, timestamp_ms=133.3)

    tm.select_by_track_id(52)

    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    calls = [0]
    def verify_ambiguous(crop: np.ndarray):
        calls[0] += 1
        if calls[0] == 1:
            return (True, 0.73)
        return (True, 0.71)

    tm.update(res, frame=frame, verify_fn=verify_ambiguous)

    # Ambiguous match must NOT be accepted
    assert tm.target.state == TargetState.LOST
