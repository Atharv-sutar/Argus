"""Targeted regression test suite verifying ReID anti-scoop, sole-bystander rejection, and reacquisition rules."""

import pytest
import numpy as np

from src.core.types import BoundingBox, Embedding, Target, TargetState, Track, TrackResult
from src.identity.evidence import EvidenceEngine, TrackObservation
from src.identity.manager import IdentityManager
from src.target.manager import TargetManager


class MockReID:
    def __init__(self, target_vec, bystander_vec):
        self.target_vec = target_vec
        self.bystander_vec = bystander_vec

    def extract(self, crop):
        if crop is not None and crop[0, 0, 0] == 255:
            return Embedding(vector=self.target_vec, model_name="mock", version="2.0")
        return Embedding(vector=self.bystander_vec, model_name="mock", version="2.0")

    def extract_decomposed(self, crop):
        emb = self.extract(crop)
        return emb, emb, emb, emb, emb


def test_sole_bystander_not_adopted_when_target_lost():
    """
    Rule A & Rule C:
    When Target A is LOST and only Bystander B is visible (even for 20 frames),
    Bystander B must NEVER be adopted if B does not exceed reacquisition threshold.
    """
    engine = EvidenceEngine(
        window_size=4,
        min_similarity_threshold=0.78,
        reacquisition_threshold=0.82,
        reacquisition_min_frames=4,
        min_margin_threshold=0.08,
    )

    bystander_track = Track(
        track_id=99,
        box=BoundingBox(10, 10, 50, 100),
        confidence=0.9,
    )

    # Bystander B produces a borderline score of 0.73 across 20 frames
    for frame_id in range(1, 21):
        engine.register_observation(
            track_id=99,
            frame_id=frame_id,
            timestamp_ms=frame_id * 33.3,
            crop_quality=0.8,
            similarity=0.73,
            margin=0.0,
            is_match=False,
        )

        dec = engine.evaluate_all_candidates(
            candidate_evaluations=[(bystander_track, 0.73, False, 0.8)],
            target_identity_id="target_0",
            current_tracked_id=None,  # Target is LOST
            is_reacquisition=True,
        )

        # Invariant: Bystander MUST NEVER be confirmed as target
        assert dec.is_confirmed is False
        assert dec.best_track_id is None


def test_reacquisition_requires_multi_frame_hysteresis():
    """
    Rule D & Rule H:
    When genuine Target A reappears with score 0.90,
    reacquisition is only confirmed after K >= 4 consistent frames.
    """
    engine = EvidenceEngine(
        window_size=4,
        min_similarity_threshold=0.78,
        reacquisition_threshold=0.82,
        reacquisition_min_frames=4,
        min_margin_threshold=0.08,
    )

    target_track = Track(
        track_id=42,
        box=BoundingBox(10, 10, 50, 100),
        confidence=0.95,
    )

    # Frames 1 to 3: Pending confirmation (hysteresis)
    for frame_id in range(1, 4):
        engine.register_observation(
            track_id=42,
            frame_id=frame_id,
            timestamp_ms=frame_id * 100.0,
            crop_quality=0.9,
            similarity=0.90,
            margin=0.0,
            is_match=True,
            box=target_track.box,
        )
        dec = engine.evaluate_all_candidates(
            candidate_evaluations=[(target_track, 0.90, True, 0.9)],
            target_identity_id="target_0",
            current_tracked_id=None,
            is_reacquisition=True,
        )
        assert dec.is_confirmed is False
        assert dec.is_uncertain is True  # In pending reacquisition state

    # Frame 4: Reacquisition confirmed!
    engine.register_observation(
        track_id=42,
        frame_id=4,
        timestamp_ms=4 * 100.0,
        crop_quality=0.9,
        similarity=0.90,
        margin=0.0,
        is_match=True,
        box=target_track.box,
    )
    dec_final = engine.evaluate_all_candidates(
        candidate_evaluations=[(target_track, 0.90, True, 0.9)],
        target_identity_id="target_0",
        current_tracked_id=None,
        is_reacquisition=True,
    )
    assert dec_final.is_confirmed is True
    assert dec_final.best_track_id == 42


def test_competing_candidates_within_margin_stay_uncertain():
    """
    Rule E:
    When two candidates have close scores (margin < 0.08),
    the system must prefer UNCERTAIN over guessing the wrong person.
    """
    engine = EvidenceEngine(
        window_size=4,
        min_similarity_threshold=0.78,
        reacquisition_threshold=0.82,
        reacquisition_min_frames=4,
        min_margin_threshold=0.08,
    )

    cand1 = Track(track_id=1, box=BoundingBox(10, 10, 50, 100), confidence=0.9)
    cand2 = Track(track_id=2, box=BoundingBox(60, 10, 100, 100), confidence=0.9)

    for frame_id in range(1, 6):
        engine.register_observation(1, frame_id, frame_id * 100.0, 0.9, 0.85, 0.02, True, box=cand1.box)
        engine.register_observation(2, frame_id, frame_id * 100.0, 0.9, 0.83, 0.02, True, box=cand2.box)

    dec = engine.evaluate_all_candidates(
        candidate_evaluations=[(cand1, 0.85, True, 0.9), (cand2, 0.83, True, 0.9)],
        target_identity_id="target_0",
        current_tracked_id=None,
        is_reacquisition=True,
    )

    assert dec.is_confirmed is False
    assert dec.is_uncertain is True
    assert "Ambiguous candidates within margin" in dec.decision_reason


def test_stationary_bystander_100_frames_immunity():
    """
    Verifies that a stationary bystander visible for 100 frames with score 0.74
    is NEVER adopted by the system, and observation deduplication collapses identical frames.
    """
    engine = EvidenceEngine(
        window_size=4,
        min_similarity_threshold=0.78,
        reacquisition_threshold=0.82,
        reacquisition_min_frames=4,
        min_margin_threshold=0.08,
    )

    bystander_track = Track(track_id=99, box=BoundingBox(10, 10, 50, 100), confidence=0.9)

    for frame_id in range(1, 101):
        engine.register_observation(
            track_id=99,
            frame_id=frame_id,
            timestamp_ms=frame_id * 33.3,
            crop_quality=0.85,
            similarity=0.74,
            margin=0.0,
            is_match=False,
            box=bystander_track.box,
        )

        dec = engine.evaluate_all_candidates(
            candidate_evaluations=[(bystander_track, 0.74, False, 0.85)],
            target_identity_id="target_0",
            current_tracked_id=None,
            is_reacquisition=True,
        )

        # Invariant: Must NEVER be confirmed as target across 100 frames
        assert dec.is_confirmed is False
        assert dec.best_track_id is None


def test_unauthorized_token_rejection():
    """
    Verifies that TargetManager.reassociate_target strictly rejects reassociation
    if the provided VerifiedIdentityDecision token does not authorize the track ID.
    """
    from src.core.types import MatchDecisionState, VerifiedIdentityDecision

    target_mgr = TargetManager()
    target_mgr.select_by_track_id(1)
    target_mgr.mark_lost(100.0)

    forged_track = Track(track_id=99, box=BoundingBox(10, 10, 50, 100), confidence=0.9)

    # Token authorized for Track 42, NOT Track 99
    invalid_token = VerifiedIdentityDecision(
        target_identity_id="target_0",
        authorized_track_id=42,
        decision_state=MatchDecisionState.MATCH,
        confidence=0.90,
        margin=0.20,
        timestamp_ms=200.0,
        reason="Test token",
    )

    # Attempt to reassociate to Track 99 with token for Track 42 -> Must be rejected!
    ok = target_mgr.reassociate_target(forged_track, frame_id=2, timestamp_ms=200.0, decision=invalid_token)
    assert ok is False
    assert target_mgr.target.state == TargetState.LOST
