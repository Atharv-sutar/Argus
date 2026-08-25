"""Unit and adversarial security tests for ReID identity tokens, anchors, and expiry."""

import time
import pytest
import numpy as np

from src.core.types import (
    BoundingBox,
    Embedding,
    MatchDecisionState,
    Target,
    TargetState,
    Track,
    VerifiedIdentityDecision,
)
from src.identity.manager import IdentityManager
from src.reid.quality import CropQualityEvaluator
from src.target.manager import TargetManager


class MockDeterministicReID:
    def __init__(self):
        self.call_count = 0

    def extract(self, crop: np.ndarray) -> Embedding:
        self.call_count += 1
        val = float(np.mean(crop)) if (crop is not None and crop.size > 0) else 0.0
        vec = np.array([val / 255.0, (255.0 - val) / 255.0, 1.0, 0.0], dtype=np.float32)
        return Embedding(vector=vec, model_name="dinov2", version="2.0")

    def extract_decomposed(self, crop: np.ndarray):
        emb = self.extract(crop)
        return emb, emb, emb, emb, emb


def test_expired_token_rejected_by_target_manager():
    """Verifies that an expired token (> 1000ms old) is strictly rejected."""
    tm = TargetManager()
    tm.select_by_track_id(1)
    tm.mark_lost(100.0)

    track_new = Track(track_id=2, box=BoundingBox(10, 10, 50, 100))

    # Create token at timestamp 100.0 ms with expiry at 1100.0 ms
    token = VerifiedIdentityDecision(
        target_identity_id="target_0",
        authorized_track_id=2,
        decision_state=MatchDecisionState.MATCH,
        confidence=0.92,
        margin=0.20,
        timestamp_ms=100.0,
        expires_at_ms=1100.0,
        reason="Test token",
    )

    # Attempt reassociation at timestamp 1200.0 ms (expired!)
    ok = tm.reassociate_target(track_new, frame_id=10, timestamp_ms=1200.0, decision=token)
    assert ok is False
    assert tm.target.state == TargetState.LOST


def test_camera_id_mismatch_token_rejected():
    """Verifies that a token issued for camera_A cannot be forged/replayed on camera_B."""
    token = VerifiedIdentityDecision(
        target_identity_id="target_0",
        authorized_track_id=5,
        source_camera_id="cam_A",
        decision_state=MatchDecisionState.MATCH,
        confidence=0.90,
        margin=0.15,
        timestamp_ms=500.0,
        reason="Camera A token",
    )

    # Valid on cam_A
    assert token.is_authorized_for("target_0", 5, current_timestamp_ms=550.0, camera_id="cam_A") is True
    # Invalid on cam_B
    assert token.is_authorized_for("target_0", 5, current_timestamp_ms=550.0, camera_id="cam_B") is False


def test_low_resolution_crop_rejected():
    """Verifies that crops below minimum height (35px) are rejected with INSUFFICIENT_QUALITY."""
    evaluator = CropQualityEvaluator(min_height=35, min_width=16)

    tiny_crop = np.zeros((20, 10, 3), dtype=np.uint8)
    is_valid, q_score, reason = evaluator.evaluate(tiny_crop)
    assert is_valid is False
    assert "HEIGHT_TOO_SMALL" in reason or "WIDTH_TOO_SMALL" in reason


def test_target_identity_anchor_immutability_over_100_cycles():
    """
    Stress test: Runs 100 repeated cycles of target disappearance, candidate searches,
    and adaptive updates, verifying that the TargetIdentityAnchor remains 100% immutable.
    """
    reid = MockDeterministicReID()
    im = IdentityManager(reid_extractor=reid, similarity_threshold=0.75, reacquisition_threshold=0.82)

    enroll_crop = np.full((80, 40, 3), 180, dtype=np.uint8)
    im.register_new_target(enroll_crop, identity_id="target_0", label="Target VIP", timestamp_ms=100.0)

    ident = im.get_identity("target_0")
    assert ident.anchor is not None
    initial_cluster_0 = ident.anchor.clusters[0].centroid.vector.copy()

    # Simulate 100 cycles of updates with different candidate crops
    for cycle in range(100):
        # Good matching crop
        c_good = np.full((80, 40, 3), 180 + (cycle % 5), dtype=np.uint8)
        im.verified_update(c_good, "target_0", timestamp_ms=200.0 + cycle * 100.0)

        # Bystander crop
        c_bystander = np.full((80, 40, 3), 30, dtype=np.uint8)
        im.verified_update(c_bystander, "target_0", timestamp_ms=250.0 + cycle * 100.0)

    # Verify that the original anchor cluster 0 was NEVER altered or corrupted
    final_cluster_0 = ident.anchor.clusters[0].centroid.vector
    assert np.allclose(initial_cluster_0, final_cluster_0, atol=1e-6)
    assert len(ident.trusted_gallery) == 1
