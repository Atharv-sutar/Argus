"""Extended multi-scenario stress tests (Scenarios A-F) and anchor hash stability."""

import hashlib
import time
import pytest
import numpy as np

from src.core.types import (
    BoundingBox,
    Embedding,
    MatchDecisionState,
    TargetState,
    Track,
    VerifiedIdentityDecision,
)
from src.identity.manager import IdentityManager
from src.reid.quality import CropQualityEvaluator
from src.target.manager import TargetManager


class MockStressReID:
    def __init__(self):
        self.call_count = 0

    def extract(self, crop: np.ndarray) -> Embedding:
        self.call_count += 1
        val = float(np.mean(crop)) if (crop is not None and crop.size > 0) else 0.0
        # 16-dimensional Gaussian RBF feature centered at val / 255.0
        centers = np.linspace(0.0, 1.0, 16)
        normalized_val = val / 255.0
        vec = np.exp(-((normalized_val - centers) ** 2) / (2 * (0.08 ** 2))).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return Embedding(vector=vec, model_name="dinov2", version="2.0")

    def extract_decomposed(self, crop: np.ndarray):
        emb = self.extract(crop)
        return emb, emb, emb, emb, emb


def compute_anchor_hash(identity_manager: IdentityManager, identity_id: str) -> str:
    """Computes a SHA-256 fingerprint of all cluster centroids in the target identity anchor."""
    ident = identity_manager.get_identity(identity_id)
    if not ident or not ident.anchor:
        return "NO_ANCHOR"
    raw = ""
    for c in ident.anchor.clusters:
        raw += f"{c.label}:" + ",".join([f"{v:.6f}" for v in c.centroid.vector]) + ";"
    return hashlib.sha256(raw.encode()).hexdigest()


def test_scenario_a_target_lost_one_bystander():
    """Scenario A: Target disappears, one bystander remains visible."""
    reid = MockStressReID()
    im = IdentityManager(reid_extractor=reid, similarity_threshold=0.80, reacquisition_threshold=0.85)

    target_crop = np.full((80, 40, 3), 200, dtype=np.uint8)
    bystander_crop = np.full((80, 40, 3), 50, dtype=np.uint8)

    im.register_new_target(target_crop, identity_id="target_0")

    for _ in range(50):
        ev = im.evaluate_candidate_crop(bystander_crop, "target_0")
        assert ev.is_match is False
        assert ev.decision == "NO_MATCH"


def test_scenario_b_target_lost_five_bystanders():
    """Scenario B: Target disappears, five distinct bystanders remain visible simultaneously."""
    reid = MockStressReID()
    im = IdentityManager(reid_extractor=reid, similarity_threshold=0.80, reacquisition_threshold=0.85)

    target_crop = np.full((80, 40, 3), 220, dtype=np.uint8)
    im.register_new_target(target_crop, identity_id="target_0")

    bystanders = [np.full((80, 40, 3), val, dtype=np.uint8) for val in [30, 60, 90, 120, 150]]

    for b_crop in bystanders:
        ev = im.evaluate_candidate_crop(b_crop, "target_0")
        assert ev.is_match is False


def test_scenario_c_visually_similar_hard_negative():
    """Scenario C: Target disappears, a visually similar hard-negative distractor is tested."""
    reid = MockStressReID()
    im = IdentityManager(reid_extractor=reid, similarity_threshold=0.82, reacquisition_threshold=0.88, min_margin=0.08)

    target_crop = np.full((80, 40, 3), 200, dtype=np.uint8)
    # Hard negative close in value (e.g. 185 vs 200)
    hard_neg_crop = np.full((80, 40, 3), 180, dtype=np.uint8)

    im.register_new_target(target_crop, identity_id="target_0")

    ev = im.evaluate_candidate_crop(hard_neg_crop, "target_0")
    # Must not falsely reacquire
    assert ev.is_match is False


def test_scenario_d_sequential_different_people_entering():
    """Scenario D: Target disappears, 20 different people sequentially enter and leave."""
    reid = MockStressReID()
    im = IdentityManager(reid_extractor=reid, similarity_threshold=0.80, reacquisition_threshold=0.85)

    target_crop = np.full((80, 40, 3), 240, dtype=np.uint8)
    im.register_new_target(target_crop, identity_id="target_0")

    for i in range(20):
        c = np.full((80, 40, 3), 10 + i * 5, dtype=np.uint8)
        ev = im.evaluate_candidate_crop(c, "target_0")
        assert ev.is_match is False


def test_scenario_e_long_interval_disappearance_and_anchor_hash_stability():
    """
    Scenario E: 100-cycle repeated disappearance and long-term anchor stability check.
    Verifies that the TargetIdentityAnchor SHA-256 hash remains 100% stable across
    Initial, 10m, 30m, and 60m simulated timestamps.
    """
    reid = MockStressReID()
    im = IdentityManager(reid_extractor=reid, similarity_threshold=0.78, reacquisition_threshold=0.82)
    tm = TargetManager()

    target_crop = np.full((80, 40, 3), 210, dtype=np.uint8)
    im.register_new_target(target_crop, identity_id="target_0", timestamp_ms=0.0)
    tm.select_by_track_id(1)

    initial_hash = compute_anchor_hash(im, "target_0")
    assert initial_hash != "NO_ANCHOR"

    hash_snapshots = {}

    # Simulate 100 disappearance and reacquisition cycles spanning 60 simulated minutes
    for cycle in range(100):
        curr_time = cycle * 36000.0  # 36 seconds per cycle = 3600s total (60 min)

        # 1. Target lost
        tm.mark_lost(curr_time)
        assert tm.target.state == TargetState.LOST

        # 2. Distractor arrives
        distractor = np.full((80, 40, 3), 40, dtype=np.uint8)
        ev_d = im.evaluate_candidate_crop(distractor, "target_0")
        assert ev_d.is_match is False

        # 3. Target returns
        target_crop_return = np.full((80, 40, 3), 210, dtype=np.uint8)
        token = VerifiedIdentityDecision(
            target_identity_id="target_0",
            authorized_track_id=100 + cycle,
            decision_state=MatchDecisionState.MATCH,
            confidence=0.95,
            margin=0.40,
            timestamp_ms=curr_time + 100.0,
            reason="Cycle reacquisition",
        )
        ok = tm.reassociate_target(
            Track(track_id=100 + cycle, box=BoundingBox(10, 10, 50, 100)),
            frame_id=cycle,
            timestamp_ms=curr_time + 100.0,
            decision=token,
        )
        assert ok is True

        # Snapshot anchor hash at intervals (0m, 10m, 30m, 60m)
        if cycle == 0:
            hash_snapshots["0m"] = compute_anchor_hash(im, "target_0")
        elif cycle == 16:  # ~10 min
            hash_snapshots["10m"] = compute_anchor_hash(im, "target_0")
        elif cycle == 50:  # ~30 min
            hash_snapshots["30m"] = compute_anchor_hash(im, "target_0")
        elif cycle == 99:  # ~60 min
            hash_snapshots["60m"] = compute_anchor_hash(im, "target_0")

    # Verify zero identity drift: all hashes must be completely identical
    assert hash_snapshots["0m"] == initial_hash
    assert hash_snapshots["10m"] == initial_hash
    assert hash_snapshots["30m"] == initial_hash
    assert hash_snapshots["60m"] == initial_hash


def test_scenario_f_cross_camera_target_reacquisition():
    """Scenario F: Target disappears from Camera 1 and reappears on Camera 2."""
    reid = MockStressReID()
    im = IdentityManager(reid_extractor=reid, similarity_threshold=0.78, reacquisition_threshold=0.82)
    tm = TargetManager()

    target_cam1 = np.full((80, 40, 3), 195, dtype=np.uint8)
    im.register_new_target(target_cam1, identity_id="target_0", timestamp_ms=100.0)
    tm.select_by_track_id(42)

    # Disappear on Cam 1
    tm.mark_lost(500.0)

    # Candidate on Cam 2 matches
    target_cam2 = np.full((80, 40, 3), 195, dtype=np.uint8)
    ev = im.evaluate_candidate_crop(target_cam2, "target_0")
    assert ev.is_match is True

    # Authorize Cam 2 track
    token = VerifiedIdentityDecision(
        target_identity_id="target_0",
        authorized_track_id=88,
        source_camera_id="cam_2",
        decision_state=MatchDecisionState.MATCH,
        confidence=0.92,
        margin=0.30,
        timestamp_ms=1500.0,
        reason="Cam 2 reacquisition",
    )
    ok = tm.reassociate_target(
        Track(track_id=88, box=BoundingBox(10, 10, 50, 100)),
        frame_id=45,
        timestamp_ms=1500.0,
        decision=token,
    )
    assert ok is True
    assert tm.target.track_id == 88
    assert tm.target.state == TargetState.TRACKING
