"""Unit tests for VectorStore and IdentityManager."""

import numpy as np
from src.core.interfaces import BaseReID
from src.core.types import Embedding
from src.identity.manager import IdentityManager
from src.identity.store import InMemoryVectorStore


class MockReID(BaseReID):
    """Deterministic mock ReID returning predefined vectors based on crop color."""

    def extract(self, crop: np.ndarray) -> Embedding:
        if crop is None or crop.size == 0:
            return Embedding(vector=np.zeros(4, dtype=np.float32))
        # Project mean color to distinct angular embedding
        mean_val = float(np.mean(crop))
        angle = (mean_val / 255.0) * (np.pi / 2.0)
        vec = np.array([np.cos(angle), np.sin(angle), mean_val / 255.0, 0.0], dtype=np.float32)
        return Embedding(vector=vec)

    def extract_batch(self, crops: list[np.ndarray]) -> list[Embedding]:
        return [self.extract(c) for c in crops]


def test_in_memory_vector_store():
    store = InMemoryVectorStore()
    assert store.count() == 0

    emb1 = Embedding(vector=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    emb2 = Embedding(vector=np.array([0.0, 1.0, 0.0], dtype=np.float32))

    store.add(emb1, "person_A")
    store.add(emb2, "person_B")
    assert store.count() == 2

    query = Embedding(vector=np.array([0.9, 0.1, 0.0], dtype=np.float32))
    matches = store.search(query, top_k=1)
    assert len(matches) == 1
    best_id, score = matches[0]
    assert best_id == "person_A"
    assert score > 0.8


def test_identity_manager_registration_and_verification():
    reid = MockReID()
    store = InMemoryVectorStore()
    im = IdentityManager(reid_extractor=reid, vector_store=store, similarity_threshold=0.7)

    # Register target appearance with blue-ish crop (mean ~200)
    crop_target = np.full((50, 50, 3), 200, dtype=np.uint8)
    im.register_or_update(crop_target, identity_id="target_0", label="VIP Target")

    ident = im.get_identity("target_0")
    assert ident is not None
    assert ident.label == "VIP Target"
    assert ident.reference_embedding is not None
    assert len(ident.embeddings) == 1

    # Verify similar candidate crop
    candidate_similar = np.full((50, 50, 3), 205, dtype=np.uint8)
    is_match, score = im.verify_candidate_crop(candidate_similar, identity_id="target_0")
    assert is_match is True
    assert score >= 0.95

    # Verify dissimilar candidate crop (mean ~10)
    candidate_different = np.full((50, 50, 3), 10, dtype=np.uint8)
    is_match_diff, score_diff = im.verify_candidate_crop(candidate_different, identity_id="target_0")
    assert is_match_diff is False
    assert score_diff < 0.7


def test_identity_manager_gallery_cap_and_reference_immutability():
    reid = MockReID()
    im = IdentityManager(reid_extractor=reid, max_gallery_size=3)

    initial_crop = np.full((20, 20, 3), 150, dtype=np.uint8)
    im.register_or_update(initial_crop, identity_id="target_0")

    ident = im.get_identity("target_0")
    ref_emb = ident.reference_embedding
    assert ref_emb is not None

    # Add 5 more updates
    for i in range(5):
        crop = np.full((20, 20, 3), 151 + i, dtype=np.uint8)
        im.register_or_update(crop, identity_id="target_0")

    # Adaptive gallery is capped at 3
    assert len(ident.embeddings) == 3
    # Permanent reference embedding is still intact!
    assert ident.reference_embedding == ref_emb


def test_gallery_contamination_rejected():
    reid = MockReID()
    im = IdentityManager(reid_extractor=reid, similarity_threshold=0.7)

    crop_person_a = np.full((50, 50, 3), 200, dtype=np.uint8)
    im.register_or_update(crop_person_a, identity_id="target_0", label="Person A")

    ident = im.get_identity("target_0")
    assert len(ident.embeddings) == 1

    # Try to add Person B (drastically different)
    crop_person_b = np.full((50, 50, 3), 20, dtype=np.uint8)
    accepted = im.verified_update(crop_person_b, identity_id="target_0")

    assert accepted is False
    assert len(ident.embeddings) == 1


def test_find_best_candidate_with_margin():
    reid = MockReID()
    im = IdentityManager(reid_extractor=reid, similarity_threshold=0.7, min_margin=0.05)

    # Register Person A (mean 200)
    crop_target = np.full((50, 50, 3), 200, dtype=np.uint8)
    im.register_or_update(crop_target, identity_id="target_0")

    # Candidate 1: Person B (mean 30 -> dissimilar)
    # Candidate 2: Person A (mean 202 -> highly similar)
    c1 = ("track_1", np.full((50, 50, 3), 30, dtype=np.uint8))
    c2 = ("track_2", np.full((50, 50, 3), 202, dtype=np.uint8))

    best_item, best_score, second_score, margin = im.find_best_candidate([c1, c2], "target_0")
    assert best_item == "track_2"
    assert best_score >= 0.95
    assert margin >= 0.05


def test_register_new_target_cleanly_replaces_identity():
    """
    When the user selects Person B after Person A was selected,
    register_new_target() must cleanly overwrite Person A's reference
    embedding with Person B's reference embedding.
    """
    reid = MockReID()
    store = InMemoryVectorStore()
    im = IdentityManager(reid_extractor=reid, vector_store=store)

    # 1. Register Person A (mean 200)
    crop_a = np.full((50, 50, 3), 200, dtype=np.uint8)
    im.register_new_target(crop_a, identity_id="target_0", label="Person A")

    ident_a = im.get_identity("target_0")
    assert ident_a.label == "Person A"
    ref_a = ident_a.reference_embedding

    # 2. Re-select and register Person B (mean 50)
    crop_b = np.full((50, 50, 3), 50, dtype=np.uint8)
    im.register_new_target(crop_b, identity_id="target_0", label="Person B")

    ident_b = im.get_identity("target_0")
    assert ident_b.label == "Person B"
    ref_b = ident_b.reference_embedding

    # Must be Person B's embedding, not Person A's!
    assert ref_b != ref_a
    assert ref_b.cosine_similarity(ref_a) < 0.80


def test_adaptive_gallery_anti_poisoning_and_flush():
    """
    Verifies that:
    1. Dissimilar scooper crops are rejected by verified_update() (anti-poisoning).
    2. Legitimate matching crops are accepted into adaptive gallery.
    3. flush_adaptive_gallery() cleanly wipes adaptive observations back to pure immutable reference.
    """
    reid = MockReID()
    store = InMemoryVectorStore()
    im = IdentityManager(reid_extractor=reid, vector_store=store, similarity_threshold=0.75, reference_threshold=0.70)

    # 1. Register Target (mean 200)
    crop_target = np.full((50, 50, 3), 200, dtype=np.uint8)
    im.register_new_target(crop_target, identity_id="target_0")
    ident = im.get_identity("target_0")
    assert len(ident.reference_gallery) == 1
    assert len(ident.adaptive_gallery) == 0

    # 2. Try to add Scooper crop (mean 20 -> dissimilar)
    crop_scooper = np.full((50, 50, 3), 20, dtype=np.uint8)
    accepted_scooper = im.verified_update(crop_scooper, identity_id="target_0")
    assert accepted_scooper is False
    assert len(ident.adaptive_gallery) == 0  # Rejection protected!

    # 3. Add legitimate matching crop (mean 202)
    crop_legit = np.full((50, 50, 3), 202, dtype=np.uint8)
    accepted_legit = im.verified_update(crop_legit, identity_id="target_0")
    assert accepted_legit is True
    assert len(ident.adaptive_gallery) == 1

    # 4. Flush adaptive gallery
    im.flush_adaptive_gallery("target_0")
    assert len(ident.adaptive_gallery) == 0
    assert len(ident.reference_gallery) == 1


def test_enroll_cross_camera_viewpoint_enforces_token_authorization():
    """
    Verifies IdentityManager.enroll_cross_camera_viewpoint validates tokens via
    VerifiedIdentityDecision.is_authorized_for(...) before enrolling view clusters.
    """
    from src.core.types import MatchDecisionState, VerifiedIdentityDecision

    reid = MockReID()
    im = IdentityManager(reid_extractor=reid, similarity_threshold=0.7)

    # Textured high-contrast crop so CropQualityEvaluator accepts it
    y_grad = np.linspace(40, 220, 120, dtype=np.uint8)[:, None, None]
    x_pat = ((np.arange(60, dtype=np.uint8) % 6) * 15)[None, :, None]
    crop = np.broadcast_to(np.clip(y_grad + x_pat, 0, 255).astype(np.uint8), (120, 60, 3)).copy()

    im.register_new_target(crop, identity_id="target_0", timestamp_ms=100.0)
    ident = im.get_identity("target_0")
    assert ident is not None and ident.anchor is not None
    initial_clusters = len(ident.anchor.clusters)

    # 1. None token rejected
    assert im.enroll_cross_camera_viewpoint(crop, "target_0", decision=None, timestamp_ms=150.0) is False

    # 2. Mismatched identity token rejected
    wrong_id_tok = VerifiedIdentityDecision(
        target_identity_id="target_other",
        authorized_track_id=7,
        decision_state=MatchDecisionState.MATCH,
        confidence=0.95,
        margin=0.2,
        timestamp_ms=100.0,
        expires_at_ms=500.0,
        reason="Wrong target identity",
    )
    assert im.enroll_cross_camera_viewpoint(crop, "target_0", decision=wrong_id_tok, timestamp_ms=150.0) is False

    # 3. Expired token rejected
    valid_tok = VerifiedIdentityDecision(
        target_identity_id="target_0",
        authorized_track_id=7,
        decision_state=MatchDecisionState.MATCH,
        confidence=0.95,
        margin=0.2,
        timestamp_ms=100.0,
        expires_at_ms=300.0,
        reason="Valid token",
    )
    assert im.enroll_cross_camera_viewpoint(crop, "target_0", decision=valid_tok, timestamp_ms=400.0) is False
    assert len(ident.anchor.clusters) == initial_clusters

    # 4. Valid token within expiry accepted
    assert im.enroll_cross_camera_viewpoint(crop, "target_0", decision=valid_tok, timestamp_ms=200.0) is True

