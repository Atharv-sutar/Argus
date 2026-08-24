"""Identity management module separating ReID appearance from identity matching."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.core.interfaces import BaseReID, BaseVectorStore
from src.core.types import Embedding, Identity
from src.identity.store import InMemoryVectorStore

logger = logging.getLogger(__name__)


class IdentityManager:
    """
    Manages identity association, appearance gallery updates, and verification.
    """

    def __init__(
        self,
        reid_extractor: BaseReID,
        vector_store: Optional[BaseVectorStore] = None,
        similarity_threshold: float = 0.65,
        min_margin: float = 0.05,
        max_gallery_size: int = 5,
    ) -> None:
        self.reid = reid_extractor
        self.vector_store = vector_store or InMemoryVectorStore()
        self.similarity_threshold = similarity_threshold
        self.min_margin = min_margin
        self.max_gallery_size = max_gallery_size
        self._identities: Dict[str, Identity] = {}

    def get_identity(self, identity_id: str) -> Optional[Identity]:
        return self._identities.get(identity_id)

    def register_new_target(
        self,
        crop: np.ndarray,
        identity_id: str,
        label: Optional[str] = None,
        timestamp_ms: float = 0.0,
    ) -> Optional[Embedding]:
        """
        Registers a brand new target identity, cleanly replacing any prior identity
        under identity_id so the permanent reference embedding and gallery are fresh.
        """
        if crop is None or crop.size == 0:
            return None

        # Clean previous vector store entries for this identity
        self.vector_store.remove_identity(identity_id)

        emb = self.reid.extract(crop)
        ident = Identity(
            identity_id=identity_id,
            label=label or identity_id,
            reference_embedding=emb,
            embeddings=[emb],
            last_seen_timestamp_ms=timestamp_ms,
        )
        self._identities[identity_id] = ident
        self.vector_store.add(emb, identity_id)
        logger.info(f"Registered fresh target identity '{identity_id}' ({ident.label}) with permanent reference embedding")
        return emb

    def register_or_update(
        self,
        crop: np.ndarray,
        identity_id: str,
        label: Optional[str] = None,
        timestamp_ms: float = 0.0,
    ) -> Optional[Embedding]:
        """
        Extracts embedding for person crop and adds to identity's appearance gallery.

        Use this for initial registration (no existing gallery to check against).
        For subsequent updates during tracking, use verified_update() instead.
        """
        if crop is None or crop.size == 0:
            return None

        if identity_id not in self._identities:
            return self.register_new_target(crop, identity_id, label, timestamp_ms)

        emb = self.reid.extract(crop)
        ident = self._identities[identity_id]
        ident.last_seen_timestamp_ms = timestamp_ms

        if len(ident.embeddings) < self.max_gallery_size:
            ident.embeddings.append(emb)
            self.vector_store.add(emb, identity_id)
        else:
            ident.embeddings.pop(0)
            ident.embeddings.append(emb)
            self.vector_store.add(emb, identity_id)

        return emb

    def verified_update(
        self,
        crop: np.ndarray,
        identity_id: str,
        timestamp_ms: float = 0.0,
    ) -> bool:
        """
        Update gallery only if the new embedding is consistent with existing identity
        and does not drift away from the permanent reference embedding.
        """
        if crop is None or crop.size == 0:
            return False

        ident = self._identities.get(identity_id)
        if ident is None or (not ident.embeddings and ident.reference_embedding is None):
            return False

        emb = self.reid.extract(crop)
        similarity = ident.compute_similarity(emb)

        if similarity < self.similarity_threshold:
            logger.warning(
                f"Gallery update REJECTED for '{identity_id}': "
                f"new embedding inconsistent with gallery (sim={similarity:.3f}, "
                f"threshold={self.similarity_threshold})"
            )
            return False

        # Additional protection: ensure crop has not drifted from permanent reference
        if ident.reference_embedding is not None:
            ref_sim = ident.reference_embedding.cosine_similarity(emb)
            if ref_sim < (self.similarity_threshold - 0.10):
                logger.warning(
                    f"Gallery update REJECTED for '{identity_id}': "
                    f"crop drifted from reference (ref_sim={ref_sim:.3f})"
                )
                return False

        ident.last_seen_timestamp_ms = timestamp_ms

        if len(ident.embeddings) < self.max_gallery_size:
            ident.embeddings.append(emb)
            self.vector_store.add(emb, identity_id)
        else:
            ident.embeddings.pop(0)
            ident.embeddings.append(emb)
            # Re-sync vector store to bound memory size and prevent leaks
            self.vector_store.remove_identity(identity_id)
            if ident.reference_embedding is not None:
                self.vector_store.add(ident.reference_embedding, identity_id)
            for g_emb in ident.embeddings:
                self.vector_store.add(g_emb, identity_id)

        return True

    def verify_candidate_crop(
        self,
        crop: np.ndarray,
        identity_id: str
    ) -> Tuple[bool, float]:
        """
        Verifies if candidate person crop matches a target identity.

        Returns:
            Tuple[bool, float]: (is_match, similarity_score)
        """
        ident = self.get_identity(identity_id)
        if ident is None or crop is None or crop.size == 0:
            return False, 0.0

        query_emb = self.reid.extract(crop)
        similarity = ident.compute_similarity(query_emb)
        is_match = similarity >= self.similarity_threshold

        return is_match, similarity

    def rank_candidate_crops(
        self,
        candidate_crops: List[Tuple[Any, np.ndarray]],
        identity_id: str,
    ) -> List[Tuple[Any, float]]:
        """
        Extracts embeddings (in batch) and ranks candidates by similarity against the target identity.

        Args:
            candidate_crops: List of tuples (candidate_object, crop_image).
            identity_id: Identity to compare against.

        Returns:
            List of (candidate_object, similarity_score) sorted descending by similarity.
        """
        ident = self.get_identity(identity_id)
        if ident is None or not candidate_crops:
            return []

        crops = [c[1] for c in candidate_crops]
        embeddings = self.reid.extract_batch(crops)

        ranked: List[Tuple[Any, float]] = []
        for (item, _), emb in zip(candidate_crops, embeddings):
            sim = ident.compute_similarity(emb)
            ranked.append((item, sim))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def find_best_candidate(
        self,
        candidate_crops: List[Tuple[Any, np.ndarray]],
        identity_id: str,
        threshold: Optional[float] = None,
        min_margin: Optional[float] = None,
    ) -> Tuple[Optional[Any], float, float, float]:
        """
        Finds the unambiguous best matching candidate from a list of crops.

        Returns:
            Tuple[best_candidate, best_score, second_best_score, margin]
            best_candidate is None if no candidate passes threshold or margin.
        """
        thresh = threshold if threshold is not None else self.similarity_threshold
        margin_req = min_margin if min_margin is not None else self.min_margin

        ranked = self.rank_candidate_crops(candidate_crops, identity_id)
        if not ranked:
            return None, 0.0, 0.0, 0.0

        best_item, best_score = ranked[0]
        second_best_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_best_score

        if best_score >= thresh and (len(ranked) == 1 or margin >= margin_req):
            return best_item, best_score, second_best_score, margin
        else:
            return None, best_score, second_best_score, margin

    def match_crop(
        self,
        crop: np.ndarray,
        threshold: Optional[float] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Searches all known identities for the best match against the query crop.

        Returns:
            Optional[Tuple[str, float]]: (identity_id, similarity) if above threshold, else None.
        """
        if crop is None or crop.size == 0:
            return None

        thresh = threshold if threshold is not None else self.similarity_threshold
        query_emb = self.reid.extract(crop)
        matches = self.vector_store.search(query_emb, top_k=1)

        if matches:
            best_id, best_score = matches[0]
            if best_score >= thresh:
                return best_id, best_score

        return None

    def clear(self) -> None:
        self._identities.clear()
        self.vector_store.clear()

