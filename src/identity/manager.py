"""Identity management module separating ReID appearance from identity matching."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.core.interfaces import BaseReID, BaseVectorStore
from src.core.types import Embedding, Identity
from src.identity.store import InMemoryVectorStore
from src.reid.quality import CropQualityEvaluator

logger = logging.getLogger(__name__)


@dataclass
class CandidateEvaluation:
    """Detailed ReID diagnostic evaluation for a candidate person crop."""
    is_match: bool
    candidate_score: float
    best_ref_sim: float
    best_adaptive_sim: float
    top2_adaptive_mean: float
    deep_sim: float
    color_sim: float
    fused_sim: float
    crop_quality_score: float
    quality_reason: str


class IdentityManager:
    """
    Manages identity association, separate immutable reference & adaptive galleries,
    appearance diversity gating, and candidate scoring.
    """

    def __init__(
        self,
        reid_extractor: BaseReID,
        vector_store: Optional[BaseVectorStore] = None,
        similarity_threshold: float = 0.65,
        reference_threshold: float = 0.60,
        min_margin: float = 0.05,
        max_reference_samples: int = 4,
        max_gallery_size: int = 5,
        redundancy_threshold: float = 0.90,
        quality_evaluator: Optional[CropQualityEvaluator] = None,
    ) -> None:
        self.reid = reid_extractor
        self.vector_store = vector_store or InMemoryVectorStore()
        self.similarity_threshold = similarity_threshold
        self.reference_threshold = reference_threshold
        self.min_margin = min_margin
        self.max_reference_samples = max_reference_samples
        self.max_gallery_size = max_gallery_size
        self.redundancy_threshold = redundancy_threshold
        self.quality = quality_evaluator or CropQualityEvaluator()
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
        Registers a fresh target identity and captures the initial reference embedding.
        """
        if crop is None or crop.size == 0:
            return None

        # Clean previous vector store entries for this identity
        self.vector_store.remove_identity(identity_id)

        emb = self.reid.extract(crop)
        ident = Identity(
            identity_id=identity_id,
            label=label or identity_id,
            reference_gallery=[emb],
            adaptive_gallery=[],
            last_seen_timestamp_ms=timestamp_ms,
        )
        self._identities[identity_id] = ident
        self.vector_store.add(emb, identity_id)
        logger.info(
            f"[IDENTITY] Registered new target identity '{identity_id}' ({ident.label}) with initial reference anchor"
        )
        return emb

    def register_or_update(
        self,
        crop: np.ndarray,
        identity_id: str,
        label: Optional[str] = None,
        timestamp_ms: float = 0.0,
    ) -> Optional[Embedding]:
        """
        Registers target appearance if not yet registered, or updates gallery if already registered.
        """
        if crop is None or crop.size == 0:
            return None

        if identity_id not in self._identities:
            return self.register_new_target(crop, identity_id, label, timestamp_ms)

        ident = self._identities[identity_id]
        emb = self.reid.extract(crop)
        ident.last_seen_timestamp_ms = timestamp_ms

        if len(ident.embeddings) < self.max_gallery_size:
            ident.adaptive_gallery.append(emb)
            self.vector_store.add(emb, identity_id)
        else:
            if ident.adaptive_gallery:
                ident.adaptive_gallery.pop(0)
            ident.adaptive_gallery.append(emb)
            self.vector_store.remove_identity(identity_id)
            for r_emb in ident.reference_gallery:
                self.vector_store.add(r_emb, identity_id)
            for a_emb in ident.adaptive_gallery:
                self.vector_store.add(a_emb, identity_id)

        return emb


    def add_reference_sample(
        self,
        crop: np.ndarray,
        identity_id: str,
        timestamp_ms: float = 0.0,
    ) -> bool:
        """
        Adds a new diverse, high-quality observation to the immutable reference gallery.
        Called during the ACQUIRING_REFERENCE window.
        """
        ident = self._identities.get(identity_id)
        if ident is None or crop is None or crop.size == 0:
            return False

        if len(ident.reference_gallery) >= self.max_reference_samples:
            return False

        is_valid, q_score, reason = self.quality.evaluate(crop)
        if not is_valid:
            logger.debug(f"[IDENTITY] Reference sample rejected for '{identity_id}': {reason}")
            return False

        emb = self.reid.extract(crop)

        # Diversity check: ensure the new reference sample is not redundant with existing references
        for existing_ref in ident.reference_gallery:
            sim = existing_ref.cosine_similarity(emb)
            if sim > self.redundancy_threshold:
                logger.debug(
                    f"[IDENTITY] Reference sample rejected for '{identity_id}': redundant (sim={sim:.3f} > {self.redundancy_threshold:.2f})"
                )
                return False

        ident.reference_gallery.append(emb)
        ident.last_seen_timestamp_ms = timestamp_ms
        self.vector_store.add(emb, identity_id)
        logger.info(
            f"[IDENTITY] Added diverse reference sample {len(ident.reference_gallery)}/{self.max_reference_samples} "
            f"for '{identity_id}' (quality={q_score:.2f})"
        )
        return True

    def is_reference_complete(self, identity_id: str) -> bool:
        """Checks whether reference gallery has acquired the target sample count."""
        ident = self._identities.get(identity_id)
        if ident is None:
            return False
        return len(ident.reference_gallery) >= self.max_reference_samples

    def verified_update(
        self,
        crop: np.ndarray,
        identity_id: str,
        timestamp_ms: float = 0.0,
    ) -> bool:
        """
        Updates the adaptive observation gallery only if:
        1. Crop passes image quality evaluation.
        2. Candidate is confirmed to match the logical identity (reference safeguard).
        3. Crop provides appearance diversity (not redundant).
        """
        if crop is None or crop.size == 0:
            return False

        ident = self._identities.get(identity_id)
        if ident is None or not ident.reference_gallery:
            return False

        # 1. Quality evaluation
        is_valid, q_score, reason = self.quality.evaluate(crop)
        if not is_valid:
            logger.debug(f"[IDENTITY] Adaptive update REJECTED for '{identity_id}': poor quality ({reason})")
            return False

        emb = self.reid.extract(crop)

        # 2. Hard reference safeguard: must match immutable reference anchors
        best_ref_sim = max(ref.cosine_similarity(emb) for ref in ident.reference_gallery)
        if best_ref_sim < self.reference_threshold:
            logger.warning(
                f"[IDENTITY] Adaptive update REJECTED for '{identity_id}': drifted from reference "
                f"(ref_sim={best_ref_sim:.3f} < {self.reference_threshold:.2f})"
            )
            return False

        # 3. Overall similarity check
        overall_sim = ident.compute_similarity(emb)
        if overall_sim < self.similarity_threshold:
            logger.warning(
                f"[IDENTITY] Adaptive update REJECTED for '{identity_id}': low similarity "
                f"(sim={overall_sim:.3f} < {self.similarity_threshold:.2f})"
            )
            return False

        # 4. Diversity check: reject near-duplicate embeddings already represented in adaptive gallery
        for existing_emb in ident.adaptive_gallery:
            sim = existing_emb.cosine_similarity(emb)
            if sim > self.redundancy_threshold:
                logger.debug(
                    f"[IDENTITY] Adaptive update SKIPPED for '{identity_id}': redundant (sim={sim:.3f} > {self.redundancy_threshold:.2f})"
                )
                return False

        ident.last_seen_timestamp_ms = timestamp_ms

        if len(ident.adaptive_gallery) < self.max_gallery_size:
            ident.adaptive_gallery.append(emb)
        else:
            ident.adaptive_gallery.pop(0)
            ident.adaptive_gallery.append(emb)

        # Re-sync vector store with reference + adaptive galleries
        self.vector_store.remove_identity(identity_id)
        for ref_emb in ident.reference_gallery:
            self.vector_store.add(ref_emb, identity_id)
        for ad_emb in ident.adaptive_gallery:
            self.vector_store.add(ad_emb, identity_id)

        logger.debug(
            f"[IDENTITY] Added adaptive observation for '{identity_id}' (ref_sim={best_ref_sim:.3f}, "
            f"adaptive_count={len(ident.adaptive_gallery)}/{self.max_gallery_size})"
        )
        return True

    def evaluate_candidate_crop(
        self,
        crop: np.ndarray,
        identity_id: str,
    ) -> CandidateEvaluation:
        """
        Performs full decomposed ReID evaluation and robust candidate scoring for a crop.
        """
        ident = self.get_identity(identity_id)
        if ident is None or crop is None or crop.size == 0 or not ident.reference_gallery:
            return CandidateEvaluation(
                is_match=False,
                candidate_score=0.0,
                best_ref_sim=0.0,
                best_adaptive_sim=0.0,
                top2_adaptive_mean=0.0,
                deep_sim=0.0,
                color_sim=0.0,
                fused_sim=0.0,
                crop_quality_score=0.0,
                quality_reason="NO_IDENTITY_OR_EMPTY_CROP",
            )

        is_valid, q_score, q_reason = self.quality.evaluate(crop)

        # Decomposed extraction
        if hasattr(self.reid, "extract_decomposed"):
            fused_emb, deep_emb, color_emb = self.reid.extract_decomposed(crop)
        else:
            fused_emb = self.reid.extract(crop)
            deep_emb = fused_emb
            color_emb = fused_emb

        # Reference gallery similarities
        ref_fused_sims = [ref.cosine_similarity(fused_emb) for ref in ident.reference_gallery]
        best_ref_sim = max(ref_fused_sims) if ref_fused_sims else 0.0

        # Decomposed similarities against the primary reference
        primary_ref = ident.reference_gallery[0]
        fused_sim = primary_ref.cosine_similarity(fused_emb)
        deep_sim = primary_ref.cosine_similarity(deep_emb) if deep_emb.dim == primary_ref.dim else fused_sim
        color_sim = primary_ref.cosine_similarity(color_emb) if color_emb.dim == primary_ref.dim else fused_sim

        # Adaptive gallery similarities
        best_adaptive_sim = 0.0
        top2_adaptive_mean = 0.0
        if ident.adaptive_gallery:
            adaptive_sims = sorted(
                [ad.cosine_similarity(fused_emb) for ad in ident.adaptive_gallery],
                reverse=True,
            )
            best_adaptive_sim = adaptive_sims[0]
            top2_adaptive_mean = sum(adaptive_sims[:2]) / len(adaptive_sims[:2])

        # Candidate consensus score
        if ident.adaptive_gallery:
            candidate_score = 0.5 * best_ref_sim + 0.5 * top2_adaptive_mean
        else:
            candidate_score = best_ref_sim

        # Hard reference safeguard + candidate threshold
        is_match = (
            best_ref_sim >= self.reference_threshold
            and candidate_score >= self.similarity_threshold
        )

        return CandidateEvaluation(
            is_match=is_match,
            candidate_score=candidate_score,
            best_ref_sim=best_ref_sim,
            best_adaptive_sim=best_adaptive_sim,
            top2_adaptive_mean=top2_adaptive_mean,
            deep_sim=deep_sim,
            color_sim=color_sim,
            fused_sim=fused_sim,
            crop_quality_score=q_score,
            quality_reason=q_reason,
        )

    def verify_candidate_crop(
        self,
        crop: np.ndarray,
        identity_id: str,
    ) -> Tuple[bool, float]:
        """Backward-compatible quick verification returning (is_match, score)."""
        evaluation = self.evaluate_candidate_crop(crop, identity_id)
        return evaluation.is_match, evaluation.candidate_score

    def rank_candidate_crops(
        self,
        candidate_crops: List[Tuple[Any, np.ndarray]],
        identity_id: str,
    ) -> List[Tuple[Any, float, CandidateEvaluation]]:
        """
        Ranks candidates by candidate_score descending.

        Returns:
            List of (candidate_object, candidate_score, CandidateEvaluation).
        """
        ident = self.get_identity(identity_id)
        if ident is None or not candidate_crops:
            return []

        ranked: List[Tuple[Any, float, CandidateEvaluation]] = []
        for item, crop in candidate_crops:
            eval_res = self.evaluate_candidate_crop(crop, identity_id)
            ranked.append((item, eval_res.candidate_score, eval_res))

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
        """
        thresh = threshold if threshold is not None else self.similarity_threshold
        margin_req = min_margin if min_margin is not None else self.min_margin

        ranked = self.rank_candidate_crops(candidate_crops, identity_id)
        if not ranked:
            return None, 0.0, 0.0, 0.0

        best_item, best_score, best_eval = ranked[0]
        second_best_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_best_score

        if best_eval.is_match and best_score >= thresh and (len(ranked) == 1 or margin >= margin_req):
            return best_item, best_score, second_best_score, margin
        else:
            return None, best_score, second_best_score, margin

    def clear(self) -> None:
        self._identities.clear()
        self.vector_store.clear()
