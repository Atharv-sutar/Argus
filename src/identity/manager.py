"""Identity management module separating ReID appearance from identity matching."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    decision: str  # "MATCH", "NO_MATCH", "AMBIGUOUS"
    candidate_score: float
    proto_sim: float
    best_ref_sim: float
    best_adaptive_sim: float
    top2_adaptive_mean: float
    deep_sim: float
    color_sim: float
    upper_sim: float
    lower_sim: float
    fused_sim: float
    feature_agreement_passed: bool
    crop_quality_score: float
    quality_reason: str
    rejection_reasons: List[str] = field(default_factory=list)


class IdentityManager:
    """
    Manages identity association, separate immutable reference & adaptive galleries,
    score-level component consensus, and robust multi-camera candidate matching.
    """

    def __init__(
        self,
        reid_extractor: BaseReID,
        vector_store: Optional[BaseVectorStore] = None,
        similarity_threshold: float = 0.78,
        reference_threshold: float = 0.72,
        upper_threshold: float = 0.60,
        min_margin: float = 0.06,
        max_reference_samples: int = 4,
        max_gallery_size: int = 5,
        redundancy_threshold: float = 0.90,
        quality_evaluator: Optional[CropQualityEvaluator] = None,
        w_upper: float = 0.45,
        w_color: float = 0.25,
        w_deep: float = 0.15,
        w_lower: float = 0.15,
    ) -> None:
        self.reid = reid_extractor
        self.vector_store = vector_store or InMemoryVectorStore()
        self.similarity_threshold = similarity_threshold
        self.reference_threshold = reference_threshold
        self.upper_threshold = upper_threshold
        self.min_margin = min_margin
        self.max_reference_samples = max_reference_samples
        self.max_gallery_size = max_gallery_size
        self.redundancy_threshold = redundancy_threshold
        self.quality = quality_evaluator or CropQualityEvaluator()
        self.w_upper = w_upper
        self.w_color = w_color
        self.w_deep = w_deep
        self.w_lower = w_lower
        self._identities: Dict[str, Identity] = {}

    def get_identity(self, identity_id: str) -> Optional[Identity]:
        return self._identities.get(identity_id)

    def _extract_all_representations(
        self, crop: np.ndarray
    ) -> Tuple[Embedding, Embedding, Embedding, Embedding, Embedding]:
        """Extracts fused and decomposed feature embeddings."""
        if hasattr(self.reid, "extract_decomposed"):
            decomposed = self.reid.extract_decomposed(crop)
            if len(decomposed) == 5:
                return decomposed
        emb = self.reid.extract(crop)
        return emb, emb, emb, emb, emb

    def register_new_target(
        self,
        crop: np.ndarray,
        identity_id: str,
        label: Optional[str] = None,
        timestamp_ms: float = 0.0,
    ) -> Optional[Embedding]:
        """
        Registers a fresh target identity, initializes decomposed reference galleries,
        and computes normalized reference prototypes.
        """
        if crop is None or crop.size == 0:
            return None

        # Clean previous vector store entries for this identity
        self.vector_store.remove_identity(identity_id)

        fused, deep, color, upper, lower = self._extract_all_representations(crop)

        ident = Identity(
            identity_id=identity_id,
            label=label or identity_id,
            reference_gallery=[fused],
            reference_deep_gallery=[deep],
            reference_color_gallery=[color],
            reference_upper_gallery=[upper],
            reference_lower_gallery=[lower],
            adaptive_gallery=[],
            last_seen_timestamp_ms=timestamp_ms,
        )
        ident.update_prototype()
        self._identities[identity_id] = ident
        self.vector_store.add(fused, identity_id)
        logger.info(
            f"[IDENTITY] Registered new target identity '{identity_id}' ({ident.label}) with initial reference prototypes"
        )
        return fused

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
        Adds a new diverse, high-quality observation to the immutable reference galleries
        and updates the reference prototype centroids.
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

        fused, deep, color, upper, lower = self._extract_all_representations(crop)

        # Diversity check against existing reference samples
        for existing_ref in ident.reference_gallery:
            sim = existing_ref.cosine_similarity(fused)
            if sim > self.redundancy_threshold:
                logger.debug(
                    f"[IDENTITY] Reference sample rejected for '{identity_id}': redundant (sim={sim:.3f} > {self.redundancy_threshold:.2f})"
                )
                return False

        ident.reference_gallery.append(fused)
        ident.reference_deep_gallery.append(deep)
        ident.reference_color_gallery.append(color)
        ident.reference_upper_gallery.append(upper)
        ident.reference_lower_gallery.append(lower)
        ident.update_prototype()
        ident.last_seen_timestamp_ms = timestamp_ms
        self.vector_store.add(fused, identity_id)
        logger.info(
            f"[IDENTITY] Added diverse reference sample {len(ident.reference_gallery)}/{self.max_reference_samples} "
            f"for '{identity_id}' (quality={q_score:.2f}, prototypes updated)"
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
        2. Candidate is confirmed to match the logical identity and prototype.
        3. Crop provides appearance diversity (not redundant).
        """
        if crop is None or crop.size == 0:
            return False

        ident = self._identities.get(identity_id)
        if ident is None or not ident.reference_gallery:
            return False

        is_valid, q_score, reason = self.quality.evaluate(crop)
        if not is_valid:
            logger.debug(f"[IDENTITY] Adaptive update REJECTED for '{identity_id}': poor quality ({reason})")
            return False

        eval_res = self.evaluate_candidate_crop(crop, identity_id)
        if not eval_res.is_match or eval_res.best_ref_sim < self.reference_threshold:
            logger.warning(
                f"[IDENTITY] Adaptive update REJECTED for '{identity_id}': failed match or drifted from reference "
                f"(score={eval_res.candidate_score:.3f}, ref_sim={eval_res.best_ref_sim:.3f} < {self.reference_threshold:.2f})"
            )
            return False

        fused_emb = self.reid.extract(crop)

        # Diversity check
        for existing_emb in ident.adaptive_gallery:
            sim = existing_emb.cosine_similarity(fused_emb)
            if sim > self.redundancy_threshold:
                logger.debug(
                    f"[IDENTITY] Adaptive update SKIPPED for '{identity_id}': redundant (sim={sim:.3f} > {self.redundancy_threshold:.2f})"
                )
                return False

        ident.last_seen_timestamp_ms = timestamp_ms

        if len(ident.adaptive_gallery) < self.max_gallery_size:
            ident.adaptive_gallery.append(fused_emb)
        else:
            ident.adaptive_gallery.pop(0)
            ident.adaptive_gallery.append(fused_emb)

        # Re-sync vector store with prototype + references + adaptive galleries
        self.vector_store.remove_identity(identity_id)
        if ident.reference_prototype is not None:
            self.vector_store.add(ident.reference_prototype, identity_id)
        for ref_emb in ident.reference_gallery:
            self.vector_store.add(ref_emb, identity_id)
        for ad_emb in ident.adaptive_gallery:
            self.vector_store.add(ad_emb, identity_id)

        logger.debug(
            f"[IDENTITY] Added adaptive observation for '{identity_id}' (candidate_score={eval_res.candidate_score:.3f}, "
            f"adaptive_count={len(ident.adaptive_gallery)}/{self.max_gallery_size})"
        )
        return True

    def evaluate_candidate_crop(
        self,
        crop: np.ndarray,
        identity_id: str,
    ) -> CandidateEvaluation:
        """
        Performs decomposed ReID extraction, score-level component consensus scoring,
        and clean gate validation without artificial substitutions.
        """
        ident = self.get_identity(identity_id)
        if ident is None or crop is None or crop.size == 0 or not ident.reference_gallery:
            return CandidateEvaluation(
                is_match=False,
                decision="NO_MATCH",
                candidate_score=0.0,
                proto_sim=0.0,
                best_ref_sim=0.0,
                best_adaptive_sim=0.0,
                top2_adaptive_mean=0.0,
                deep_sim=0.0,
                color_sim=0.0,
                upper_sim=0.0,
                lower_sim=0.0,
                fused_sim=0.0,
                feature_agreement_passed=False,
                crop_quality_score=0.0,
                quality_reason="NO_IDENTITY_OR_EMPTY_CROP",
                rejection_reasons=["No identity registered or empty crop"],
            )

        is_valid, q_score, q_reason = self.quality.evaluate(crop)

        # Decomposed feature extraction
        fused_emb, deep_emb, color_emb, upper_emb, lower_emb = self._extract_all_representations(crop)

        # 1. Component similarities against individual prototypes & reference galleries (no cross-fallbacks)
        # Deep
        if ident.reference_deep_proto is not None and ident.reference_deep_proto.dim == deep_emb.dim:
            deep_proto_sim = ident.reference_deep_proto.cosine_similarity(deep_emb)
        else:
            deep_proto_sim = 0.0

        if ident.reference_deep_gallery and ident.reference_deep_gallery[0].dim == deep_emb.dim:
            best_ref_deep = max(r.cosine_similarity(deep_emb) for r in ident.reference_deep_gallery)
        else:
            best_ref_deep = deep_proto_sim
        deep_sim = max(deep_proto_sim, best_ref_deep)

        # Upper Body
        if ident.reference_upper_proto is not None and ident.reference_upper_proto.dim == upper_emb.dim:
            upper_proto_sim = ident.reference_upper_proto.cosine_similarity(upper_emb)
        else:
            upper_proto_sim = 0.0

        if ident.reference_upper_gallery and ident.reference_upper_gallery[0].dim == upper_emb.dim:
            best_ref_upper = max(r.cosine_similarity(upper_emb) for r in ident.reference_upper_gallery)
        else:
            best_ref_upper = upper_proto_sim
        upper_sim = max(upper_proto_sim, best_ref_upper)

        # Lower Body
        if ident.reference_lower_proto is not None and ident.reference_lower_proto.dim == lower_emb.dim:
            lower_proto_sim = ident.reference_lower_proto.cosine_similarity(lower_emb)
        else:
            lower_proto_sim = 0.0

        if ident.reference_lower_gallery and ident.reference_lower_gallery[0].dim == lower_emb.dim:
            best_ref_lower = max(r.cosine_similarity(lower_emb) for r in ident.reference_lower_gallery)
        else:
            best_ref_lower = lower_proto_sim
        lower_sim = max(lower_proto_sim, best_ref_lower)

        # Color & Texture
        if ident.reference_color_proto is not None and ident.reference_color_proto.dim == color_emb.dim:
            color_proto_sim = ident.reference_color_proto.cosine_similarity(color_emb)
        else:
            color_proto_sim = 0.0

        if ident.reference_color_gallery and ident.reference_color_gallery[0].dim == color_emb.dim:
            best_ref_color = max(r.cosine_similarity(color_emb) for r in ident.reference_color_gallery)
        else:
            best_ref_color = color_proto_sim
        color_sim = max(color_proto_sim, best_ref_color)

        # Fused & Adaptive Gallery similarities
        proto_sim = ident.reference_prototype.cosine_similarity(fused_emb) if (ident.reference_prototype and ident.reference_prototype.dim == fused_emb.dim) else 0.0
        ref_fused_sims = [ref.cosine_similarity(fused_emb) for ref in ident.reference_gallery if ref.dim == fused_emb.dim]
        best_ref_sim = max(ref_fused_sims) if ref_fused_sims else proto_sim
        fused_sim = proto_sim

        best_adaptive_sim = 0.0
        top2_adaptive_mean = 0.0
        if ident.adaptive_gallery:
            adaptive_sims = sorted(
                [ad.cosine_similarity(fused_emb) for ad in ident.adaptive_gallery if ad.dim == fused_emb.dim],
                reverse=True,
            )
            if adaptive_sims:
                best_adaptive_sim = adaptive_sims[0]
                top2_adaptive_mean = sum(adaptive_sims[:2]) / len(adaptive_sims[:2])

        # 2. Score-Level Consensus
        component_score = (
            self.w_upper * upper_sim
            + self.w_color * color_sim
            + self.w_deep * deep_sim
            + self.w_lower * lower_sim
        )

        if ident.adaptive_gallery and top2_adaptive_mean > 0:
            candidate_score = 0.70 * component_score + 0.30 * top2_adaptive_mean
        else:
            candidate_score = component_score

        # 3. Decision Evaluation
        reasons: List[str] = []
        ref_anchor_score = max(proto_sim, best_ref_sim, deep_sim)

        is_ref_pass = (ref_anchor_score >= self.reference_threshold or deep_sim >= self.reference_threshold)
        is_score_pass = (candidate_score >= self.similarity_threshold)
        is_upper_pass = (upper_sim >= self.upper_threshold)
        is_qual_pass = is_valid

        if not is_qual_pass:
            reasons.append(f"Poor crop quality: {q_reason}")
        if not is_ref_pass:
            reasons.append(f"RefAnchor {ref_anchor_score:.3f} < {self.reference_threshold:.2f}")
        if not is_upper_pass:
            reasons.append(f"UpperBody Sim {upper_sim:.3f} < {self.upper_threshold:.2f}")
        if not is_score_pass:
            reasons.append(f"CandidateScore {candidate_score:.3f} < {self.similarity_threshold:.2f}")

        is_match = (is_qual_pass and is_ref_pass and is_score_pass and is_upper_pass)
        decision = "MATCH" if is_match else "NO_MATCH"

        return CandidateEvaluation(
            is_match=is_match,
            decision=decision,
            candidate_score=candidate_score,
            proto_sim=proto_sim,
            best_ref_sim=best_ref_sim,
            best_adaptive_sim=best_adaptive_sim,
            top2_adaptive_mean=top2_adaptive_mean,
            deep_sim=deep_sim,
            color_sim=color_sim,
            upper_sim=upper_sim,
            lower_sim=lower_sim,
            fused_sim=fused_sim,
            feature_agreement_passed=True,
            crop_quality_score=q_score,
            quality_reason=q_reason,
            rejection_reasons=reasons,
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
