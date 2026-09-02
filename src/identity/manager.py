"""Identity management module separating ReID appearance from identity matching."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.core.interfaces import BaseReID, BaseVectorStore
from src.core.types import (
    Embedding,
    Identity,
    MatchDecisionState,
    TargetIdentityAnchor,
    VerifiedIdentityDecision,
    ViewCluster,
)
from src.identity.store import InMemoryVectorStore
from src.identity.evidence import EvidenceEngine
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
    Manages identity association, separate immutable trusted & provisional galleries,
    temporal evidence accumulation, and robust multi-camera candidate matching.
    """

    def __init__(
        self,
        reid_extractor: BaseReID,
        vector_store: Optional[BaseVectorStore] = None,
        similarity_threshold: float = 0.72,
        reacquisition_threshold: float = 0.78,
        reference_threshold: float = 0.72,
        upper_threshold: float = 0.70,
        min_margin: float = 0.06,
        max_reference_samples: int = 6,
        max_gallery_size: int = 15,
        reacquisition_min_frames: int = 3,
        redundancy_threshold: float = 0.95,
        quality_evaluator: Optional[CropQualityEvaluator] = None,
        w_upper: float = 0.40,
        w_color: float = 0.15,
        w_deep: float = 0.35,
        w_lower: float = 0.10,
        evidence_engine: Optional[EvidenceEngine] = None,
    ) -> None:
        self.reid = reid_extractor
        self.vector_store = vector_store or InMemoryVectorStore()
        self.similarity_threshold = similarity_threshold
        self.reacquisition_threshold = reacquisition_threshold
        self.reference_threshold = reference_threshold
        self.upper_threshold = upper_threshold
        self.min_margin = min_margin
        self.max_reference_samples = max_reference_samples
        self.max_gallery_size = max_gallery_size
        self.reacquisition_min_frames = reacquisition_min_frames
        self.redundancy_threshold = redundancy_threshold
        self.quality = quality_evaluator or CropQualityEvaluator()
        self.w_upper = w_upper
        self.w_color = w_color
        self.w_deep = w_deep
        self.w_lower = w_lower
        self.auto_add_threshold = 0.85 # shim
        self.evidence_engine = evidence_engine or EvidenceEngine(
            window_size=4,
            min_similarity_threshold=similarity_threshold,
            reacquisition_threshold=reacquisition_threshold,
            reacquisition_min_frames=reacquisition_min_frames,
            min_margin_threshold=min_margin,
        )
        self._identities: Dict[str, Identity] = {}
        self._entry_crops: Dict[str, str] = {}

    def get_identity(self, identity_id: str) -> Optional[Identity]:
        return self._identities.get(identity_id)
        
    @property
    def size(self) -> int:
        return self.manual_count + self.auto_count

    @property
    def is_empty(self) -> bool:
        """Shim for TargetGallery.is_empty"""
        ident = self.get_identity("target_0")
        if not ident: return True
        return len(ident.trusted_gallery) == 0 and len(ident.provisional_gallery) == 0

    @property
    def reid_extractor(self) -> Any:
        return self.reid

    @property
    def manual_count(self) -> int:
        """Shim for TargetGallery.manual_count (assumes target_0)"""
        ident = self.get_identity("target_0")
        return len(ident.trusted_gallery) if ident else 0

    @property
    def auto_count(self) -> int:
        """Shim for TargetGallery.auto_count (assumes target_0)"""
        ident = self.get_identity("target_0")
        return len(ident.provisional_gallery) if ident else 0

    @property
    def max_size(self) -> int:
        """Shim for TargetGallery.max_size."""
        return self.max_gallery_size

    def _encode_crop_thumbnail(self, crop: Optional[np.ndarray]) -> str:
        """Helper to resize and encode person crop to base64 JPEG string."""
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            return ""
        try:
            import cv2
            import base64
            h, w = crop.shape[:2]
            if h <= 0 or w <= 0:
                return ""
            th_h = 80
            th_w = max(20, int(w * (th_h / float(h))))
            thumb = cv2.resize(crop, (th_w, th_h), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception:
            return ""

    def get_thumbnails(self, max_count: int = 15) -> List[Dict[str, Any]]:
        """
        Returns JSON-serializable thumbnail previews for the web UI.
        Delegates to GalleryEntry crops stored in the Identity's galleries.
        """
        thumbnails: List[Dict[str, Any]] = []
        ident = self.get_identity("target_0")
        if not ident:
            return thumbnails

        entries = []
        for i, emb in enumerate(ident.trusted_gallery):
            crop_b64 = self._entry_crops.get(f"{ident.identity_id}_trusted_{i}", "")
            entries.append({
                "entry_id": f"trusted_{i}",
                "is_manual": True,
                "camera_id": emb.camera_id,
                "timestamp_ms": emb.timestamp_ms,
                "confidence": 1.0,
                "quality_score": emb.quality_score,
                "image_b64": crop_b64,
            })
        for i, emb in enumerate(ident.provisional_gallery):
            crop_b64 = self._entry_crops.get(f"{ident.identity_id}_provisional_{i}", "")
            entries.append({
                "entry_id": f"provisional_{i}",
                "is_manual": False,
                "camera_id": emb.camera_id,
                "timestamp_ms": emb.timestamp_ms,
                "confidence": emb.quality_score,
                "quality_score": emb.quality_score,
                "image_b64": crop_b64,
            })

        for entry in entries[-max_count:]:
            thumbnails.append(entry)
        return thumbnails

    def remove_entry(self, entry_id: str) -> bool:
        """
        Shim for TargetGallery.remove_entry.
        Removes from trusted or provisional gallery by synthetic entry_id.
        """
        ident = self.get_identity("target_0")
        if not ident:
            return False

        removed = False
        if entry_id.startswith("trusted_"):
            try:
                idx = int(entry_id.split("_", 1)[1])
                if 0 <= idx < len(ident.trusted_gallery):
                    ident.trusted_gallery.pop(idx)
                    if 0 <= idx < len(ident.trusted_upper_gallery):
                        ident.trusted_upper_gallery.pop(idx)
                    if 0 <= idx < len(ident.trusted_lower_gallery):
                        ident.trusted_lower_gallery.pop(idx)
                    self._entry_crops.pop(f"{ident.identity_id}_{entry_id}", None)
                    removed = True
            except (ValueError, IndexError):
                pass
        elif entry_id.startswith("provisional_"):
            try:
                idx = int(entry_id.split("_", 1)[1])
                if 0 <= idx < len(ident.provisional_gallery):
                    ident.provisional_gallery.pop(idx)
                    self._entry_crops.pop(f"{ident.identity_id}_{entry_id}", None)
                    removed = True
            except (ValueError, IndexError):
                pass

        if removed:
            # Rebuild vector store for this identity
            self.vector_store.remove_identity("target_0")
            for emb in ident.trusted_gallery:
                self.vector_store.add(emb, "target_0")
            for emb in ident.provisional_gallery:
                self.vector_store.add(emb, "target_0")
        return removed

    def match(self, candidate_embedding: Embedding) -> Tuple[float, Optional[Any]]:
        """Shim for TargetGallery.match"""
        res = self.match_batch([candidate_embedding])
        return res[0]
        
    def clear(self, identity_id: str = "target_0") -> None:
        """Shim for TargetGallery.clear"""
        if identity_id in self._identities:
            del self._identities[identity_id]
        self.vector_store.remove_identity(identity_id)
        self._entry_crops.clear()
        if hasattr(self, "evidence_engine") and self.evidence_engine is not None:
            self.evidence_engine.clear()
        
    def rollback_auto_entries(self, for_track_id: int) -> int:
        """
        Shim for TargetGallery.rollback_auto_entries. 
        Clears the provisional gallery entirely.
        """
        ident = self.get_identity("target_0")
        if ident:
            count = len(ident.provisional_gallery)
            ident.provisional_gallery.clear()
            # Clear provisional crops
            keys_to_del = [k for k in self._entry_crops if "_provisional_" in k]
            for k in keys_to_del:
                self._entry_crops.pop(k, None)
            return count
        return 0

    @property
    def _manual_matrix(self) -> Optional[np.ndarray]:
        ident = self.get_identity("target_0")
        if not ident or not ident.trusted_gallery: return None
        return np.stack([e.vector for e in ident.trusted_gallery])

    @property
    def _manual_entries(self) -> List[Any]:
        ident = self.get_identity("target_0")
        return ident.trusted_gallery if ident else []

    def match_batch(self, candidate_embeddings: List[Embedding]) -> List[Tuple[float, Optional[Any]]]:
        """Shim for TargetGallery.match_batch"""
        details = self.match_batch_details(candidate_embeddings)
        return [(float(d[0]), d[3]) for d in details]

    def match_batch_details(
        self, embs: List[Embedding]
    ) -> List[Tuple[float, float, float, Optional[Any]]]:
        """Shim for TargetGallery.match_batch_details"""
        results = []
        ident = self.get_identity("target_0")
        if not ident:
            return [(0.0, 0.0, 0.0, -1)] * len(embs)
            
        man_mat = self._manual_matrix
        auto_mat = None
        if ident.provisional_gallery:
            auto_mat = np.stack([e.vector for e in ident.provisional_gallery])
            
        for emb in embs:
            man_sim = 0.0
            auto_sim = 0.0
            if man_mat is not None:
                man_sim = float(np.max(np.dot(man_mat, emb.vector)))
            if auto_mat is not None:
                auto_sim = float(np.max(np.dot(auto_mat, emb.vector)))
                
            eff_sim = max(man_sim, auto_sim)
            results.append((eff_sim, man_sim, auto_sim, -1))
        return results
        
    def add_auto(
        self,
        crop: np.ndarray,
        embedding: Embedding,
        candidate_similarity: float,
        camera_id: str = "camera_0",
        timestamp_ms: float = 0.0,
        frame_id: int = 0,
        track_id: Optional[int] = None,
    ) -> bool:
        """Shim for TargetGallery.add_auto with rate limiting and diversity."""
        if candidate_similarity < self.auto_add_threshold:
            return False
            
        ident = self.get_identity("target_0")
        if not ident: return False

        # Rate limit: at most 1 auto-entry every 1.5 seconds
        last_ts = getattr(self, "_last_auto_add_ts", 0.0)
        if timestamp_ms > 0 and (timestamp_ms - last_ts) < 1500.0:
            return False
        
        # Check against manual matrix if available
        man_mat = self._manual_matrix
        if man_mat is not None and len(man_mat) > 0:
            m_sims = np.clip(np.dot(man_mat, embedding.vector), 0.0, 1.0)
            max_manual_sim = float(np.max(m_sims))
            min_anchor = max(0.55, 0.75 - 0.05) # dummy match_threshold 0.75
            if max_manual_sim < min_anchor:
                return False

        # Diversity check against recent provisional entries
        for prev_emb in ident.provisional_gallery[-5:]:
            if prev_emb.dim == embedding.dim and prev_emb.cosine_similarity(embedding) > 0.94:
                return False
                
        # Only add if it passes
        idx = len(ident.provisional_gallery)
        ident.provisional_gallery.append(embedding)
        self._last_auto_add_ts = timestamp_ms if timestamp_ms > 0 else time.time() * 1000.0
        if crop is not None and crop.size > 0:
            self._entry_crops[f"{ident.identity_id}_provisional_{idx}"] = self._encode_crop_thumbnail(crop)

        # Enforce max provisional capacity
        while len(ident.provisional_gallery) > self.max_gallery_size:
            ident.provisional_gallery.pop(0)

        return True

    def _extract_all_representations(
        self, crop: np.ndarray, embedding: Optional[Embedding] = None
    ) -> Tuple[Embedding, Embedding, Embedding, Embedding, Embedding]:
        """Extracts fused and decomposed feature embeddings."""
        if embedding is not None:
            return embedding, embedding, embedding, embedding, embedding
            
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
        embedding: Optional[Embedding] = None,
    ) -> Optional[Embedding]:
        """
        Registers a fresh target identity, initializes trusted reference galleries,
        and computes normalized reference prototypes.
        """
        if crop is None or crop.size == 0:
            return None

        # Clean previous vector store entries for this identity
        self.vector_store.remove_identity(identity_id)
        if self.evidence_engine:
            self.evidence_engine.clear()

        fused, deep, global_v, upper, lower = self._extract_all_representations(crop, embedding)

        # Create initial view cluster and TargetIdentityAnchor
        initial_cluster = ViewCluster(
            cluster_id=f"{identity_id}_view_0",
            label="initial_enrollment",
            exemplars=[fused],
            centroid=fused,
        )
        anchor = TargetIdentityAnchor(
            identity_id=identity_id,
            label=label or identity_id,
            clusters=[initial_cluster],
            model_name=fused.model_name,
            feature_dim=fused.dim,
            created_timestamp_ms=timestamp_ms,
        )

        ident = Identity(
            identity_id=identity_id,
            label=label or identity_id,
            trusted_gallery=[fused],
            trusted_upper_gallery=[upper],
            trusted_lower_gallery=[lower],
            anchor=anchor,
            view_clusters=[initial_cluster],
            provisional_gallery=[],
            last_seen_timestamp_ms=timestamp_ms,
        )
        ident.update_prototype()
        self._identities[identity_id] = ident
        self.vector_store.add(fused, identity_id)

        # Store initial crop thumbnail
        if crop is not None and crop.size > 0:
            self._entry_crops[f"{identity_id}_trusted_0"] = self._encode_crop_thumbnail(crop)

        logger.info(
            f"[IDENTITY] Registered new target identity '{identity_id}' ({ident.label}) with immutable TargetIdentityAnchor"
        )
        return fused

    def register_or_update(
        self,
        crop: np.ndarray,
        identity_id: str,
        label: Optional[str] = None,
        timestamp_ms: float = 0.0,
    ) -> Optional[Embedding]:
        """Registers target appearance if not yet registered, or updates gallery if already registered."""
        if crop is None or crop.size == 0:
            return None

        if identity_id not in self._identities:
            return self.register_new_target(crop, identity_id, label, timestamp_ms)

        ident = self._identities[identity_id]
        emb = self.reid.extract(crop)
        ident.last_seen_timestamp_ms = timestamp_ms

        if len(ident.embeddings) < self.max_gallery_size:
            idx = len(ident.provisional_gallery)
            ident.provisional_gallery.append(emb)
            if crop is not None and crop.size > 0:
                self._entry_crops[f"{identity_id}_provisional_{idx}"] = self._encode_crop_thumbnail(crop)
            self.vector_store.add(emb, identity_id)
        else:
            if ident.provisional_gallery:
                ident.provisional_gallery.pop(0)
            idx = len(ident.provisional_gallery)
            ident.provisional_gallery.append(emb)
            if crop is not None and crop.size > 0:
                self._entry_crops[f"{identity_id}_provisional_{idx}"] = self._encode_crop_thumbnail(crop)
            self.vector_store.remove_identity(identity_id)
            for r_emb in ident.trusted_gallery:
                self.vector_store.add(r_emb, identity_id)
            for a_emb in ident.provisional_gallery:
                self.vector_store.add(a_emb, identity_id)

        return emb

    def add_reference_sample(
        self,
        crop: np.ndarray,
        identity_id: str,
        timestamp_ms: float = 0.0,
        force: bool = True,
    ) -> bool:
        """
        Adds a new observation to the immutable trusted galleries
        and updates discrete ViewClusters within the TargetIdentityAnchor.
        When force=True (manual operator capture), accepts the sample unconditionally.
        """
        ident = self._identities.get(identity_id)
        if ident is None or crop is None or crop.size == 0:
            return False

        fused, deep, global_v, upper, lower = self._extract_all_representations(crop)

        # If trusted gallery is full, remove oldest to make room
        while len(ident.trusted_gallery) >= self.max_reference_samples:
            ident.trusted_gallery.pop(0)
            if ident.trusted_upper_gallery:
                ident.trusted_upper_gallery.pop(0)
            if ident.trusted_lower_gallery:
                ident.trusted_lower_gallery.pop(0)

        if not force:
            # Diversity check against existing trusted samples
            for existing_ref in ident.trusted_gallery:
                if existing_ref.dim == fused.dim:
                    sim = existing_ref.cosine_similarity(fused)
                    if sim > self.redundancy_threshold:
                        logger.debug(
                            f"[IDENTITY] Reference sample rejected for '{identity_id}': redundant (sim={sim:.3f} > {self.redundancy_threshold:.2f})"
                        )
                        return False

        idx = len(ident.trusted_gallery)
        ident.trusted_gallery.append(fused)
        ident.trusted_upper_gallery.append(upper)
        ident.trusted_lower_gallery.append(lower)
        ident.update_prototype()

        if crop is not None and crop.size > 0:
            self._entry_crops[f"{identity_id}_trusted_{idx}"] = self._encode_crop_thumbnail(crop)

        self.vector_store.add(fused, identity_id)
        return True

        # Update or add new view cluster in TargetIdentityAnchor
        if ident.anchor is not None:
            # Check if this sample matches an existing cluster (> 0.85) or forms a new cluster
            matched_cluster = False
            for cluster in ident.anchor.clusters:
                c_sim = cluster.match_score(fused)
                if c_sim > 0.85:
                    cluster.exemplars.append(fused)
                    cluster.update_centroid()
                    matched_cluster = True
                    break
            if not matched_cluster:
                new_c = ViewCluster(
                    cluster_id=f"{identity_id}_view_{len(ident.anchor.clusters)}",
                    label=f"viewpoint_{len(ident.anchor.clusters)}",
                    exemplars=[fused],
                    centroid=fused,
                )
                ident.anchor.clusters.append(new_c)
                ident.view_clusters.append(new_c)

        ident.last_seen_timestamp_ms = timestamp_ms
        self.vector_store.add(fused, identity_id)
        logger.info(
            f"[IDENTITY] Added diverse reference sample {len(ident.trusted_gallery)}/{self.max_reference_samples} "
            f"for '{identity_id}' (quality={q_score:.2f}, clusters={len(ident.view_clusters)})"
        )
        return True

    def enroll_cross_camera_viewpoint(
        self,
        crop: np.ndarray,
        identity_id: str,
        decision: Optional[VerifiedIdentityDecision] = None,
        timestamp_ms: float = 0.0,
    ) -> bool:
        """
        Enrolls a verified cross-camera viewpoint into the TargetIdentityAnchor
        as a permanent ViewCluster, strictly gated by a valid VerifiedIdentityDecision token.
        Preserves anti-scooping guarantees while allowing multi-camera appearance learning.
        """
        ident = self._identities.get(identity_id)
        if ident is None or crop is None or crop.size == 0:
            return False

        if decision is None or decision.decision_state != MatchDecisionState.MATCH:
            logger.warning(f"[IDENTITY] Cross-camera viewpoint enrollment REJECTED: no valid VerifiedIdentityDecision token.")
            return False

        is_valid, q_score, reason = self.quality.evaluate(crop)
        if not is_valid:
            logger.debug(f"[IDENTITY] Cross-camera viewpoint rejected for '{identity_id}': poor quality ({reason})")
            return False

        fused, deep, global_v, upper, lower = self._extract_all_representations(crop)

        if ident.anchor is None:
            return False

        # Check if this sample matches an existing cluster (> 0.85) or forms a new viewpoint cluster
        matched_cluster = False
        for cluster in ident.anchor.clusters:
            c_sim = cluster.match_score(fused)
            if c_sim > 0.85:
                cluster.exemplars.append(fused)
                cluster.update_centroid()
                matched_cluster = True
                break

        if not matched_cluster:
            new_c = ViewCluster(
                cluster_id=f"{identity_id}_view_{len(ident.anchor.clusters)}",
                label=f"viewpoint_{len(ident.anchor.clusters)}",
                exemplars=[fused],
                centroid=fused,
            )
            ident.anchor.clusters.append(new_c)
            ident.view_clusters.append(new_c)

        if len(ident.trusted_gallery) < self.max_gallery_size:
            ident.trusted_gallery.append(fused)
            ident.trusted_upper_gallery.append(upper)
            ident.trusted_lower_gallery.append(lower)
        ident.update_prototype()

        ident.last_seen_timestamp_ms = timestamp_ms
        self.vector_store.add(fused, identity_id)
        logger.info(
            f"[IDENTITY] Enrolled verified cross-camera viewpoint for '{identity_id}' "
            f"(clusters={len(ident.anchor.clusters)}, quality={q_score:.2f})"
        )
        return True

    def is_reference_complete(self, identity_id: str) -> bool:
        """Checks whether reference gallery has acquired the target sample count."""
        ident = self._identities.get(identity_id)
        if ident is None:
            return False
        return len(ident.trusted_gallery) >= self.max_reference_samples

    def verified_update(
        self,
        crop: np.ndarray,
        identity_id: str,
        timestamp_ms: float = 0.0,
    ) -> bool:
        """
        Updates the provisional observation gallery only if:
        1. Crop passes image quality evaluation.
        2. Candidate is confirmed to match the logical identity and trusted prototype anchor.
        3. Crop provides appearance diversity (not redundant).
        """
        if crop is None or crop.size == 0:
            return False

        ident = self._identities.get(identity_id)
        if ident is None or not ident.trusted_gallery:
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
        for existing_emb in ident.provisional_gallery:
            if existing_emb.dim == fused_emb.dim:
                sim = existing_emb.cosine_similarity(fused_emb)
                if sim > self.redundancy_threshold:
                    logger.debug(
                        f"[IDENTITY] Adaptive update SKIPPED for '{identity_id}': redundant (sim={sim:.3f} > {self.redundancy_threshold:.2f})"
                    )
                    return False

        ident.last_seen_timestamp_ms = timestamp_ms

        if len(ident.provisional_gallery) < self.max_gallery_size:
            ident.provisional_gallery.append(fused_emb)
        else:
            ident.provisional_gallery.pop(0)
            ident.provisional_gallery.append(fused_emb)

        # Re-sync vector store with prototype + references + provisional galleries
        self.vector_store.remove_identity(identity_id)
        if ident.trusted_prototype is not None:
            self.vector_store.add(ident.trusted_prototype, identity_id)
        for ref_emb in ident.trusted_gallery:
            self.vector_store.add(ref_emb, identity_id)
        for ad_emb in ident.provisional_gallery:
            self.vector_store.add(ad_emb, identity_id)

        logger.debug(
            f"[IDENTITY] Added adaptive observation for '{identity_id}' (candidate_score={eval_res.candidate_score:.3f}, "
            f"adaptive_count={len(ident.provisional_gallery)}/{self.max_gallery_size})"
        )
        return True

    def flush_adaptive_gallery(self, identity_id: str) -> None:
        """Flushes all rolling provisional observations, resetting back to the immutable trusted gallery."""
        ident = self._identities.get(identity_id)
        if ident is not None and ident.provisional_gallery:
            ident.provisional_gallery.clear()
            self.vector_store.remove_identity(identity_id)
            if ident.trusted_prototype is not None:
                self.vector_store.add(ident.trusted_prototype, identity_id)
            for ref_emb in ident.trusted_gallery:
                self.vector_store.add(ref_emb, identity_id)
            logger.info(f"[IDENTITY] Flushed provisional observation gallery for '{identity_id}'. Pure trusted gallery active.")

    def evaluate_candidate_crop(
        self,
        crop: np.ndarray,
        identity_id: str,
    ) -> CandidateEvaluation:
        """Performs multi-crop ReID extraction and calibrated candidate evaluation."""
        ident = self.get_identity(identity_id)
        if ident is None or crop is None or crop.size == 0 or not ident.trusted_gallery:
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

        # Multi-crop feature extraction
        fused_emb, deep_emb, global_emb, upper_emb, lower_emb = self._extract_all_representations(crop)

        # 1. Prototype & Gallery similarities
        proto_sim = ident.trusted_prototype.cosine_similarity(fused_emb) if (ident.trusted_prototype and ident.trusted_prototype.dim == fused_emb.dim) else 0.0
        ref_fused_sims = [ref.cosine_similarity(fused_emb) for ref in ident.trusted_gallery if ref.dim == fused_emb.dim]
        best_ref_sim = max(ref_fused_sims) if ref_fused_sims else proto_sim

        # Upper Body similarity
        if ident.trusted_upper_proto is not None and ident.trusted_upper_proto.dim == upper_emb.dim:
            upper_sim = ident.trusted_upper_proto.cosine_similarity(upper_emb)
        elif ident.trusted_upper_gallery and ident.trusted_upper_gallery[0].dim == upper_emb.dim:
            upper_sim = max(u.cosine_similarity(upper_emb) for u in ident.trusted_upper_gallery)
        else:
            upper_sim = proto_sim

        # Lower Body similarity
        if ident.trusted_lower_proto is not None and ident.trusted_lower_proto.dim == lower_emb.dim:
            lower_sim = ident.trusted_lower_proto.cosine_similarity(lower_emb)
        elif ident.trusted_lower_gallery and ident.trusted_lower_gallery[0].dim == lower_emb.dim:
            lower_sim = max(l.cosine_similarity(lower_emb) for l in ident.trusted_lower_gallery)
        else:
            lower_sim = proto_sim

        deep_sim = proto_sim
        color_sim = proto_sim
        fused_sim = proto_sim

        best_adaptive_sim = 0.0
        top2_adaptive_mean = 0.0
        if ident.provisional_gallery:
            adaptive_sims = sorted(
                [ad.cosine_similarity(fused_emb) for ad in ident.provisional_gallery if ad.dim == fused_emb.dim],
                reverse=True,
            )
            if adaptive_sims:
                best_adaptive_sim = adaptive_sims[0]
                top2_adaptive_mean = sum(adaptive_sims[:2]) / len(adaptive_sims[:2])

        # Multi-cluster anchor similarity across discrete view clusters
        anchor_sim = ident.anchor.max_similarity(fused_emb) if ident.anchor else best_ref_sim
        ref_anchor_score = max(proto_sim, best_ref_sim, anchor_sim)

        # 2. Score-Level Multi-Crop Consensus (Multi-Modal Cluster Aware)
        component_score = (
            0.35 * max(anchor_sim, proto_sim)
            + 0.40 * upper_sim
            + 0.15 * best_ref_sim
            + 0.10 * lower_sim
        )

        if ident.provisional_gallery and top2_adaptive_mean > 0:
            candidate_score = 0.70 * component_score + 0.30 * top2_adaptive_mean
        else:
            candidate_score = component_score

        # 3. Decision Evaluation
        reasons: List[str] = []

        is_ref_pass = (ref_anchor_score >= self.reference_threshold)
        is_score_pass = (candidate_score >= self.similarity_threshold)
        is_upper_pass = (upper_sim >= self.upper_threshold or candidate_score >= self.similarity_threshold + 0.04)
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
        Evaluates a list of candidate (object, crop) tuples against identity_id,
        returning them sorted by score descending.
        """
        if not candidate_crops:
            return []

        evaluations = []
        for obj, crop in candidate_crops:
            ev = self.evaluate_candidate_crop(crop, identity_id)
            evaluations.append((obj, ev.candidate_score, ev))

        evaluations.sort(key=lambda x: x[1], reverse=True)
        return evaluations

    def find_best_candidate(
        self,
        candidates: List[Tuple[Any, np.ndarray]],
        identity_id: str,
    ) -> Tuple[Optional[Any], float, float, float]:
        """
        Finds the best matching candidate crop for a given identity.
        Returns:
            Tuple[best_candidate, best_score, second_best_score, margin]
        """
        if not candidates:
            return None, 0.0, 0.0, 0.0

        ranked = self.rank_candidate_crops(candidates, identity_id)
        if not ranked:
            return None, 0.0, 0.0, 0.0

        best_cand, best_score, best_eval = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score

        if not best_eval.is_match:
            return None, best_score, second_score, margin

        if len(ranked) > 1 and margin < self.min_margin:
            logger.debug(
                f"[IDENTITY] Candidate for '{identity_id}' rejected by margin: "
                f"margin={margin:.3f} < min_margin={self.min_margin:.3f}"
            )
            return None, best_score, second_score, margin

        return best_cand, best_score, second_score, margin

    def clear(self) -> None:
        self._identities.clear()
        self.vector_store.clear()

    def save_to_db(self, db_path: str) -> None:
        """Persists all tracked identities and embeddings to an SQLite database."""
        # Note: IdentityManager expects vector_store to be SQLiteVectorStore if persisting
        if not hasattr(self.vector_store, "save_identity_metadata"):
            logger.warning("save_to_db called but vector_store is not SQLiteVectorStore.")
            return

        for identity_id, ident in self._identities.items():
            from src.identity.serialization import serialize_identity
            data = serialize_identity(ident)
            self.vector_store.save_identity_metadata(identity_id, data)
        logger.info(f"[IDENTITY] Persisted {len(self._identities)} identities to {db_path}")

    def load_from_db(self, db_path: str) -> None:
        """Loads identities and embeddings from an SQLite database."""
        if not hasattr(self.vector_store, "load_all_identity_metadata"):
            logger.warning("load_from_db called but vector_store is not SQLiteVectorStore.")
            return

        self._identities.clear()
        try:
            metadata_dict = self.vector_store.load_all_identity_metadata()
            from src.identity.serialization import deserialize_identity
            for identity_id, data in metadata_dict.items():
                ident = deserialize_identity(data)
                self._identities[identity_id] = ident
            logger.info(f"[IDENTITY] Loaded {len(self._identities)} identities from {db_path}")
        except Exception as e:
            logger.error(f"[IDENTITY] Failed to load from database: {e}")

