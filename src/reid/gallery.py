"""Target-only multi-image appearance gallery with exact in-memory max-similarity matching."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.core.types import Embedding, GalleryEntry
from src.reid.extractor import PyTorchReIDExtractor
from src.reid.quality import ReIDCropQuality

logger = logging.getLogger(__name__)


class TargetGallery:
    """
    In-memory multi-image gallery for the single active target.

    Invariants:
    1. Single Target Focus: Maintains a capped gallery (e.g. 20-30 entries) representing
       different viewpoints and angles of the current locked target.
    2. Protected Manual Entries: Human-confirmed additions (seed, hotkey 'a', right-click)
       have is_manual=True and are NEVER evicted to make room for auto-added entries.
    3. Auto-Growth with Eviction Policy: When the gallery reaches capacity, auto-additions
       evict the oldest auto-added entry. If all entries are manual, auto-add is skipped.
    4. Exact Vectorized Matching: Multi-image matching uses direct matrix multiplication
       (candidate @ gallery.T) taking the maximum cosine similarity.
    """

    def __init__(
        self,
        reid_extractor: Optional[PyTorchReIDExtractor] = None,
        quality_evaluator: Optional[ReIDCropQuality] = None,
        max_size: int = 25,
        match_threshold: float = 0.65,
        auto_add_threshold: float = 0.80,
        auto_add_min_consecutive: int = 3,
        diversity_threshold: float = 0.92,
    ) -> None:
        self.reid_extractor = reid_extractor
        self.quality_evaluator = quality_evaluator or ReIDCropQuality()
        self.max_size = max(5, max_size)
        self.match_threshold = match_threshold
        self.auto_add_threshold = auto_add_threshold
        self.auto_add_min_consecutive = auto_add_min_consecutive
        self.diversity_threshold = diversity_threshold

        self._entries: List[GalleryEntry] = []
        self._manual_entries: List[GalleryEntry] = []
        self._auto_entries: List[GalleryEntry] = []
        self._matrix: Optional[np.ndarray] = None  # shape: (N, 512)
        self._manual_matrix: Optional[np.ndarray] = None  # shape: (M, 512)
        self._auto_matrix: Optional[np.ndarray] = None  # shape: (A, 512)
        self._consecutive_matches: Dict[int, int] = {}  # track_id -> consecutive frames
        self._target_label: str = "target_0"

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def is_empty(self) -> bool:
        return len(self._entries) == 0

    @property
    def manual_count(self) -> int:
        return len(self._manual_entries)

    @property
    def auto_count(self) -> int:
        return len(self._auto_entries)

    def _rebuild_matrix(self) -> None:
        """Rebuilds the cached 2D numpy matrices for pooled, manual, and auto entries."""
        self._manual_entries = [e for e in self._entries if e.is_manual]
        self._auto_entries = [e for e in self._entries if not e.is_manual]

        if self._entries:
            vectors = [e.embedding.vector for e in self._entries]
            self._matrix = np.stack(vectors, axis=0)  # shape: (N, dim)
        else:
            self._matrix = None

        if self._manual_entries:
            m_vectors = [e.embedding.vector for e in self._manual_entries]
            self._manual_matrix = np.stack(m_vectors, axis=0)  # shape: (M, dim)
        else:
            self._manual_matrix = None

        if self._auto_entries:
            a_vectors = [e.embedding.vector for e in self._auto_entries]
            self._auto_matrix = np.stack(a_vectors, axis=0)  # shape: (A, dim)
        else:
            self._auto_matrix = None

    def seed(
        self,
        crop: np.ndarray,
        embedding: Optional[Embedding] = None,
        camera_id: str = "camera_0",
        timestamp_ms: float = 0.0,
        frame_id: int = 0,
        target_label: str = "target_0",
    ) -> bool:
        """
        Clears the existing gallery and seeds it with a newly selected target.
        Bypasses similarity thresholds (human-confirmed ground truth).
        Always enrolled as a permanent, protected manual entry (is_manual=True).
        """
        self.clear()
        self._target_label = target_label

        if embedding is None:
            if self.reid_extractor is None or crop is None or crop.size == 0:
                logger.error("Cannot seed gallery: missing crop or extractor")
                return False
            embedding = self.reid_extractor.extract(crop)

        q_score = 1.0
        if crop is not None and crop.size > 0:
            is_valid, q_val, _ = self.quality_evaluator.evaluate(crop)
            q_score = q_val if is_valid else 0.5

        entry = GalleryEntry(
            entry_id=f"seed_{uuid.uuid4().hex[:8]}",
            embedding=embedding,
            crop=crop.copy() if crop is not None else None,
            is_manual=True,
            timestamp_ms=timestamp_ms,
            camera_id=camera_id,
            frame_id=frame_id,
            confidence=1.0,
            quality_score=q_score,
            track_id=None,
        )
        self._entries.append(entry)
        self._rebuild_matrix()
        logger.info(
            f"[TARGET_GALLERY] Seeded new target '{self._target_label}' on camera '{camera_id}' "
            f"(dim={embedding.dim}, quality={q_score:.2f}, protected_manual=True)"
        )
        return True

    def add_manual(
        self,
        crop: np.ndarray,
        embedding: Optional[Embedding] = None,
        camera_id: str = "camera_0",
        timestamp_ms: float = 0.0,
        frame_id: int = 0,
    ) -> bool:
        """
        Manually captures and adds a new viewpoint to the existing target gallery.
        Protected entry (is_manual=True) that will not be evicted by automatic updates.
        """
        if embedding is None:
            if self.reid_extractor is None or crop is None or crop.size == 0:
                logger.error("Cannot add manual sample: missing crop or extractor")
                return False
            embedding = self.reid_extractor.extract(crop)

        q_score = 1.0
        if crop is not None and crop.size > 0:
            is_valid, q_val, _ = self.quality_evaluator.evaluate(crop)
            q_score = q_val if is_valid else 0.5

        # Capacity management for manual additions
        if len(self._entries) >= self.max_size:
            # First attempt to evict oldest auto-added entry
            auto_indices = [i for i, e in enumerate(self._entries) if not e.is_manual]
            if auto_indices:
                evicted = self._entries.pop(auto_indices[0])
                logger.info(f"[TARGET_GALLERY] Evicted auto-entry '{evicted.entry_id}' to make room for manual addition")
            else:
                # If all entries are manual, evict the oldest manual entry
                evicted = self._entries.pop(0)
                logger.info(f"[TARGET_GALLERY] Evicted oldest manual entry '{evicted.entry_id}' (gallery at capacity {self.max_size})")

        entry = GalleryEntry(
            entry_id=f"manual_{uuid.uuid4().hex[:8]}",
            embedding=embedding,
            crop=crop.copy() if crop is not None else None,
            is_manual=True,
            timestamp_ms=timestamp_ms,
            camera_id=camera_id,
            frame_id=frame_id,
            confidence=1.0,
            quality_score=q_score,
            track_id=None,
        )
        self._entries.append(entry)
        self._rebuild_matrix()
        logger.info(
            f"[TARGET_GALLERY] Added MANUAL protected sample ({self.size}/{self.max_size}) "
            f"on '{camera_id}' (manual={self.manual_count}, auto={self.auto_count})"
        )
        return True

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
        """
        Automatically accumulates a verified viewpoint when sightings meet strict criteria:
        1. candidate_similarity >= auto_add_threshold
        2. Ground-truth anchor: similarity against manual entries >= match_threshold (if manual entries exist)
        3. Consecutive match hold >= auto_add_min_consecutive
        4. Quality passes quality_evaluator
        5. Diversity check: similarity to all existing gallery entries <= diversity_threshold
        6. Evicts oldest auto-added entry if gallery is full (never evicts manual entries)
        """
        if self.is_empty:
            return False

        # 1. Similarity threshold check
        if candidate_similarity < self.auto_add_threshold:
            if track_id is not None:
                self._consecutive_matches.pop(track_id, None)
            return False

        # 2. Ground-truth manual anchor check (Anti-Contamination Guard)
        if self._manual_matrix is not None and len(self._manual_matrix) > 0:
            raw_m_dots = self._manual_matrix @ embedding.vector
            m_sims = np.clip(raw_m_dots, 0.0, 1.0)  # raw cosine for anchor check
            max_manual_sim = float(np.max(m_sims))
            min_anchor = max(0.55, self.match_threshold - 0.05)
            if max_manual_sim < min_anchor:
                logger.debug(
                    f"[TARGET_GALLERY] Auto-add rejected for Track #{track_id}: "
                    f"failed broad manual anchor check (sim_manual={max_manual_sim:.3f} < {min_anchor:.2f})"
                )
                if track_id is not None:
                    self._consecutive_matches.pop(track_id, None)
                return False
        else:
            max_manual_sim = 1.0

        # 3. Consecutive match hold
        if track_id is not None:
            self._consecutive_matches[track_id] = self._consecutive_matches.get(track_id, 0) + 1
            if self._consecutive_matches[track_id] < self.auto_add_min_consecutive:
                return False

        # 4. Quality evaluation
        if crop is not None and crop.size > 0:
            is_valid, q_val, reason = self.quality_evaluator.evaluate(crop)
            if not is_valid:
                logger.debug(f"[TARGET_GALLERY] Auto-add skipped for Track #{track_id}: poor quality ({reason})")
                return False
            q_score = q_val
        else:
            return False

        # 5. Diversity check against existing entries
        if self._matrix is not None and len(self._matrix) > 0:
            raw_dots = self._matrix @ embedding.vector
            sims = np.clip(raw_dots, 0.0, 1.0)
            max_sim_to_gallery = float(np.max(sims))
            if max_sim_to_gallery >= self.diversity_threshold:
                logger.debug(
                    f"[TARGET_GALLERY] Auto-add skipped for Track #{track_id}: redundant appearance "
                    f"(sim={max_sim_to_gallery:.3f} >= {self.diversity_threshold:.2f})"
                )
                return False

        # 6. Capacity management (evict oldest auto-entry only)
        if len(self._entries) >= self.max_size:
            auto_indices = [i for i, e in enumerate(self._entries) if not e.is_manual]
            if not auto_indices:
                logger.debug("[TARGET_GALLERY] Auto-add skipped: gallery full of protected manual entries")
                return False
            evicted = self._entries.pop(auto_indices[0])
            logger.debug(f"[TARGET_GALLERY] Evicted auto-entry '{evicted.entry_id}' for new auto viewpoint")

        entry = GalleryEntry(
            entry_id=f"auto_{uuid.uuid4().hex[:8]}",
            embedding=embedding,
            crop=crop.copy() if crop is not None else None,
            is_manual=False,
            timestamp_ms=timestamp_ms,
            camera_id=camera_id,
            frame_id=frame_id,
            confidence=candidate_similarity,
            quality_score=q_score,
            track_id=track_id,
        )
        self._entries.append(entry)
        self._rebuild_matrix()
        logger.info(
            f"[TARGET_GALLERY] Auto-added verified viewpoint ({self.size}/{self.max_size}) "
            f"on '{camera_id}' Track #{track_id} (sim={candidate_similarity:.3f}, manual_anchor={max_manual_sim:.3f})"
        )
        return True

    def match(self, candidate_embedding: Embedding) -> Tuple[float, Optional[GalleryEntry]]:
        """
        Computes ground-truth anchored max-similarity of a candidate embedding.
        Returns (effective_cosine_similarity, best_matching_entry).
        """
        if self._matrix is None or len(self._entries) == 0:
            return 0.0, None

        res = self.match_batch([candidate_embedding])
        return res[0]

    def match_batch(
        self, candidate_embeddings: List[Embedding]
    ) -> List[Tuple[float, Optional[GalleryEntry]]]:
        """
        Batch ground-truth anchored max-similarity matching for candidate persons.
        Returns list of (effective_cosine_similarity, best_matching_entry).
        """
        details = self.match_batch_details(candidate_embeddings)
        return [(d[0], d[3]) for d in details]

    def match_batch_details(
        self, candidate_embeddings: List[Embedding]
    ) -> List[Tuple[float, float, float, Optional[GalleryEntry]]]:
        """
        Detailed batch matching returning (effective_score, manual_score, auto_score, best_entry).

        Uses raw cosine similarity (dot product on L2-normalized embeddings) clamped to [0, 1].
        Every gallery image is compared individually — the MAX similarity across all gallery
        entries of each type (manual / auto) is taken as the score for that type.

        Anchors composite similarity strongly to the human-confirmed manual seed:
            effective_score = 0.70 * manual_score + 0.30 * auto_score
        """
        if not candidate_embeddings:
            return []
        if self._matrix is None or len(self._entries) == 0:
            return [(0.0, 0.0, 0.0, None) for _ in candidate_embeddings]

        # Stack candidate vectors: shape (K, dim)
        c_matrix = np.stack([c.vector for c in candidate_embeddings], axis=0)

        has_manual = self._manual_matrix is not None and len(self._manual_entries) > 0
        has_auto = self._auto_matrix is not None and len(self._auto_entries) > 0

        # Similarity floor for linear rescaling: OSNet cosine dots sit in ~0.85-1.0
        # range even across different people. Rescale [floor, 1.0] -> [0.0, 1.0] so
        # that target (raw ~0.97) scores ~0.90 while non-target (raw ~0.90) scores ~0.67.
        sim_floor = 0.70
        sim_range = 1.0 - sim_floor

        # 1. Compute manual similarities — linearly rescaled cosine dot
        if has_manual:
            raw_m_dots = c_matrix @ self._manual_matrix.T  # (K, M)
            cal_m_sim = np.clip((raw_m_dots - sim_floor) / sim_range, 0.0, 1.0).astype(np.float32)
            m_best_indices = np.argmax(cal_m_sim, axis=1)
            m_max_scores = np.max(cal_m_sim, axis=1)
        else:
            m_best_indices = np.zeros(len(candidate_embeddings), dtype=int)
            m_max_scores = np.zeros(len(candidate_embeddings), dtype=np.float32)

        # 2. Compute auto similarities — linearly rescaled cosine dot
        if has_auto:
            raw_a_dots = c_matrix @ self._auto_matrix.T  # (K, A)
            cal_a_sim = np.clip((raw_a_dots - sim_floor) / sim_range, 0.0, 1.0).astype(np.float32)
            a_best_indices = np.argmax(cal_a_sim, axis=1)
            a_max_scores = np.max(cal_a_sim, axis=1)
        else:
            a_best_indices = np.zeros(len(candidate_embeddings), dtype=int)
            a_max_scores = np.zeros(len(candidate_embeddings), dtype=np.float32)

        results: List[Tuple[float, float, float, Optional[GalleryEntry]]] = []

        for i in range(len(candidate_embeddings)):
            m_score = float(m_max_scores[i]) if has_manual else 0.0
            a_score = float(a_max_scores[i]) if has_auto else 0.0

            if has_manual and has_auto:
                # Anti-drift anchoring: 70% manual ground-truth, 30% temporal adaptive
                effective_score = 0.70 * m_score + 0.30 * a_score
                best_entry = self._manual_entries[int(m_best_indices[i])] if m_score >= a_score else self._auto_entries[int(a_best_indices[i])]
            elif has_manual:
                effective_score = m_score
                best_entry = self._manual_entries[int(m_best_indices[i])]
            elif has_auto:
                effective_score = a_score
                best_entry = self._auto_entries[int(a_best_indices[i])]
            else:
                effective_score = 0.0
                best_entry = None

            results.append((effective_score, m_score, a_score, best_entry))

        return results

    def rollback_auto_entries(
        self,
        for_track_id: Optional[int] = None,
    ) -> int:
        """
        Removes auto-enrolled entries associated with for_track_id (or all auto entries if None)
        following a lock-switch or detected drift. NEVER removes protected manual entries.

        Returns:
            Number of purged auto entries.
        """
        initial_count = len(self._entries)
        kept_entries = []
        purged_count = 0

        for entry in self._entries:
            if entry.is_manual:
                kept_entries.append(entry)
            else:
                if for_track_id is None or entry.track_id == for_track_id:
                    purged_count += 1
                else:
                    kept_entries.append(entry)

        if purged_count > 0:
            self._entries = kept_entries
            self._rebuild_matrix()
            if for_track_id is not None:
                self._consecutive_matches.pop(for_track_id, None)
            else:
                self._consecutive_matches.clear()

            logger.info(
                f"[TARGET_GALLERY] Rollback: purged {purged_count} auto-enrolled entries "
                f"(track_filter={for_track_id}). Remaining gallery size={self.size} "
                f"(manual={self.manual_count}, auto={self.auto_count})"
            )

        return purged_count

    def reset_track_consensus(self, track_id: Optional[int] = None) -> None:
        """Resets consecutive match counter for a specific track or all tracks."""
        if track_id is not None:
            self._consecutive_matches.pop(track_id, None)
        else:
            self._consecutive_matches.clear()

    def remove_entry(self, entry_id: str) -> bool:
        """Removes a specific gallery entry by entry_id and rebuilds the cosine matrix."""
        for i, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                self._entries.pop(i)
                self._rebuild_matrix()
                logger.info(f"[TARGET_GALLERY] Removed entry '{entry_id}' (remaining={self.size})")
                return True
        return False


    def clear(self) -> None:
        """Clears all entries, matrix, and state from the gallery."""
        self._entries.clear()
        self._matrix = None
        self._consecutive_matches.clear()
        self._target_label = "target_0"
        logger.info("[TARGET_GALLERY] Gallery cleared.")

    def get_entries(self) -> List[GalleryEntry]:
        """Returns shallow copy of all active gallery entries."""
        return list(self._entries)

    def get_thumbnails(self, max_count: int = 15) -> List[Dict[str, Any]]:
        """
        Returns JSON-serializable list of thumbnail previews for the web UI.
        """
        import base64
        thumbnails: List[Dict[str, Any]] = []

        # Return latest entries up to max_count
        for entry in self._entries[-max_count:]:
            b64_crop = ""
            if entry.crop is not None and entry.crop.size > 0:
                try:
                    # Resize thumbnail to standard height 80px preserving aspect ratio
                    h, w = entry.crop.shape[:2]
                    th_h = 80
                    th_w = max(20, int(w * (th_h / float(h))))
                    thumb = cv2.resize(entry.crop, (th_w, th_h), interpolation=cv2.INTER_AREA)
                    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    b64_crop = base64.b64encode(buf.tobytes()).decode("utf-8")
                except Exception:
                    pass

            thumbnails.append({
                "entry_id": entry.entry_id,
                "is_manual": entry.is_manual,
                "camera_id": entry.camera_id,
                "timestamp_ms": entry.timestamp_ms,
                "confidence": entry.confidence,
                "quality_score": entry.quality_score,
                "image_b64": b64_crop,
            })
        return thumbnails
