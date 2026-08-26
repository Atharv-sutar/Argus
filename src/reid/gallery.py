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
        match_threshold: float = 0.85,
        auto_add_threshold: float = 0.90,
        auto_add_min_consecutive: int = 3,
        diversity_threshold: float = 0.96,
    ) -> None:
        self.reid_extractor = reid_extractor
        self.quality_evaluator = quality_evaluator or ReIDCropQuality()
        self.max_size = max(5, max_size)
        self.match_threshold = match_threshold
        self.auto_add_threshold = auto_add_threshold
        self.auto_add_min_consecutive = auto_add_min_consecutive
        self.diversity_threshold = diversity_threshold

        self._entries: List[GalleryEntry] = []
        self._matrix: Optional[np.ndarray] = None  # shape: (N, 512)
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
        return sum(1 for e in self._entries if e.is_manual)

    @property
    def auto_count(self) -> int:
        return sum(1 for e in self._entries if not e.is_manual)

    def _rebuild_matrix(self) -> None:
        """Rebuilds the cached 2D numpy matrix of normalized embedding vectors."""
        if not self._entries:
            self._matrix = None
            return
        vectors = [e.embedding.vector for e in self._entries]
        self._matrix = np.stack(vectors, axis=0)  # shape: (N, dim)

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
        )
        self._entries.append(entry)
        self._rebuild_matrix()
        logger.info(
            f"[TARGET_GALLERY] Seeded new target '{self._target_label}' on camera '{camera_id}' "
            f"(dim={embedding.dim}, quality={q_score:.2f})"
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
        2. Consecutive match hold >= auto_add_min_consecutive
        3. Quality passes quality_evaluator
        4. Diversity check: similarity to all existing gallery entries <= diversity_threshold
        5. Evicts oldest auto-added entry if gallery is full (never evicts manual entries)
        """
        if self.is_empty:
            return False

        # 1. Similarity threshold check
        if candidate_similarity < self.auto_add_threshold:
            if track_id is not None:
                self._consecutive_matches.pop(track_id, None)
            return False

        # 2. Consecutive match hold
        if track_id is not None:
            self._consecutive_matches[track_id] = self._consecutive_matches.get(track_id, 0) + 1
            if self._consecutive_matches[track_id] < self.auto_add_min_consecutive:
                return False
        
        # 3. Quality evaluation
        if crop is not None and crop.size > 0:
            is_valid, q_val, reason = self.quality_evaluator.evaluate(crop)
            if not is_valid:
                logger.debug(f"[TARGET_GALLERY] Auto-add skipped: poor quality ({reason})")
                return False
            q_score = q_val
        else:
            return False

        # 4. Diversity check against existing entries
        if self._matrix is not None and len(self._matrix) > 0:
            sims = self._matrix @ embedding.vector
            max_sim_to_gallery = float(np.max(sims))
            if max_sim_to_gallery >= self.diversity_threshold:
                logger.debug(
                    f"[TARGET_GALLERY] Auto-add skipped: redundant appearance "
                    f"(sim={max_sim_to_gallery:.3f} >= {self.diversity_threshold:.2f})"
                )
                return False

        # 5. Capacity management (evict oldest auto-entry only)
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
        )
        self._entries.append(entry)
        self._rebuild_matrix()
        logger.info(
            f"[TARGET_GALLERY] Auto-enrolled verified viewpoint ({self.size}/{self.max_size}) "
            f"on '{camera_id}' (sim={candidate_similarity:.3f}, quality={q_score:.2f}, "
            f"manual={self.manual_count}, auto={self.auto_count})"
        )
        return True

    def match(self, candidate_embedding: Embedding) -> Tuple[float, Optional[GalleryEntry]]:
        """
        Computes max-similarity of a candidate embedding against all gallery entries.
        Returns (max_cosine_similarity, best_matching_entry).
        """
        if self._matrix is None or len(self._entries) == 0:
            return 0.0, None

        if candidate_embedding.dim != self._matrix.shape[1]:
            logger.warning(
                f"[TARGET_GALLERY] Dim mismatch in match: {candidate_embedding.dim} vs {self._matrix.shape[1]}"
            )
            return 0.0, None

        # Direct in-memory dot product against all N gallery vectors
        sims = self._matrix @ candidate_embedding.vector
        best_idx = int(np.argmax(sims))
        max_sim = float(sims[best_idx])
        return max_sim, self._entries[best_idx]

    def match_batch(
        self, candidate_embeddings: List[Embedding]
    ) -> List[Tuple[float, Optional[GalleryEntry]]]:
        """
        Batch max-similarity matching for multiple person candidates in a frame.
        Returns list of (max_cosine_similarity, best_matching_entry).
        """
        if not candidate_embeddings:
            return []
        if self._matrix is None or len(self._entries) == 0:
            return [(0.0, None) for _ in candidate_embeddings]

        # Stack candidate vectors: shape (K, dim)
        c_matrix = np.stack([c.vector for c in candidate_embeddings], axis=0)
        # Matrix multiply: (K, dim) @ (dim, N) -> (K, N)
        sim_matrix = c_matrix @ self._matrix.T

        best_indices = np.argmax(sim_matrix, axis=1)
        max_scores = np.max(sim_matrix, axis=1)

        return [
            (float(max_scores[i]), self._entries[int(best_indices[i])])
            for i in range(len(candidate_embeddings))
        ]

    def reset_track_consensus(self, track_id: Optional[int] = None) -> None:
        """Resets consecutive match counter for a specific track or all tracks."""
        if track_id is not None:
            self._consecutive_matches.pop(track_id, None)
        else:
            self._consecutive_matches.clear()

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
