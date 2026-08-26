"""Core data types and domain contracts for Argus surveillance system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TrackState(str, Enum):
    """Lifecycle state of an individual track."""
    NEW = "NEW"
    TRACKED = "TRACKED"
    LOST = "LOST"
    REMOVED = "REMOVED"


@dataclass(frozen=True)
class BoundingBox:
    """Represents a 2D bounding box in pixel coordinates (x1, y1, x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.x2 < self.x1:
            raise ValueError(f"Invalid BoundingBox: x2 ({self.x2}) cannot be less than x1 ({self.x1})")
        if self.y2 < self.y1:
            raise ValueError(f"Invalid BoundingBox: y2 ({self.y2}) cannot be less than y1 ({self.y1})")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Invalid BoundingBox: confidence ({self.confidence}) must be in [0.0, 1.0]")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def iou(self, other: BoundingBox) -> float:
        """Computes Intersection-over-Union (IoU) with another bounding box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih

        if intersection == 0.0:
            return 0.0

        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class Detection:
    """Represents a detected object within a single frame."""
    box: BoundingBox
    class_id: int = 0
    class_name: str = "person"
    confidence: float = 1.0


@dataclass(frozen=True)
class DetectionResult:
    """Collection of detections for a specific video frame."""
    detections: List[Detection] = field(default_factory=list)
    frame_id: int = 0
    timestamp_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)


@dataclass(frozen=True)
class Track:
    """Represents a tracked target with a consistent ID across frames."""
    track_id: int
    box: BoundingBox
    class_id: int = 0
    confidence: float = 1.0
    state: TrackState = TrackState.TRACKED
    age: int = 1
    hits: int = 1


@dataclass(frozen=True)
class TrackResult:
    """Collection of active/updated tracks for a specific video frame."""
    tracks: List[Track] = field(default_factory=list)
    frame_id: int = 0
    timestamp_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.tracks)


@dataclass
class FrameData:
    """Holds a captured image frame and associated acquisition metadata."""
    frame: np.ndarray
    frame_id: int = 0
    timestamp_ms: float = 0.0
    camera_id: str = "camera_0"


class TargetState(str, Enum):
    """Lifecycle state of the user-selected tracking target."""
    UNSELECTED = "UNSELECTED"
    LOCKED = "LOCKED"
    ACQUIRING_REFERENCE = "ACQUIRING_REFERENCE"
    TRACKING = "TRACKING"
    OCCLUDED = "OCCLUDED"
    UNCERTAIN = "UNCERTAIN"
    LOST = "LOST"
    RECOVERING = "RECOVERING"


@dataclass
class Target:
    """Represents the user-selected focus target."""
    track_id: Optional[int] = None
    state: TargetState = TargetState.UNSELECTED
    last_known_box: Optional[BoundingBox] = None
    last_seen_frame: int = 0
    last_seen_timestamp_ms: float = 0.0
    lost_duration_ms: float = 0.0


@dataclass
class Embedding:
    """Represents a normalized feature vector extracted from a person crop with rich metadata."""
    vector: np.ndarray
    dim: int = field(init=False)
    model_name: str = "reid"
    version: str = "2.0"
    crop_type: str = "full"  # "full", "upper", "lower"
    quality_score: float = 1.0
    camera_id: str = "camera_0"
    timestamp_ms: float = 0.0

    def __post_init__(self) -> None:
        flat = np.asarray(self.vector, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(flat))
        if norm > 0.0:
            self.vector = flat / norm
        else:
            self.vector = flat
        object.__setattr__(self, "dim", self.vector.shape[0])

    def cosine_similarity(self, other: Embedding) -> float:
        """Computes cosine similarity between two normalized embedding vectors."""
        if self.dim != other.dim:
            raise ValueError(f"Embedding dimension mismatch: {self.dim} vs {other.dim}")
        return float(np.dot(self.vector, other.vector))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Embedding):
            return False
        return np.array_equal(self.vector, other.vector)


@dataclass
class GalleryEntry:
    """A confirmed appearance sample of the active target."""
    entry_id: str
    embedding: Embedding
    crop: Optional[np.ndarray] = None
    is_manual: bool = False  # True = Human confirmed (protected from auto eviction)
    timestamp_ms: float = 0.0
    camera_id: str = "camera_0"
    frame_id: int = 0
    confidence: float = 1.0
    quality_score: float = 1.0


@dataclass
class ViewCluster:
    """Discrete multi-modal viewpoint cluster (e.g. Front, Rear, Side, Upper Torso)."""
    cluster_id: str
    label: str = "general"
    exemplars: List[Embedding] = field(default_factory=list)
    centroid: Optional[Embedding] = None

    def update_centroid(self) -> None:
        if not self.exemplars:
            self.centroid = None
            return
        mat = np.stack([e.vector for e in self.exemplars], axis=0)
        mean_vec = np.mean(mat, axis=0)
        self.centroid = Embedding(
            vector=mean_vec,
            model_name=self.exemplars[0].model_name,
            version=self.exemplars[0].version,
            crop_type=self.exemplars[0].crop_type,
        )

    def match_score(self, candidate_emb: Embedding) -> float:
        """Computes max similarity against all exemplars in this cluster."""
        if not self.exemplars:
            return 0.0
        scores = [e.cosine_similarity(candidate_emb) for e in self.exemplars if e.dim == candidate_emb.dim]
        return max(scores) if scores else 0.0


@dataclass
class TargetIdentityAnchor:
    """Immutable ground-truth identity anchor established at target selection."""
    identity_id: str
    label: str = "selected_target"
    clusters: List[ViewCluster] = field(default_factory=list)
    model_name: str = "osnet_x0_25"
    feature_dim: int = 512
    created_timestamp_ms: float = 0.0
    anchor_hash: str = ""

    def max_similarity(self, candidate_emb: Embedding) -> float:
        """Finds max similarity across all discrete appearance clusters in the anchor."""
        if not self.clusters:
            return 0.0
        scores = [c.match_score(candidate_emb) for c in self.clusters]
        return max(scores) if scores else 0.0


class MatchDecisionState(str, Enum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_QUALITY = "INSUFFICIENT_QUALITY"


@dataclass
class VerifiedIdentityDecision:
    """Structurally authorized decision for target identity assignment and reassociation."""
    target_identity_id: str
    authorized_track_id: Optional[int]
    decision_state: MatchDecisionState
    confidence: float
    margin: float
    timestamp_ms: float
    reason: str
    decision_id: str = ""
    source_camera_id: str = "camera_0"
    model_name: str = "osnet_x0_25"
    model_version: str = "2.0"
    expires_at_ms: float = 0.0
    evidence_hash: str = ""
    decision_token: str = ""

    def __post_init__(self) -> None:
        if not self.decision_id:
            import uuid
            self.decision_id = uuid.uuid4().hex[:12]
        if self.expires_at_ms <= 0.0:
            self.expires_at_ms = self.timestamp_ms + 1000.0  # 1.0 second authorization validity window
        if not self.decision_token and self.authorized_track_id is not None:
            import hashlib
            raw = f"{self.decision_id}:{self.target_identity_id}:{self.authorized_track_id}:{self.source_camera_id}:{self.model_name}:{self.decision_state.value}:{self.confidence:.4f}:{self.timestamp_ms}:{self.expires_at_ms}"
            self.decision_token = hashlib.sha256(raw.encode()).hexdigest()[:24]

    def is_authorized_for(
        self,
        target_identity_id: str,
        track_id: int,
        current_timestamp_ms: float = 0.0,
        camera_id: Optional[str] = None,
    ) -> bool:
        """
        Validates that this token explicitly authorizes target_identity_id to bind to track_id,
        is not expired, matches camera if specified, and has a valid decision token.
        """
        if self.decision_state != MatchDecisionState.MATCH:
            return False
        if self.target_identity_id != target_identity_id:
            return False
        if self.authorized_track_id != track_id:
            return False
        if not self.decision_token:
            return False
        if current_timestamp_ms > 0.0 and self.expires_at_ms > 0.0 and current_timestamp_ms > self.expires_at_ms:
            return False
        if camera_id is not None and self.source_camera_id != camera_id:
            return False
        return True


@dataclass
class Identity:
    """
    Represents a unique known identity with dual-gallery memory architecture:
    1. trusted_gallery: Immutable multi-view enrollment cluster (front, side, angle views).
    2. provisional_gallery: Bounded rolling observations strictly verified against the trusted anchor.
    3. rejected_gallery: Hard-negative impostor embeddings for negative evidence comparison.
    """
    identity_id: str
    label: str = "target_0"
    trusted_gallery: List[Embedding] = field(default_factory=list)
    provisional_gallery: List[Embedding] = field(default_factory=list)
    rejected_gallery: List[Embedding] = field(default_factory=list)

    # Multi-crop galleries
    trusted_upper_gallery: List[Embedding] = field(default_factory=list)
    trusted_lower_gallery: List[Embedding] = field(default_factory=list)

    # Immutable identity anchor & multi-view clusters
    anchor: Optional[TargetIdentityAnchor] = None
    view_clusters: List[ViewCluster] = field(default_factory=list)

    # Normalized centroid prototypes
    trusted_prototype: Optional[Embedding] = None
    trusted_upper_proto: Optional[Embedding] = None
    trusted_lower_proto: Optional[Embedding] = None

    # Confidence and metadata
    confidence: float = 0.0
    last_seen_timestamp_ms: float = 0.0
    last_camera_id: str = "camera_0"

    # Backward-compatible property aliases
    @property
    def reference_gallery(self) -> List[Embedding]:
        return self.trusted_gallery

    @reference_gallery.setter
    def reference_gallery(self, val: List[Embedding]) -> None:
        self.trusted_gallery = val

    @property
    def adaptive_gallery(self) -> List[Embedding]:
        return self.provisional_gallery

    @adaptive_gallery.setter
    def adaptive_gallery(self, val: List[Embedding]) -> None:
        self.provisional_gallery = val

    @property
    def reference_upper_gallery(self) -> List[Embedding]:
        return self.trusted_upper_gallery

    @reference_upper_gallery.setter
    def reference_upper_gallery(self, val: List[Embedding]) -> None:
        self.trusted_upper_gallery = val

    @property
    def reference_lower_gallery(self) -> List[Embedding]:
        return self.trusted_lower_gallery

    @reference_lower_gallery.setter
    def reference_lower_gallery(self, val: List[Embedding]) -> None:
        self.trusted_lower_gallery = val

    @property
    def reference_deep_gallery(self) -> List[Embedding]:
        return self.trusted_gallery

    @reference_deep_gallery.setter
    def reference_deep_gallery(self, val: List[Embedding]) -> None:
        self.trusted_gallery = val

    @property
    def reference_color_gallery(self) -> List[Embedding]:
        return self.trusted_gallery

    @reference_color_gallery.setter
    def reference_color_gallery(self, val: List[Embedding]) -> None:
        pass

    @property
    def reference_prototype(self) -> Optional[Embedding]:
        return self.trusted_prototype

    @reference_prototype.setter
    def reference_prototype(self, val: Optional[Embedding]) -> None:
        self.trusted_prototype = val

    @property
    def reference_deep_proto(self) -> Optional[Embedding]:
        return self.trusted_prototype

    @reference_deep_proto.setter
    def reference_deep_proto(self, val: Optional[Embedding]) -> None:
        pass

    @property
    def reference_color_proto(self) -> Optional[Embedding]:
        return self.trusted_prototype

    @reference_color_proto.setter
    def reference_color_proto(self, val: Optional[Embedding]) -> None:
        pass

    @property
    def reference_upper_proto(self) -> Optional[Embedding]:
        return self.trusted_upper_proto

    @reference_upper_proto.setter
    def reference_upper_proto(self, val: Optional[Embedding]) -> None:
        self.trusted_upper_proto = val

    @property
    def reference_lower_proto(self) -> Optional[Embedding]:
        return self.trusted_lower_proto

    @reference_lower_proto.setter
    def reference_lower_proto(self, val: Optional[Embedding]) -> None:
        self.trusted_lower_proto = val

    @property
    def reference_embedding(self) -> Optional[Embedding]:
        if self.trusted_prototype is not None:
            return self.trusted_prototype
        return self.trusted_gallery[0] if self.trusted_gallery else None

    @reference_embedding.setter
    def reference_embedding(self, emb: Optional[Embedding]) -> None:
        if emb is not None:
            if not self.trusted_gallery:
                self.trusted_gallery.append(emb)
            else:
                self.trusted_gallery[0] = emb
            self.update_prototype()
        else:
            self.trusted_gallery.clear()
            self.trusted_prototype = None

    @property
    def embeddings(self) -> List[Embedding]:
        """Combined list of all active embeddings (trusted references + verified provisional)."""
        return self.trusted_gallery + self.provisional_gallery

    @embeddings.setter
    def embeddings(self, embs: List[Embedding]) -> None:
        self.provisional_gallery = list(embs)

    @staticmethod
    def _compute_centroid(gallery: List[Embedding]) -> Optional[Embedding]:
        if not gallery:
            return None
        vectors = [emb.vector for emb in gallery]
        mean_vec = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            mean_vec = mean_vec / norm
        return Embedding(
            vector=mean_vec,
            model_name=gallery[0].model_name,
            version=gallery[0].version,
            crop_type=gallery[0].crop_type,
        )

    def update_prototype(self) -> None:
        """Calculates and updates normalized centroid prototypes for all trusted representations."""
        self.trusted_prototype = self._compute_centroid(self.trusted_gallery)
        self.trusted_upper_proto = self._compute_centroid(self.trusted_upper_gallery)
        self.trusted_lower_proto = self._compute_centroid(self.trusted_lower_gallery)

    def compute_detailed_similarity(self, query: Embedding) -> Tuple[float, float, float, float]:
        """
        Computes detailed similarity against trusted prototype, trusted gallery,
        and provisional rolling gallery.

        Returns:
            Tuple[proto_sim, best_trusted_sim, best_provisional_sim, top2_provisional_mean]
        """
        proto_sim = 0.0
        if self.trusted_prototype is not None and self.trusted_prototype.dim == query.dim:
            proto_sim = self.trusted_prototype.cosine_similarity(query)
        elif self.trusted_gallery and self.trusted_gallery[0].dim == query.dim:
            proto_sim = self.trusted_gallery[0].cosine_similarity(query)

        best_trusted_sim = 0.0
        if self.trusted_gallery:
            valid_trusted = [emb for emb in self.trusted_gallery if emb.dim == query.dim]
            if valid_trusted:
                best_trusted_sim = max(emb.cosine_similarity(query) for emb in valid_trusted)

        best_provisional_sim = 0.0
        top2_provisional_mean = 0.0
        if self.provisional_gallery:
            valid_prov = [emb for emb in self.provisional_gallery if emb.dim == query.dim]
            if valid_prov:
                prov_sims = sorted([emb.cosine_similarity(query) for emb in valid_prov], reverse=True)
                best_provisional_sim = prov_sims[0]
                top2_provisional_mean = sum(prov_sims[:2]) / len(prov_sims[:2])

        return proto_sim, best_trusted_sim, best_provisional_sim, top2_provisional_mean

    def compute_similarity(self, query: Embedding) -> float:
        """Returns highest appearance similarity score against stored identity."""
        proto_sim, best_trust, best_prov, _ = self.compute_detailed_similarity(query)
        return max(proto_sim, best_trust, best_prov)





