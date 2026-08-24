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
    TRACKING = "TRACKING"
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
    """Represents a feature vector extracted from a person observation crop."""
    vector: np.ndarray
    dim: int = field(init=False)

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
        return bool(np.array_equal(self.vector, other.vector))


@dataclass
class Identity:
    """Represents a unique known identity with an appearance gallery."""
    identity_id: str
    label: str = "target_0"
    reference_embedding: Optional[Embedding] = None
    embeddings: List[Embedding] = field(default_factory=list)
    last_seen_timestamp_ms: float = 0.0

    def compute_similarity(self, query: Embedding) -> float:
        """
        Returns similarity score against stored identity.
        Gives primary weight to the immutable reference embedding while
        allowing verified gallery observations to assist in pose/viewpoint shifts.
        """
        if self.reference_embedding is None:
            return max(emb.cosine_similarity(query) for emb in self.embeddings) if self.embeddings else 0.0

        ref_sim = self.reference_embedding.cosine_similarity(query)
        if not self.embeddings:
            return ref_sim

        # Drift protection gate: query must have a minimum baseline similarity to reference
        # to prevent matching against a contaminated/drifted gallery.
        if ref_sim < 0.72:
            return ref_sim

        gallery_max = max(emb.cosine_similarity(query) for emb in self.embeddings)
        return max(ref_sim, gallery_max)


