"""Data types and schemas for ReID benchmark evaluation and ground-truth validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.types import BoundingBox, TargetState


@dataclass
class GroundTruthObservation:
    """A ground truth observation of a person in a specific camera and frame."""
    camera_id: str
    frame_id: int
    person_id: str
    box: BoundingBox
    occluded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "person_id": self.person_id,
            "box": [self.box.x1, self.box.y1, self.box.x2, self.box.y2],
            "confidence": self.box.confidence,
            "occluded": self.occluded,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundTruthObservation:
        b = data["box"]
        conf = data.get("confidence", 1.0)
        box = BoundingBox(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), confidence=conf)
        return cls(
            camera_id=data["camera_id"],
            frame_id=int(data["frame_id"]),
            person_id=str(data["person_id"]),
            box=box,
            occluded=bool(data.get("occluded", False)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GroundTruthDataset:
    """Collection of ground-truth observations across multiple cameras."""
    name: str = "benchmark"
    target_person_id: str = "person_1"
    observations: List[GroundTruthObservation] = field(default_factory=list)
    _index: Dict[Tuple[str, int], List[GroundTruthObservation]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._build_index()

    def _build_index(self) -> None:
        self._index = {}
        for obs in self.observations:
            key = (obs.camera_id, obs.frame_id)
            self._index.setdefault(key, []).append(obs)

    def add_observation(self, obs: GroundTruthObservation) -> None:
        self.observations.append(obs)
        key = (obs.camera_id, obs.frame_id)
        self._index.setdefault(key, []).append(obs)

    def get_for_frame(self, camera_id: str, frame_id: int) -> List[GroundTruthObservation]:
        """Returns all ground-truth persons visible in the specified camera and frame."""
        return self._index.get((camera_id, frame_id), [])

    def get_target_for_frame(self, camera_id: str, frame_id: int) -> Optional[GroundTruthObservation]:
        """Returns the target person observation if present in the given frame."""
        for obs in self.get_for_frame(camera_id, frame_id):
            if obs.person_id == self.target_person_id:
                return obs
        return None

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        data = {
            "name": self.name,
            "target_person_id": self.target_person_id,
            "observations": [obs.to_dict() for obs in self.observations],
        }
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> GroundTruthDataset:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        observations = [GroundTruthObservation.from_dict(d) for d in data.get("observations", [])]
        return cls(
            name=data.get("name", "benchmark"),
            target_person_id=data.get("target_person_id", "person_1"),
            observations=observations,
        )


@dataclass
class FrameEvaluationResult:
    """Frame-level evaluation metrics."""
    frame_id: int
    camera_id: str
    target_present: bool
    system_has_target: bool
    system_target_state: TargetState
    system_active_cam: Optional[str]
    system_track_id: Optional[int]
    system_box: Optional[BoundingBox]
    matched_person_id: Optional[str]
    iou_with_target: float
    is_true_positive: bool
    is_false_positive: bool
    is_false_negative: bool
    is_true_negative: bool
    is_id_switch: bool
    candidate_scores: Dict[int, float] = field(default_factory=dict)
    target_sim: Optional[float] = None
    best_bystander_sim: Optional[float] = None


@dataclass
class EvaluationMetrics:
    """Consolidated summary metrics for ReID and multi-camera target tracking."""
    total_frames: int = 0
    target_present_frames: int = 0
    true_positives: int = 0   # Target correctly identified and tracked (IoU >= 0.4)
    false_positives: int = 0  # System locked onto bystander / wrong person
    false_negatives: int = 0  # Target present in view but system LOST / UNCERTAIN / missed
    true_negatives: int = 0   # Non-targets in view correctly not locked
    id_switches: int = 0      # Swapped between different persons while tracking
    recovery_events: int = 0  # Legitimate reacquisition after absence
    state_frames: Dict[str, int] = field(default_factory=dict)
    target_similarities: List[float] = field(default_factory=list)
    bystander_similarities: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return (self.true_positives / denom) if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return (self.true_positives / denom) if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 = 2 * (P * R) / (P + R)"""
        p, r = self.precision, self.recall
        denom = p + r
        return (2.0 * p * r / denom) if denom > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        """FPR = FP / (FP + TN)"""
        denom = self.false_positives + self.true_negatives
        return (self.false_positives / denom) if denom > 0 else 0.0

    @property
    def mean_target_sim(self) -> float:
        return float(np.mean(self.target_similarities)) if self.target_similarities else 0.0

    @property
    def mean_bystander_sim(self) -> float:
        return float(np.mean(self.bystander_similarities)) if self.bystander_similarities else 0.0

    @property
    def separation_gap(self) -> float:
        """Separation margin = mean_target_sim - mean_bystander_sim"""
        return self.mean_target_sim - self.mean_bystander_sim

    def summary_table(self) -> str:
        """Formats the evaluation metrics into an ASCII summary table."""
        lines = [
            "+" + "-" * 50 + "+",
            f"| {'ReID & TRACKING EVALUATION METRICS':^48} |",
            "+" + "-" * 50 + "+",
            f"| Total Frames Evaluated      : {self.total_frames:<20} |",
            f"| Target Present Frames       : {self.target_present_frames:<20} |",
            "+" + "-" * 50 + "+",
            f"| Precision (TP / (TP + FP))  : {self.precision * 100:>6.2f}%              |",
            f"| Recall    (TP / (TP + FN))  : {self.recall * 100:>6.2f}%              |",
            f"| F1 Score                    : {self.f1_score * 100:>6.2f}%              |",
            f"| False Positive Rate (Scoops): {self.false_positive_rate * 100:>6.2f}%              |",
            "+" + "-" * 50 + "+",
            f"| True Positives (TP)         : {self.true_positives:<20} |",
            f"| False Positives (FP)        : {self.false_positives:<20} |",
            f"| False Negatives (FN)        : {self.false_negatives:<20} |",
            f"| True Negatives (TN)         : {self.true_negatives:<20} |",
            f"| Identity Switches (ID-Swaps): {self.id_switches:<20} |",
            f"| Recovery Events             : {self.recovery_events:<20} |",
            "+" + "-" * 50 + "+",
            f"| Mean Target Similarity      : {self.mean_target_sim:>6.3f}               |",
            f"| Mean Bystander Similarity   : {self.mean_bystander_sim:>6.3f}               |",
            f"| Discrimination Gap (Δ sim)  : {self.separation_gap:>+6.3f}               |",
            "+" + "-" * 50 + "+",
            "| State Distribution:                              |",
        ]
        tot = max(1, self.total_frames)
        for st, count in sorted(self.state_frames.items()):
            pct = (count / tot) * 100.0
            lines.append(f"|   {st:<16}: {count:>5} ({pct:>5.1f}%)               |")
        lines.append("+" + "-" * 50 + "+")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "target_present_frames": self.target_present_frames,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "id_switches": self.id_switches,
            "recovery_events": self.recovery_events,
            "mean_target_sim": round(self.mean_target_sim, 4),
            "mean_bystander_sim": round(self.mean_bystander_sim, 4),
            "separation_gap": round(self.separation_gap, 4),
            "state_frames": self.state_frames,
        }
