"""Types and dataclasses for the production ReID benchmark and evaluation suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.core.types import BoundingBox, MatchDecisionState, TargetState


class ViewpointType(str, Enum):
    FRONT = "FRONT"
    REAR = "REAR"
    SIDE_LEFT = "SIDE_LEFT"
    SIDE_RIGHT = "SIDE_RIGHT"
    OBLIQUE = "OBLIQUE"
    UNKNOWN = "UNKNOWN"


class ResolutionLevel(str, Enum):
    HIGH = "HIGH"        # > 120px
    MEDIUM = "MEDIUM"    # 60px - 120px
    LOW = "LOW"          # 35px - 60px
    TINY = "TINY"        # < 35px


class OcclusionLevel(str, Enum):
    NONE = "NONE"        # 0%
    PARTIAL = "PARTIAL"  # 1% - 40%
    HEAVY = "HEAVY"      # > 40%


class FailureAttribution(str, Enum):
    NONE = "NONE"
    DETECTION_FAILURE = "DETECTION_FAILURE"
    TRACK_FAILURE = "TRACK_FAILURE"
    CROP_QUALITY_FAILURE = "CROP_QUALITY_FAILURE"
    EMBEDDING_FAILURE = "EMBEDDING_FAILURE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    REACQUISITION_FAILURE = "REACQUISITION_FAILURE"
    TARGET_STATE_FAILURE = "TARGET_STATE_FAILURE"


class SystemReacquisitionOutcome(str, Enum):
    CORRECT_REACQUISITION = "CORRECT_REACQUISITION"
    FALSE_REACQUISITION = "FALSE_REACQUISITION"
    TARGET_LOST = "TARGET_LOST"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_QUALITY = "INSUFFICIENT_QUALITY"


@dataclass
class BenchmarkObservation:
    """Rich metadata for an annotated person observation in the benchmark dataset."""
    identity_id: str
    sequence_id: str
    camera_id: str
    frame_id: int
    timestamp_ms: float
    track_id: int
    bbox: BoundingBox
    image_path: str = ""
    crop: Optional[np.ndarray] = None
    viewpoint: ViewpointType = ViewpointType.UNKNOWN
    resolution: ResolutionLevel = ResolutionLevel.MEDIUM
    occlusion: OcclusionLevel = OcclusionLevel.NONE
    lighting: str = "normal"
    quality_score: float = 1.0
    clothing_description: str = ""
    is_hard_negative: bool = False
    hard_negative_target_id: Optional[str] = None


@dataclass
class EvaluationEvent:
    """Represents a single target evaluation or reacquisition event."""
    event_id: str
    target_identity_id: str
    candidate_observation: BenchmarkObservation
    ground_truth_is_target: bool
    candidate_score: float
    margin: float
    cluster_scores: Dict[str, float] = field(default_factory=dict)
    decision_state: MatchDecisionState = MatchDecisionState.NO_MATCH
    system_outcome: SystemReacquisitionOutcome = SystemReacquisitionOutcome.TARGET_LOST
    failure_attribution: FailureAttribution = FailureAttribution.NONE
    decision_reason: str = ""
    execution_time_ms: float = 0.0


@dataclass
class ConfusionMatrix:
    """Confusion matrix tracking verification and reacquisition outcomes."""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    ambiguous_count: int = 0
    insufficient_quality_count: int = 0

    @property
    def total(self) -> int:
        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
            + self.ambiguous_count
            + self.insufficient_quality_count
        )

    @property
    def tpr(self) -> float:
        denom = self.true_positives + self.false_negatives
        return (self.true_positives / denom) * 100.0 if denom > 0 else 0.0

    @property
    def fmr(self) -> float:
        denom = self.false_positives + self.true_negatives
        return (self.false_positives / denom) * 100.0 if denom > 0 else 0.0

    @property
    def fnmr(self) -> float:
        denom = self.true_positives + self.false_negatives
        return (self.false_negatives / denom) * 100.0 if denom > 0 else 0.0


@dataclass
class ProductionReIDReport:
    """Complete structured benchmark report."""
    num_identities: int
    num_sequences: int
    num_cameras: int
    num_genuine_events: int
    num_impostor_events: int
    num_hard_negatives: int
    num_cross_camera_events: int

    # Retrieval
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float

    # Verification
    tpr: float
    tpr_ci_low: float
    tpr_ci_high: float
    fmr: float
    fmr_ci_low: float
    fmr_ci_high: float
    fnmr: float
    eer: float

    # Reacquisition
    reacquisition_success_rate: float
    false_reacquisition_rate: float

    # Scenario breakdowns
    scenario_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    failure_attribution_counts: Dict[str, int] = field(default_factory=dict)

    # Operational metrics
    avg_latency_ms: float = 0.0
    approx_fps: float = 0.0
    peak_vram_mb: float = 0.0
    ram_mb: float = 0.0
