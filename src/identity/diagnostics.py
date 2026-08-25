"""Forensic diagnostic recorder for ReID evaluation, anomalies, and uncertain events."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import cv2
import numpy as np

from src.core.types import BoundingBox, MatchDecisionState

logger = logging.getLogger(__name__)


@dataclass
class ForensicDiagnosticRecord:
    """Detailed forensic snapshot of an uncertain, rejected, or ambiguous ReID decision."""
    timestamp_ms: float
    target_identity_id: str
    candidate_track_id: Optional[int]
    decision_state: str
    candidate_score: float
    margin: float
    cluster_scores: Dict[str, float]
    quality_score: float
    quality_reason: str
    decision_reason: str
    bounding_box: Optional[Dict[str, float]] = None
    candidate_crop_path: Optional[str] = None


class ForensicDiagnosticRecorder:
    """
    Automatically logs rich forensic snapshots when uncertain or borderline
    reacquisition events occur, enabling rapid root-cause analysis without debug noise.
    """

    def __init__(self, log_dir: str = "diagnostics/reid") -> None:
        self.log_dir = log_dir
        self.records: List[ForensicDiagnosticRecord] = []
        os.makedirs(self.log_dir, exist_ok=True)

    def record_event(
        self,
        target_identity_id: str,
        candidate_track_id: Optional[int],
        decision_state: MatchDecisionState,
        candidate_score: float,
        margin: float,
        cluster_scores: Dict[str, float],
        quality_score: float,
        quality_reason: str,
        decision_reason: str,
        crop: Optional[np.ndarray] = None,
        box: Optional[BoundingBox] = None,
        timestamp_ms: Optional[float] = None,
    ) -> ForensicDiagnosticRecord:
        """Records a forensic snapshot and optionally saves the candidate crop image."""
        ts = timestamp_ms if timestamp_ms is not None else (time.time() * 1000.0)
        crop_path = None

        # Save image crop if provided
        if crop is not None and crop.size > 0:
            crop_filename = f"diag_{int(ts)}_{target_identity_id}_track_{candidate_track_id}.png"
            crop_path = os.path.join(self.log_dir, crop_filename)
            try:
                cv2.imwrite(crop_path, crop)
            except Exception as e:
                logger.warning(f"Failed to write forensic crop to {crop_path}: {e}")

        box_dict = asdict(box) if box is not None else None

        record = ForensicDiagnosticRecord(
            timestamp_ms=ts,
            target_identity_id=target_identity_id,
            candidate_track_id=candidate_track_id,
            decision_state=decision_state.value,
            candidate_score=candidate_score,
            margin=margin,
            cluster_scores=cluster_scores,
            quality_score=quality_score,
            quality_reason=quality_reason,
            decision_reason=decision_reason,
            bounding_box=box_dict,
            candidate_crop_path=crop_path,
        )
        self.records.append(record)

        # Append to JSONL log
        jsonl_path = os.path.join(self.log_dir, "forensic_events.jsonl")
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except Exception as e:
            logger.debug(f"Failed writing forensic event to {jsonl_path}: {e}")

        return record
