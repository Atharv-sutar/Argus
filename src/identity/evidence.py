"""Temporal evidence accumulation, hard-negative margins, and explainability engine for ReID."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.core.types import (
    BoundingBox,
    Embedding,
    MatchDecisionState,
    Track,
    VerifiedIdentityDecision,
)

logger = logging.getLogger(__name__)


@dataclass
class TrackObservation:
    """Individual frame observation for a tracked candidate with spatial coordinates."""
    frame_id: int
    timestamp_ms: float
    crop_quality: float
    similarity: float
    margin: float
    is_match: bool
    center_x: float = 0.0
    center_y: float = 0.0


@dataclass
class EvidenceDecision:
    """Consolidated multi-frame identity decision for candidate tracks."""
    target_identity_id: str
    best_track_id: Optional[int]
    best_score: float
    second_best_score: float
    margin: float
    is_confirmed: bool
    is_uncertain: bool
    confidence: float
    decision_reason: str
    decision_state: MatchDecisionState = MatchDecisionState.NO_MATCH
    verified_token: Optional[VerifiedIdentityDecision] = None
    ranked_tracks: List[Tuple[int, float, bool]] = field(default_factory=list)
    diagnostic_log: str = ""


class EvidenceEngine:
    """
    Accumulates multi-frame temporal evidence per track with spatio-temporal
    independence safeguards, evaluates winner margins, applies hard-negative
    protections, and provides structured explainability.
    """

    def __init__(
        self,
        window_size: int = 4,
        min_similarity_threshold: float = 0.78,
        reacquisition_threshold: float = 0.82,
        reacquisition_min_frames: int = 4,
        min_margin_threshold: float = 0.08,
        min_consistency_ratio: float = 0.75,
        min_spatial_displacement_px: float = 0.0,
        min_time_gap_ms: float = 20.0,
    ) -> None:
        self.window_size = window_size
        self.min_similarity_threshold = min_similarity_threshold
        self.reacquisition_threshold = reacquisition_threshold
        self.reacquisition_min_frames = reacquisition_min_frames
        self.min_margin_threshold = min_margin_threshold
        self.min_consistency_ratio = min_consistency_ratio
        self.min_spatial_displacement_px = min_spatial_displacement_px
        self.min_time_gap_ms = min_time_gap_ms
        self._history: Dict[int, List[TrackObservation]] = {}

    def register_observation(
        self,
        track_id: int,
        frame_id: int,
        timestamp_ms: float,
        crop_quality: float,
        similarity: float,
        margin: float,
        is_match: bool,
        box: Optional[BoundingBox] = None,
    ) -> None:
        """
        Adds a new observation to the sliding window for track_id.
        Enforces observation independence: consecutive identical/stationary frames
        update the latest sample rather than falsely inflating sample counts.
        """
        cx = (box.x1 + box.x2) / 2.0 if box is not None else 0.0
        cy = (box.y1 + box.y2) / 2.0 if box is not None else 0.0

        obs = TrackObservation(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            crop_quality=crop_quality,
            similarity=similarity,
            margin=margin,
            is_match=is_match,
            center_x=cx,
            center_y=cy,
        )
        if track_id not in self._history:
            self._history[track_id] = []

        history = self._history[track_id]
        if history:
            last = history[-1]
            dt = timestamp_ms - last.timestamp_ms
            disp = np.hypot(cx - last.center_x, cy - last.center_y)
            dsim = abs(similarity - last.similarity)

            # If the candidate is virtually identical in pose, position, and time, update in-place
            if dt < self.min_time_gap_ms and disp < self.min_spatial_displacement_px and dsim < 0.025:
                history[-1] = obs
                return

        history.append(obs)
        if len(history) > self.window_size:
            history.pop(0)

    def prune_stale_tracks(self, active_track_ids: List[int]) -> None:
        """Removes history for tracks that are no longer active."""
        active_set = set(active_track_ids)
        to_delete = [t_id for t_id in self._history if t_id not in active_set]
        for t_id in to_delete:
            del self._history[t_id]

    def clear(self) -> None:
        """Clears all accumulated history."""
        self._history.clear()

    def get_track_evidence(self, track_id: int) -> Tuple[float, float, int]:
        """
        Returns:
            Tuple[weighted_mean_score, consistency_ratio, sample_count]
        """
        history = self._history.get(track_id, [])
        if not history:
            return 0.0, 0.0, 0

        n = len(history)
        weights = np.linspace(0.8, 1.2, n)
        weights = weights / np.sum(weights)

        scores = np.array([obs.similarity for obs in history], dtype=np.float32)
        weighted_mean = float(np.sum(scores * weights))

        matches = sum(1 for obs in history if obs.similarity >= self.min_similarity_threshold)
        consistency_ratio = float(matches / n)

        return weighted_mean, consistency_ratio, n

    def evaluate_all_candidates(
        self,
        candidate_evaluations: List[Tuple[Track, float, bool, float]],
        target_identity_id: str,
        current_tracked_id: Optional[int] = None,
        is_reacquisition: bool = False,
    ) -> EvidenceDecision:
        """
        Consolidates frame-level evaluations across all active candidate tracks.
        Enforces strict reacquisition thresholds, multi-frame hysteresis, and prevents
        silent adoption of lone bystanders.
        """
        if not candidate_evaluations:
            return EvidenceDecision(
                target_identity_id=target_identity_id,
                best_track_id=None,
                best_score=0.0,
                second_best_score=0.0,
                margin=0.0,
                is_confirmed=False,
                is_uncertain=False,
                confidence=0.0,
                decision_reason="NO_CANDIDATES_IN_SCENE",
            )

        # 1. Compute multi-frame temporal scores
        scored_tracks: List[Tuple[Track, float, float, int, bool]] = []
        for track, instant_score, instant_match, q_score in candidate_evaluations:
            t_mean, t_ratio, n_samples = self.get_track_evidence(track.track_id)
            fused_score = t_mean if n_samples > 1 else instant_score
            is_valid_candidate = instant_match and (fused_score >= self.min_similarity_threshold)
            scored_tracks.append((track, fused_score, t_ratio, n_samples, is_valid_candidate))

        # 2. Sort by fused score descending
        scored_tracks.sort(key=lambda x: x[1], reverse=True)

        top_track, top_score, top_ratio, top_samples, top_valid = scored_tracks[0]
        second_score = scored_tracks[1][1] if len(scored_tracks) > 1 else 0.0
        margin = top_score - second_score if len(scored_tracks) > 1 else 0.0

        ranked_summary = [(t[0].track_id, t[1], t[4]) for t in scored_tracks]

        is_switching_or_reacquiring = (
            is_reacquisition
            or current_tracked_id is None
            or top_track.track_id != current_tracked_id
        )

        diag_lines = [
            f"[EVIDENCE_ENGINE] Target={target_identity_id} | Candidates={len(scored_tracks)} | Mode={'REACQUISITION' if is_switching_or_reacquiring else 'TRACKING'}",
            f"  Top Candidate: Track {top_track.track_id} (Score={top_score:.3f}, Consistency={top_ratio:.2f}, Samples={top_samples})",
            f"  Second Candidate: Score={second_score:.3f} | Margin={margin:.3f}",
        ]

        # 3. REACQUISITION GATING (Strict Anti-Scoop & Anti-Adoption)
        if is_switching_or_reacquiring:
            # Rule 1: High Absolute Threshold for Reacquisition
            if top_score < self.reacquisition_threshold:
                diag_lines.append(
                    f"  Decision: NO_MATCH (Reacquisition score {top_score:.3f} < {self.reacquisition_threshold:.2f})"
                )
                return EvidenceDecision(
                    target_identity_id=target_identity_id,
                    best_track_id=None,
                    best_score=top_score,
                    second_best_score=second_score,
                    margin=margin,
                    is_confirmed=False,
                    is_uncertain=(top_score >= self.min_similarity_threshold),
                    decision_state=MatchDecisionState.NO_MATCH if top_score < self.min_similarity_threshold else MatchDecisionState.AMBIGUOUS,
                    confidence=top_score,
                    decision_reason=f"Candidate below reacquisition threshold ({top_score:.3f} < {self.reacquisition_threshold:.2f})",
                    ranked_tracks=ranked_summary,
                    diagnostic_log="\n".join(diag_lines),
                )

            # Rule 2: Multi-Frame Temporal Hysteresis Requirement
            if top_samples < self.reacquisition_min_frames or top_ratio < self.min_consistency_ratio:
                diag_lines.append(
                    f"  Decision: UNCERTAIN/PENDING (Accumulating temporal evidence: {top_samples}/{self.reacquisition_min_frames} frames, ratio={top_ratio:.2f})"
                )
                return EvidenceDecision(
                    target_identity_id=target_identity_id,
                    best_track_id=top_track.track_id,
                    best_score=top_score,
                    second_best_score=second_score,
                    margin=margin,
                    is_confirmed=False,
                    is_uncertain=True,
                    decision_state=MatchDecisionState.AMBIGUOUS,
                    confidence=top_score,
                    decision_reason=f"Target candidate pending temporal confirmation ({top_samples}/{self.reacquisition_min_frames} frames)",
                    ranked_tracks=ranked_summary,
                    diagnostic_log="\n".join(diag_lines),
                )

            # Rule 3: Competitive Margin Check when competitors exist
            if len(scored_tracks) > 1 and margin < self.min_margin_threshold:
                diag_lines.append(
                    f"  Decision: UNCERTAIN (Competing candidates within margin: {margin:.3f} < {self.min_margin_threshold:.2f})"
                )
                return EvidenceDecision(
                    target_identity_id=target_identity_id,
                    best_track_id=top_track.track_id,
                    best_score=top_score,
                    second_best_score=second_score,
                    margin=margin,
                    is_confirmed=False,
                    is_uncertain=True,
                    decision_state=MatchDecisionState.AMBIGUOUS,
                    confidence=top_score,
                    decision_reason=f"Ambiguous candidates within margin (Top={top_score:.3f}, 2nd={second_score:.3f}, Margin={margin:.3f})",
                    ranked_tracks=ranked_summary,
                    diagnostic_log="\n".join(diag_lines),
                )

            # Reacquisition Confirmed
            diag_lines.append("  Decision: CONFIRMED_REACQUISITION")
            import time
            token = VerifiedIdentityDecision(
                target_identity_id=target_identity_id,
                authorized_track_id=top_track.track_id,
                decision_state=MatchDecisionState.MATCH,
                confidence=top_score,
                margin=margin,
                timestamp_ms=time.time() * 1000.0,
                reason="Reacquired target with high confidence and verified temporal consistency",
            )
            return EvidenceDecision(
                target_identity_id=target_identity_id,
                best_track_id=top_track.track_id,
                best_score=top_score,
                second_best_score=second_score,
                margin=margin,
                is_confirmed=True,
                is_uncertain=False,
                decision_state=MatchDecisionState.MATCH,
                verified_token=token,
                confidence=top_score,
                decision_reason="Reacquired target with high confidence and verified temporal consistency",
                ranked_tracks=ranked_summary,
                diagnostic_log="\n".join(diag_lines),
            )

        # 4. ROUTINE TRACKING CONTINUITY GATING
        if not top_valid or top_score < self.min_similarity_threshold:
            diag_lines.append(f"  Decision: NO_MATCH (Score {top_score:.3f} < {self.min_similarity_threshold:.2f})")
            return EvidenceDecision(
                target_identity_id=target_identity_id,
                best_track_id=None,
                best_score=top_score,
                second_best_score=second_score,
                margin=margin,
                is_confirmed=False,
                is_uncertain=False,
                decision_state=MatchDecisionState.NO_MATCH,
                confidence=top_score,
                decision_reason=f"Track verification below threshold ({top_score:.3f} < {self.min_similarity_threshold:.2f})",
                ranked_tracks=ranked_summary,
                diagnostic_log="\n".join(diag_lines),
            )

        if len(scored_tracks) > 1 and second_score >= self.min_similarity_threshold and margin < self.min_margin_threshold:
            diag_lines.append(f"  Decision: UNCERTAIN (Margin {margin:.3f} < {self.min_margin_threshold:.2f})")
            return EvidenceDecision(
                target_identity_id=target_identity_id,
                best_track_id=top_track.track_id,
                best_score=top_score,
                second_best_score=second_score,
                margin=margin,
                is_confirmed=False,
                is_uncertain=True,
                decision_state=MatchDecisionState.AMBIGUOUS,
                confidence=top_score,
                decision_reason=f"Ambiguous candidates within margin ({margin:.3f} < {self.min_margin_threshold:.2f})",
                ranked_tracks=ranked_summary,
                diagnostic_log="\n".join(diag_lines),
            )

        diag_lines.append("  Decision: CONFIRMED_TRACKING")
        import time
        token = VerifiedIdentityDecision(
            target_identity_id=target_identity_id,
            authorized_track_id=top_track.track_id,
            decision_state=MatchDecisionState.MATCH,
            confidence=top_score,
            margin=margin,
            timestamp_ms=time.time() * 1000.0,
            reason="Confirmed tracking target with verified similarity",
        )
        return EvidenceDecision(
            target_identity_id=target_identity_id,
            best_track_id=top_track.track_id,
            best_score=top_score,
            second_best_score=second_score,
            margin=margin,
            is_confirmed=True,
            is_uncertain=False,
            decision_state=MatchDecisionState.MATCH,
            verified_token=token,
            confidence=top_score,
            decision_reason="Confirmed tracking target with verified similarity",
            ranked_tracks=ranked_summary,
            diagnostic_log="\n".join(diag_lines),
        )
