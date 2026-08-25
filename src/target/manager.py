"""Target management module for manual selection, target locking, and re-association."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple
import numpy as np

from src.core.types import BoundingBox, Target, TargetState, Track, TrackResult

logger = logging.getLogger(__name__)

# Type alias for the appearance verification callback.
# Signature: (crop: np.ndarray) -> Tuple[bool, float]
#   Returns (is_match, similarity_score).
AppearanceVerifier = Callable[[np.ndarray], Tuple[bool, float]]


class TargetManager:
    """
    Manages user-selected focus target across frames.
    Tracks state transitions: UNSELECTED -> LOCKED -> TRACKING -> UNCERTAIN -> LOST -> RECOVERING.
    """

    def __init__(
        self,
        lost_timeout_ms: float = 2000.0,
        reassociation_iou_thresh: float = 0.3,
        min_margin: float = 0.05,
    ) -> None:
        self.lost_timeout_ms = lost_timeout_ms
        self.reassociation_iou_thresh = reassociation_iou_thresh
        self.min_margin = min_margin
        self._target = Target(state=TargetState.UNSELECTED)

    @property
    def target(self) -> Target:
        """Returns the current Target state."""
        return self._target

    def is_active(self) -> bool:
        """Returns True if a target has been selected."""
        return self._target.state != TargetState.UNSELECTED

    def select_by_track_id(
        self,
        track_id: int,
        track_result: Optional[TrackResult] = None
    ) -> bool:
        """
        Manually select and lock onto a specific track ID.

        Args:
            track_id: ID of the track to select.
            track_result: Optional current TrackResult to capture initial box/timestamp.

        Returns:
            bool: True if selection was successfully initialized.
        """
        matched_track: Optional[Track] = None
        frame_id = 0
        timestamp_ms = 0.0

        if track_result is not None:
            frame_id = track_result.frame_id
            timestamp_ms = track_result.timestamp_ms
            for track in track_result.tracks:
                if track.track_id == track_id:
                    matched_track = track
                    break

        box = matched_track.box if matched_track else (self._target.last_known_box if self._target else None)

        self._target = Target(
            track_id=track_id,
            state=TargetState.LOCKED,
            last_known_box=box,
            last_seen_frame=frame_id,
            last_seen_timestamp_ms=timestamp_ms,
            lost_duration_ms=0.0,
        )
        logger.info(f"LogicalTarget=selected Tracker={track_id} Decision=SELECTED")
        return True

    def select_by_point(
        self,
        x: float,
        y: float,
        track_result: TrackResult
    ) -> Optional[int]:
        """
        Select a target by clicking on a 2D pixel coordinate (x, y).

        Args:
            x: X-coordinate of click in pixels.
            y: Y-coordinate of click in pixels.
            track_result: Current TrackResult containing candidate tracks.

        Returns:
            Optional[int]: Selected track ID if a track contains (x, y), else None.
        """
        candidates = []
        for track in track_result.tracks:
            b = track.box
            if b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2:
                candidates.append(track)

        if not candidates:
            logger.info(f"No track found at point ({x:.1f}, {y:.1f})")
            return None

        # Pick candidate with smallest area (tightest box) if multiple intersect
        candidates.sort(key=lambda t: t.box.area)
        selected = candidates[0]
        self.select_by_track_id(selected.track_id, track_result)
        return selected.track_id

    def mark_acquiring_reference(self, track: Track, frame_id: int, timestamp_ms: float) -> Target:
        """Transitions target state to ACQUIRING_REFERENCE while building initial reference gallery."""
        self._target.track_id = track.track_id
        self._target.last_known_box = track.box
        self._target.last_seen_frame = frame_id
        self._target.last_seen_timestamp_ms = timestamp_ms
        self._target.lost_duration_ms = 0.0
        self._target.state = TargetState.ACQUIRING_REFERENCE
        return self._target

    def mark_tracking(self, track: Track, frame_id: int, timestamp_ms: float) -> Target:
        """Transitions target state to confirmed TRACKING."""
        self._target.track_id = track.track_id
        self._target.last_known_box = track.box
        self._target.last_seen_frame = frame_id
        self._target.last_seen_timestamp_ms = timestamp_ms
        self._target.lost_duration_ms = 0.0
        self._target.state = TargetState.TRACKING
        return self._target


    def mark_uncertain(self, timestamp_ms: float) -> Target:
        """Transitions target state to UNCERTAIN when verification is weak/pending."""
        if self._target.last_seen_timestamp_ms > 0:
            elapsed = max(0.0, timestamp_ms - self._target.last_seen_timestamp_ms)
        else:
            elapsed = self._target.lost_duration_ms + 33.3
        self._target.lost_duration_ms = elapsed
        self._target.state = TargetState.UNCERTAIN
        return self._target

    def mark_lost(self, timestamp_ms: float) -> Target:
        """Transitions target state to LOST when target cannot be verified or found."""
        if self._target.last_seen_timestamp_ms > 0:
            elapsed = max(0.0, timestamp_ms - self._target.last_seen_timestamp_ms)
        else:
            elapsed = self._target.lost_duration_ms + 33.3
        self._target.lost_duration_ms = elapsed
        self._target.state = TargetState.LOST
        return self._target

    def clear(self) -> None:
        """Deselect the current target."""
        self._target = Target(state=TargetState.UNSELECTED)
        logger.info("LogicalTarget=cleared Decision=DESELECTED")

    def update(
        self,
        track_result: TrackResult,
        frame: Optional[np.ndarray] = None,
        verify_fn: Optional[AppearanceVerifier] = None,
        min_margin: Optional[float] = None,
    ) -> Target:
        """
        Update the target state against the latest TrackResult.

        When verify_fn is provided:
        1. Checks if the currently assigned track ID still looks like the selected person.
        2. If verified -> remains TRACKING.
        3. If mismatched or missing -> searches all other candidate tracks in the frame,
           ranks by appearance similarity, and requires a margin over the second-best candidate.
        4. If no candidate qualifies -> transitions to LOST (never adopts the wrong person).
        """
        if self._target.state == TargetState.UNSELECTED or self._target.track_id is None:
            return self._target

        margin_req = min_margin if min_margin is not None else self.min_margin

        # --- Appearance-verified path ---
        if verify_fn is not None and frame is not None:
            current_track: Optional[Track] = None
            for track in track_result.tracks:
                if track.track_id == self._target.track_id:
                    current_track = track
                    break

            # 1. Verify current track first
            if current_track is not None:
                crop = self._extract_crop(frame, current_track.box)
                if crop is not None:
                    is_match, score = verify_fn(crop)
                    if is_match:
                        return self.mark_tracking(current_track, track_result.frame_id, track_result.timestamp_ms)
                    else:
                        logger.warning(
                            f"LogicalTarget=selected CurrentTracker={self._target.track_id} "
                            f"VerificationSimilarity={score:.3f} Decision=TARGET_IDENTITY_MISMATCH"
                        )

            # 2. Current track missing or failed verification: search all candidate tracks
            candidate_tracks = [
                t for t in track_result.tracks
                if t.track_id != self._target.track_id or current_track is None
            ]
            scored_candidates: List[Tuple[Track, float]] = []
            for t in candidate_tracks:
                c = self._extract_crop(frame, t.box)
                if c is not None:
                    is_match, score = verify_fn(c)
                    if is_match:
                        scored_candidates.append((t, score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            if scored_candidates:
                best_track, best_score = scored_candidates[0]
                second_score = scored_candidates[1][1] if len(scored_candidates) > 1 else 0.0
                margin = best_score - second_score

                if len(scored_candidates) == 1 or margin >= margin_req:
                    logger.info(
                        f"LogicalTarget=selected OldTracker={self._target.track_id} "
                        f"NewTracker={best_track.track_id} BestSimilarity={best_score:.3f} "
                        f"SecondBest={second_score:.3f} Margin={margin:.3f} Decision=REASSOCIATE"
                    )
                    return self.mark_tracking(best_track, track_result.frame_id, track_result.timestamp_ms)
                else:
                    logger.info(
                        f"LogicalTarget=selected BestSimilarity={best_score:.3f} "
                        f"SecondBest={second_score:.3f} Margin={margin:.3f} Decision=REJECT_AMBIGUOUS"
                    )

            return self.mark_lost(track_result.timestamp_ms)

        # --- Legacy path without appearance verifier ---
        found_track: Optional[Track] = None
        for track in track_result.tracks:
            if track.track_id == self._target.track_id:
                found_track = track
                break

        if found_track is None and self._target.last_known_box is not None:
            found_track = self._try_reassociate(track_result, frame, verify_fn)

        if found_track is not None:
            return self.mark_tracking(found_track, track_result.frame_id, track_result.timestamp_ms)
        else:
            return self.mark_lost(track_result.timestamp_ms)

    def _try_reassociate(
        self,
        track_result: TrackResult,
        frame: Optional[np.ndarray],
        verify_fn: Optional[AppearanceVerifier],
    ) -> Optional[Track]:
        """Legacy spatial IoU re-association."""
        if self._target.lost_duration_ms > self.lost_timeout_ms or self._target.last_known_box is None:
            return None

        candidates: list[Tuple[Track, float]] = []
        for track in track_result.tracks:
            iou = self._target.last_known_box.iou(track.box)
            if iou >= self.reassociation_iou_thresh:
                candidates.append((track, iou))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_track_legacy, best_iou = candidates[0]
        logger.warning(
            f"LogicalTarget=selected Tracker={self._target.track_id} "
            f"CandidateTracker={best_track_legacy.track_id} "
            f"IoU={best_iou:.2f} Decision=LEGACY_IOU_REASSOCIATE "
            f"(no appearance verifier available)"
        )
        self._target.track_id = best_track_legacy.track_id
        return best_track_legacy

    @staticmethod
    def _extract_crop(frame: np.ndarray, box: BoundingBox) -> Optional[np.ndarray]:
        """Safely extracts bounded crop from frame."""
        if frame is None or box is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box.as_xyxy())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
