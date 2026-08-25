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

    def reassociate_target(
        self,
        track: Track,
        frame_id: int,
        timestamp_ms: float,
        decision: Optional[VerifiedIdentityDecision] = None,
        reid_verified: bool = False,
    ) -> bool:
        """
        Reassociates a lost target with a new tracker ID.
        STRICT SAFETY INVARIANT: Reassociation MUST have positive ReID verification
        via a VerifiedIdentityDecision token or explicit verification flag.
        """
        if decision is not None:
            if not decision.is_authorized_for(decision.target_identity_id, track.track_id, current_timestamp_ms=timestamp_ms):
                logger.error(
                    f"[TARGET_SAFETY] Reassociation REJECTED: VerifiedIdentityDecision token "
                    f"is invalid or expired for track {track.track_id} and target '{decision.target_identity_id}'!"
                )
                return False
        elif not reid_verified:
            logger.error(
                f"[TARGET_SAFETY] Attempted to reassociate target '{self._target.track_id}' to "
                f"new tracker '{track.track_id}' WITHOUT positive ReID verification! REJECTED."
            )
            return False

        old_tr = self._target.track_id
        self._target.track_id = track.track_id
        self._target.last_known_box = track.box
        self._target.last_seen_frame = frame_id
        self._target.last_seen_timestamp_ms = timestamp_ms
        self._target.lost_duration_ms = 0.0
        self._target.state = TargetState.TRACKING
        logger.info(
            f"[TARGET_REASSOCIATE] Logical target reassociated: OldTracker={old_tr} -> "
            f"NewTracker={track.track_id} | State=TRACKING | ReIDVerified=True"
        )
        return True

    def update(
        self,
        track_result: TrackResult,
        frame: Optional[np.ndarray] = None,
        verify_fn: Optional[AppearanceVerifier] = None,
        min_margin: Optional[float] = None,
    ) -> Target:
        """
        Updates the target state against the latest TrackResult.
        TargetManager maintains spatial motion continuity for the actively locked track ID.
        If the track is missing or fails verification, transitions to LOST.
        TargetManager NEVER independently assigns another tracker ID.
        """
        if self._target.state == TargetState.UNSELECTED or self._target.track_id is None:
            return self._target

        # Check current track if present in track_result
        current_track: Optional[Track] = None
        for track in track_result.tracks:
            if track.track_id == self._target.track_id:
                current_track = track
                break

        if current_track is not None:
            if verify_fn is not None and frame is not None:
                crop = self._extract_crop(frame, current_track.box)
                if crop is not None:
                    is_match, score = verify_fn(crop)
                    if is_match:
                        return self.mark_tracking(current_track, track_result.frame_id, track_result.timestamp_ms)
                    else:
                        logger.warning(
                            f"[TARGET] LogicalTarget={self._target.track_id} CurrentTracker={current_track.track_id} "
                            f"Score={score:.3f} | Verification failed -> State=LOST"
                        )
                        return self.mark_lost(track_result.timestamp_ms)
            return self.mark_tracking(current_track, track_result.frame_id, track_result.timestamp_ms)

        # Track is not in track_result -> Target is LOST
        return self.mark_lost(track_result.timestamp_ms)



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
