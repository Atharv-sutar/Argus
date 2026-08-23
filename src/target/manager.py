"""Target management module for manual selection, target locking, and re-association."""

from __future__ import annotations

import logging
from typing import Optional, Tuple
from src.core.types import BoundingBox, Target, TargetState, Track, TrackResult

logger = logging.getLogger(__name__)


class TargetManager:
    """
    Manages user-selected focus target across frames.
    Tracks state transitions: UNSELECTED -> LOCKED -> TRACKING -> LOST -> RECOVERING.
    """

    def __init__(
        self,
        lost_timeout_ms: float = 2000.0,
        reassociation_iou_thresh: float = 0.3,
    ) -> None:
        self.lost_timeout_ms = lost_timeout_ms
        self.reassociation_iou_thresh = reassociation_iou_thresh
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

        self._target = Target(
            track_id=track_id,
            state=TargetState.LOCKED,
            last_known_box=matched_track.box if matched_track else None,
            last_seen_frame=frame_id,
            last_seen_timestamp_ms=timestamp_ms,
            lost_duration_ms=0.0,
        )
        logger.info(f"Target selected with Track ID: {track_id}")
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

    def clear(self) -> None:
        """Deselect the current target."""
        self._target = Target(state=TargetState.UNSELECTED)
        logger.info("Target deselected / cleared.")

    def update(self, track_result: TrackResult) -> Target:
        """
        Update the target state against the latest TrackResult from the active camera.

        Args:
            track_result: TrackResult from the current frame.

        Returns:
            Target: Updated target state.
        """
        if self._target.state == TargetState.UNSELECTED or self._target.track_id is None:
            return self._target

        # 1. Look for the exact same track ID in current tracks
        found_track: Optional[Track] = None
        for track in track_result.tracks:
            if track.track_id == self._target.track_id:
                found_track = track
                break

        # 2. If track ID was not found, attempt spatial IoU re-association
        if found_track is None and self._target.last_known_box is not None:
            best_iou = 0.0
            best_track: Optional[Track] = None
            for track in track_result.tracks:
                iou = self._target.last_known_box.iou(track.box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is not None and best_iou >= self.reassociation_iou_thresh:
                logger.info(
                    f"Re-associated target: ID {self._target.track_id} -> ID {best_track.track_id} (IoU={best_iou:.2f})"
                )
                self._target.track_id = best_track.track_id
                found_track = best_track

        # 3. Update target state based on search result
        if found_track is not None:
            self._target.last_known_box = found_track.box
            self._target.last_seen_frame = track_result.frame_id
            self._target.last_seen_timestamp_ms = track_result.timestamp_ms
            self._target.lost_duration_ms = 0.0
            self._target.state = TargetState.TRACKING
        else:
            # Target is not visible in the current frame
            if self._target.last_seen_timestamp_ms > 0:
                elapsed = max(0.0, track_result.timestamp_ms - self._target.last_seen_timestamp_ms)
            else:
                elapsed = self._target.lost_duration_ms + 33.3  # Fallback frame dt
            self._target.lost_duration_ms = elapsed

            if self._target.lost_duration_ms >= self.lost_timeout_ms:
                self._target.state = TargetState.LOST
            else:
                self._target.state = TargetState.LOST

        return self._target
