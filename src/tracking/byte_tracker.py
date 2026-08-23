"""Tracking module implementing BaseTracker using IoU association and track lifecycle management."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.core.interfaces import BaseTracker
from src.core.types import BoundingBox, Detection, DetectionResult, Track, TrackResult, TrackState

logger = logging.getLogger(__name__)


class SingleTrackState:
    """Internal state for a single active or lost track."""

    def __init__(self, track_id: int, detection: Detection, frame_id: int) -> None:
        self.track_id = track_id
        self.box = detection.box
        self.class_id = detection.class_id
        self.confidence = detection.confidence
        self.state = TrackState.NEW
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.start_frame = frame_id

    def update(self, detection: Detection, frame_id: int) -> None:
        self.box = detection.box
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.NEW and self.hits >= 2:
            self.state = TrackState.TRACKED
        elif self.state == TrackState.LOST:
            self.state = TrackState.TRACKED

    def mark_missed(self) -> None:
        self.age += 1
        self.time_since_update += 1
        if self.state == TrackState.NEW:
            self.state = TrackState.REMOVED
        elif self.state == TrackState.TRACKED:
            self.state = TrackState.LOST

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            box=self.box,
            class_id=self.class_id,
            confidence=self.confidence,
            state=self.state,
            age=self.age,
            hits=self.hits,
        )


class ByteTracker(BaseTracker):
    """
    ByteTrack-inspired multi-object tracker based on two-stage IoU association.
    Associates high-confidence detections first, then low-confidence detections,
    maintaining robust track IDs without unnecessary complexity.
    """

    def __init__(
        self,
        track_thresh: float = 0.4,
        match_thresh: float = 0.5,
        track_buffer: int = 30,
        min_box_area: float = 100.0,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area

        self._next_id = 1
        self._tracks: Dict[int, SingleTrackState] = {}
        self._frame_id = 0

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()
        self._frame_id = 0

    def _compute_iou_matrix(
        self,
        tracks: List[SingleTrackState],
        detections: List[Detection]
    ) -> np.ndarray:
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                iou_matrix[i, j] = t.box.iou(d.box)
        return iou_matrix

    def _linear_assignment(
        self,
        iou_matrix: np.ndarray,
        threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy IoU bipartite matching."""
        if iou_matrix.size == 0:
            return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))

        matched_tracks = set()
        matched_dets = set()
        matches: List[Tuple[int, int]] = []

        # Find best matches in descending order of IoU
        flat_indices = np.argsort(-iou_matrix, axis=None)
        num_tracks, num_dets = iou_matrix.shape

        for idx in flat_indices:
            r = idx // num_dets
            c = idx % num_dets
            iou = iou_matrix[r, c]

            if iou < threshold:
                break

            if r not in matched_tracks and c not in matched_dets:
                matched_tracks.add(r)
                matched_dets.add(c)
                matches.append((r, c))

        unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
        unmatched_dets = [j for j in range(num_dets) if j not in matched_dets]

        return matches, unmatched_tracks, unmatched_dets

    def update(
        self,
        detection_result: DetectionResult,
        frame: Optional[np.ndarray] = None
    ) -> TrackResult:
        self._frame_id = detection_result.frame_id
        timestamp_ms = detection_result.timestamp_ms

        # Filter out detections below min box area
        valid_detections = [
            d for d in detection_result.detections
            if d.box.area >= self.min_box_area
        ]

        # Partition into high and low confidence detections
        high_dets = [d for d in valid_detections if d.confidence >= self.track_thresh]
        low_dets = [d for d in valid_detections if d.confidence < self.track_thresh]

        active_track_ids = [
            tid for tid, t in self._tracks.items()
            if t.state in (TrackState.TRACKED, TrackState.NEW, TrackState.LOST)
        ]
        active_tracks = [self._tracks[tid] for tid in active_track_ids]

        # Stage 1: Associate active tracks with high-confidence detections
        iou_matrix_high = self._compute_iou_matrix(active_tracks, high_dets)
        matches_1, unmatched_t_indices_1, unmatched_d_indices_1 = self._linear_assignment(
            iou_matrix_high, self.match_thresh
        )

        for t_idx, d_idx in matches_1:
            active_tracks[t_idx].update(high_dets[d_idx], self._frame_id)

        # Stage 2: Associate remaining tracks with low-confidence detections
        remaining_tracks = [active_tracks[i] for i in unmatched_t_indices_1]
        iou_matrix_low = self._compute_iou_matrix(remaining_tracks, low_dets)
        matches_2, unmatched_t_indices_2, _ = self._linear_assignment(
            iou_matrix_low, self.match_thresh
        )

        for t_idx, d_idx in matches_2:
            remaining_tracks[t_idx].update(low_dets[d_idx], self._frame_id)

        # Handle unmatched tracks from Stage 2
        for t_idx in unmatched_t_indices_2:
            remaining_tracks[t_idx].mark_missed()

        # Stage 3: Initialize new tracks for unmatched high-confidence detections
        for d_idx in unmatched_d_indices_1:
            det = high_dets[d_idx]
            new_track = SingleTrackState(self._next_id, det, self._frame_id)
            self._tracks[self._next_id] = new_track
            self._next_id += 1

        # Clean up stale/removed tracks exceeding track_buffer
        to_delete = []
        for tid, t in self._tracks.items():
            if t.time_since_update > self.track_buffer or t.state == TrackState.REMOVED:
                to_delete.append(tid)

        for tid in to_delete:
            del self._tracks[tid]

        # Output current active tracks
        current_tracks = [
            t.to_track() for t in self._tracks.values()
            if t.state in (TrackState.TRACKED, TrackState.NEW)
        ]

        return TrackResult(
            tracks=current_tracks,
            frame_id=self._frame_id,
            timestamp_ms=timestamp_ms
        )
