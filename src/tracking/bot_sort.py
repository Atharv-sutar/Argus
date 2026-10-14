"""BoT-SORT Tracker Implementation.

BoT-SORT (Robust Multi-Object Tracking with BoT-SORT):
1. Global Camera Motion Compensation (GMC / CMC) via sparse optical flow or ORB.
2. Kalman Filter with state vector [x_c, y_c, w, h, dx, dy, dw, dh].
3. Vectorized IoU computation (C-speed NumPy broadcasting, 0 Python loops).
4. Optimal assignment via SciPy linear_sum_assignment (Hungarian algorithm).
5. Exponential Moving Average (EMA) appearance feature fusion for robust identity preservation.
6. Two-stage data association (high-confidence detections first, then low-confidence detections).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from src.core.interfaces import BaseTracker, BaseReID
from src.core.types import (
    BoundingBox,
    Detection,
    DetectionResult,
    Embedding,
    Track,
    TrackResult,
    TrackState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global Motion Compensation (GMC)
# ---------------------------------------------------------------------------

class CameraMotionCompensator:
    """Estimates rigid 2D affine transformation between consecutive frames."""

    def __init__(self, method: str = "sparseOptFlow") -> None:
        self.method = method.lower()
        self.prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_gray = None

    def estimate_motion(self, frame: Optional[np.ndarray]) -> np.ndarray:
        """
        Estimate 2x3 affine transformation matrix M from previous frame to current.
        Returns identity matrix if frame is None or motion cannot be computed.
        """
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        if frame is None or self.method in ("none", "off", "disabled"):
            return identity

        # Downsample frame for fast feature matching
        h, w = frame.shape[:2]
        scale = min(1.0, 640.0 / max(h, w))
        if scale < 1.0:
            small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small_frame = frame

        if len(small_frame.shape) == 3:
            curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = small_frame

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return identity

        M = identity
        try:
            if self.method == "sparseoptflow":
                # Shi-Tomasi corner detection + Lucas-Kanade optical flow
                p0 = cv2.goodFeaturesToTrack(
                    self.prev_gray,
                    maxCorners=200,
                    qualityLevel=0.01,
                    minDistance=10,
                    blockSize=3,
                )
                if p0 is not None and len(p0) >= 8:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, curr_gray, p0, None,
                        winSize=(15, 15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                    )
                    good_old = p0[st == 1]
                    good_new = p1[st == 1]
                    if len(good_old) >= 6:
                        affine_matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                        if affine_matrix is not None:
                            M = affine_matrix.astype(np.float32)
                            # Rescale translation vector back to full frame size if downsampled
                            if scale < 1.0:
                                M[0, 2] /= scale
                                M[1, 2] /= scale

            elif self.method == "orb":
                orb = cv2.ORB_create(nfeatures=300)
                kp1, des1 = orb.detectAndCompute(self.prev_gray, None)
                kp2, des2 = orb.detectAndCompute(curr_gray, None)
                if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    if len(matches) >= 8:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        affine_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
                        if affine_matrix is not None:
                            M = affine_matrix.astype(np.float32)
                            if scale < 1.0:
                                M[0, 2] /= scale
                                M[1, 2] /= scale
        except Exception as e:
            logger.debug("GMC motion estimation failed, using identity: %s", e)
            M = identity

        self.prev_gray = curr_gray
        return M


# ---------------------------------------------------------------------------
# Kalman Filter for BoT-SORT
# ---------------------------------------------------------------------------

class BoTSORTKalmanFilter:
    """
    Kalman filter tracking bounding box state directly as [xc, yc, w, h, dx, dy, dw, dh].
    Unlike standard SORT which tracks aspect ratio, tracking [xc, yc, w, h] avoids
    severe aspect ratio oscillation when subjects rotate or bend.
    """

    def __init__(self) -> None:
        self.ndim = 4
        self.dt = 1.0

        # State transition matrix F (8x8)
        self.F = np.eye(2 * self.ndim, 2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self.F[i, self.ndim + i] = self.dt

        # Measurement matrix H (4x8)
        self.H = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)

        # Standard noise scaling factors
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track state from measurement [xc, yc, w, h].
        Returns (mean, covariance).
        """
        mean_pos = measurement.astype(np.float32)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman prediction step."""
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel]))).astype(np.float32)

        predicted_mean = np.dot(self.F, mean)
        predicted_covariance = np.dot(np.dot(self.F, covariance), self.F.T) + motion_cov
        return predicted_mean, predicted_covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std)).astype(np.float32)

        mean_projected = np.dot(self.H, mean)
        covariance_projected = np.dot(np.dot(self.H, covariance), self.H.T) + innovation_cov
        return mean_projected, covariance_projected

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman measurement correction step."""
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain: K = P * H.T * inv(S)
        kalman_gain = np.dot(
            np.dot(covariance, self.H.T),
            np.linalg.inv(projected_cov)
        )
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.dot(np.dot(kalman_gain, projected_cov), kalman_gain.T)
        return new_mean, new_covariance

    def apply_motion(
        self, mean: np.ndarray, covariance: np.ndarray, M: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply camera motion affine transformation to Kalman state."""
        R = M[:2, :2]
        T = M[:2, 2]

        # Rotate position
        mean[:2] = np.dot(R, mean[:2]) + T
        # Rotate velocity
        mean[4:6] = np.dot(R, mean[4:6])

        # Rotate covariance sub-blocks
        covariance[:2, :2] = np.dot(np.dot(R, covariance[:2, :2]), R.T)
        covariance[4:6, 4:6] = np.dot(np.dot(R, covariance[4:6, 4:6]), R.T)
        return mean, covariance


# ---------------------------------------------------------------------------
# Single Track Representation
# ---------------------------------------------------------------------------

class SingleBoTTrack:
    """Represents a single track lifecycle within BoT-SORT."""

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        frame_id: int,
        kalman_filter: BoTSORTKalmanFilter,
        embedding: Optional[Embedding] = None,
        ema_alpha: float = 0.9,
    ) -> None:
        self.track_id = track_id
        self.class_id = detection.class_id
        self.confidence = detection.confidence
        self.state = TrackState.NEW
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.start_frame = frame_id
        self.kalman_filter = kalman_filter
        self.ema_alpha = ema_alpha

        # Initialize Kalman state from detection box
        measurement = self._box_to_z(detection.box)
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)

        # Appearance embedding bank with EMA
        self.embedding: Optional[np.ndarray] = None
        if embedding is not None and embedding.vector is not None:
            v = np.asarray(embedding.vector, dtype=np.float32)
            norm = np.linalg.norm(v)
            self.embedding = v / (norm + 1e-6)

    @staticmethod
    def _box_to_z(box: BoundingBox) -> np.ndarray:
        w = max(1.0, float(box.width))
        h = max(1.0, float(box.height))
        xc = float(box.x1) + w / 2.0
        yc = float(box.y1) + h / 2.0
        return np.array([xc, yc, w, h], dtype=np.float32)

    def to_box(self) -> BoundingBox:
        xc, yc, w, h = self.mean[:4]
        x1 = max(0.0, float(xc - w / 2.0))
        y1 = max(0.0, float(yc - h / 2.0))
        x2 = max(x1 + 1.0, float(xc + w / 2.0))
        y2 = max(y1 + 1.0, float(yc + h / 2.0))
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=self.confidence)

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            box=self.to_box(),
            class_id=self.class_id,
            confidence=self.confidence,
            state=self.state,
            age=self.age,
            hits=self.hits,
        )

    def predict(self) -> None:
        """Run Kalman motion prediction."""
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def apply_camera_motion(self, M: np.ndarray) -> None:
        """Apply global camera motion compensation to Kalman state."""
        self.mean, self.covariance = self.kalman_filter.apply_motion(self.mean, self.covariance, M)

    def update(
        self,
        detection: Detection,
        frame_id: int,
        embedding: Optional[Embedding] = None,
    ) -> None:
        """Update track with matched detection."""
        measurement = self._box_to_z(detection.box)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, measurement)
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0

        # Update appearance feature using EMA
        if embedding is not None and embedding.vector is not None:
            v = np.asarray(embedding.vector, dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-6)
            if self.embedding is None:
                self.embedding = v
            else:
                self.embedding = self.ema_alpha * self.embedding + (1.0 - self.ema_alpha) * v
                self.embedding /= (np.linalg.norm(self.embedding) + 1e-6)

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


# ---------------------------------------------------------------------------
# Vectorized Association Math
# ---------------------------------------------------------------------------

def compute_iou_matrix_vectorized(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorized Intersection-over-Union calculation using NumPy broadcasting.
    boxes_a: (N, 4) in [x1, y1, x2, y2]
    boxes_b: (M, 4) in [x1, y1, x2, y2]
    Returns: (N, M) matrix of IoU values in [0.0, 1.0].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # (N, 1, 2) vs (1, M, 2)
    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    inter_hw = np.maximum(0.0, br - tl)
    inter_area = inter_hw[:, :, 0] * inter_hw[:, :, 1]

    area_a = np.maximum(0.0, (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))
    area_b = np.maximum(0.0, (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    iou = np.where(union_area > 0, inter_area / np.maximum(1e-6, union_area), 0.0)
    return iou.astype(np.float32)


def compute_cosine_distance_matrix(embs_a: np.ndarray, embs_b: np.ndarray) -> np.ndarray:
    """
    Computes cosine distance matrix (1.0 - similarity) between two feature banks.
    embs_a: (N, D) normalized
    embs_b: (M, D) normalized
    Returns: (N, M) matrix in range [0.0, 2.0].
    """
    if len(embs_a) == 0 or len(embs_b) == 0:
        return np.ones((len(embs_a), len(embs_b)), dtype=np.float32)

    cos_sim = np.dot(embs_a, embs_b.T)
    return np.clip(1.0 - cos_sim, 0.0, 2.0).astype(np.float32)


def linear_assignment(
    cost_matrix: np.ndarray, threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solves linear sum assignment using SciPy Hungarian algorithm with threshold gating.
    Falls back to vectorized greedy assignment if SciPy is not installed.
    cost_matrix: (N, M) cost matrix where lower is better.
    threshold: maximum acceptable cost.
    Returns: (matches, unmatched_a, unmatched_b)
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    num_a, num_b = cost_matrix.shape

    if _SCIPY_AVAILABLE:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches: List[Tuple[int, int]] = []
        matched_a = set()
        matched_b = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= threshold:
                matches.append((r, c))
                matched_a.add(r)
                matched_b.add(c)

        unmatched_a = [i for i in range(num_a) if i not in matched_a]
        unmatched_b = [j for j in range(num_b) if j not in matched_b]
        return matches, unmatched_a, unmatched_b

    # Fallback greedy matching
    flat_indices = np.argsort(cost_matrix, axis=None)
    matched_a = set()
    matched_b = set()
    matches = []

    for idx in flat_indices:
        r = idx // num_b
        c = idx % num_b
        if cost_matrix[r, c] > threshold:
            break
        if r not in matched_a and c not in matched_b:
            matched_a.add(r)
            matched_b.add(c)
            matches.append((r, c))

    unmatched_a = [i for i in range(num_a) if i not in matched_a]
    unmatched_b = [j for j in range(num_b) if j not in matched_b]
    return matches, unmatched_a, unmatched_b


# ---------------------------------------------------------------------------
# BoT-SORT Tracker Class
# ---------------------------------------------------------------------------

class BoTSORTTracker(BaseTracker):
    """
    State-of-the-Art BoT-SORT Tracker implementing BaseTracker.
    Combines:
    - Camera Motion Compensation (GMC)
    - Kalman filtering on [xc, yc, w, h]
    - Vectorized IoU & Cosine distance computation
    - SciPy Hungarian data association
    - Two-stage association (ByteTrack style)
    """

    def __init__(
        self,
        track_thresh: float = 0.4,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_box_area: float = 100.0,
        gmc_method: str = "sparseOptFlow",
        reid_extractor: Optional[BaseReID] = None,
        reid_weight: float = 0.5,
        appearance_thresh: float = 0.35,
        ema_alpha: float = 0.9,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.reid_extractor = reid_extractor
        self.reid_weight = reid_weight
        self.appearance_thresh = appearance_thresh
        self.ema_alpha = ema_alpha

        self.kalman_filter = BoTSORTKalmanFilter()
        self.gmc = CameraMotionCompensator(method=gmc_method)

        self._next_id = 1
        self._tracks: Dict[int, SingleBoTTrack] = {}
        self._frame_id = 0

    def reset(self) -> None:
        """Reset internal tracker state."""
        self._next_id = 1
        self._tracks.clear()
        self._frame_id = 0
        self.gmc.reset()

    def update(
        self,
        detection_result: DetectionResult,
        frame: Optional[np.ndarray] = None,
        det_embeddings: Optional[List[Optional[Embedding]]] = None,
    ) -> TrackResult:
        """
        Process new detections and update tracking state with BoT-SORT.
        """
        self._frame_id = detection_result.frame_id
        timestamp_ms = detection_result.timestamp_ms

        # 1. Camera Motion Compensation
        M = self.gmc.estimate_motion(frame)

        # 2. Kalman Prediction and Motion Compensation for all active tracks
        active_track_ids = [
            tid for tid, t in self._tracks.items()
            if t.state in (TrackState.TRACKED, TrackState.NEW, TrackState.LOST)
        ]
        active_tracks = [self._tracks[tid] for tid in active_track_ids]

        for track in active_tracks:
            track.predict()
            track.apply_camera_motion(M)

        # 3. Filter valid detections by area
        valid_detections = [
            d for d in detection_result.detections
            if d.box.area >= self.min_box_area
        ]

        # 4. Partition detections into high and low confidence sets
        high_dets = [d for d in valid_detections if d.confidence >= self.track_thresh]
        low_dets = [d for d in valid_detections if d.confidence < self.track_thresh]

        # 5. Appearance embeddings for high-confidence detections
        if self.reid_extractor is None:
            det_embeddings = [None] * len(high_dets)
        elif det_embeddings is None:
            det_embeddings = [None] * len(high_dets)
            if frame is not None and high_dets:
                crops = []
                valid_indices = []
                for i, d in enumerate(high_dets):
                    x1, y1, x2, y2 = map(int, d.box.as_xyxy())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0 and crop.shape[0] >= 16 and crop.shape[1] >= 16:
                        crops.append(crop)
                        valid_indices.append(i)

                if crops:
                    try:
                        embs = self.reid_extractor.extract_batch(crops)
                        for emb, idx in zip(embs, valid_indices):
                            det_embeddings[idx] = emb
                    except Exception as e:
                        logger.debug("ReID extraction in BoT-SORT failed: %s", e)
        elif len(det_embeddings) < len(high_dets):
            # Pad with None if fewer embeddings provided
            det_embeddings = list(det_embeddings) + [None] * (len(high_dets) - len(det_embeddings))

        # 6. Stage 1: Associate Active Tracks with High-Confidence Detections
        matches_1: List[Tuple[int, int]] = []
        unmatched_tracks_1: List[int] = list(range(len(active_tracks)))
        unmatched_high_dets: List[int] = list(range(len(high_dets)))

        if active_tracks and high_dets:
            track_boxes = np.array([t.to_box().as_xyxy() for t in active_tracks], dtype=np.float32)
            det_boxes = np.array([d.box.as_xyxy() for d in high_dets], dtype=np.float32)

            iou_matrix = compute_iou_matrix_vectorized(track_boxes, det_boxes)
            # Cost is 1.0 - IoU (lower is better)
            cost_matrix = 1.0 - iou_matrix

            # Fuse appearance features if available
            has_track_embs = any(t.embedding is not None for t in active_tracks)
            has_det_embs = any(e is not None for e in det_embeddings)

            if has_track_embs and has_det_embs and self.reid_extractor is not None:
                # Build embedding matrices
                dim = None
                for t in active_tracks:
                    if t.embedding is not None:
                        dim = len(t.embedding)
                        break
                if dim is None:
                    for e in det_embeddings:
                        if e is not None and e.vector is not None:
                            dim = len(e.vector)
                            break

                if dim is not None:
                    track_emb_mat = np.zeros((len(active_tracks), dim), dtype=np.float32)
                    det_emb_mat = np.zeros((len(high_dets), dim), dtype=np.float32)

                    for i, t in enumerate(active_tracks):
                        if t.embedding is not None:
                            track_emb_mat[i] = t.embedding
                    for j, e in enumerate(det_embeddings):
                        if e is not None and e.vector is not None:
                            v = np.asarray(e.vector, dtype=np.float32)
                            det_emb_mat[j] = v / (np.linalg.norm(v) + 1e-6)

                    emb_cost_matrix = compute_cosine_distance_matrix(track_emb_mat, det_emb_mat)

                    # BoT-SORT cost fusion: combine IoU and appearance
                    # Gating: where IoU > 0, fuse with weight; where IoU == 0, keep spatial cost high
                    fused_cost = (1.0 - self.reid_weight) * cost_matrix + self.reid_weight * emb_cost_matrix
                    cost_matrix = np.where(iou_matrix > 0.05, fused_cost, cost_matrix)

            matches_1, unmatched_tracks_1, unmatched_high_dets = linear_assignment(
                cost_matrix, threshold=(1.0 - self.match_thresh) if self.match_thresh < 1.0 else self.match_thresh
            )

        for t_idx, d_idx in matches_1:
            track = active_tracks[t_idx]
            det = high_dets[d_idx]
            emb = det_embeddings[d_idx] if (det_embeddings is not None and d_idx < len(det_embeddings)) else None
            track.update(det, self._frame_id, embedding=emb)

        # 7. Stage 2: Associate Remaining Tracks with Low-Confidence Detections (Pure IoU)
        remaining_tracks = [active_tracks[i] for i in unmatched_tracks_1]
        matches_2: List[Tuple[int, int]] = []
        unmatched_tracks_2: List[int] = list(range(len(remaining_tracks)))

        if remaining_tracks and low_dets:
            rem_boxes = np.array([t.to_box().as_xyxy() for t in remaining_tracks], dtype=np.float32)
            low_boxes = np.array([d.box.as_xyxy() for d in low_dets], dtype=np.float32)

            low_iou = compute_iou_matrix_vectorized(rem_boxes, low_boxes)
            low_cost = 1.0 - low_iou

            # For low confidence detections, use spatial match threshold (0.5 max cost = 0.5 min IoU)
            matches_2, unmatched_tracks_2, _ = linear_assignment(low_cost, threshold=0.5)

            for t_idx, d_idx in matches_2:
                remaining_tracks[t_idx].update(low_dets[d_idx], self._frame_id)

        # Mark unassigned tracks as missed
        for t_idx in unmatched_tracks_2:
            remaining_tracks[t_idx].mark_missed()

        # 8. Stage 3: Initialize New Tracks for Unmatched High-Confidence Detections
        for d_idx in unmatched_high_dets:
            det = high_dets[d_idx]
            emb = det_embeddings[d_idx] if (det_embeddings is not None and d_idx < len(det_embeddings)) else None
            new_track = SingleBoTTrack(
                track_id=self._next_id,
                detection=det,
                frame_id=self._frame_id,
                kalman_filter=self.kalman_filter,
                embedding=emb,
                ema_alpha=self.ema_alpha,
            )
            self._tracks[self._next_id] = new_track
            self._next_id += 1

        # 9. Clean up stale/removed tracks exceeding track_buffer
        to_delete = []
        for tid, t in self._tracks.items():
            if t.time_since_update > self.track_buffer or t.state == TrackState.REMOVED:
                to_delete.append(tid)

        for tid in to_delete:
            del self._tracks[tid]

        # 10. Output current active tracks (TRACKED or NEW)
        current_tracks = [
            t.to_track() for t in self._tracks.values()
            if t.state in (TrackState.TRACKED, TrackState.NEW)
        ]

        return TrackResult(
            tracks=current_tracks,
            frame_id=self._frame_id,
            timestamp_ms=timestamp_ms
        )
