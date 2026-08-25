from __future__ import annotations

import logging
import time
import hashlib
import os
from typing import Generator, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.core.interfaces import BaseCamera, BaseDetector, BaseTracker
from src.core.types import DetectionResult, Target, TargetState, Track, TrackResult
from src.identity.manager import CandidateEvaluation, IdentityManager
from src.identity.evidence import EvidenceDecision
from src.target.manager import TargetManager
from src.visualization.annotator import FrameAnnotator

logger = logging.getLogger(__name__)

# Identity key used for the single-camera selected target.
_TARGET_IDENTITY_ID = "target_0"


def get_embedding_hash(emb) -> str:
    if emb is None:
        return "None"
    rounded = np.round(emb.vector, 4)
    m = hashlib.md5()
    m.update(rounded.tobytes())
    return m.hexdigest()[:8]


def log_reid_test(message: str) -> None:
    log_dir = r'C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'reid_test_log.txt')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    logger.info(message)


class SingleCameraPipeline:
    """
    Orchestrates real-time single-camera processing:
    Camera -> Detection -> Tracking -> Target Management (Dual-Gallery ReID) -> Visualization

    Core invariant: once a person is selected as the target, the system builds an immutable
    multi-observation reference gallery over an acquisition window, and continuously verifies
    identity using reference safeguards and diversity-gated adaptive observations.
    """

    def __init__(
        self,
        camera: BaseCamera,
        detector: BaseDetector,
        tracker: BaseTracker,
        target_manager: Optional[TargetManager] = None,
        identity_manager: Optional[IdentityManager] = None,
        annotator: Optional[FrameAnnotator] = None,
        camera_id: str = "camera_0",
        reid_interval: int = 10,
        min_margin: float = 0.05,
        identity_key: str = "target_0",
        reference_window_frames: int = 20,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.target_manager = target_manager or TargetManager(min_margin=min_margin)
        self.identity_manager = identity_manager
        self.annotator = annotator or FrameAnnotator()
        self.camera_id = camera_id
        self.reid_interval = reid_interval
        self.min_margin = min_margin
        self._identity_key = identity_key
        self.reference_window_frames = reference_window_frames

        self._frame_id = 0
        self._fps = 0.0
        self._last_time = time.time()
        self._fps_smoothing = 0.9
        self._last_track_result: Optional[TrackResult] = None
        self._last_frame: Optional[np.ndarray] = None
        self._acquisition_start_frame: int = 0
        self._consecutive_mismatches: int = 0
        self._max_mismatches_before_lost: int = 3
        self.target_evaluation_enabled: bool = True

    @property
    def current_target(self) -> Target:
        """Returns the current target state."""
        return self.target_manager.target

    def select_target_by_point(self, x: float, y: float) -> Optional[int]:
        """Manually select a target at pixel coordinates (x, y)."""
        if self._last_track_result is None:
            return None
        old_tr = self.target_manager.target.track_id
        old_st = self.target_manager.target.state.value

        selected_id = self.target_manager.select_by_point(x, y, self._last_track_result)
        if selected_id is not None:
            self._acquisition_start_frame = self._frame_id
            self._register_target_appearance(selected_id)

            ident = self.identity_manager.get_identity(self._identity_key) if self.identity_manager else None
            ref_hash = get_embedding_hash(ident.reference_embedding) if (ident and ident.reference_embedding) else "None"

            log_reid_test(
                f"\n[REID_TEST] TARGET_ASSIGNMENT\n"
                f"LogicalTarget={self._identity_key}\n"
                f"OldTracker={old_tr}\n"
                f"NewTracker={selected_id}\n"
                f"CurrentTargetState={self.target_manager.target.state.value}\n"
                f"SourceFunction=select_target_by_point\n"
                f"File=single_camera.py\n"
                f"ReIDDecision=ACCEPT\n"
                f"ReIDSimilarity=1.000\n"
                f"ReferenceEmbeddingHash={ref_hash}\n"
                f"Reason=MANUAL_SELECTION_POINT\n"
            )
        return selected_id

    def select_target_by_id(self, track_id: int) -> bool:
        """Manually select and lock onto a specific track ID."""
        old_tr = self.target_manager.target.track_id
        old_st = self.target_manager.target.state.value

        result = self.target_manager.select_by_track_id(track_id, self._last_track_result)
        if result:
            self._acquisition_start_frame = self._frame_id
            self._register_target_appearance(track_id)

            ident = self.identity_manager.get_identity(self._identity_key) if self.identity_manager else None
            ref_hash = get_embedding_hash(ident.reference_embedding) if (ident and ident.reference_embedding) else "None"

            log_reid_test(
                f"\n[REID_TEST] TARGET_ASSIGNMENT\n"
                f"LogicalTarget={self._identity_key}\n"
                f"OldTracker={old_tr}\n"
                f"NewTracker={track_id}\n"
                f"CurrentTargetState={self.target_manager.target.state.value}\n"
                f"SourceFunction=select_target_by_id\n"
                f"File=single_camera.py\n"
                f"ReIDDecision=ACCEPT\n"
                f"ReIDSimilarity=1.000\n"
                f"ReferenceEmbeddingHash={ref_hash}\n"
                f"Reason=MANUAL_SELECTION_ID\n"
            )
        return result

    def _register_target_appearance(self, track_id: int) -> None:
        """Initializes the reference gallery and sets target to ACQUIRING_REFERENCE."""
        if self.identity_manager is None or self._last_frame is None or self._last_track_result is None:
            return
        for track in self._last_track_result.tracks:
            if track.track_id == track_id:
                crop = self._extract_crop(self._last_frame, track.box)
                if crop is not None:
                    # Save diagnostic initial crop
                    if cv2 is not None:
                        os.makedirs(r'C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch', exist_ok=True)
                        cv2.imwrite(r'C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch\reid_ref_A.png', crop)

                    emb = self.identity_manager.register_new_target(
                        crop=crop,
                        identity_id=self._identity_key,
                        label="selected_target",
                    )
                    ident = self.identity_manager.get_identity(self._identity_key)
                    dim = emb.dim if emb else 0
                    hsh = get_embedding_hash(emb)
                    log_reid_test(
                        f"\n[REID_TEST] TARGET_SELECTED\n"
                        f"LogicalTarget={self._identity_key}\n"
                        f"Tracker={track_id}\n"
                        f"InitialReferenceCreated=true\n"
                        f"Dimension={dim}\n"
                        f"Hash={hsh}\n"
                    )
                break

    def clear_target(self, clear_identity: bool = True) -> None:
        """Deselect the current target on this camera, optionally clearing the shared identity."""
        self.target_manager.clear()
        if clear_identity and self.identity_manager is not None:
            self.identity_manager.clear()

    def _extract_crop(self, frame: np.ndarray, box) -> Optional[np.ndarray]:
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

    def _is_occluded(self, track: Track, all_tracks: List[Track], iou_threshold: float = 0.30) -> bool:
        """Detects whether a track's bounding box significantly overlaps with any other track."""
        if not all_tracks or len(all_tracks) <= 1:
            return False
        for other in all_tracks:
            if other.track_id == track.track_id:
                continue
            if track.box.iou(other.box) > iou_threshold:
                return True
        return False

    def _manage_target_identity(
        self,
        track_result: TrackResult,
        frame: np.ndarray,
        timestamp_ms: float
    ) -> Target:
        """
        Evaluates the selected target's identity against the current frame tracks.

        1. Handles reference acquisition over initial window to build diverse 3-5 sample reference gallery.
        2. Performs competitive all-candidate ReID sweeps on interval and post-occlusion to prevent ID-swaps / scoops.
        3. Protects against heavy occlusion and recovers lost targets with margin requirements.
        """
        if not self.target_evaluation_enabled:
            # Passive monitoring camera: purely update spatial tracker without running global target re-association
            return self.target_manager.update(track_result, frame=frame)

        target = self.target_manager.target
        ident = self.identity_manager.get_identity(self._identity_key) if self.identity_manager else None
        has_global_identity = ident is not None and bool(ident.reference_gallery)

        if not self.target_manager.is_active() and not has_global_identity:
            return target

        if self.identity_manager is None:
            return self.target_manager.update(track_result, frame=frame)

        current_track_id = target.track_id
        current_track: Optional[Track] = None
        for track in track_result.tracks:
            if track.track_id == current_track_id:
                current_track = track
                break

        # Auto-capture initial appearance if target was pre-selected (e.g. via CLI)
        if not has_global_identity and current_track is not None:
            crop = self._extract_crop(frame, current_track.box)
            if crop is not None:
                self.identity_manager.register_new_target(
                    crop=crop,
                    identity_id=self._identity_key,
                    label="selected_target",
                    timestamp_ms=timestamp_ms,
                )
                self._acquisition_start_frame = self._frame_id
                ident = self.identity_manager.get_identity(self._identity_key)
                has_global_identity = ident is not None and bool(ident.reference_gallery)

        if not has_global_identity:
            return self.target_manager.update(track_result, frame=frame)

        ref_hash = get_embedding_hash(ident.reference_embedding) if (ident and ident.reference_embedding) else "None"

        current_verified = False
        should_run_reid = (
            current_track is None
            or target.state in (TargetState.UNSELECTED, TargetState.LOST, TargetState.LOCKED, TargetState.ACQUIRING_REFERENCE, TargetState.UNCERTAIN, TargetState.OCCLUDED)
            or (self._frame_id % self.reid_interval == 0)
        )

        is_occluded = self._is_occluded(current_track, track_result.tracks, iou_threshold=0.35) if current_track is not None else False

        if is_occluded and current_track is not None:
            # Active heavy occlusion / path crossing: freeze gallery updates, rely on spatial Kalman tracking
            current_verified = True
            self.target_manager.mark_tracking(current_track, track_result.frame_id, timestamp_ms)
            self.target_manager.target.state = TargetState.OCCLUDED
            logger.debug(
                f"[REID] Target ID {current_track.track_id} is occluded by another person. "
                f"Entering OCCLUDED state, freezing gallery update and maintaining spatial tracking lock."
            )
        elif should_run_reid:
            # Multi-Candidate Evaluation Sweep
            all_candidate_crops = []
            cand_eval_records = []
            for t in track_result.tracks:
                c = self._extract_crop(frame, t.box)
                if c is not None and c.size > 0:
                    all_candidate_crops.append((t, c))

            if all_candidate_crops:
                ranked = self.identity_manager.rank_candidate_crops(all_candidate_crops, self._identity_key)
                for rank_idx, (cand_track, cand_score, cand_eval) in enumerate(ranked):
                    cand_decision = "ACCEPT" if cand_eval.is_match else "REJECT"
                    second_s = ranked[1][1] if len(ranked) > 1 and rank_idx == 0 else (ranked[0][1] if len(ranked) > 1 else 0.0)
                    cand_margin = cand_score - second_s if rank_idx == 0 else (cand_score - ranked[0][1])

                    # Feed observation to temporal EvidenceEngine
                    if self.identity_manager.evidence_engine:
                        self.identity_manager.evidence_engine.register_observation(
                            track_id=cand_track.track_id,
                            frame_id=self._frame_id,
                            timestamp_ms=timestamp_ms,
                            crop_quality=cand_eval.crop_quality_score,
                            similarity=cand_score,
                            margin=cand_margin,
                            is_match=cand_eval.is_match,
                            box=cand_track.box,
                        )

                    cand_eval_records.append((cand_track, cand_score, cand_eval.is_match, cand_eval.crop_quality_score))

                    log_reid_test(
                        f"\n[REID_TEST] CANDIDATE\n"
                        f"LogicalTarget={self._identity_key}\n"
                        f"CandidateTracker={cand_track.track_id}\n"
                        f"Crop: {cand_track.box.x1:.1f} {cand_track.box.y1:.1f} {cand_track.box.x2:.1f} {cand_track.box.y2:.1f}\n"
                        f"Quality: score={cand_eval.crop_quality_score:.2f} reason={cand_eval.quality_reason}\n"
                        f"Decomposed: DeepSim={cand_eval.deep_sim:.3f} ColorSim={cand_eval.color_sim:.3f} FusedSim={cand_eval.fused_sim:.3f}\n"
                        f"Parts: UpperSim={cand_eval.upper_sim:.3f} LowerSim={cand_eval.lower_sim:.3f} Agreement={cand_eval.feature_agreement_passed}\n"
                        f"Gallery: ProtoSim={cand_eval.proto_sim:.3f} BestRefSim={cand_eval.best_ref_sim:.3f} BestAdaptiveSim={cand_eval.best_adaptive_sim:.3f}\n"
                        f"CandidateScore={cand_score:.3f}\n"
                        f"Rank={rank_idx + 1}\n"
                        f"Decision={cand_decision}\n"
                    )

                # Prune old tracks from evidence engine
                if self.identity_manager.evidence_engine:
                    self.identity_manager.evidence_engine.prune_stale_tracks([t.track_id for t in track_result.tracks])
                    evidence_dec: EvidenceDecision = self.identity_manager.evidence_engine.evaluate_all_candidates(
                        cand_eval_records,
                        self._identity_key,
                        current_tracked_id=current_track_id,
                        is_reacquisition=(current_track is None or target.state in (TargetState.LOST, TargetState.UNCERTAIN)),
                    )
                else:
                    top_cand, top_score, top_eval = ranked[0]
                    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
                    margin = top_score - second_score
                    is_conf = top_eval.is_match and (len(ranked) == 1 or margin >= self.min_margin)
                    evidence_dec = EvidenceDecision(
                        target_identity_id=self._identity_key,
                        best_track_id=top_cand.track_id if is_conf else None,
                        best_score=top_score,
                        second_best_score=second_score,
                        margin=margin,
                        is_confirmed=is_conf,
                        is_uncertain=not is_conf and top_eval.is_match,
                        confidence=top_score,
                        decision_reason="Instant evaluation fallback",
                    )

                if evidence_dec.is_confirmed and evidence_dec.best_track_id is not None:
                    # Find track object for best candidate
                    top_track_obj = next((t for t in track_result.tracks if t.track_id == evidence_dec.best_track_id), None)
                    if top_track_obj is not None:
                        if top_track_obj.track_id == current_track_id:
                            # Current track verified
                            self._consecutive_mismatches = 0
                            current_verified = True
                            old_state = target.state.value
                            self.target_manager.mark_tracking(top_track_obj, track_result.frame_id, timestamp_ms)
                            new_state = target.state.value
                            if old_state != new_state:
                                log_reid_test(
                                    f"\n[REID_TEST] STATE_CHANGE\n"
                                    f"LogicalTarget={self._identity_key}\n"
                                    f"OldState={old_state}\n"
                                    f"NewState={new_state}\n"
                                    f"TrackerID={top_track_obj.track_id}\n"
                                    f"Reason=verification_passed\n"
                                    f"ReIDScore={evidence_dec.best_score:.3f}\n"
                                )
                            c_top = self._extract_crop(frame, top_track_obj.box)
                            if not self.identity_manager.is_reference_complete(self._identity_key):
                                # Multi-view reference auto-acquisition during enrollment window
                                if not is_occluded and c_top is not None:
                                    self.identity_manager.add_reference_sample(c_top, self._identity_key, timestamp_ms)
                            else:
                                # Rolling observation gallery update with strict anti-poisoning safeguards
                                if not is_occluded and evidence_dec.margin >= 0.08 and evidence_dec.best_score >= 0.70 and c_top is not None:
                                    self.identity_manager.verified_update(c_top, self._identity_key, timestamp_ms)
                        else:
                            # Target recovered on different track ID (ANTI-SCOOP CORRECTION / HANDOFF)
                            self._consecutive_mismatches = 0
                            current_verified = True
                            old_tr = target.track_id
                            old_st = target.state.value
                            self.identity_manager.flush_adaptive_gallery(self._identity_key)
                            reassociated = self.target_manager.reassociate_target(
                                top_track_obj,
                                track_result.frame_id,
                                timestamp_ms,
                                decision=evidence_dec.verified_token,
                                reid_verified=True,
                            )
                            if reassociated:
                                new_st = target.state.value
                                log_reid_test(
                                    f"\n[REID_TEST] TARGET_ASSIGNMENT\n"
                                    f"LogicalTarget={self._identity_key}\n"
                                    f"OldTracker={old_tr}\n"
                                    f"NewTracker={top_track_obj.track_id}\n"
                                    f"CurrentTargetState={new_st}\n"
                                    f"SourceFunction=_manage_target_identity\n"
                                    f"File=single_camera.py\n"
                                    f"ReIDDecision=ACCEPT\n"
                                    f"ReIDSimilarity={evidence_dec.best_score:.3f}\n"
                                    f"ReferenceEmbeddingHash={ref_hash}\n"
                                    f"Reason=ANTI_SCOOP_CORRECTION\n"
                                )
                                log_reid_test(
                                    f"\n[REID_TEST] STATE_CHANGE\n"
                                    f"LogicalTarget={self._identity_key}\n"
                                    f"OldState={old_st}\n"
                                    f"NewState={new_st}\n"
                                    f"TrackerID={top_track_obj.track_id}\n"
                                    f"Reason=anti_scoop_target_recovered\n"
                                    f"ReIDScore={evidence_dec.best_score:.3f}\n"
                                )
                elif evidence_dec.is_uncertain:
                    # Ambiguous / close candidates -> enter UNCERTAIN state without switching targets
                    if current_track is not None:
                        current_verified = True
                        self.target_manager.mark_tracking(current_track, track_result.frame_id, timestamp_ms)
                        self.target_manager.target.state = TargetState.UNCERTAIN
                    else:
                        current_verified = False
                else:
                    current_verified = False
            else:
                current_verified = False
        else:
            # Spatial tracking continuity maintained between ReID intervals
            if current_track is not None:
                current_verified = True
                self.target_manager.mark_tracking(current_track, track_result.frame_id, timestamp_ms)
            else:
                current_verified = False

        if not current_verified:
            if current_track is not None:
                self._consecutive_mismatches += 1
                if self._consecutive_mismatches < self._max_mismatches_before_lost:
                    self.target_manager.mark_tracking(current_track, track_result.frame_id, timestamp_ms)
                    logger.debug(
                        f"[REID] Minor verification mismatch ({self._consecutive_mismatches}/{self._max_mismatches_before_lost}), keeping spatial tracking."
                    )
                else:
                    self.identity_manager.flush_adaptive_gallery(self._identity_key)
                    old_st = target.state.value
                    self.target_manager.mark_lost(timestamp_ms)
                    new_st = target.state.value
                    if old_st != new_st:
                        log_reid_test(
                            f"\n[REID_TEST] STATE_CHANGE\n"
                            f"LogicalTarget={self._identity_key}\n"
                            f"OldState={old_st}\n"
                            f"NewState={new_st}\n"
                            f"TrackerID={target.track_id}\n"
                            f"Reason=consecutive_mismatches_exceeded\n"
                            f"ReIDScore=0.000\n"
                        )
            else:
                self.identity_manager.flush_adaptive_gallery(self._identity_key)
                old_st = target.state.value
                if old_st != TargetState.UNSELECTED.value:
                    self.target_manager.mark_lost(timestamp_ms)
                new_st = target.state.value
                if old_st != new_st:
                    log_reid_test(
                        f"\n[REID_TEST] STATE_CHANGE\n"
                        f"LogicalTarget={self._identity_key}\n"
                        f"OldState={old_st}\n"
                        f"NewState={new_st}\n"
                        f"TrackerID={target.track_id}\n"
                        f"Reason=no_candidate_passed_reid\n"
                        f"ReIDScore=0.000\n"
                    )

        return self.target_manager.target

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float
    ) -> Tuple[DetectionResult, TrackResult, Target, np.ndarray]:
        """Processes a single video frame with full visualization."""
        self._frame_id += 1
        self._last_frame = frame

        # 1. Detection
        det_result = self.detector.detect(
            frame=frame,
            frame_id=self._frame_id,
            timestamp_ms=timestamp_ms
        )

        # 2. Tracking
        track_result = self.tracker.update(
            detection_result=det_result,
            frame=frame
        )
        self._last_track_result = track_result

        # 3. Target Identity Management
        target = self._manage_target_identity(track_result, frame, timestamp_ms)

        # 4. Dynamic FPS
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        current_fps = (1.0 / dt) if dt > 0 else 0.0
        self._fps = self._fps * self._fps_smoothing + current_fps * (1.0 - self._fps_smoothing)

        # 5. Visualization Annotation
        annotated_frame = self.annotator.annotate(
            frame=frame,
            track_result=track_result,
            target=target,
            fps=self._fps,
            camera_id=self.camera_id
        )

        return det_result, track_result, target, annotated_frame

    def process_frame_headless(
        self,
        frame: np.ndarray,
        timestamp_ms: float
    ) -> Tuple[DetectionResult, TrackResult, Target]:
        """Processes a single video frame without rendering annotations."""
        self._frame_id += 1
        self._last_frame = frame

        det_result = self.detector.detect(
            frame=frame,
            frame_id=self._frame_id,
            timestamp_ms=timestamp_ms
        )

        track_result = self.tracker.update(
            detection_result=det_result,
            frame=frame
        )
        self._last_track_result = track_result

        target = self._manage_target_identity(track_result, frame, timestamp_ms)
        return det_result, track_result, target

    def stream(self) -> Generator[Tuple[np.ndarray, TrackResult, Target], None, None]:
        """Continuously yields frames from camera."""
        while self.camera.is_opened():
            success, frame, timestamp_ms = self.camera.read()
            if not success or frame is None:
                logger.info("Camera stream ended or read failed.")
                break

            _, track_result, target, annotated_frame = self.process_frame(frame, timestamp_ms)
            yield annotated_frame, track_result, target

    def stop(self) -> None:
        """Stops the pipeline and releases resources."""
        self.camera.release()
        logger.info("SingleCameraPipeline stopped and camera released.")
