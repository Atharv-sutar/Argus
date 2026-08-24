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
from src.identity.manager import IdentityManager
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
    Camera -> Detection -> Tracking -> Target Management (ReID-gated) -> Visualization

    Core invariant: once a person is selected as the target, the system is
    permanently committed to that person. It will never silently switch to
    a different person. If the target cannot be confidently identified,
    the system enters LOST state rather than tracking the wrong person.
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

        self._frame_id = 0
        self._fps = 0.0
        self._last_time = time.time()
        self._fps_smoothing = 0.9
        self._last_track_result: Optional[TrackResult] = None
        self._last_frame: Optional[np.ndarray] = None

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
            self._register_target_appearance(selected_id)

            ident = self.identity_manager.get_identity(_TARGET_IDENTITY_ID) if self.identity_manager else None
            ref_hash = get_embedding_hash(ident.reference_embedding) if (ident and ident.reference_embedding) else "None"

            log_reid_test(
                f"\n[REID_TEST] TARGET_ASSIGNMENT\n"
                f"LogicalTarget={_TARGET_IDENTITY_ID}\n"
                f"OldTracker={old_tr}\n"
                f"NewTracker={selected_id}\n"
                f"CurrentTargetState={self.target_manager.target.state.value}\n"
                f"SourceFunction=select_target_by_point\n"
                f"File=single_camera.py\n"
                f"Line=73\n"
                f"ReIDDecision=ACCEPT\n"
                f"ReIDSimilarity=1.000\n"
                f"ReferenceEmbeddingHash={ref_hash}\n"
                f"Reason=MANUAL_SELECTION_POINT\n"
            )

            log_reid_test(
                f"\n[REID_TEST] STATE_CHANGE\n"
                f"LogicalTarget={_TARGET_IDENTITY_ID}\n"
                f"OldState={old_st}\n"
                f"NewState={self.target_manager.target.state.value}\n"
                f"TrackerID={selected_id}\n"
                f"Reason=manual_selection_point\n"
                f"ReIDScore=1.000\n"
            )
        return selected_id

    def select_target_by_id(self, track_id: int) -> bool:
        """Manually select and lock onto a specific track ID."""
        old_tr = self.target_manager.target.track_id
        old_st = self.target_manager.target.state.value

        result = self.target_manager.select_by_track_id(track_id, self._last_track_result)
        if result:
            self._register_target_appearance(track_id)

            ident = self.identity_manager.get_identity(_TARGET_IDENTITY_ID) if self.identity_manager else None
            ref_hash = get_embedding_hash(ident.reference_embedding) if (ident and ident.reference_embedding) else "None"

            log_reid_test(
                f"\n[REID_TEST] TARGET_ASSIGNMENT\n"
                f"LogicalTarget={_TARGET_IDENTITY_ID}\n"
                f"OldTracker={old_tr}\n"
                f"NewTracker={track_id}\n"
                f"CurrentTargetState={self.target_manager.target.state.value}\n"
                f"SourceFunction=select_target_by_id\n"
                f"File=single_camera.py\n"
                f"Line=113\n"
                f"ReIDDecision=ACCEPT\n"
                f"ReIDSimilarity=1.000\n"
                f"ReferenceEmbeddingHash={ref_hash}\n"
                f"Reason=MANUAL_SELECTION_ID\n"
            )

            log_reid_test(
                f"\n[REID_TEST] STATE_CHANGE\n"
                f"LogicalTarget={_TARGET_IDENTITY_ID}\n"
                f"OldState={old_st}\n"
                f"NewState={self.target_manager.target.state.value}\n"
                f"TrackerID={track_id}\n"
                f"Reason=manual_selection_id\n"
                f"ReIDScore=1.000\n"
            )
        return result

    def _register_target_appearance(self, track_id: int) -> None:
        """Immediately extract and register the target's appearance at selection time."""
        if self.identity_manager is None or self._last_frame is None or self._last_track_result is None:
            return
        for track in self._last_track_result.tracks:
            if track.track_id == track_id:
                crop = self._extract_crop(self._last_frame, track.box)
                if crop is not None:
                    # Save diagnostic crop
                    if cv2 is not None:
                        os.makedirs(r'C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch', exist_ok=True)
                        cv2.imwrite(r'C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch\reid_ref_A.png', crop)

                    emb = self.identity_manager.register_new_target(
                        crop=crop,
                        identity_id=_TARGET_IDENTITY_ID,
                        label="selected_target",
                    )
                    ident = self.identity_manager.get_identity(_TARGET_IDENTITY_ID)
                    dim = emb.dim if emb else 0
                    hsh = get_embedding_hash(emb)
                    gallery_count = len(ident.embeddings) if ident else 1
                    log_reid_test(
                        f"\n[REID_TEST] TARGET_SELECTED\n"
                        f"LogicalTarget={_TARGET_IDENTITY_ID}\n"
                        f"Tracker={track_id}\n"
                        f"ReferenceEmbeddingCreated=true\n"
                        f"ReferenceEmbeddingDimension={dim}\n"
                        f"ReferenceEmbeddingHash={hsh}\n"
                        f"GallerySize={gallery_count}\n"
                    )
                break

    def clear_target(self) -> None:
        """Deselect the current target."""
        self.target_manager.clear()
        if self.identity_manager is not None:
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

    def _make_appearance_verifier(self):
        """
        Creates a verifier closure backed by IdentityManager.

        Returns None if no identity manager or no registered identity exists.
        """
        if self.identity_manager is None:
            return None
        ident = self.identity_manager.get_identity(_TARGET_IDENTITY_ID)
        if ident is None or (not ident.embeddings and ident.reference_embedding is None):
            return None

        im = self.identity_manager

        def verify(crop: np.ndarray) -> Tuple[bool, float]:
            return im.verify_candidate_crop(crop, identity_id=_TARGET_IDENTITY_ID)

        return verify

    def _manage_target_identity(
        self,
        track_result: TrackResult,
        frame: np.ndarray,
        timestamp_ms: float
    ) -> Target:
        """
        Evaluates the selected target's identity against the current frame tracks.

        1. If the current tracker ID is present, extracts its crop and tests identity.
        2. If verified, confirms TRACKING and periodically updates gallery.
        3. If current tracker ID is missing OR its occupant fails identity verification:
           searches ALL other visible tracks in the frame, ranks them by similarity,
           and checks margin against ambiguity.
        4. If an unambiguous match is found, re-associates target to the new tracker ID.
        5. If no candidate passes or match is ambiguous, transitions to LOST.
        """
        target = self.target_manager.target
        if not self.target_manager.is_active():
            return target

        if self.identity_manager is None:
            return self.target_manager.update(track_result, frame=frame)

        current_track_id = target.track_id
        current_track: Optional[Track] = None
        for track in track_result.tracks:
            if track.track_id == current_track_id:
                current_track = track
                break

        # Auto-capture initial appearance if target was pre-selected (e.g. via CLI --target-id)
        ident = self.identity_manager.get_identity(_TARGET_IDENTITY_ID)
        if (ident is None or (not ident.embeddings and ident.reference_embedding is None)) and current_track is not None:
            crop = self._extract_crop(frame, current_track.box)
            if crop is not None:
                self.identity_manager.register_new_target(
                    crop=crop,
                    identity_id=_TARGET_IDENTITY_ID,
                    label="selected_target",
                    timestamp_ms=timestamp_ms,
                )
                ident = self.identity_manager.get_identity(_TARGET_IDENTITY_ID)
                logger.info(
                    f"[REID] Target initial appearance auto-captured | LogicalTarget={_TARGET_IDENTITY_ID} | Tracker={current_track_id}"
                )

        if ident is None or (not ident.embeddings and ident.reference_embedding is None):
            return self.target_manager.update(track_result, frame=frame)

        ref_hash = get_embedding_hash(ident.reference_embedding) if (ident and ident.reference_embedding) else "None"

        current_verified = False
        if current_track is not None:
            crop = self._extract_crop(frame, current_track.box)
            if crop is not None:
                # Generate query embedding
                query_emb = self.identity_manager.reid.extract(crop)
                query_hash = get_embedding_hash(query_emb)

                is_match, score = self.identity_manager.verify_candidate_crop(crop, _TARGET_IDENTITY_ID)

                ref_sim = ident.reference_embedding.cosine_similarity(query_emb) if (ident and ident.reference_embedding) else 0.0
                gallery_sims = [emb.cosine_similarity(query_emb) for emb in ident.embeddings] if ident else []
                best_gallery_sim = max(gallery_sims) if gallery_sims else 0.0

                decision = "ACCEPT" if is_match else "REJECT"

                log_reid_test(
                    f"\n[REID_TEST] TARGET_VERIFICATION\n"
                    f"LogicalTarget={_TARGET_IDENTITY_ID}\n"
                    f"Tracker={current_track_id}\n"
                    f"Frame={self._frame_id}\n"
                    f"State={target.state.value}\n"
                    f"Crop: {current_track.box.x1:.1f} {current_track.box.y1:.1f} {current_track.box.x2:.1f} {current_track.box.y2:.1f}\n"
                    f"Embedding: generated=true dimension={query_emb.dim} hash={query_hash}\n"
                    f"Reference: hash={ref_hash}\n"
                    f"Similarity: reference_similarity={ref_sim:.3f} best_gallery_similarity={best_gallery_sim:.3f}\n"
                    f"Decision={decision}\n"
                )

                if is_match:
                    current_verified = True
                    old_state = target.state.value
                    self.target_manager.mark_tracking(current_track, track_result.frame_id, timestamp_ms)
                    new_state = target.state.value
                    if old_state != new_state:
                        log_reid_test(
                            f"\n[REID_TEST] STATE_CHANGE\n"
                            f"LogicalTarget={_TARGET_IDENTITY_ID}\n"
                            f"OldState={old_state}\n"
                            f"NewState={new_state}\n"
                            f"TrackerID={current_track.track_id}\n"
                            f"Reason=verification_passed\n"
                            f"ReIDScore={score:.3f}\n"
                        )
                    if self._frame_id % self.reid_interval == 0:
                        self.identity_manager.verified_update(crop, _TARGET_IDENTITY_ID, timestamp_ms)
                else:
                    logger.warning(
                        f"[REID] LogicalTarget={_TARGET_IDENTITY_ID} | CurrentTracker={current_track_id} | "
                        f"Similarity={score:.3f} (< {self.identity_manager.similarity_threshold:.2f}) | "
                        f"Decision=TARGET_IDENTITY_MISMATCH"
                    )

        if not current_verified:
            old_tr = target.track_id
            old_st = target.state.value

            if current_track is None:
                logger.info(
                    f"[REID] LogicalTarget={_TARGET_IDENTITY_ID} | Tracker={current_track_id} | "
                    f"Current target unavailable | State=LOST"
                )

            # Search all other candidate tracks in the frame
            candidate_tracks = [
                t for t in track_result.tracks
                if t.track_id != current_track_id or current_track is None
            ]
            candidate_crops = []
            for t in candidate_tracks:
                c = self._extract_crop(frame, t.box)
                if c is not None:
                    candidate_crops.append((t, c))

            best_track = None
            if candidate_crops:
                ranked = self.identity_manager.rank_candidate_crops(candidate_crops, _TARGET_IDENTITY_ID)
                for rank_idx, (cand_track, sim_score) in enumerate(ranked):
                    cand_crop = next(c for t, c in candidate_crops if t.track_id == cand_track.track_id)
                    cand_emb = self.identity_manager.reid.extract(cand_crop)
                    cand_hash = get_embedding_hash(cand_emb)

                    cand_ref_sim = ident.reference_embedding.cosine_similarity(cand_emb) if (ident and ident.reference_embedding) else 0.0
                    cand_gallery_sims = [emb.cosine_similarity(cand_emb) for emb in ident.embeddings] if ident else []
                    cand_best_gallery_sim = max(cand_gallery_sims) if cand_gallery_sims else 0.0

                    cand_decision = "ACCEPT" if sim_score >= self.identity_manager.similarity_threshold else "REJECT"

                    # Save diagnostic candidate crop
                    if cv2 is not None:
                        cv2.imwrite(rf'C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch\reid_cand_frame_{self._frame_id}_track_{cand_track.track_id}.png', cand_crop)

                    log_reid_test(
                        f"\n[REID_TEST] CANDIDATE\n"
                        f"LogicalTarget={_TARGET_IDENTITY_ID}\n"
                        f"CandidateTracker={cand_track.track_id}\n"
                        f"Crop: {cand_track.box.x1:.1f} {cand_track.box.y1:.1f} {cand_track.box.x2:.1f} {cand_track.box.y2:.1f}\n"
                        f"EmbeddingGenerated=true\n"
                        f"EmbeddingDimension={cand_emb.dim}\n"
                        f"ReferenceEmbeddingHash={ref_hash}\n"
                        f"CandidateEmbeddingHash={cand_hash}\n"
                        f"SimilarityToReference={cand_ref_sim:.3f}\n"
                        f"SimilarityToAdaptiveGallery={cand_best_gallery_sim:.3f}\n"
                        f"Rank={rank_idx + 1}\n"
                        f"Decision={cand_decision}\n"
                    )

                best_item, best_score, second_score, margin = self.identity_manager.find_best_candidate(
                    candidate_crops, _TARGET_IDENTITY_ID, min_margin=self.min_margin
                )
                if best_item is not None:
                    best_track = best_item

            if best_track is not None:
                new_tr = best_track.track_id
                self.target_manager.select_by_track_id(best_track.track_id, track_result)
                self.target_manager.mark_tracking(best_track, track_result.frame_id, timestamp_ms)
                new_st = target.state.value

                log_reid_test(
                    f"\n[REID_TEST] TARGET_ASSIGNMENT\n"
                    f"LogicalTarget={_TARGET_IDENTITY_ID}\n"
                    f"OldTracker={old_tr}\n"
                    f"NewTracker={new_tr}\n"
                    f"CurrentTargetState={new_st}\n"
                    f"SourceFunction=_manage_target_identity\n"
                    f"File=single_camera.py\n"
                    f"Line=257\n"
                    f"ReIDDecision=ACCEPT\n"
                    f"ReIDSimilarity={best_score:.3f}\n"
                    f"ReferenceEmbeddingHash={ref_hash}\n"
                    f"Reason=RECOVERY\n"
                )

                log_reid_test(
                    f"\n[REID_TEST] STATE_CHANGE\n"
                    f"LogicalTarget={_TARGET_IDENTITY_ID}\n"
                    f"OldState={old_st}\n"
                    f"NewState={new_st}\n"
                    f"TrackerID={new_tr}\n"
                    f"Reason=target_recovered\n"
                    f"ReIDScore={best_score:.3f}\n"
                )
            else:
                self.target_manager.mark_lost(timestamp_ms)
                new_st = target.state.value
                if old_st != new_st:
                    log_reid_test(
                        f"\n[REID_TEST] STATE_CHANGE\n"
                        f"LogicalTarget={_TARGET_IDENTITY_ID}\n"
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
        """
        Processes a single video frame through detection, tracking,
        target identity management, and visualization.
        """
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

        # 4. Calculate dynamic FPS
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

    def stream(self) -> Generator[Tuple[np.ndarray, TrackResult, Target], None, None]:
        """
        Generator yielding (annotated_frame, track_result, target) continuously from the camera.
        """
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

