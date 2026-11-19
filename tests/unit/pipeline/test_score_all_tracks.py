"""Step 4: target.score_all_tracks — every track competes on ReID frames, a consistent blur
gate, and a forced full re-verification right after an occlusion ends."""

import numpy as np
import pytest

from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.core.types import BoundingBox, Detection, DetectionResult, Embedding, TargetState
from src.detection.yolo_detector import BaseDetector
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.tracking.byte_tracker import ByteTracker

from tests.unit.pipeline._lock_harness import build, eval_frame, seed_and_lock

TRUE_TARGET, IMPOSTOR = 1, 2


def _cfg(score_all: bool, scope: str = "none") -> AppConfig:
    cfg = AppConfig()
    cfg.target.score_all_tracks = score_all
    cfg.target.blur_gate_scope = scope
    return cfg


def test_flags_default_off_in_code_and_yaml():
    for c in (AppConfig(), AppConfig.from_yaml("configs/default.yaml")):
        assert c.target.score_all_tracks is False
        assert c.target.blur_gate_scope == "none"


# ---------------------------------------------------------------- step(): who gets embedded
class _Det(BaseDetector):
    def __init__(self, boxes):
        self.dets = [Detection(box=BoundingBox(*b), class_id=0, confidence=0.95) for b in boxes]

    def detect(self, frame, timestamp_ms=0.0):
        return DetectionResult(detections=self.dets, frame_id=0, timestamp_ms=timestamp_ms)

    def detect_batch(self, frames, frame_ids=None, timestamps_ms=None):
        frame_ids = frame_ids or [0] * len(frames)
        timestamps_ms = timestamps_ms or [0.0] * len(frames)
        return [DetectionResult(detections=self.dets, frame_id=f, timestamp_ms=t) for f, t in zip(frame_ids, timestamps_ms)]


class _ReID:
    def __init__(self):
        self.v = np.zeros(8, np.float32)
        self.v[0] = 1.0

    def extract(self, crop):
        return Embedding(vector=self.v.copy())

    def extract_batch(self, crops):
        return [self.extract(c) for c in crops]


def _scored_track_counts(score_all: bool):
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_A", name="A", source="synthetic", source_type=SourceType.SYNTHETIC))
    cfg = _cfg(score_all)
    cfg.reid.extract_interval_frames = 1
    cfg.reid.min_sharpness = 0.0
    det = _Det([(50, 50, 150, 300), (400, 50, 500, 300)])
    pipe = MultiCameraPipeline(
        graph=graph, config=cfg, reid_extractor=_ReID(),
        camera_factory=lambda n: SyntheticCamera(width=640, height=480, fps=30),
        tracker_factory=lambda: ByteTracker(track_thresh=0.4, match_thresh=0.5),
        shared_detector=det,
    )
    try:
        pipe._get_or_create_worker("cam_A")
        pipe.step()
        pipe.step()
        assert pipe.select_target_on_camera("cam_A", 100.0, 100.0) is not None
        seen = []
        orig = pipe._evaluate_active_camera_target

        def spy(**kw):
            seen.append(len(kw.get("precomputed_candidates") or []))
            return orig(**kw)

        pipe._evaluate_active_camera_target = spy
        for _ in range(3):
            pipe.target_manager.target.state = TargetState.TRACKING  # steady-tracking state under test
            pipe.step()
        return seen
    finally:
        pipe.stop()


def test_steady_tracking_scores_only_locked_track_when_flag_off():
    assert _scored_track_counts(False) == [1, 1, 1]


def test_steady_tracking_scores_every_track_when_flag_on():
    assert _scored_track_counts(True) == [2, 2, 2]


# ---------------------------------------------------------------- blur gate consistency
def test_scope_all_blurred_locked_track_defers_instead_of_counting_a_miss():
    p, im, tm = build(_cfg(True, "all"))
    seed_and_lock(tm, TRUE_TARGET)
    for i in range(1, 11):
        p._locked_track_blurred = True  # what step() reports when the locked crop fails the gate
        # step() excluded the blurred locked crop; only the bystander was embedded
        tid, st = eval_frame(p, i * 3, {TRUE_TARGET: 0.9, IMPOSTOR: 0.30}, scored_ids=[IMPOSTOR])
        assert tid == TRUE_TARGET
        assert st not in (TargetState.REJECTED, TargetState.SEARCHING, TargetState.LOST)


def test_legacy_missing_locked_crop_is_treated_as_a_miss():
    # Contrast: without the deferral, a locked track with no embedding scores 0 and is rejected.
    p, im, tm = build(_cfg(False))
    seed_and_lock(tm, TRUE_TARGET)
    states = [eval_frame(p, i * 3, {TRUE_TARGET: 0.9, IMPOSTOR: 0.30}, scored_ids=[IMPOSTOR])[1] for i in range(1, 4)]
    assert TargetState.SEARCHING in states


def test_scope_none_scores_blurred_locked_track_normally():
    p, im, tm = build(_cfg(True, "none"))
    seed_and_lock(tm, TRUE_TARGET)
    p._locked_track_blurred = True  # ignored when scope is "none"
    tid, st = eval_frame(p, 3, {TRUE_TARGET: 0.9, IMPOSTOR: 0.30})
    assert (tid, st) == (TRUE_TARGET, TargetState.CONFIRMED)


# ---------------------------------------------------------------- post-occlusion re-verification
def _swap_during_occlusion(score_all: bool):
    """Tracker swapped IDs during an occlusion: lock now sits on the impostor (0.55); the true
    target (0.93) is visible. The occlusion ends on frame 4, which is NOT a ReID frame (interval 3)."""
    p, im, tm = build(_cfg(score_all))
    seed_and_lock(tm, IMPOSTOR, state=TargetState.OCCLUDED)
    sims = {TRUE_TARGET: 0.93, IMPOSTOR: 0.55}
    return [eval_frame(p, f, sims) for f in (4, 5, 6)]


def test_first_frame_after_occlusion_is_fully_reverified_when_flag_on():
    hist = _swap_during_occlusion(True)
    assert hist[0] == (TRUE_TARGET, TargetState.CONFIRMED)  # fixed on the first post-occlusion frame


def test_legacy_trusts_tracker_id_until_next_reid_frame():
    hist = _swap_during_occlusion(False)
    assert hist[0] == (IMPOSTOR, TargetState.TRACKING)  # trust-by-ID on frames 4 and 5
    assert hist[1] == (IMPOSTOR, TargetState.TRACKING)
    assert hist[2][0] == TRUE_TARGET  # only corrected on ReID frame 6


def test_reverify_happens_once_then_cadence_resumes():
    p, im, tm = build(_cfg(True))
    seed_and_lock(tm, TRUE_TARGET, state=TargetState.OCCLUDED)
    sims = {TRUE_TARGET: 0.93, IMPOSTOR: 0.30}
    eval_frame(p, 4, sims)  # forced re-verification
    assert p._post_occlusion_reverify_pending is False
    p._last_candidate_scores.clear()
    eval_frame(p, 5, sims)  # ordinary non-ReID frame: no scoring
    assert p._last_candidate_scores == {}
