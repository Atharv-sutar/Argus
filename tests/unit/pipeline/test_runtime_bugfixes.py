"""Regression tests for runtime bugs found in the clean-up audit (see fixes.* in configs/default.yaml)."""

import threading
import time

import numpy as np
import pytest

from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig, SearchConfig
from src.core.multi_camera_types import CameraNodeConfig, HandoffDecision, SourceType
from src.core.types import BoundingBox, Detection, DetectionResult, Embedding, TargetState, TrackResult
from src.detection.yolo_detector import BaseDetector
from src.multi_camera.camera_graph import CameraGraph
from src.multi_camera.search_manager import SearchManager
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline

from tests.unit.pipeline._lock_harness import FRAME, build, emb_with_sim, seed_and_lock, tracks


class _Det(BaseDetector):
    def __init__(self, boxes, gate: threading.Event = None, entered: threading.Event = None):
        self.dets = [Detection(box=BoundingBox(*b), class_id=0, confidence=0.95) for b in boxes]
        self.gate, self.entered = gate, entered

    def detect(self, frame, timestamp_ms=0.0):
        return DetectionResult(detections=self.dets, frame_id=0, timestamp_ms=timestamp_ms)

    def detect_batch(self, frames, frame_ids=None, timestamps_ms=None):
        if self.gate is not None:
            self.entered.set()
            self.gate.wait(5.0)
        frame_ids = frame_ids or [0] * len(frames)
        timestamps_ms = timestamps_ms or [0.0] * len(frames)
        return [DetectionResult(detections=self.dets, frame_id=f, timestamp_ms=t) for f, t in zip(frame_ids, timestamps_ms)]


class _ReID:
    def extract(self, crop):
        v = np.zeros(8, np.float32)
        v[0] = 1.0
        return Embedding(vector=v)

    def extract_batch(self, crops):
        return [self.extract(c) for c in crops]

    def extract_decomposed(self, crop):
        e = self.extract(crop)
        return e, e, e, e, e


class _NoTracks:
    """Tracker that never confirms anything, forcing the click-selection detection fallback."""

    reid_extractor = None

    def update(self, det, frame, det_embeddings=None):
        return TrackResult(tracks=[], frame_id=det.frame_id, timestamp_ms=det.timestamp_ms)


def _pipe(cfg, det, tracker_factory=None):
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_A", name="A", source="synthetic", source_type=SourceType.SYNTHETIC))
    cfg.storage.enabled = False
    return MultiCameraPipeline(
        graph=graph, config=cfg, reid_extractor=_ReID(),
        camera_factory=lambda n: SyntheticCamera(width=640, height=480, fps=30),
        tracker_factory=tracker_factory or (lambda: _NoTracks()),
        shared_detector=det,
    )


def test_fixes_flags_default_off_in_code_and_yaml():
    for c in (AppConfig(), AppConfig.from_yaml("configs/default.yaml")):
        f = c.fixes
        assert not any([f.serialize_pipeline_access, f.reset_state_on_target_change, f.auto_add_handles_time_reset,
                        f.per_track_handoff_confirmation, f.apply_settings_live])


# ------------------------------------------------------------ direct crash fixes
def test_click_fallback_with_no_track_no_longer_crashes():
    # Previously built Track(state=TrackState.CONFIRMED), which does not exist -> AttributeError.
    pipe = _pipe(AppConfig(), _Det([(50, 50, 150, 300)]))
    try:
        pipe._get_or_create_worker("cam_A")
        pipe.step()
        assert pipe.select_target_on_camera("cam_A", 100.0, 100.0) == 1
        assert pipe.target_manager.target.state == TargetState.LOCKED
    finally:
        pipe.stop()


def test_accept_handoff_exists_and_delegates(monkeypatch):
    # POST /api/handoff/confirm calls pipeline.accept_handoff(), which did not exist.
    p = object.__new__(MultiCameraPipeline)
    called = []
    monkeypatch.setattr(MultiCameraPipeline, "confirm_handoff", lambda self: called.append(1) or True)
    assert p.accept_handoff() is True and called == [1]


# ------------------------------------------------------------ fixes.serialize_pipeline_access
@pytest.mark.parametrize("flag", [False, True])
def test_operator_actions_wait_for_in_flight_step(flag):
    cfg = AppConfig()
    cfg.fixes.serialize_pipeline_access = flag
    gate, entered = threading.Event(), threading.Event()
    det = _Det([(50, 50, 150, 300)])
    pipe = _pipe(cfg, det)
    try:
        pipe._get_or_create_worker("cam_A")
        pipe.step()
        det.gate, det.entered = gate, entered
        t_step = threading.Thread(target=pipe.step, daemon=True)
        t_step.start()
        assert entered.wait(5.0)  # step() is now blocked inside detection
        done = threading.Event()
        threading.Thread(target=lambda: (pipe.clear_target(), done.set()), daemon=True).start()
        finished_during_step = done.wait(0.5)
        gate.set()
        t_step.join(5.0)
        assert done.wait(5.0)
        assert finished_during_step is (not flag)
    finally:
        gate.set()
        pipe.stop()


# ------------------------------------------------------------ fixes.reset_state_on_target_change
@pytest.mark.parametrize("flag", [False, True])
def test_clear_target_drops_previous_target_state(flag):
    cfg = AppConfig()
    cfg.fixes.reset_state_on_target_change = flag
    p, im, tm = build(cfg)
    im.reset_state_on_target_change = flag
    p._evidence_engine.clear = lambda: None  # not under test here
    p.search_manager = SearchManager(CameraGraph(), SearchConfig())
    p.graph, p._nodes, p._transit_history = CameraGraph(), {}, []
    p.route_recorder, p.playback_controller = None, None
    p._frame_lock, p._latest_jpegs = threading.Lock(), {}
    p.forensic_recorder.discard = lambda: None
    seed_and_lock(tm, 1)
    im._entry_crops["target_0_provisional_3"] = "old-thumbnail"
    p._switch_consensus[7] = 1
    p._current_track_misses = 1
    p._pending_handoff_cam = "cam_B"
    p.clear_target()
    assert ("target_0_provisional_3" in im._entry_crops) is (not flag)
    assert (p._switch_consensus == {}) is flag
    assert (getattr(p, "_pending_handoff_cam", None) is None) is flag


@pytest.mark.parametrize("flag", [False, True])
def test_new_target_does_not_inherit_old_thumbnails(flag):
    p, im, tm = build(AppConfig())
    im.reset_state_on_target_change = flag
    seed_and_lock(tm, 1)
    im._entry_crops["target_0_trusted_1"] = "previous-target-crop"
    seed_and_lock(tm, 2)  # operator selects someone new: gallery re-seeded with clear_existing=True
    assert ("target_0_trusted_1" in im._entry_crops) is (not flag)


# ------------------------------------------------------------ fixes.auto_add_handles_time_reset
@pytest.mark.parametrize("flag", [False, True])
def test_auto_add_after_timestamp_goes_backwards(flag):
    p, im, tm = build(AppConfig())
    im.auto_add_handles_time_reset = flag
    seed_and_lock(tm, 1)
    crop = FRAME[200:440, 100:180]
    assert im.add_auto(crop, emb_with_sim(0.95, 20), 0.95, timestamp_ms=60_000.0, track_id=1)
    # video file loops back to t=0.5 s with a different-looking (diverse) sample
    added = im.add_auto(crop, emb_with_sim(0.93, 21), 0.93, timestamp_ms=500.0, track_id=1)
    assert added is flag


# ------------------------------------------------------------ fixes.per_track_handoff_confirmation
@pytest.mark.parametrize("flag", [False, True])
def test_handoff_confirmations_must_come_from_one_track(flag):
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_B", name="B", source="synthetic", source_type=SourceType.SYNTHETIC))
    sm = SearchManager(graph, SearchConfig(confirmation_frames=3))
    sm.per_track_confirmation = flag
    sm.on_target_lost("cam_B")
    decisions = [sm.on_candidate_found("cam_B", 0.90, 0.73, 0.82, track_id=tid) for tid in (5, 6, 5)]
    assert (decisions[-1] == HandoffDecision.CONFIRMED) is (not flag)
    # the same track three times still confirms
    sm.reset()
    sm.on_target_lost("cam_B")
    same = [sm.on_candidate_found("cam_B", 0.90, 0.73, 0.82, track_id=9) for _ in range(3)]
    assert same[-1] == HandoffDecision.CONFIRMED


# ------------------------------------------------------------ fixes.apply_settings_live
@pytest.mark.parametrize("flag", [False, True])
def test_settings_update_propagates_match_threshold(flag):
    cfg = AppConfig()
    cfg.fixes.apply_settings_live = flag
    p, im, tm = build(cfg)
    p.identity_manager = im
    before = im.similarity_threshold
    cfg.reid.match_threshold = 0.66
    p.apply_config_update(cfg)
    assert (im.similarity_threshold == pytest.approx(0.66)) is flag
    assert (p._evidence_engine.min_similarity_threshold == pytest.approx(0.66)) is flag
    if not flag:
        assert im.similarity_threshold == before
