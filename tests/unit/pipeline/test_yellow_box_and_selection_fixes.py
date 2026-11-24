from __future__ import annotations

import numpy as np
import pytest

from src.core.config import AppConfig
from src.core.interfaces import BaseDetector, BaseReID
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.core.types import (
    BoundingBox,
    Detection,
    DetectionResult,
    Embedding,
    Target,
    TargetState,
    Track,
    TrackResult,
    TrackState,
)
from src.identity.manager import IdentityManager
from src.identity.sqlite_store import SQLiteVectorStore
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.visualization.annotator import FrameAnnotator


def _count_bgr_pixels(img: np.ndarray, bgr: tuple[int, int, int]) -> int:
    mask = np.all(img == np.array(bgr, dtype=np.uint8), axis=2)
    return int(np.sum(mask))


def test_annotator_never_draws_solid_yellow_when_searching_lost_or_rejected():
    """Regression: yellow locked box must never stick on last_known_box or rejected track."""
    annotator = FrameAnnotator(draw_boxes=True, draw_ids=True, box_thickness=2)
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    # Use lower canvas (below top-left HUD text y<80? Note boxes sit at y=90..220) to isolate bbox pixels
    stale_box = BoundingBox(20, 100, 80, 210)
    moved_track = Track(track_id=1, box=BoundingBox(200, 100, 260, 210), state=TrackState.TRACKED, confidence=0.95)
    tr_other = TrackResult(tracks=[Track(track_id=2, box=BoundingBox(200, 100, 260, 210), state=TrackState.TRACKED, confidence=0.95)])

    yellow = (0, 215, 255)

    # 1. Actively LOCKED with live track -> SHOULD draw yellow box around moved_track (NOT stale_box!)
    t_locked = Target(state=TargetState.LOCKED, track_id=1, last_known_box=stale_box)
    img_locked = annotator.annotate(blank, TrackResult(tracks=[moved_track]), target=t_locked, camera_id="cam_0")
    assert _count_bgr_pixels(img_locked[95:220, 195:265], yellow) > 100
    assert _count_bgr_pixels(img_locked[95:220, 15:85], yellow) == 0

    # 2. SEARCHING with track missing (last_known_box fallback) -> NO solid yellow at all in box region!
    t_searching_missing = Target(state=TargetState.SEARCHING, track_id=1, last_known_box=stale_box, lost_duration_ms=1500.0)
    img_search_missing = annotator.annotate(blank, tr_other, target=t_searching_missing, camera_id="cam_0")
    assert _count_bgr_pixels(img_search_missing[80:, :], yellow) == 0

    # 3. SEARCHING while old track_id=1 still exists (e.g. after mark_rejected -> mark_searching) -> NO yellow!
    t_searching_stale_id = Target(state=TargetState.SEARCHING, track_id=1, last_known_box=stale_box)
    img_search_stale = annotator.annotate(blank, TrackResult(tracks=[moved_track]), target=t_searching_stale_id, camera_id="cam_0")
    assert _count_bgr_pixels(img_search_stale[80:, :], yellow) == 0

    # 4. REJECTED / LOST_PERMANENTLY -> NO yellow and NO frozen box around stale_box!
    for st in (TargetState.REJECTED, TargetState.LOST_PERMANENTLY):
        t_dead = Target(state=st, track_id=1, last_known_box=stale_box)
        img_dead = annotator.annotate(blank, TrackResult(tracks=[moved_track]), target=t_dead, camera_id="cam_0")
        assert _count_bgr_pixels(img_dead[80:, :], yellow) == 0
        # Nothing drawn in stale_box region
        stale_patch = img_dead[100:210, 20:80]
        assert int(np.sum(stale_patch)) == 0


def test_identity_manager_set_db_path_and_weights(tmp_path):
    """IdentityManager.set_db_path switches SQLite store cleanly; w_* weights affect candidate_score."""
    class _FixedReID(BaseReID):
        def __init__(self, map_fn):
            self._map_fn = map_fn
        def extract(self, crop):
            return self._map_fn("full")
        def extract_decomposed(self, crop):
            return (
                self._map_fn("fused"),
                self._map_fn("deep"),
                self._map_fn("color"),
                self._map_fn("upper"),
                self._map_fn("lower"),
            )
        def extract_batch(self, crops):
            return [self._map_fn("full") for _ in crops]

    def _emb(vec):
        v = np.array(vec, dtype=np.float32)
        v /= np.linalg.norm(v)
        return Embedding(vector=v, model_name="mock", version="2.0", crop_type="full")

    # Anchor along axis 0
    store1 = SQLiteVectorStore(db_path=str(tmp_path / "case1.db"))
    im = IdentityManager(
        reid_extractor=_FixedReID(lambda k: _emb([1, 0, 0, 0])),
        vector_store=store1,
        w_upper=1.0, w_color=0.0, w_deep=0.0, w_lower=0.0,
        crops_dir=str(tmp_path / "crops"),
    )
    crop = np.full((80, 40, 3), 128, dtype=np.uint8)
    im.register_new_target(crop, "target_0")
    im.save_to_db(str(tmp_path / "case1.db"))

    # Custom weights where only upper matches
    im.reid = _FixedReID(lambda k: _emb([1, 0, 0, 0]) if k == "upper" else _emb([0, 1, 0, 0]))
    ev_upper_only = im.evaluate_candidate_crop(crop, "target_0")
    assert ev_upper_only.candidate_score == pytest.approx(1.0, abs=1e-4)

    # Switch DB path to empty case2 -> clears case1 identity cleanly
    im.set_db_path(str(tmp_path / "case2.db"))
    assert im.get_identity("target_0") is None
    assert str(im.vector_store.db_path).endswith("case2.db")

    # Switch back to case1 -> reloads saved case1 identity
    im.set_db_path(str(tmp_path / "case1.db"))
    assert im.get_identity("target_0") is not None


def test_select_target_fallback_avoids_id_collision_and_searching_runs_reid_every_frame():
    """Verify non-colliding synthetic ID fallback and SEARCHING Phase-4 ReID every frame."""
    class _CountingReID(BaseReID):
        def __init__(self):
            self.batch_calls = 0
        def extract(self, crop):
            return Embedding(vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), model_name="m", version="2.0")
        def extract_decomposed(self, crop):
            e = self.extract(crop)
            return e, e, e, e, e
        def extract_batch(self, crops):
            self.batch_calls += 1
            return [self.extract(c) for c in crops]

    class _Det(BaseDetector):
        def detect(self, frame, frame_id: int = 0, timestamp_ms: float = 0.0):
            return DetectionResult(
                detections=[
                    Detection(box=BoundingBox(10, 20, 60, 140), class_id=0, confidence=0.95),
                    Detection(box=BoundingBox(200, 20, 250, 140), class_id=0, confidence=0.95),
                ],
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
            )

    class _Cam:
        def __init__(self, frame):
            self._f = frame
            self._idx = 0
        def open(self): return True
        def read(self):
            self._idx += 1
            f = self._f.copy()
            f[0, 0, 0] = self._idx % 255
            return True, f, 100.0 * self._idx
        def release(self): pass
        def is_opened(self): return True
        def get_fps(self): return 30.0
        def get_resolution(self): return (320, 240)

    cfg = AppConfig()
    cfg.storage.enabled = False
    cfg.recording.enabled = False
    cfg.reid.extract_interval_frames = 3
    cfg.reid.min_sharpness = 0.0

    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_0", name="c0", source="0", source_type=SourceType.WEBCAM))

    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    reid = _CountingReID()
    pipe = MultiCameraPipeline(
        graph=graph,
        config=cfg,
        reid_extractor=reid,
        shared_detector=_Det(),
        camera_factory=lambda n: _Cam(frame),
    )
    try:
        w = pipe._get_or_create_worker("cam_0")
        # Seed existing Track #1 at x=10..60, then click unrelated det at x=225 without track in _last_track_result
        w._last_frame = frame.copy()
        w._last_track_result = TrackResult(tracks=[
            Track(track_id=1, box=BoundingBox(10, 20, 60, 140), state=TrackState.TRACKED, confidence=0.95)
        ])
        # Mock process_frame returning only Track #1 so fallback path executes on detection (200..250)
        w.process_frame = lambda f, ts, det, det_embeddings=None: TrackResult(
            tracks=[Track(track_id=1, box=BoundingBox(10, 20, 60, 140), state=TrackState.TRACKED, confidence=0.95)],
            frame_id=1,
            timestamp_ms=ts,
        )
        sel_id = pipe.select_target_on_camera("cam_0", 225.0, 80.0)
        assert sel_id is not None and sel_id != 1, f"Fallback ID collided with Track #1 (got {sel_id})"
        assert {t.track_id for t in w._last_track_result.tracks} >= {1, sel_id}

        # Verify SEARCHING state forces ReID even on non-interval frames when reid_interval == 3
        pipe.target_manager.target.state = TargetState.SEARCHING
        pipe._frame_count = 0
        reid.batch_calls = 0
        for _ in range(3):
            pipe.step()
        assert reid.batch_calls >= 3, f"Expected ReID on all 3 SEARCHING frames, got {reid.batch_calls}"
    finally:
        pipe.stop()
