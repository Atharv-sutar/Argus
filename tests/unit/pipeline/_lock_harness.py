"""Minimal harness that drives the real MultiCameraPipeline._evaluate_active_camera_target
with synthetic embeddings of exactly known gallery similarity (no camera, detector or model)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.config import AppConfig
from src.core.types import BoundingBox, Embedding, TargetState, Track, TrackResult, TrackState
from src.identity.evidence import EvidenceEngine
from src.identity.manager import IdentityManager
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.target.manager import TargetManager

DIM = 512
_E = np.eye(DIM, dtype=np.float32)


def emb_with_sim(sim: float, axis: int) -> Embedding:
    """Unit vector whose cosine similarity to the gallery anchor (axis 0) is exactly `sim`."""
    v = sim * _E[0] + np.sqrt(max(0.0, 1.0 - sim ** 2)) * _E[axis]
    return Embedding(vector=v.astype(np.float32), model_name="synthetic", version="1", crop_type="full")


class _NullStore:
    def add(self, *a, **k):
        pass

    def remove_identity(self, *a, **k):
        pass

    def clear(self, *a, **k):
        pass

    def __str__(self) -> str:
        return "null"


class _AnchorReID:
    """Seeds the trusted gallery with the anchor vector (axis 0)."""

    def extract_decomposed(self, crop):
        e = Embedding(vector=_E[0].copy(), model_name="synthetic", version="1", crop_type="fused")
        return e, e, e, e, e

    def extract(self, crop):
        return self.extract_decomposed(crop)[0]

    def extract_batch(self, crops):
        return [self.extract(c) for c in crops]


class _Worker:
    def __init__(self):
        self._last_frame = None

    @staticmethod
    def extract_crop(frame, box):
        x1, y1, x2, y2 = map(int, box.as_xyxy())
        return frame[max(0, y1):y2, max(0, x1):x2]


class _NullRecorder:
    def on_target_state_changed(self, *a, **k):
        pass

    def feed_frame(self, *a, **k):
        pass

    def discard(self, *a, **k):
        pass


BOXES: Dict[int, BoundingBox] = {
    1: BoundingBox(100, 200, 180, 440),
    2: BoundingBox(600, 200, 680, 440),
}


def build(config: Optional[AppConfig] = None) -> Tuple[MultiCameraPipeline, IdentityManager, TargetManager]:
    cfg = config or AppConfig()
    im = IdentityManager(
        reid_extractor=_AnchorReID(),
        vector_store=_NullStore(),
        similarity_threshold=cfg.reid.match_threshold,
        reacquisition_threshold=cfg.reid.reacquisition_threshold,
        min_margin=cfg.reid.min_margin,
        auto_add_threshold=cfg.reid.auto_add_threshold,
    )
    tm = TargetManager(identity_manager=im, min_margin=cfg.reid.min_margin)
    p = object.__new__(MultiCameraPipeline)
    p.config = cfg
    p.target_manager = tm
    p.reid_extractor = _AnchorReID()
    p._last_candidate_scores = {}
    p._switch_consensus = {}
    p._current_track_misses = 0
    p._frame_count = 0
    p.reid_interval = cfg.reid.extract_interval_frames
    p._active_camera_id = "cam_0"
    p._reacquisition_threshold = cfg.reid.reacquisition_threshold
    p._evidence_engine = EvidenceEngine(
        window_size=5,
        min_similarity_threshold=cfg.reid.match_threshold,
        reacquisition_threshold=p._reacquisition_threshold,
        reacquisition_min_frames=3,
        min_margin_threshold=cfg.reid.min_margin,
        min_consistency_ratio=0.70,
    )
    import collections
    import threading
    from src.multi_camera.camera_graph import CameraGraph
    from src.multi_camera.search_manager import SearchManager
    from src.core.config import SearchConfig
    p.route_recorder = None
    p.forensic_recorder = _NullRecorder()
    p.decision_logger = None
    p._bystander_history = {}
    p._worker_lock = threading.Lock()
    p._pipeline_lock = threading.RLock()
    p._frame_lock = threading.Lock()
    p._latest_jpegs = {}
    p._workers = {"cam_0": _Worker()}
    p._transit_history = collections.deque(maxlen=500)
    p.graph = CameraGraph()
    p._nodes = {}
    p.search_manager = SearchManager(p.graph, SearchConfig())
    return p, im, tm


FRAME = np.full((720, 1280, 3), 127, np.uint8)


def tracks(ids=(1, 2)) -> List[Track]:
    return [Track(track_id=i, box=BOXES[i], state=TrackState.TRACKED, confidence=0.9) for i in ids]


def seed_and_lock(tm: TargetManager, locked_id: int, state: TargetState = TargetState.CONFIRMED) -> None:
    """Seed the gallery from the TRUE target (track 1), then put the lock on `locked_id`."""
    tm.start_new_investigation(1, TrackResult(tracks=tracks(), frame_id=0, timestamp_ms=0.0), FRAME, "cam_0")
    tm.target.track_id = locked_id
    tm.target.state = state


def run_reid_frames(p: MultiCameraPipeline, sims: Dict[int, float], n: int, start: int = 1, dt_ms: float = 100.0):
    """Feed n ReID frames; sims maps track_id -> similarity to the gallery anchor."""
    history = []
    for i in range(start, start + n):
        p._frame_count = i * max(1, p.reid_interval)  # every call is a ReID frame
        tr = TrackResult(tracks=tracks(tuple(sims.keys())), frame_id=i, timestamp_ms=i * dt_ms)
        cands = []
        for t in tr.tracks:
            crop = _Worker.extract_crop(FRAME, t.box)
            cands.append((t, crop, emb_with_sim(sims[t.track_id], axis=10 + t.track_id)))
        p._evaluate_active_camera_target(_Worker(), FRAME, tr, i * dt_ms, precomputed_candidates=cands)
        history.append((p.target_manager.target.track_id, p.target_manager.target.state))
    return history


def enrolled_track_ids(im: IdentityManager) -> List[Optional[int]]:
    ident = im.get_identity("target_0")
    if not ident:
        return []
    return [e[1] if isinstance(e, tuple) else getattr(e, "track_id", None) for e in ident.provisional_gallery]


def eval_frame(p: MultiCameraPipeline, frame_count: int, sims: Dict[int, float], scored_ids=None, dt_ms: float = 100.0):
    """Run one _evaluate_active_camera_target call at an explicit _frame_count.
    `scored_ids` restricts which tracks have precomputed embeddings (default: all)."""
    p._frame_count = frame_count
    tr = TrackResult(tracks=tracks(tuple(sims.keys())), frame_id=frame_count, timestamp_ms=frame_count * dt_ms)
    ids = set(sims.keys()) if scored_ids is None else set(scored_ids)
    cands = [
        (t, _Worker.extract_crop(FRAME, t.box), emb_with_sim(sims[t.track_id], axis=10 + t.track_id))
        for t in tr.tracks if t.track_id in ids
    ]
    p._evaluate_active_camera_target(_Worker(), FRAME, tr, frame_count * dt_ms, precomputed_candidates=cands)
    return p.target_manager.target.track_id, p.target_manager.target.state
