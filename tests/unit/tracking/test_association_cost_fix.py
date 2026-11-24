"""Step 3: tracking.fix_association_cost — ID fragmentation, dead low-confidence stage, duplicate frames."""

from typing import List, Optional

import numpy as np
import pytest

from src.core.config import AppConfig
from src.core.types import BoundingBox, Detection, DetectionResult, Embedding
from src.pipeline.camera_worker import CameraWorker
from src.tracking.bot_sort import BoTSORTTracker

FRAMES = 60


def _walk_ids(speed_px: float, fix: bool, conf_fn=lambda f: 0.9, with_embeddings: bool = False) -> List[Optional[int]]:
    """One person walking right at speed_px per frame; returns the ID covering them each frame."""
    rng = np.random.default_rng(0)
    proto = rng.normal(size=512).astype(np.float32)
    proto /= np.linalg.norm(proto)

    class _Dummy:  # non-None extractor so appearance fusion is active; embeddings are passed in
        def extract_batch(self, crops):
            raise AssertionError("embeddings are precomputed in this test")

    tr = BoTSORTTracker(
        track_thresh=0.4, match_thresh=0.8, track_buffer=30, min_box_area=100.0,
        reid_extractor=_Dummy() if with_embeddings else None, reid_weight=0.5,
        fix_association_cost=fix,
    )
    ids: List[Optional[int]] = []
    for f in range(1, FRAMES + 1):
        x = 50.0 + speed_px * f
        conf = conf_fn(f)
        det = Detection(box=BoundingBox(x, 200.0, x + 80.0, 440.0), class_id=0, confidence=conf)
        embs = None
        if with_embeddings and conf >= 0.4:
            v = proto + 0.3 * rng.normal(size=512).astype(np.float32) / np.sqrt(512)
            embs = [Embedding(vector=(v / np.linalg.norm(v)).astype(np.float32))]
        res = tr.update(DetectionResult(detections=[det], frame_id=f, timestamp_ms=f * 33.3), None, det_embeddings=embs)
        best, best_d = None, 1e9
        for t in res.tracks:
            d = abs(t.box.x1 - x)
            if d < best_d:
                best_d, best = d, t.track_id
        ids.append(best if best_d < 0.75 * 80 else None)
    return ids


def id_switches(ids: List[Optional[int]]) -> int:
    seen = [i for i in ids if i is not None]
    return len(set(seen)) - 1 if seen else 0


def test_flag_defaults_on_in_code_and_yaml():
    assert AppConfig().tracking.fix_association_cost is True
    y = AppConfig.from_yaml("configs/default.yaml").tracking
    assert y.fix_association_cost is True
    assert y.low_conf_detection_threshold == AppConfig().tracking.low_conf_detection_threshold == 0.2


@pytest.mark.parametrize("speed", [10, 20, 40])
@pytest.mark.parametrize("with_embeddings", [False, True])
def test_single_id_over_60_frames_with_flag_on(speed, with_embeddings):
    ids = _walk_ids(speed, fix=True, with_embeddings=with_embeddings)
    assert None not in ids
    assert id_switches(ids) == 0


def test_flag_off_reproduces_fragmentation():
    # Documents the legacy bug the flag fixes (IoU >= 0.8 requirement).
    assert id_switches(_walk_ids(10, fix=False)) > 0
    assert id_switches(_walk_ids(40, fix=False)) > 0


def test_stage1_threshold_semantics():
    assert BoTSORTTracker(match_thresh=0.8, fix_association_cost=False)._stage1_max_cost() == pytest.approx(0.2)
    assert BoTSORTTracker(match_thresh=0.8, fix_association_cost=True)._stage1_max_cost() == pytest.approx(0.8)


def test_low_confidence_detections_keep_track_alive():
    # Confidence dips to 0.25 (below track_thresh 0.4) for frames 20-29, e.g. partial occlusion.
    # With the detector at 0.2 these reach BoT-SORT's low-score stage and extend the track.
    dip = lambda f: 0.25 if 20 <= f < 30 else 0.9
    ids = _walk_ids(10, fix=True, conf_fn=dip)
    assert None not in ids and id_switches(ids) == 0


def test_pipeline_lowers_detector_confidence_only_with_flag():
    from src.pipeline.multi_camera_pipeline import MultiCameraPipeline

    p = object.__new__(MultiCameraPipeline)
    p.config = AppConfig()
    p.config.tracking.fix_association_cost = False
    assert p._effective_detection_confidence() == pytest.approx(0.4)
    p.config.tracking.fix_association_cost = True
    assert p._effective_detection_confidence() == pytest.approx(0.2)
    # never raises the threshold above the configured detector confidence
    p.config.detection.confidence_threshold = 0.1
    assert p._effective_detection_confidence() == pytest.approx(0.1)


class _ThreadedCam:
    """Mimics OpenCVCamera(use_thread=True): read() returns the latest frame and records its seq."""

    use_thread = True

    def __init__(self):
        self._frame_seq = 0
        self._last_read_seq = 0

    def publish(self):
        self._frame_seq += 1

    def is_opened(self):
        return True

    def read(self):
        self._last_read_seq = self._frame_seq
        return True, np.zeros((4, 4, 3), np.uint8), float(self._frame_seq)

    def release(self):
        pass


@pytest.mark.parametrize("skip", [False, True])
def test_camera_worker_duplicate_frames(skip):
    cam = _ThreadedCam()
    w = CameraWorker(camera=cam, tracker=None, camera_id="c", skip_duplicate_frames=skip)
    cam.publish()
    ok1, f1, _ = w.read_frame()
    ok2, f2, _ = w.read_frame()  # no new frame published
    cam.publish()
    ok3, f3, _ = w.read_frame()
    assert ok1 and f1 is not None and ok3 and f3 is not None
    if skip:
        assert ok2 is False and f2 is None and w.last_read_duplicate is False  # reset by 3rd read
    else:
        assert ok2 is True and f2 is not None


def test_camera_worker_ignores_unthreaded_cameras():
    cam = _ThreadedCam()
    cam.use_thread = False
    w = CameraWorker(camera=cam, tracker=None, camera_id="c", skip_duplicate_frames=True)
    cam.publish()
    assert w.read_frame()[0] and w.read_frame()[0]


def _crossing(speed: float, fix: bool, with_embeddings: bool):
    """Two people walking towards each other and crossing; returns (ids_of_A, ids_of_B)."""
    rng = np.random.default_rng(1)
    pa, pb = rng.normal(size=512), rng.normal(size=512)
    pa, pb = pa / np.linalg.norm(pa), pb / np.linalg.norm(pb)

    def emb(p):
        v = p + 0.3 * rng.normal(size=512) / np.sqrt(512)
        return Embedding(vector=(v / np.linalg.norm(v)).astype(np.float32))

    class _Dummy:
        def extract_batch(self, crops):
            raise AssertionError

    tr = BoTSORTTracker(match_thresh=0.8, reid_extractor=_Dummy() if with_embeddings else None,
                        fix_association_cost=fix)
    n = int(600 / speed) + 20
    ids_a, ids_b = [], []
    for f in range(1, n + 1):
        ax, bx = 50 + speed * f, 50 + speed * n - speed * f
        dets = [Detection(box=BoundingBox(ax, 200, ax + 80, 440), class_id=0, confidence=0.9),
                Detection(box=BoundingBox(bx, 205, bx + 80, 445), class_id=0, confidence=0.9)]
        res = tr.update(DetectionResult(detections=dets, frame_id=f, timestamp_ms=f * 33.3), None,
                        det_embeddings=[emb(pa), emb(pb)] if with_embeddings else None)
        for x, y, out in ((ax, 200, ids_a), (bx, 205, ids_b)):
            m = [t.track_id for t in res.tracks if abs(t.box.x1 - x) < 2 and abs(t.box.y1 - y) < 3]
            out.append(m[0] if m else None)
    return ids_a, ids_b


@pytest.mark.parametrize("speed", [10, 20])
@pytest.mark.parametrize("with_embeddings", [False, True])
def test_crossing_keeps_identities_with_flag_on(speed, with_embeddings):
    # The looser max cost must not introduce identity swaps when two people cross.
    a, b = _crossing(speed, fix=True, with_embeddings=with_embeddings)
    assert len(set(i for i in a if i is not None)) == 1
    assert len(set(i for i in b if i is not None)) == 1
    assert set(a) & set(b) <= {None}
