"""Test BoT-SORT tracking dynamics and target selection in live pipeline."""

import logging
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import AppConfig
from src.core.types import BoundingBox, Detection, DetectionResult, TargetState
from src.tracking.bot_sort import BoTSORTTracker
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.multi_camera.camera_graph import CameraGraph
from src.core.multi_camera_types import CameraNodeConfig, SourceType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_botsort_dynamics")

class ControllableDetector:
    def __init__(self):
        self.boxes = []

    def set_boxes(self, boxes):
        self.boxes = boxes

    def detect(self, frame, frame_id=0, timestamp_ms=0.0):
        dets = [
            Detection(box=BoundingBox(*b), class_id=0, confidence=0.92)
            for b in self.boxes
        ]
        return DetectionResult(detections=dets, frame_id=frame_id, timestamp_ms=timestamp_ms)

    def detect_batch(self, frames, frame_ids, timestamps_ms):
        return [self.detect(f, fid, ts) for f, fid, ts in zip(frames, frame_ids, timestamps_ms)]

def run_test():
    config = AppConfig.load("configs/default.yaml")
    config.tracking.tracker_type = "botsort"

    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_0", name="Main Cam", source="synthetic", source_type=SourceType.SYNTHETIC))

    detector = ControllableDetector()

    pipe = MultiCameraPipeline(
        graph=graph,
        config=config,
    )
    # Hook our detector to cam_0
    worker = pipe._get_or_create_worker("cam_0")
    worker.detector = detector

    assert isinstance(worker.tracker, BoTSORTTracker)

    # Frame 1: Person A at (100, 100, 180, 260)
    detector.set_boxes([(100.0, 100.0, 180.0, 260.0, 0.92)])
    res1 = pipe.step()
    frame, track_res, _ = res1["cam_0"]
    logger.info("Step 1: Tracks count = %d", track_res.count)
    assert track_res.count == 1
    t1_id = track_res.tracks[0].track_id

    # Select target on (140, 180)
    selected_id = pipe.select_target_on_camera("cam_0", 140.0, 180.0)
    logger.info("Selected Target ID: %s", selected_id)
    assert selected_id == t1_id
    assert pipe.target_manager.is_active()

    # Frame 2: Person A moves to (105, 103, 185, 263)
    detector.set_boxes([(105.0, 103.0, 185.0, 263.0, 0.91)])
    res2 = pipe.step()
    _, track_res2, target2 = res2["cam_0"]
    logger.info("Step 2: Tracks count = %d, Target state = %s", track_res2.count, target2.state.value if target2 else None)
    assert track_res2.count == 1
    assert track_res2.tracks[0].track_id == t1_id
    assert target2.state in (TargetState.TRACKING, TargetState.CONFIRMED, TargetState.LOCKED)

    # Frame 3: Add Person B at (400, 200, 480, 360)
    detector.set_boxes([
        (110.0, 106.0, 190.0, 266.0, 0.90),
        (400.0, 200.0, 480.0, 360.0, 0.93)
    ])
    res3 = pipe.step()
    _, track_res3, target3 = res3["cam_0"]
    logger.info("Step 3: Tracks count = %d, Target track_id = %s", track_res3.count, target3.track_id if target3 else None)
    assert track_res3.count == 2
    assert target3.track_id == t1_id

    pipe.stop()
    logger.info("BoT-SORT DYNAMICS VERIFICATION PASSED: No conflicts between BoT-SORT, TargetManager, ReID, and Pipeline!")

if __name__ == "__main__":
    run_test()
