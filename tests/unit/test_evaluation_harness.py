"""Unit tests for the ReID Evaluation Harness and ground-truth validation."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.benchmark.evaluator import ReIDEvaluationHarness, compute_iou
from src.benchmark.types import (
    EvaluationMetrics,
    GroundTruthDataset,
    GroundTruthObservation,
)
from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig
from src.core.interfaces import BaseDetector, BaseReID
from src.core.multi_camera_types import (
    CameraEdgeConfig,
    CameraNodeConfig,
    EdgeType,
    SourceType,
)
from src.core.types import BoundingBox, Detection, DetectionResult, Embedding, TargetState
from src.multi_camera.camera_graph import CameraGraph
from src.tracking.byte_tracker import ByteTracker


def test_iou_computation():
    box1 = BoundingBox(10.0, 10.0, 50.0, 50.0)
    box2 = BoundingBox(10.0, 10.0, 50.0, 50.0)
    assert compute_iou(box1, box2) == pytest.approx(1.0)

    box3 = BoundingBox(60.0, 60.0, 100.0, 100.0)
    assert compute_iou(box1, box3) == 0.0

    # Half overlap: box1 area=1600, box4 (30, 10, 70, 50) area=1600, inter=(20*40)=800
    box4 = BoundingBox(30.0, 10.0, 70.0, 50.0)
    # union = 1600 + 1600 - 800 = 2400. IoU = 800 / 2400 = 1/3
    assert compute_iou(box1, box4) == pytest.approx(1.0 / 3.0)


def test_ground_truth_dataset_indexing_and_io():
    dataset = GroundTruthDataset(name="test_gt", target_person_id="target_person")
    obs1 = GroundTruthObservation("cam_A", 1, "target_person", BoundingBox(10, 10, 50, 50))
    obs2 = GroundTruthObservation("cam_A", 1, "bystander_1", BoundingBox(100, 10, 150, 50))
    obs3 = GroundTruthObservation("cam_B", 2, "target_person", BoundingBox(20, 20, 60, 60))

    dataset.add_observation(obs1)
    dataset.add_observation(obs2)
    dataset.add_observation(obs3)

    assert len(dataset.get_for_frame("cam_A", 1)) == 2
    assert dataset.get_target_for_frame("cam_A", 1) == obs1
    assert dataset.get_target_for_frame("cam_A", 2) is None
    assert dataset.get_target_for_frame("cam_B", 2) == obs3

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        dataset.save_to_file(tmp_path)
        loaded = GroundTruthDataset.load_from_file(tmp_path)
        assert loaded.name == "test_gt"
        assert loaded.target_person_id == "target_person"
        assert len(loaded.observations) == 3
        assert loaded.get_target_for_frame("cam_A", 1).box.x1 == 10.0
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_evaluation_metrics_math():
    metrics = EvaluationMetrics(
        total_frames=100,
        target_present_frames=80,
        true_positives=70,
        false_positives=10,
        false_negatives=10,
        true_negatives=10,
        id_switches=1,
        recovery_events=2,
        target_similarities=[0.85, 0.88, 0.82],
        bystander_similarities=[0.60, 0.65, 0.58],
        state_frames={"CONFIRMED": 65, "TRACKING": 15, "LOST": 20},
    )

    # Precision = 70 / (70 + 10) = 70 / 80 = 0.875
    assert metrics.precision == pytest.approx(0.875)
    # Recall = 70 / (70 + 10) = 70 / 80 = 0.875
    assert metrics.recall == pytest.approx(0.875)
    # F1 = 0.875
    assert metrics.f1_score == pytest.approx(0.875)
    # FPR = 10 / (10 + 10) = 0.5
    assert metrics.false_positive_rate == pytest.approx(0.5)

    assert metrics.mean_target_sim == pytest.approx(0.85)
    assert metrics.mean_bystander_sim == pytest.approx(0.61)
    assert metrics.separation_gap == pytest.approx(0.24)

    # Summary table should format cleanly
    table = metrics.summary_table()
    assert "87.50%" in table
    assert "CONFIRMED" in table


class HarnessMockDetector(BaseDetector):
    def __init__(self):
        self.detections = []

    def set_boxes(self, boxes: list[tuple[float, float, float, float]]):
        self.detections = [
            Detection(box=BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], confidence=0.95), class_id=0, confidence=0.95)
            for b in boxes
        ]

    def detect(self, frame, frame_id=0, timestamp_ms=0.0):
        return DetectionResult(detections=list(self.detections), frame_id=frame_id, timestamp_ms=timestamp_ms)


class HarnessMockReID(BaseReID):
    def __init__(self, target_vec: np.ndarray, bystander_vec: np.ndarray):
        self.target_vec = target_vec
        self.bystander_vec = bystander_vec
        self.is_target = True

    def extract(self, crop: np.ndarray) -> Embedding:
        vec = self.target_vec if self.is_target else self.bystander_vec
        return Embedding(vector=vec)

    def extract_batch(self, crops: list[np.ndarray]) -> list[Embedding]:
        return [self.extract(c) for c in crops]


def test_harness_execution_with_mock_pipeline():
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_A", name="Lobby", source="synthetic", source_type=SourceType.SYNTHETIC))

    config = AppConfig()
    config.reid.match_threshold = 0.75
    config.reid.extract_interval_frames = 1

    t_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    mock_reid = HarnessMockReID(t_vec, b_vec)
    mock_det = HarnessMockDetector()

    # Ground truth: target person at (50, 50, 100, 150) for frames 0 to 10
    dataset = GroundTruthDataset(name="mock_test", target_person_id="target_person")
    for f in range(10):
        dataset.add_observation(GroundTruthObservation(
            camera_id="cam_A",
            frame_id=f,
            person_id="target_person",
            box=BoundingBox(50.0, 50.0, 100.0, 150.0),
        ))

    mock_det.set_boxes([(50.0, 50.0, 100.0, 150.0)])

    harness = ReIDEvaluationHarness(
        graph=graph,
        config=config,
        camera_sources={"cam_A": "synthetic"},
        dataset=dataset,
        target_person_id="target_person",
        reid_extractor=mock_reid,
        shared_detector=mock_det,
        tracker_factory=lambda: ByteTracker(track_thresh=0.4, match_thresh=0.5),
    )

    # Patch harness's camera factory to return SyntheticCamera
    harness._create_camera_factory = lambda: (lambda node_cfg: SyntheticCamera(width=320, height=240, fps=30))

    metrics = harness.run(
        seed_camera_id="cam_A",
        seed_frame_id=0,
        seed_box=BoundingBox(50.0, 50.0, 100.0, 150.0),
        max_frames=8,
    )

    assert metrics.total_frames == 8
    assert metrics.true_positives > 0
    assert metrics.false_positives == 0
    assert metrics.precision == 1.0
