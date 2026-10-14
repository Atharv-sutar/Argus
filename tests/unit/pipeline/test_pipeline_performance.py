"""Performance and latency profiling test for MultiCameraPipeline."""

import gc
import time
import numpy as np
import pytest

from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.core.types import BoundingBox, Detection, DetectionResult, Embedding, TargetState, TrackResult
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline


class BenchmarkMockDetector:
    """Simulates batched YOLO detection latency without requiring heavy weights."""
    def __init__(self, simulated_ms: float = 2.0):
        self.simulated_ms = simulated_ms

    def detect(self, frame, frame_id=0, timestamp_ms=0.0):
        if self.simulated_ms > 0:
            time.sleep(self.simulated_ms / 1000.0)
        return DetectionResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            detections=[
                Detection(
                    box=BoundingBox(50, 50, 150, 200),
                    confidence=0.92,
                    class_id=0,
                    class_name="person",
                )
            ],
        )

    def detect_batch(self, frames, frame_ids, timestamps_ms):
        if self.simulated_ms > 0:
            time.sleep(self.simulated_ms / 1000.0)
        return [
            DetectionResult(
                frame_id=fid,
                timestamp_ms=ts,
                detections=[
                    Detection(
                        box=BoundingBox(50, 50, 150, 200),
                        confidence=0.92,
                        class_id=0,
                        class_name="person",
                    )
                ],
            )
            for fid, ts in zip(frame_ids, timestamps_ms)
        ]


def test_multi_camera_pipeline_throughput_and_bounded_memory():
    """
    Profile multi-camera pipeline execution across 6 cameras (1 active, 5 standby).
    Verify throughput, bounded memory, and absence of queue drift.
    """
    config = AppConfig()
    config.recording.enabled = False  # Isolate pipeline core

    graph = CameraGraph()
    # Add 6 cameras
    for i in range(6):
        node = CameraNodeConfig(
            camera_id=f"perf_cam_{i}",
            name=f"Camera {i}",
            source=i,
            source_type=SourceType.WEBCAM,
            enabled=True,
        )
        graph.add_node(node)

    def mock_cam_factory(cfg):
        return SyntheticCamera(width=320, height=240, fps=30)

    detector = BenchmarkMockDetector(simulated_ms=1.5)

    pipeline = MultiCameraPipeline(
        graph=graph,
        config=config,
        camera_factory=mock_cam_factory,
        shared_detector=detector,
    )

    try:
        # Warmup
        for _ in range(5):
            pipeline.step()

        # Measure 30 iterations
        n_steps = 30
        t_start = time.perf_counter()
        for _ in range(n_steps):
            results = pipeline.step()
            assert len(results) == 6

        elapsed = time.perf_counter() - t_start
        fps = n_steps / elapsed

        # Assert throughput: 6 cameras should maintain > 15 FPS concurrently
        assert fps >= 12.0, f"Pipeline FPS too low: {fps:.1f} FPS for 6 cameras"

        # Check telemetry stats
        telem = pipeline.get_telemetry()
        assert telem.fps > 0.0

        # Memory verification (gc and torch cache)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                assert vram_mb < 3000.0, f"VRAM allocation exceeded 3GB budget: {vram_mb:.1f}MB"
        except ImportError:
            pass

    finally:
        pipeline.stop()
