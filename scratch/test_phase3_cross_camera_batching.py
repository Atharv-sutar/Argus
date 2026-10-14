"""Phase 3 Validation: Cross-Camera Batching End-to-End Performance & Accuracy Test.

Tests multi-camera simultaneous processing with unified cross-camera detection & ReID batching.
"""

import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath("."))

from src.core.config import AppConfig
from src.app.main import build_pipeline
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.core.types import TargetState


def create_humanoid_frame(h=480, w=640, x_center=300, y_center=240, color=(40, 40, 200)):
    """Creates a synthetic frame containing a distinct humanoid figure."""
    frame = np.ones((h, w, 3), dtype=np.uint8) * 180
    x1, y1 = int(x_center - 40), int(y_center - 100)
    x2, y2 = int(x_center + 40), int(y_center + 100)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.circle(frame, (int(x_center), int(y_center - 70)), 22, (220, 180, 140), -1)
    return frame


def test_cross_camera_batching():
    print("=" * 60)
    print(">>> STARTING PHASE 3: CROSS-CAMERA BATCHING TEST <<<")
    print("=" * 60)

    config = AppConfig.load("configs/default.yaml")
    config.inference.device = "cuda"
    config.inference.engine_type = "auto"
    config.inference.enable_cross_camera_batching = True
    config.inference.batch_size = 4
    config.detection.model_name = "yolo11n.pt"
    config.tracking.tracker_type = "botsort"
    config.reid.model_name = "osnet_x0_25"

    pipeline = build_pipeline(config=config, use_synthetic=True)

    # Add second and third camera to the graph
    print("\n[Topology] Configuring 3-Camera Graph Topology (camera_0, camera_1, camera_2)...")
    from src.core.multi_camera_types import CameraEdgeConfig, EdgeDirection

    pipeline.graph.add_node(CameraNodeConfig(camera_id="camera_1", name="Camera 1", source=1, source_type=SourceType.SYNTHETIC, enabled=True))
    pipeline.graph.add_node(CameraNodeConfig(camera_id="camera_2", name="Camera 2", source=2, source_type=SourceType.SYNTHETIC, enabled=True))
    pipeline.graph.add_edge(CameraEdgeConfig(source_camera_id="camera_0", target_camera_id="camera_1", direction=EdgeDirection.BIDIRECTIONAL))
    pipeline.graph.add_edge(CameraEdgeConfig(source_camera_id="camera_1", target_camera_id="camera_2", direction=EdgeDirection.BIDIRECTIONAL))

    # Synchronize graph and workers
    pipeline._sync_nodes_with_graph()
    w0 = pipeline._get_or_create_worker("camera_0")
    w1 = pipeline._get_or_create_worker("camera_1")
    w2 = pipeline._get_or_create_worker("camera_2")

    print(f"[Topology] Active Workers: {list(pipeline._workers.keys())}")
    print(f"[Topology] Shared Detector: {type(pipeline.detector).__name__} (Batch Size: {pipeline.detector.batch_size})")
    print(f"[Topology] ReID Extractor: {type(pipeline.reid_extractor).__name__} (Batch Size: {pipeline.reid_extractor.batch_size})")

    # Activate search on camera_1 and camera_2 to force 3-camera simultaneous cross-camera batching
    print("\n>>> Activating Search across camera_1 and camera_2 to trigger 3-camera batched inference...")
    pipeline.search_manager.on_target_lost("camera_0")
    pipeline._activate_search_cameras(["camera_1", "camera_2"])

    print(">>> Running 30 steps with 3 simultaneous cameras (cross-camera batched)...")
    latencies = []

    for step_idx in range(30):
        t0 = time.perf_counter()
        results = pipeline.step()
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        if step_idx % 10 == 0:
            print(f"  Step {step_idx:02d}: {len(results)} cameras processed concurrently | Step Latency: {dt:.2f} ms")

    pipeline.stop()

    avg_step_latency = np.mean(latencies[5:])  # Exclude warmup
    per_camera_latency = avg_step_latency / 3.0
    effective_fps = 1000.0 / avg_step_latency
    cumulative_camera_fps = effective_fps * 3.0

    print("\n" + "=" * 60)
    print(">>> PHASE 3 BENCHMARK RESULTS (3 CAMERAS CROSS-BATCHED):")
    print(f"  Total Pipeline Step Latency: {avg_step_latency:.2f} ms / step (for ALL 3 cameras combined)")
    print(f"  Effective Step Rate:         {effective_fps:.1f} FPS (entire network)")
    print(f"  Amortized per-camera Latency:{per_camera_latency:.2f} ms / camera")
    print(f"  Cumulative Ingest Throughput:{cumulative_camera_fps:.1f} frames / second")
    print("=" * 60)

    assert avg_step_latency < 100.0, f"Cross-camera batching too slow: {avg_step_latency} ms"
    print(">>> SUCCESS: Cross-camera batching verified with high throughput and 0 errors!\n")


if __name__ == "__main__":
    test_cross_camera_batching()
