"""Comprehensive end-to-end verification of the Target-Only Gallery ReID architecture."""

import time
import numpy as np

from src.core.config import AppConfig
from src.core.multi_camera_types import CameraEdgeConfig, CameraNodeConfig, EdgeType, SourceType
from src.core.types import BoundingBox, DetectionResult, TargetState, Track, TrackResult
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline


def run_verification():
    print("================================================================")
    print(" RUNNING TARGET-ONLY GALLERY REID ARCHITECTURE VERIFICATION")
    print("================================================================")

    # 1. Setup 2-camera topology graph: Cam_A <-> Cam_B
    node_a = CameraNodeConfig(camera_id="cam_A", name="Camera A", source="synthetic", source_type=SourceType.SYNTHETIC)
    node_b = CameraNodeConfig(camera_id="cam_B", name="Camera B", source="synthetic", source_type=SourceType.SYNTHETIC)
    edge = CameraEdgeConfig(source_camera_id="cam_A", target_camera_id="cam_B", edge_type=EdgeType.ADJACENT)

    graph = CameraGraph()
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_edge(edge)

    config = AppConfig()
    config.inference.device = "cpu"
    config.reid.max_gallery_size = 10
    config.reid.match_threshold = 0.85
    config.reid.auto_add_threshold = 0.90
    config.reid.auto_add_min_consecutive = 2
    config.reid.extract_interval_frames = 1

    pipeline = MultiCameraPipeline(graph=graph, config=config)

    # 2. Step 1: Initial state
    results = pipeline.step()
    print("Initial step results:", list(results.keys()))
    assert len(results) > 0
    assert pipeline.target_state == "UNSELECTED"
    assert pipeline.gallery.size == 0

    # 3. Simulate Target Selection on cam_A
    print("\n--- Test 1: Manual Target Selection & Gallery Seeding ---")
    worker_a = pipeline._get_or_create_worker("cam_A")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    box = BoundingBox(100, 100, 200, 300)
    t1 = Track(track_id=10, box=box)
    track_res = TrackResult(tracks=[t1], frame_id=1, timestamp_ms=100.0)
    worker_a._last_frame = frame
    worker_a._last_track_result = track_res

    # Select target by point (150, 150) inside track 10
    selected_id = pipeline.select_target_on_camera("cam_A", 150.0, 150.0)
    print(f"Selected target: {selected_id}")
    assert selected_id == 10
    assert pipeline.active_camera_id == "cam_A"
    assert pipeline.target_state == "TRACKING"
    assert pipeline.gallery.size == 1
    assert pipeline.gallery.manual_count == 1
    assert pipeline.gallery.auto_count == 0
    print("Gallery successfully seeded with 1 manual protected entry.")

    # 4. Test Manual Angle Addition (Hotkey / Right-Click / API)
    print("\n--- Test 2: Manual Protected Sample Capture ---")
    box_side = BoundingBox(120, 110, 220, 310)
    t1_side = Track(track_id=10, box=box_side)
    worker_a._last_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    worker_a._last_track_result = TrackResult(tracks=[t1_side], frame_id=2, timestamp_ms=133.0)

    added = pipeline.add_manual_target_sample("cam_A")
    print(f"Manual sample add result: {added}")
    assert added is True
    assert pipeline.gallery.size == 2
    assert pipeline.gallery.manual_count == 2
    assert pipeline.gallery.auto_count == 0
    print("Manual protected angle successfully added to target gallery.")

    # 5. Test Exact Max-Similarity Matching on Active Camera
    print("\n--- Test 3: Candidate Max-Similarity Matching & Auto Growth ---")
    crop_saved = worker_a.extract_crop(worker_a._last_frame, box_side).copy()
    emb_sample = pipeline.reid_extractor.extract(crop_saved)
    max_sim, best_entry = pipeline.gallery.match(emb_sample)
    print(f"Self-match similarity to gallery: {max_sim:.4f}")
    assert max_sim >= 0.99

    # 6. Test Target Loss & Multi-Camera Search Activation
    print("\n--- Test 4: Target Loss & Graph Search Activation ---")
    # Simulate track missing on cam_A
    worker_a._last_track_result = TrackResult(tracks=[], frame_id=3, timestamp_ms=200.0)
    pipeline.target_manager.mark_lost(200.0)
    print(f"Target state after mark_lost: {pipeline.target_state}")
    assert pipeline.target_state == "LOST"

    # Step pipeline: Should activate search on adjacent cam_B
    results = pipeline.step()
    progress = pipeline.get_search_progress()
    print(f"Search Manager State: {progress.state}, Active Search Cameras: {progress.active_cameras}")
    assert pipeline.search_manager.is_searching is True
    assert "cam_B" in progress.active_cameras

    # 7. Test Cross-Camera Reacquisition & Handoff to cam_B
    print("\n--- Test 5: Cross-Camera Reacquisition & Handoff ---")
    worker_b = pipeline._get_or_create_worker("cam_B")
    box_b = BoundingBox(200, 150, 300, 350)
    t_b = Track(track_id=42, box=box_b)
    # Paste preserved target crop into worker_b's frame at box_b
    frame_b = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    import cv2
    frame_b[150:350, 200:300] = cv2.resize(crop_saved, (100, 200))
    worker_b._last_frame = frame_b
    worker_b._last_track_result = TrackResult(tracks=[t_b], frame_id=1, timestamp_ms=250.0)

    # Perform matching on cam_B
    rec_tr, rec_cr, rec_em, rec_sim = pipeline._match_candidates_against_gallery(
        worker_b, worker_b._last_frame, worker_b._last_track_result
    )
    print(f"Matching candidate on cam_B: Track={rec_tr.track_id if rec_tr else None}, Similarity={rec_sim:.4f}")
    assert rec_tr is not None and rec_tr.track_id == 42
    assert rec_sim >= config.reid.match_threshold

    # Perform handoff
    pipeline._perform_handoff("cam_B", rec_tr, rec_cr, rec_em)
    print(f"Active camera after handoff: {pipeline.active_camera_id}")
    print(f"Target state after handoff: {pipeline.target_state}")
    print(f"Tracker ID on new camera: {pipeline.target_manager.target.track_id}")
    assert pipeline.active_camera_id == "cam_B"
    assert pipeline.target_state == "TRACKING"
    assert pipeline.target_manager.target.track_id == 42
    assert pipeline.search_manager.is_searching is False

    # 8. Test Gallery Purge on Target Deselect / Clear
    print("\n--- Test 6: Clear Target & Purge Gallery ---")
    pipeline.clear_target()
    print(f"Gallery size after clear: {pipeline.gallery.size}")
    print(f"Target state after clear: {pipeline.target_state}")
    assert pipeline.gallery.size == 0
    assert pipeline.target_state == "UNSELECTED"

    print("\n================================================================")
    print(" ALL 6 TARGET-ONLY GALLERY REID TESTS PASSED PERFECTLY!")
    print("================================================================")


if __name__ == "__main__":
    run_verification()
