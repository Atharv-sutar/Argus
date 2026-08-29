import os
import sys
sys.path.insert(0, ".")
import time
import numpy as np
import cv2

from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline

print("Testing Unified MultiCameraPipeline End-to-End...")

config = AppConfig()
config.camera.width = 640
config.camera.height = 480
config.reid.model_name = "osnet_x1_0"
config.reid.match_threshold = 0.72
config.reid.auto_add_threshold = 0.82

graph = CameraGraph()
graph.add_node(CameraNodeConfig(
    camera_id="cam_0",
    name="Test Camera 0",
    source="synthetic",
    source_type=SourceType.SYNTHETIC,
    enabled=True
))
graph.add_node(CameraNodeConfig(
    camera_id="cam_1",
    name="Test Camera 1",
    source="synthetic",
    source_type=SourceType.SYNTHETIC,
    enabled=True
))
from src.core.multi_camera_types import CameraEdgeConfig, EdgeDirection

graph.add_edge(CameraEdgeConfig(
    source_camera_id="cam_0",
    target_camera_id="cam_1",
    direction=EdgeDirection.BIDIRECTIONAL,
    expected_min_transition_s=1.0,
    expected_max_transition_s=10.0,
))

pipeline = MultiCameraPipeline(
    graph=graph,
    config=config,
)

print(f"Pipeline created. Active camera: {pipeline.active_camera_id}")
print(f"ReID extractor model: {pipeline.reid_extractor.model_name}, dim: {pipeline.reid_extractor.feature_dim}")

# Run 10 steps
for i in range(10):
    results = pipeline.step()

print("Step 10 complete. Camera statuses:")
for cid in graph.all_camera_ids():
    st = pipeline.get_camera_status(cid)
    print(f"  {cid}: {st.value if st else 'None'}")

# Test target selection on cam_0
print("\nSelecting target on cam_0...")
selected = pipeline.select_target_on_camera("cam_0", 320.0, 240.0)
print(f"Selected target ID: {selected}, Target state: {pipeline.target_state}, Gallery size: {pipeline.gallery.size}")

# Run another 10 steps with active target
for i in range(10):
    results = pipeline.step()

print(f"After 10 active steps: Target state: {pipeline.target_state}, Gallery size: {pipeline.gallery.size} (man={pipeline.gallery.manual_count}, auto={pipeline.gallery.auto_count})")
print(f"Last candidate scores: {pipeline.last_candidate_scores}")

# Test adding manual sample
pipeline.add_manual_target_sample("cam_0")
print(f"After manual sample: Gallery size: {pipeline.gallery.size} (man={pipeline.gallery.manual_count}, auto={pipeline.gallery.auto_count})")

# Test clean stop
pipeline.stop()
print("Pipeline stopped cleanly. End-to-end test SUCCESS!")
