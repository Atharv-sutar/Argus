import os
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2

from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline

print("=== Testing Active Target Tracking and Gallery Seeding ===")

config = AppConfig()
config.camera.width = 640
config.camera.height = 480
config.reid.model_name = "osnet_x1_0"
config.reid.match_threshold = 0.72
config.reid.auto_add_threshold = 0.82
config.reid.extract_interval_frames = 2

graph = CameraGraph()
graph.add_node(CameraNodeConfig(
    camera_id="cam_0",
    name="Test Camera 0",
    source="synthetic",
    source_type=SourceType.SYNTHETIC,
    enabled=True
))

pipeline = MultiCameraPipeline(
    graph=graph,
    config=config,
)

# Step once to generate tracks
results = pipeline.step()
worker = pipeline._workers["cam_0"]
tracks = worker._last_track_result.tracks if worker._last_track_result else []
print(f"Generated {len(tracks)} tracks on cam_0:")
for t in tracks:
    print(f"  Track #{t.track_id}: box={t.box.as_xyxy()}")

if tracks:
    target_tid = tracks[0].track_id
    print(f"\nSelecting Track #{target_tid} on cam_0...")
    pipeline.select_target_by_id("cam_0", target_tid)
    print(f"Target selected. State: {pipeline.target_state}, Gallery size: {pipeline.gallery.size} (man={pipeline.gallery.manual_count})")
    
    # Run 10 steps
    for i in range(10):
        res = pipeline.step()
        
    print(f"\nAfter 10 steps:")
    print(f"  Target State: {pipeline.target_state}")
    print(f"  Gallery Size: {pipeline.gallery.size} (manual={pipeline.gallery.manual_count}, auto={pipeline.gallery.auto_count})")
    print(f"  Candidate Scores: {pipeline.last_candidate_scores}")

pipeline.stop()
print("\nActive tracking test completed successfully!")
