import os
import sys
sys.path.insert(0, ".")
import time
import urllib.request
import json

from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.multi_camera.ui_server import run_ui_server

print("Testing UI Server API integration...")

config = AppConfig()
config.camera.width = 640
config.camera.height = 480

graph = CameraGraph()
graph.add_node(CameraNodeConfig(
    camera_id="cam_0",
    name="Front Gate",
    source="synthetic",
    source_type=SourceType.SYNTHETIC,
    enabled=True
))

pipeline = MultiCameraPipeline(graph=graph, config=config)

# Start UI server non-blocking on port 8769
server = run_ui_server(port=8769, pipeline=pipeline, block=False)
time.sleep(0.5)

try:
    # 1. Test /api/cameras/live
    req = urllib.request.urlopen("http://127.0.0.1:8769/api/cameras/live")
    live_data = json.loads(req.read().decode("utf-8"))
    print("Live cameras count:", len(live_data.get("cameras", [])))
    
    # 2. Test /api/status
    req = urllib.request.urlopen("http://127.0.0.1:8769/api/status")
    status_data = json.loads(req.read().decode("utf-8"))
    print("Status target_state:", status_data.get("target_state"))
    print("Status gallery size:", status_data.get("gallery_size"))
    
    # 3. Test /api/target/gallery
    req = urllib.request.urlopen("http://127.0.0.1:8769/api/target/gallery")
    gallery_data = json.loads(req.read().decode("utf-8"))
    print("Gallery API size:", gallery_data.get("size"))

    print("All UI Server API endpoints tested successfully!")
finally:
    pipeline.stop()
    server.shutdown()
    server.server_close()
