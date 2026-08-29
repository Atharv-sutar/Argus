import os
import sys
sys.path.insert(0, ".")
import time
import urllib.request

from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.multi_camera.ui_server import run_ui_server

config = AppConfig()
graph = CameraGraph()
graph.add_node(CameraNodeConfig(
    camera_id="cam_1",
    name="Test Camera 1",
    source="synthetic",
    source_type=SourceType.SYNTHETIC,
    enabled=True
))

pipeline = MultiCameraPipeline(graph=graph, config=config)
server = run_ui_server(port=8771, pipeline=pipeline, block=False)
time.sleep(0.5)

# Step the pipeline a few times to get frames
for _ in range(5):
    pipeline.step()

try:
    # Test frame.jpg
    req = urllib.request.urlopen("http://127.0.0.1:8771/api/camera/cam_1/frame.jpg")
    content = req.read()
    print(f"frame.jpg returned {len(content)} bytes, Content-Type: {req.headers.get('Content-Type')}")

    # Test stream connection
    req_stream = urllib.request.urlopen("http://127.0.0.1:8771/api/camera/cam_1/stream", timeout=2)
    first_chunk = req_stream.read(1024)
    print(f"Stream connection opened, read {len(first_chunk)} bytes. First 100 bytes:\n{first_chunk[:100]}")
finally:
    pipeline.stop()
    server.shutdown()
    server.server_close()
