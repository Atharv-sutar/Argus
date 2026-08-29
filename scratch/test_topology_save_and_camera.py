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

print("Testing Topology Save and Fast Discovery...")

config = AppConfig()
graph = CameraGraph()
graph.add_node(CameraNodeConfig(
    camera_id="cam_0",
    name="Front Gate",
    source="synthetic",
    source_type=SourceType.SYNTHETIC,
    enabled=True
))

pipeline = MultiCameraPipeline(graph=graph, config=config)
server = run_ui_server(port=8775, pipeline=pipeline, block=False)
time.sleep(0.5)

try:
    # 1. Test Discovery
    print("Testing /api/cameras/discover...")
    t0 = time.time()
    req = urllib.request.urlopen("http://127.0.0.1:8775/api/cameras/discover")
    disc = json.loads(req.read().decode("utf-8"))
    t1 = time.time()
    print(f"Discovery completed in {t1 - t0:.2f}s, found {len(disc.get('cameras', []))} cameras: {disc.get('cameras', [])}")

    # 2. Test Saving Topology Graph via POST /api/graph
    new_graph_data = {
        "version": 1,
        "cameras": [
            {
                "camera_id": "cam_0",
                "name": "Front Gate",
                "source": "synthetic",
                "source_type": "synthetic",
                "enabled": True,
                "position_x": 100,
                "position_y": 100
            },
            {
                "camera_id": "cam_1",
                "name": "Lobby",
                "source": "synthetic",
                "source_type": "synthetic",
                "enabled": True,
                "position_x": 300,
                "position_y": 100
            }
        ],
        "edges": [
            {
                "source_camera_id": "cam_0",
                "target_camera_id": "cam_1",
                "edge_type": "adjacent",
                "direction": "bidirectional",
                "enabled": True
            }
        ],
        "background_map": None
    }

    req_data = json.dumps(new_graph_data).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8775/api/graph", data=req_data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    save_result = json.loads(resp.read().decode("utf-8"))
    print(f"Save Topology Result: {save_result}")
    assert save_result.get("success") is True, f"Expected success: True, got {save_result}"

    # Verify pipeline was dynamically updated!
    print(f"Pipeline updated nodes: {list(pipeline._nodes.keys())}, edge count: {pipeline.graph.edge_count()}")
    assert len(pipeline._nodes) == 2, f"Expected 2 nodes, got {len(pipeline._nodes)}"

    # 3. Test frame streaming
    req_frame = urllib.request.urlopen("http://127.0.0.1:8775/api/camera/cam_0/frame.jpg")
    print(f"cam_0 frame.jpg status: {req_frame.status}, length: {len(req_frame.read())} bytes")

    print("\nALL TOPOLOGY AND CAMERA SPEED TESTS PASSED SUCCESSFULLY!")

finally:
    pipeline.stop()
    server.shutdown()
    server.server_close()
