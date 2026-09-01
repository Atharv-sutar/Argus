"""Unit tests for safe quit endpoint and resource cleanup (Issue 2)."""

import json
import threading
import time
import urllib.request

from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.multi_camera.camera_graph import CameraGraph
from src.multi_camera.ui_server import run_ui_server
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline


def test_safe_quit_endpoint_stops_pipeline():
    """Verify that calling POST /api/system/quit cleanly stops the pipeline and closes server."""
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_A", name="Lobby", source="synthetic", source_type=SourceType.SYNTHETIC))

    config = AppConfig()
    pipe = MultiCameraPipeline(
        graph=graph,
        config=config,
        camera_factory=lambda cfg: SyntheticCamera(width=320, height=240, fps=30),
    )

    pipe.step()
    assert len(pipe._workers) > 0

    # Start UI server on a random test port
    port = 8799
    server = run_ui_server(port=port, graph_file="configs/camera_graph.json", pipeline=pipe, block=False)

    time.sleep(0.2)

    try:
        # Call POST /api/system/quit
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/system/quit",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True

        time.sleep(0.6)

        # Verify pipeline workers are cleaned up
        assert len(pipe._workers) == 0
    finally:
        from src.multi_camera.ui_server import _SHUTDOWN_EVENT
        _SHUTDOWN_EVENT.clear()
        try:
            server.shutdown()
        except Exception:
            pass
