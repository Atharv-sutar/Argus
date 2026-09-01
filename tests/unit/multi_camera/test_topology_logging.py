"""Unit tests for topology and live matrix logging and server API."""

import json
import logging
import time
import urllib.request
from pathlib import Path

from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig
from src.core.logging_config import setup_logging
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.multi_camera.camera_graph import CameraGraph
from src.multi_camera.ui_server import probe_local_webcams, run_ui_server
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline


def test_logging_setup_creates_log_file(tmp_path):
    """Verify that setup_logging configures file handler and writes debug entries."""
    log_file = tmp_path / "test_topology.log"
    setup_logging(log_file=str(log_file), console_level=logging.INFO, file_level=logging.DEBUG)

    test_logger = logging.getLogger("argus.test")
    test_logger.debug("DEBUG_LOG_ENTRY_FOR_TOPOLOGY")
    test_logger.info("INFO_LOG_ENTRY_FOR_MATRIX")

    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert "DEBUG_LOG_ENTRY_FOR_TOPOLOGY" in content
    assert "INFO_LOG_ENTRY_FOR_MATRIX" in content


def test_probe_local_webcams_bounded_and_logged(tmp_path):
    """Verify probe_local_webcams completes within timeout without hanging."""
    log_file = tmp_path / "test_probe.log"
    setup_logging(log_file=str(log_file), console_level=logging.INFO, file_level=logging.DEBUG)

    t0 = time.time()
    cameras = probe_local_webcams(max_indices=2, pipeline=None, timeout_per_index=1.0)
    elapsed = time.time() - t0

    # Must complete within bounded time (e.g. < 5s)
    assert elapsed < 5.0
    assert isinstance(cameras, list)

    content = log_file.read_text(encoding="utf-8")
    assert "[TOPOLOGY/PROBE]" in content


def test_topology_save_and_live_matrix_logging(tmp_path):
    """Verify topology save and live matrix operations write detailed entries to log file."""
    log_file = tmp_path / "topology_matrix_test.log"
    graph_file = tmp_path / "camera_graph.json"

    setup_logging(log_file=str(log_file), console_level=logging.INFO, file_level=logging.DEBUG)

    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_0", name="Entrance", source="synthetic", source_type=SourceType.SYNTHETIC))

    pipe = MultiCameraPipeline(
        graph=graph,
        config=AppConfig(),
        camera_factory=lambda cfg: SyntheticCamera(width=320, height=240, fps=30),
    )
    pipe.step()

    port = 8798
    server = run_ui_server(port=port, graph_file=str(graph_file), pipeline=pipe, block=False)
    time.sleep(0.2)

    try:
        # 1. GET /api/cameras/live
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/cameras/live", timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert "cameras" in data
            assert len(data["cameras"]) == 1

        # 2. POST /api/graph (Save topology)
        new_graph_data = {
            "version": 1,
            "cameras": [
                {
                    "camera_id": "cam_0",
                    "name": "Entrance",
                    "source": "synthetic",
                    "source_type": "synthetic",
                    "enabled": True,
                    "position_x": 100.0,
                    "position_y": 100.0,
                },
                {
                    "camera_id": "cam_1",
                    "name": "Hallway",
                    "source": "synthetic",
                    "source_type": "synthetic",
                    "enabled": True,
                    "position_x": 300.0,
                    "position_y": 100.0,
                },
            ],
            "edges": [
                {
                    "source_camera_id": "cam_0",
                    "target_camera_id": "cam_1",
                    "direction": "bidirectional",
                    "edge_type": "adjacent",
                    "enabled": True,
                }
            ],
            "background_map": None,
        }

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/graph",
            data=json.dumps(new_graph_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            save_resp = json.loads(resp.read().decode("utf-8"))
            assert save_resp["success"] is True

        # Verify saved file exists on disk
        assert graph_file.is_file()

        # 3. GET /api/graph (Load topology)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/graph", timeout=3.0) as resp:
            graph_data = json.loads(resp.read().decode("utf-8"))
            assert len(graph_data["cameras"]) == 2
            assert len(graph_data["edges"]) == 1

        # 4. Check log file contents
        content = log_file.read_text(encoding="utf-8")
        assert "[LIVE MATRIX]" in content
        assert "[TOPOLOGY]" in content
        assert "Entrance" in content or "cam_0" in content
        assert "Hallway" in content or "cam_1" in content
    finally:
        try:
            pipe.stop()
            server.shutdown()
        except Exception:
            pass
