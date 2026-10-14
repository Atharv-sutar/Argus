"""Integration tests for auto-loading video directories into MultiCameraPipeline and UI server."""

import json
from io import BytesIO
from pathlib import Path
import pytest

from src.camera.directory_loader import auto_load_video_directory
from src.core.config import AppConfig
from src.multi_camera.camera_graph import CameraGraph
from src.multi_camera.ui_server import MappingAPIHandler
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline


@pytest.fixture
def epfl_video_dir(tmp_path):
    """Provides either real EPFL directory or a simulated video directory for test isolation."""
    real_path = Path("test_footage/EPFL campus")
    if real_path.is_dir() and len(list(real_path.glob("*.avi"))) >= 3:
        return real_path

    # Fallback to simulated directory with dummy video files
    vdir = tmp_path / "simulated_videos"
    vdir.mkdir()
    (vdir / "campus4-c0.avi").write_text("mock")
    (vdir / "campus4-c1.avi").write_text("mock")
    (vdir / "campus4-c2.avi").write_text("mock")
    return vdir


def test_pipeline_integration_with_directory_graph(epfl_video_dir):
    graph = auto_load_video_directory(epfl_video_dir, default_transit_time_s=5.0)
    assert graph.node_count() >= 3

    config = AppConfig()
    pipeline = MultiCameraPipeline(graph=graph, config=config)

    try:
        cards = pipeline.get_all_camera_cards()
        assert len(cards) == graph.node_count()
        card_cids = [c["camera_id"] for c in cards]
        assert "campus4_c0" in card_cids
        assert "campus4_c1" in card_cids
        assert "campus4_c2" in card_cids

        # Set active camera
        ok = pipeline.set_active_camera("campus4_c0")
        assert ok is True
        assert pipeline.active_camera_id == "campus4_c0"

        # Check card active flags
        updated_cards = pipeline.get_all_camera_cards()
        active_card = next(c for c in updated_cards if c["camera_id"] == "campus4_c0")
        assert active_card["is_active"] is True

        standby_card = next(c for c in updated_cards if c["camera_id"] == "campus4_c1")
        assert standby_card["is_active"] is False
    finally:
        pipeline.stop()


def test_ui_server_load_directory_endpoint(epfl_video_dir, tmp_path):
    temp_graph_file = tmp_path / "test_graph.json"

    # Setup handler state
    MappingAPIHandler.graph_file = temp_graph_file
    MappingAPIHandler.runtime_pipeline = None

    class MockSocket:
        def __init__(self, data: bytes):
            self.data = data
            self.output = BytesIO()

        def makefile(self, mode: str, bufsize: int = -1):
            if "r" in mode:
                return BytesIO(self.data)
            return self.output

        def sendall(self, data: bytes) -> None:
            self.output.write(data)

        def send(self, data: bytes) -> int:
            self.output.write(data)
            return len(data)

    # Create dummy POST request for /api/cameras/load_directory
    payload = json.dumps({"directory": str(epfl_video_dir), "transit_time_s": 6.0}).encode("utf-8")
    req = (
        b"POST /api/cameras/load_directory HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n"
        b"\r\n" + payload
    )

    sock = MockSocket(req)
    handler = MappingAPIHandler(sock, ("127.0.0.1", 12345), None)

    # Check response in sock.output
    response_bytes = sock.output.getvalue()
    assert b"200 OK" in response_bytes
    assert b"Successfully loaded" in response_bytes
    assert temp_graph_file.is_file()

    # Verify persisted graph on disk
    saved_graph = CameraGraph.load(temp_graph_file)
    assert saved_graph.node_count() >= 3
