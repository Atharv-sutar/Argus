"""Unit tests for topology and live matrix logging and server API."""

import json
import logging
import time
import urllib.request

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


def test_probe_single_device_robust_json_parsing(monkeypatch):
    """Verify _probe_single_device handles extraneous stdout lines gracefully."""
    from unittest.mock import MagicMock
    import subprocess
    from src.multi_camera.ui_server import _probe_single_device

    # Mock subprocess.run returning noisy output before JSON
    fake_stdout = "[WARN:0@0.002] global cap_ffmpeg: OpenCV warning\n{\"source\": 0, \"width\": 1280, \"height\": 720, \"fps\": 30}\n"
    mock_proc = MagicMock(returncode=0, stdout=fake_stdout)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_proc)

    res = _probe_single_device(0, timeout=1.0)
    assert res is not None
    assert res["source"] == 0
    assert res["width"] == 1280
    assert res["height"] == 720
    assert res["fps"] == 30
    assert res["status"] == "available"


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


def test_case_open_and_close_restores_default_db_paths(tmp_path, monkeypatch):
    """Verify POST /api/cases/<id>/open switches DB paths and /close restores defaults."""
    from types import SimpleNamespace
    from src.multi_camera.ui_server import MappingAPIHandler

    monkeypatch.chdir(tmp_path)

    class _StubIdentityManager:
        def __init__(self, db_path):
            self.db_path = db_path
            self.calls = []

        def set_db_path(self, db_path):
            self.db_path = db_path
            self.calls.append(db_path)

    default_id_db = str(tmp_path / "default_identities.db")
    stub_im = _StubIdentityManager(default_id_db)
    stub_pipe = SimpleNamespace(
        identity_manager=stub_im,
        config=SimpleNamespace(storage=SimpleNamespace(db_path=default_id_db)),
    )

    saved_attrs = (
        MappingAPIHandler.graph_file,
        MappingAPIHandler.runtime_pipeline,
        MappingAPIHandler.annotation_store,
        MappingAPIHandler.default_annotation_db_path,
        MappingAPIHandler.default_identity_db_path,
        MappingAPIHandler.case_manager,
        getattr(MappingAPIHandler, "audit_logger", None),
    )

    server = run_ui_server(
        port=0,
        graph_file=str(tmp_path / "camera_graph.json"),
        pipeline=stub_pipe,
        block=False,
    )
    port = server.server_port
    time.sleep(0.05)

    try:
        default_anno_db = MappingAPIHandler.annotation_store.db_path
        MappingAPIHandler.case_manager.create_case(case_id="case_001", mode="live")

        # Open case -> switches annotation & identity DBs
        open_req = urllib.request.Request(f"http://127.0.0.1:{port}/api/cases/case_001/open", data=b"{}", method="POST")
        with urllib.request.urlopen(open_req, timeout=3.0) as resp:
            assert json.loads(resp.read().decode("utf-8"))["success"] is True

        assert "case_001" in MappingAPIHandler.annotation_store.db_path
        assert "case_001" in stub_im.db_path

        # Close case -> restores original DB paths
        close_req = urllib.request.Request(f"http://127.0.0.1:{port}/api/cases/case_001/close", data=b"{}", method="POST")
        with urllib.request.urlopen(close_req, timeout=3.0) as resp:
            assert json.loads(resp.read().decode("utf-8"))["success"] is True

        assert MappingAPIHandler.case_manager.active_case is None
        assert MappingAPIHandler.annotation_store.db_path == default_anno_db
        assert stub_im.db_path == default_id_db
    finally:
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass
        (
            MappingAPIHandler.graph_file,
            MappingAPIHandler.runtime_pipeline,
            MappingAPIHandler.annotation_store,
            MappingAPIHandler.default_annotation_db_path,
            MappingAPIHandler.default_identity_db_path,
            MappingAPIHandler.case_manager,
            MappingAPIHandler.audit_logger,
        ) = saved_attrs


def test_switch_source_recorded_to_live_clears_playback_and_streams(tmp_path, monkeypatch):
    """
    Regression test for switching from recorded video view -> live view and apply_discovered:
    1. Does not pass unsupported width/height/fps kwargs to CameraNodeConfig.
    2. Releases and clears ended/paused PlaybackController so live frames stream immediately.
    """
    from src.multi_camera.ui_server import MappingAPIHandler
    from src.playback.controller import PlaybackController

    monkeypatch.setattr(
        "src.multi_camera.ui_server.probe_local_webcams",
        lambda **kwargs: [
            {"source": 0, "name": "Integrated Cam", "width": 640, "height": 480, "fps": 30},
            {"source": 1, "name": "USB Cam", "width": 1280, "height": 720, "fps": 30},
        ],
    )

    graph_file = tmp_path / "camera_graph.json"
    rec_graph = CameraGraph()
    rec_graph.add_node(CameraNodeConfig(camera_id="campus4_c0", name="Cam 0", source="fake0.avi", source_type=SourceType.VIDEO_FILE))
    rec_graph.add_node(CameraNodeConfig(camera_id="campus4_c1", name="Cam 1", source="fake1.avi", source_type=SourceType.VIDEO_FILE))
    rec_graph.save(graph_file)

    pipe = MultiCameraPipeline(
        graph=rec_graph,
        config=AppConfig(),
        camera_factory=lambda cfg: SyntheticCamera(width=320, height=240, fps=30),
    )
    pc = PlaybackController()
    pc.is_playing = False  # Simulate finished/paused recorded playback
    pipe.playback_controller = pc

    saved_attrs = (
        MappingAPIHandler.graph_file,
        MappingAPIHandler.runtime_pipeline,
        MappingAPIHandler.annotation_store,
        MappingAPIHandler.default_annotation_db_path,
        MappingAPIHandler.default_identity_db_path,
        MappingAPIHandler.case_manager,
        getattr(MappingAPIHandler, "audit_logger", None),
    )

    server = run_ui_server(port=0, graph_file=str(graph_file), pipeline=pipe, block=False)
    port = server.server_port
    time.sleep(0.05)

    try:
        # Before switch: ended/paused PlaybackController pauses step()
        assert pipe.step() == {}

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/system/switch_source",
            data=json.dumps({"mode": "live"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["success"] is True
        assert body["mode"] == "live"
        assert body["updated"] == ["cam_0", "cam_1"]

        # PlaybackController must be cleared and live step() must now yield frames
        assert pipe.playback_controller is None
        res = pipe.step()
        assert "cam_0" in res and res["cam_0"][0] is not None
        assert "cam_1" in res and res["cam_1"][0] is not None

        # apply_discovered with 2 cameras (including width/height/fps metadata) must also succeed cleanly
        apply_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/cameras/apply_discovered",
            data=json.dumps({
                "cameras": [
                    {"source": 0, "name": "Cam 0", "width": 640, "height": 480, "fps": 30},
                    {"source": 1, "name": "Cam 1", "width": 1280, "height": 720, "fps": 30},
                ]
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(apply_req, timeout=5.0) as resp:
            apply_body = json.loads(resp.read().decode("utf-8"))
        assert apply_body["success"] is True
        assert apply_body["cameras"] == ["cam_0", "cam_1"]
    finally:
        try:
            pipe.stop()
            server.shutdown()
            server.server_close()
        except Exception:
            pass
        (
            MappingAPIHandler.graph_file,
            MappingAPIHandler.runtime_pipeline,
            MappingAPIHandler.annotation_store,
            MappingAPIHandler.default_annotation_db_path,
            MappingAPIHandler.default_identity_db_path,
            MappingAPIHandler.case_manager,
            MappingAPIHandler.audit_logger,
        ) = saved_attrs
