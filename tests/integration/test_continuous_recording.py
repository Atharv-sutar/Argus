"""Integration tests for continuous recording NVR in MultiCameraPipeline."""

import os
import shutil
import tempfile
import time
import numpy as np
import pytest

from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.recording.manager import ContinuousRecordingManager
from src.camera.capture import SyntheticCamera


@pytest.fixture
def temp_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath, ignore_errors=True)


def test_recording_eligibility():
    """Verify that only live surveillance sources are eligible for continuous recording."""
    mgr = ContinuousRecordingManager(enabled=True)

    # Eligible live sources
    assert mgr.is_eligible_source(SourceType.WEBCAM) is True
    assert mgr.is_eligible_source(SourceType.RTSP) is True
    assert mgr.is_eligible_source("webcam") is True
    assert mgr.is_eligible_source("rtsp") is True
    assert mgr.is_eligible_source("local") is True
    assert mgr.is_eligible_source("usb") is True

    # Ineligible non-live sources (Strict Rule)
    assert mgr.is_eligible_source(SourceType.VIDEO_FILE) is False
    assert mgr.is_eligible_source(SourceType.SYNTHETIC) is False
    assert mgr.is_eligible_source("video_file") is False
    assert mgr.is_eligible_source("synthetic") is False
    assert mgr.is_eligible_source("file") is False


def test_pipeline_continuous_recording_live_tap(temp_dir, monkeypatch):
    """
    Verify that live camera feeds are tapped directly from pipeline acquisition
    into continuous recording writers, while offline video files are skipped.
    """
    config = AppConfig()
    config.recording.enabled = True
    config.recording.output_dir = temp_dir
    config.recording.segment_duration_s = 1.0  # short segment for test
    config.recording.retention.max_age_days = 1.0
    config.recording.retention.max_storage_gb = 5.0

    graph = CameraGraph()

    # Camera 1: Live webcam source (should record)
    cfg_live = CameraNodeConfig(
        camera_id="cam_live",
        name="Live Entrance",
        source=0,
        source_type=SourceType.WEBCAM,
        enabled=True,
    )
    graph.add_node(cfg_live)

    # Camera 2: Video file source (must NEVER record)
    cfg_file = CameraNodeConfig(
        camera_id="cam_file",
        name="Offline Video",
        source="test.mp4",
        source_type=SourceType.VIDEO_FILE,
        enabled=True,
    )
    graph.add_node(cfg_file)

    # Mock camera factory to return open synthetic cameras for both
    def mock_cam_factory(node_cfg):
        return SyntheticCamera(width=320, height=240, fps=10)

    # Mock detector and tracker to keep test fast
    class FastMockDetector:
        def detect(self, frame, frame_id=0, timestamp_ms=0.0):
            from src.core.types import DetectionResult
            return DetectionResult(frame_id=frame_id, timestamp_ms=timestamp_ms, detections=[])
        def detect_batch(self, frames, frame_ids, timestamps_ms):
            from src.core.types import DetectionResult
            return [DetectionResult(frame_id=fid, timestamp_ms=ts, detections=[]) for fid, ts in zip(frame_ids, timestamps_ms)]

    class FastMockTracker:
        def update(self, detection_result, frame=None):
            from src.core.types import TrackResult
            return TrackResult(frame_id=detection_result.frame_id, timestamp_ms=detection_result.timestamp_ms, tracks=[])

    pipeline = MultiCameraPipeline(
        graph=graph,
        config=config,
        camera_factory=mock_cam_factory,
        shared_detector=FastMockDetector(),
        tracker_factory=lambda: FastMockTracker(),
    )

    try:
        # Step pipeline to trigger worker initialization and frame acquisition
        for _ in range(5):
            pipeline.step()
            time.sleep(0.05)

        # Verify recording manager state:
        # 'cam_live' should be in active_cameras (writer started)
        # 'cam_file' must NOT be in active_cameras
        stats = pipeline.recording_manager.get_storage_stats()
        assert "cam_live" in stats["active_cameras"]
        assert "cam_file" not in stats["active_cameras"]

        # Telemetry should include recording stats
        telem = pipeline.get_telemetry()
        assert telem.recording_stats is not None
        assert telem.recording_stats["enabled"] is True
        assert "cam_live" in telem.recording_stats["active_cameras"]
        assert "cam_file" not in telem.recording_stats["active_cameras"]

    finally:
        pipeline.stop()
