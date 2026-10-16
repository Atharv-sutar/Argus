import pytest
import numpy as np
from unittest.mock import MagicMock
from src.playback.controller import PlaybackController
from src.playback.journey_stitcher import JourneyStitcher
from src.playback.route_recorder import RouteEvent
from src.core.types import BoundingBox, Detection, DetectionResult, Track, TrackResult, TrackState
from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.multi_camera.camera_graph import CameraGraph
from src.multi_camera.camera_node import CameraNode
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline


def test_playback_controller_eof_rewind_and_seek():
    """Verify that when playback reaches the end, play() and toggle() rewind to start."""
    controller = PlaybackController()
    controller.max_duration_ms = 10000.0
    controller.current_time_ms = 10000.0
    controller.is_playing = False

    # Calling play at EOF should reset current_time_ms to 0.0 and start playing
    controller.play()
    assert controller.current_time_ms == 0.0
    assert controller.is_playing is True

    # Seek backwards while playing
    controller.seek(4000.0)
    assert controller.current_time_ms == 4000.0

    # Fast forward to end and pause
    controller.current_time_ms = 10000.0
    controller.pause()
    assert controller.is_playing is False

    # Calling toggle_play_pause at EOF should restart from 0.0
    controller.toggle_play_pause()
    assert controller.current_time_ms == 0.0
    assert controller.is_playing is True


def test_journey_stitcher_cross_camera_segments():
    """Verify JourneyStitcher builds proper multi-camera segments across handoffs."""
    events = [
        RouteEvent(camera_id="cam_0", track_id=1, timestamp_ms=1000.0, event_type="enter", bbox=(10, 10, 50, 50), confidence=1.0, was_human_corrected=False),
        RouteEvent(camera_id="cam_0", track_id=1, timestamp_ms=5000.0, event_type="exit", bbox=(0, 0, 0, 0), confidence=1.0, was_human_corrected=False),
        RouteEvent(camera_id="cam_1", track_id=2, timestamp_ms=5000.0, event_type="handoff", bbox=(20, 20, 60, 60), confidence=1.0, was_human_corrected=False),
    ]
    stitcher = JourneyStitcher(
        video_sources={"cam_0": "fake0.mp4", "cam_1": "fake1.mp4"},
        output_path="test_out.mp4",
    )
    segments = stitcher._build_segments(events, final_time_ms=9000.0)
    assert len(segments) == 2
    assert segments[0]["camera_id"] == "cam_0"
    assert segments[0]["start_ms"] == 1000.0
    assert segments[0]["end_ms"] >= 5000.0
    assert segments[1]["camera_id"] == "cam_1"
    assert segments[1]["start_ms"] == 5000.0
    assert segments[1]["end_ms"] >= 9000.0


def test_normalized_coord_selection_and_detection_fallback():
    """Verify target selection supports normalized coordinates and falls back to detections."""
    config = AppConfig()
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_0", name="TestCam", source="0", source_type=SourceType.WEBCAM))

    pipeline = MultiCameraPipeline(config=config, graph=graph)
    worker = pipeline._get_or_create_worker("cam_0")
    assert worker is not None

    # Simulate worker having a frame (1280x720) and no tracks yet
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    worker._last_frame = test_frame
    worker._last_track_result = TrackResult(tracks=[], frame_id=1, timestamp_ms=100.0)

    # Mock detector to return a person detection in the center (x=640, y=360)
    mock_detector = MagicMock()
    mock_box = BoundingBox(x1=600, y1=300, x2=680, y2=420)
    mock_detector.detect.return_value = DetectionResult(
        detections=[Detection(box=mock_box, confidence=0.92, class_id=0, class_name="person")]
    )
    worker.detector = mock_detector

    # Operator clicks at normalized center: norm_x=0.5, norm_y=0.5
    selected_id = pipeline.select_target_on_camera("cam_0", x=0.0, y=0.0, norm_x=0.5, norm_y=0.5)

    assert selected_id is not None
    assert pipeline.active_camera_id == "cam_0"
    assert pipeline.target_manager.is_active() is True
    # Verify route recorder logged enter event
    events = pipeline.route_recorder.get_events()
    assert len(events) >= 1
    assert events[-1].camera_id == "cam_0"
    assert events[-1].event_type == "enter"

    pipeline.stop()
