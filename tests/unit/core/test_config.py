"""Unit tests for configuration loading and validation."""

import pytest
from src.core.config import AppConfig


def test_default_config():
    config = AppConfig()
    assert config.camera.name == "camera_0"
    assert config.detection.model_name == "yolov8n.pt"
    assert config.detection.confidence_threshold == 0.4
    assert config.tracking.track_thresh == 0.4
    assert config.visualization.show_window is True


def test_config_from_dict():
    data = {
        "camera": {"source": "test.mp4", "fps": 25},
        "detection": {"confidence_threshold": 0.5},
        "tracking": {"track_buffer": 45},
    }
    config = AppConfig.from_dict(data)
    assert config.camera.source == "test.mp4"
    assert config.camera.fps == 25
    assert config.detection.confidence_threshold == 0.5
    assert config.tracking.track_buffer == 45
