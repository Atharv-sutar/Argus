"""Configuration loader and schema validation for Argus."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class CameraConfig:
    source: Union[int, str] = 0
    name: str = "camera_0"
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class InferenceConfig:
    device: str = "auto"
    half_precision: bool = False


@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.45
    target_classes: List[int] = field(default_factory=lambda: [0])
    image_size: int = 640


@dataclass
class TrackingConfig:
    track_thresh: float = 0.4
    match_thresh: float = 0.8
    track_buffer: int = 30
    min_box_area: float = 100.0


@dataclass
class VisualizationConfig:
    show_window: bool = True
    window_name: str = "Argus Surveillance"
    draw_fps: bool = True
    draw_boxes: bool = True
    draw_ids: bool = True
    box_thickness: int = 2
    font_scale: float = 0.6


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> AppConfig:
        """Load configuration from a YAML file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppConfig:
        """Construct AppConfig from a dictionary with nested dataclasses."""
        camera_data = data.get("camera", {})
        inference_data = data.get("inference", {})
        detection_data = data.get("detection", {})
        tracking_data = data.get("tracking", {})
        vis_data = data.get("visualization", {})

        return cls(
            camera=CameraConfig(**camera_data),
            inference=InferenceConfig(**inference_data),
            detection=DetectionConfig(**detection_data),
            tracking=TrackingConfig(**tracking_data),
            visualization=VisualizationConfig(**vis_data),
        )
