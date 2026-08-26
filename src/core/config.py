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
class ReIDConfig:
    model_name: str = "osnet_x0_25"
    similarity_threshold: float = 0.88
    reacquisition_threshold: float = 0.90
    reference_threshold: float = 0.85
    upper_threshold: float = 0.82
    min_margin: float = 0.05
    w_upper: float = 0.40
    w_color: float = 0.15
    w_deep: float = 0.35
    w_lower: float = 0.10
    extract_interval_frames: int = 5

    gallery_size: int = 6
    reference_samples: int = 4
    reference_window_frames: int = 30
    adaptive_gallery_size: int = 6
    min_crop_width: int = 24
    min_crop_height: int = 60
    min_sharpness: float = 25.0
    redundancy_threshold: float = 0.90
    temporal_window_size: int = 5
    reacquisition_min_frames: int = 5






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
class SearchConfig:
    """Configuration for multi-camera progressive search."""
    initial_radius: int = 1
    radius_increment: int = 1
    per_radius_timeout_s: float = 5.0
    max_radius: int = 3
    total_recovery_timeout_s: float = 30.0
    confirmation_frames: int = 3
    handoff_confirm_delay_s: float = 2.0



@dataclass
class MultiCameraConfig:
    """Configuration for the multi-camera graph system."""
    enabled: bool = False
    graph_file: str = "configs/camera_graph.json"
    search: SearchConfig = field(default_factory=SearchConfig)
    ui_port: int = 8765


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    multi_camera: MultiCameraConfig = field(default_factory=MultiCameraConfig)

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
        reid_data = data.get("reid", {})
        vis_data = data.get("visualization", {})
        mc_data = data.get("multi_camera", {})

        search_data = mc_data.pop("search", {}) if isinstance(mc_data, dict) else {}
        search_config = SearchConfig(**search_data) if search_data else SearchConfig()
        mc_config = MultiCameraConfig(**mc_data, search=search_config) if mc_data else MultiCameraConfig()

        return cls(
            camera=CameraConfig(**camera_data),
            inference=InferenceConfig(**inference_data),
            detection=DetectionConfig(**detection_data),
            tracking=TrackingConfig(**tracking_data),
            reid=ReIDConfig(**reid_data),
            visualization=VisualizationConfig(**vis_data),
            multi_camera=mc_config,
        )

