"""Configuration loader and schema validation for Argus."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
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
    engine_type: str = "auto"
    engine_cache_dir: str = "models/trt_cache"
    batch_size: int = 4
    enable_cross_camera_batching: bool = True
    max_batch_size: int = 8


@dataclass
class DetectionConfig:
    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.45
    target_classes: List[int] = field(default_factory=lambda: [0])
    image_size: int = 640


@dataclass
class TrackingConfig:
    tracker_type: str = "botsort"
    track_thresh: float = 0.4
    match_thresh: float = 0.8
    track_buffer: int = 30
    min_box_area: float = 100.0
    gmc_method: str = "sparseOptFlow"
    reid_weight: float = 0.5
    appearance_thresh: float = 0.35
    ema_alpha: float = 0.9
    # Step 3 (default OFF): treat match_thresh as a ByteTrack-style maximum association
    # cost (0.8 => cost <= 0.8) instead of 1 - match_thresh (0.8 => IoU >= 0.8); run YOLO at
    # low_conf_detection_threshold so the low-score association stage receives detections
    # (track_thresh still gates new tracks); and skip camera frames whose sequence number
    # has not changed so the tracker never sees duplicates.
    fix_association_cost: bool = False
    low_conf_detection_threshold: float = 0.2


@dataclass
class ReIDConfig:
    model_name: str = "osnet_x0_25"
    require_pretrained_weights: bool = True
    match_threshold: float = 0.75
    reacquisition_threshold: float = 0.78
    auto_add_threshold: float = 0.82
    auto_add_min_consecutive: int = 3
    diversity_threshold: float = 0.92
    max_gallery_size: int = 25
    min_margin: float = 0.05
    lock_switch_margin: float = 0.08
    extract_interval_frames: int = 3
    min_crop_width: int = 32
    min_crop_height: int = 64
    event_driven_reid: bool = True  # Uncertainty-triggered ReID: bypass redundant per-frame extraction during steady tracking
    
    # PROVISIONAL / INITIAL thresholds pending calibration against real camera footage
    # (resolution, lighting, and noise vary significantly per camera)
    min_sharpness: float = 60.0 # Laplacian variance threshold for blur filtering
    tracklet_cleave_threshold: float = 0.3 # Cosine shift threshold to split poisoned tracklets
    uncertain_band_min: float = 0.73
    uncertain_band_max: float = 0.78

    # Backward-compatible property aliases
    @property
    def similarity_threshold(self) -> float:
        return self.match_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, val: float) -> None:
        self.match_threshold = val

    @property
    def gallery_size(self) -> int:
        return self.max_gallery_size

    @gallery_size.setter
    def gallery_size(self, val: int) -> None:
        self.max_gallery_size = val






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
    
    # PROVISIONAL / INITIAL threshold pending Phase 5 calibration
    # Do not hardcode any logic that assumes this is strictly 50%
    candidate_filter_rate: float = 0.5
    handoff_confirm_delay_s: float = 2.0



@dataclass
class PlaybackConfig:
    """Configuration for playback and fast-scan processing modes."""
    fast_scan_auto_accept: float = 0.82
    fast_scan_auto_reject: float = 0.78
    fast_scan_rewind_s: float = 5.0

@dataclass
class StorageConfig:
    """Configuration for persistent storage of targets and embeddings."""
    db_path: str = "data/identities.db"
    enabled: bool = True

@dataclass
class MultiCameraConfig:
    """Configuration for the multi-camera graph system."""
    enabled: bool = False
    graph_file: str = "configs/camera_graph.json"
    search: SearchConfig = field(default_factory=SearchConfig)
    ui_port: int = 8765


@dataclass
class ForensicConfig:
    """Configuration for forensic clip recording."""
    target_fps: float = 15.0
    resolution: Tuple[int, int] = (1280, 720)
    max_memory_frames: int = 9000
    export_dir: str = "exports"


@dataclass
class RecordingRetentionConfig:
    """Configuration for retention cleanup of continuous recordings."""
    max_storage_gb: float = 50.0
    max_age_days: float = 7.0
    cleanup_interval_s: float = 300.0


@dataclass
class RecordingConfig:
    """Configuration for continuous background NVR recording."""
    enabled: bool = True
    output_dir: str = "recordings"
    segment_duration_s: float = 300.0
    container_format: str = "mp4"
    retention: RecordingRetentionConfig = field(default_factory=RecordingRetentionConfig)


@dataclass
class TargetConfig:
    """Target-lock decision behaviour flags (all default OFF = legacy behaviour)."""
    # Step 2: an incumbent above match_threshold can still lose the lock to a candidate that
    # beats it by lock_switch_margin (existing consensus + manual-anchor gates apply), and
    # auto-enrollment only happens when the incumbent is top-1 by min_margin.
    switch_when_better: bool = False
    # Step 4: on ReID frames embed/score every track (not only the locked one) during
    # TRACKING/LOCKED, apply the blur gate per blur_gate_scope, and force one full
    # re-verification on the first frames after an occlusion ends.
    score_all_tracks: bool = False
    # Used only when score_all_tracks is on: "none" = no blur gate for anyone,
    # "all" = blur gate for everyone incl. the locked track (a blurred locked track defers
    # the decision for that frame instead of counting as a miss).
    blur_gate_scope: str = "none"


@dataclass
class FixesConfig:
    """Bug fixes that change runtime behaviour (all default OFF = legacy behaviour)."""
    # Serialize step() with operator actions from the HTTP thread (select/correct/clear target,
    # manual sample, handoff confirm/reject, undo/redo, camera focus) on the pipeline lock.
    serialize_pipeline_access: bool = False
    # On new target selection / clear: drop the previous target's gallery thumbnails, evidence,
    # switch consensus, miss counter, pending handoff and auto-add rate-limit clock.
    reset_state_on_target_change: bool = False
    # Auto-add rate limiter: a timestamp that jumps backwards (video loop, new clip) starts a
    # new timeline instead of blocking enrollment until the old timestamp is passed again.
    auto_add_handles_time_reset: bool = False
    # Cross-camera handoff: the N confirmation frames must come from the same track ID.
    per_track_handoff_confirmation: bool = False
    # /api/settings: write total_recovery_timeout_s (was a typo'd attribute) and push a new
    # match_threshold into the identity manager and evidence engine, not just the config.
    apply_settings_live: bool = False


@dataclass
class DiagnosticsConfig:
    """Configuration for ReID decision logging and system diagnostics."""
    decision_log: bool = False
    decision_log_path: str = "logs/decision_log.jsonl"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    multi_camera: MultiCameraConfig = field(default_factory=MultiCameraConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    forensic: ForensicConfig = field(default_factory=ForensicConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    fixes: FixesConfig = field(default_factory=FixesConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> AppConfig:
        """Load configuration from a YAML file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    load = from_yaml

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

        playback_data = data.get("playback", {})
        playback_config = PlaybackConfig(**playback_data) if playback_data else PlaybackConfig()

        storage_data = data.get("storage", {})
        storage_config = StorageConfig(**storage_data) if storage_data else StorageConfig()

        forensic_data = data.get("forensic", {})
        
        # Handle tuple conversion for resolution
        if "resolution" in forensic_data and isinstance(forensic_data["resolution"], list):
            forensic_data["resolution"] = tuple(forensic_data["resolution"])
            
        forensic_config = ForensicConfig(**forensic_data) if forensic_data else ForensicConfig()

        rec_data = data.get("recording", {})
        ret_data = rec_data.pop("retention", {}) if isinstance(rec_data, dict) else {}
        ret_config = RecordingRetentionConfig(**ret_data) if ret_data else RecordingRetentionConfig()
        recording_config = RecordingConfig(**rec_data, retention=ret_config) if rec_data else RecordingConfig()

        diag_data = data.get("diagnostics", {})
        diagnostics_config = DiagnosticsConfig(**diag_data) if diag_data else DiagnosticsConfig()

        target_data = data.get("target", {})
        target_config = TargetConfig(**target_data) if target_data else TargetConfig()

        fixes_data = data.get("fixes", {})
        fixes_config = FixesConfig(**fixes_data) if fixes_data else FixesConfig()

        return cls(
            camera=CameraConfig(**camera_data),
            inference=InferenceConfig(**inference_data),
            detection=DetectionConfig(**detection_data),
            tracking=TrackingConfig(**tracking_data),
            reid=ReIDConfig(**reid_data),
            visualization=VisualizationConfig(**vis_data),
            multi_camera=mc_config,
            storage=storage_config,
            playback=playback_config,
            forensic=forensic_config,
            recording=recording_config,
            diagnostics=diagnostics_config,
            target=target_config,
            fixes=fixes_config,
        )


    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        import yaml
        file_path = Path(path)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
