"""Multi-camera domain types for the camera graph topology and search system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union


class SourceType(str, Enum):
    """Type of video source for a camera."""
    WEBCAM = "webcam"
    RTSP = "rtsp"
    VIDEO_FILE = "video_file"
    SYNTHETIC = "synthetic"


class EdgeType(str, Enum):
    """Physical/operational relationship between two cameras."""
    OVERLAP = "overlap"    # Fields of view partially overlap — person may be visible in both
    ADJACENT = "adjacent"  # Nearby but non-overlapping — person can walk between them
    TRAVEL = "travel"      # Connected only through a longer physical route


class EdgeDirection(str, Enum):
    """Directional constraint on a camera edge."""
    BIDIRECTIONAL = "bidirectional"
    A_TO_B = "a_to_b"
    B_TO_A = "b_to_a"


class CameraStatus(str, Enum):
    """Runtime operational state of a camera node."""
    ONLINE = "online"          # Connected and grabbing frames
    OFFLINE = "offline"        # Not reachable / disconnected
    DISABLED = "disabled"      # Manually disabled by user
    SEARCHING = "searching"    # Actively running AI pipeline for target search
    ACTIVE_TARGET = "active_target"  # Currently tracking the target


class SearchState(str, Enum):
    """Global multi-camera search lifecycle state."""
    IDLE = "idle"              # No search in progress — target is tracked or unselected
    SEARCHING = "searching"    # Actively searching neighbor cameras
    EXPANDED = "expanded"      # Search radius has been expanded beyond initial neighbors
    TARGET_FOUND = "target_found"  # Candidate confirmed on a new camera
    TIMEOUT = "timeout"        # Search exhausted without finding the target


@dataclass
class CameraNodeConfig:
    """Static configuration for a single camera node in the topology graph."""
    camera_id: str
    name: str
    source: Union[int, str]  # Webcam index, RTSP URL, or video file path
    source_type: SourceType = SourceType.WEBCAM
    enabled: bool = True
    position_x: float = 0.0  # Logical X position on the map canvas
    position_y: float = 0.0  # Logical Y position on the map canvas
    floor: Optional[str] = None
    zone: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source": self.source,
            "source_type": self.source_type.value,
            "enabled": self.enabled,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "floor": self.floor,
            "zone": self.zone,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraNodeConfig:
        raw_st = str(data.get("source_type", "webcam")).lower().strip()
        if raw_st in ("local", "usb", "cam", "webcam"):
            st = SourceType.WEBCAM
        elif raw_st in ("rtsp", "http", "https", "stream"):
            st = SourceType.RTSP
        elif raw_st in ("video", "video_file", "file"):
            st = SourceType.VIDEO_FILE
        elif raw_st in ("synthetic", "synth"):
            st = SourceType.SYNTHETIC
        else:
            try:
                st = SourceType(raw_st)
            except ValueError:
                st = SourceType.WEBCAM

        return cls(
            camera_id=data["camera_id"],
            name=data["name"],
            source=data["source"],
            source_type=st,
            enabled=data.get("enabled", True),
            position_x=data.get("position_x", 0.0),
            position_y=data.get("position_y", 0.0),
            floor=data.get("floor"),
            zone=data.get("zone"),
            description=data.get("description"),
        )


@dataclass
class CameraEdgeConfig:
    """Configuration for a directed or bidirectional connection between two cameras."""
    source_camera_id: str
    target_camera_id: str
    edge_type: EdgeType = EdgeType.ADJACENT
    direction: EdgeDirection = EdgeDirection.BIDIRECTIONAL
    enabled: bool = True
    expected_min_transition_s: Optional[float] = None
    expected_typical_transition_s: Optional[float] = None
    expected_max_transition_s: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "source_camera_id": self.source_camera_id,
            "target_camera_id": self.target_camera_id,
            "edge_type": self.edge_type.value,
            "direction": self.direction.value,
            "enabled": self.enabled,
            "expected_min_transition_s": self.expected_min_transition_s,
            "expected_typical_transition_s": self.expected_typical_transition_s,
            "expected_max_transition_s": self.expected_max_transition_s,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraEdgeConfig:
        return cls(
            source_camera_id=data["source_camera_id"],
            target_camera_id=data["target_camera_id"],
            edge_type=EdgeType(data.get("edge_type", "adjacent")),
            direction=EdgeDirection(data.get("direction", "bidirectional")),
            enabled=data.get("enabled", True),
            expected_min_transition_s=data.get("expected_min_transition_s"),
            expected_typical_transition_s=data.get("expected_typical_transition_s"),
            expected_max_transition_s=data.get("expected_max_transition_s"),
        )


@dataclass
class SearchProgress:
    """Observable snapshot of the current multi-camera search state."""
    current_camera_id: Optional[str] = None
    search_radius: int = 0
    elapsed_s: float = 0.0
    active_cameras: List[str] = field(default_factory=list)
    standby_cameras: List[str] = field(default_factory=list)
    searched_cameras: List[str] = field(default_factory=list)
    state: SearchState = SearchState.IDLE
    candidate_camera_id: Optional[str] = None
    candidate_confirmation_count: int = 0

    def to_dict(self) -> dict:
        return {
            "current_camera_id": self.current_camera_id,
            "search_radius": self.search_radius,
            "elapsed_s": round(self.elapsed_s, 2),
            "active_cameras": self.active_cameras,
            "standby_cameras": self.standby_cameras,
            "searched_cameras": self.searched_cameras,
            "state": self.state.value,
            "candidate_camera_id": self.candidate_camera_id,
            "candidate_confirmation_count": self.candidate_confirmation_count,
        }
