"""Auto-discovery and graph generation for directories of recorded video files."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Set, Union

from src.core.multi_camera_types import (
    CameraEdgeConfig,
    CameraNodeConfig,
    EdgeDirection,
    EdgeType,
    SourceType,
)
from src.multi_camera.camera_graph import CameraGraph

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS: Set[str] = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".m4v",
    ".webm",
}


def _natural_sort_key(path: Path) -> List[Union[int, str]]:
    """Generates natural sort keys so campus4-c2 comes before campus4-c10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _sanitize_camera_id(stem: str) -> str:
    """Produces a clean, alphanumeric + underscore camera ID."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", stem).strip("_").lower()
    if not clean:
        clean = "cam"
    if clean[0].isdigit():
        clean = f"cam_{clean}"
    return clean


def _friendly_camera_name(stem: str) -> str:
    """Produces a human-readable camera name from a filename stem."""
    name = re.sub(r"[_\-]+", " ", stem).strip()
    return name.title() if name else "Camera"


def is_video_directory(dir_path: Union[str, Path]) -> bool:
    """Checks whether the given path is a directory containing at least one supported video file."""
    p = Path(dir_path)
    if not p.is_dir():
        return False
    try:
        for entry in p.iterdir():
            if entry.is_file() and entry.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                return True
    except OSError:
        return False
    return False


def discover_video_files(dir_path: Union[str, Path]) -> List[Path]:
    """
    Scans a directory for supported video files, returning them naturally sorted.
    """
    p = Path(dir_path)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: '{dir_path}'")
    if not p.is_dir():
        raise ValueError(f"Path is not a directory: '{dir_path}'")

    video_files = [
        entry
        for entry in p.iterdir()
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]
    video_files.sort(key=_natural_sort_key)
    return video_files


def auto_load_video_directory(
    dir_path: Union[str, Path],
    default_transit_time_s: float = 5.0,
    max_transit_time_s: float = 60.0,
    connect_all: bool = True,
) -> CameraGraph:
    """
    Discovers all video files in a directory and builds a fully configured CameraGraph.

    Parameters:
        dir_path: Path to directory containing video files.
        default_transit_time_s: Expected typical transition time between adjacent cameras.
        max_transit_time_s: Maximum transit time before target is deemed lost.
        connect_all: If True, adds bidirectional edges between all discovered cameras.

    Returns:
        CameraGraph populated with CameraNodeConfig and CameraEdgeConfig objects.
    """
    video_files = discover_video_files(dir_path)
    if not video_files:
        raise ValueError(
            f"No supported video files ({', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}) "
            f"found in directory: '{dir_path}'"
        )

    graph = CameraGraph()
    used_ids: Set[str] = set()
    node_configs: List[CameraNodeConfig] = []

    cols = 3 if len(video_files) >= 3 else max(1, len(video_files))

    for idx, fpath in enumerate(video_files):
        stem = fpath.stem
        raw_id = _sanitize_camera_id(stem)
        cid = raw_id
        counter = 1
        while cid in used_ids:
            cid = f"{raw_id}_{counter}"
            counter += 1
        used_ids.add(cid)

        col = idx % cols
        row = idx // cols
        pos_x = 180.0 + (col * 280.0)
        pos_y = 140.0 + (row * 200.0)

        # Normalize path to forward slashes for cross-platform portability
        source_str = str(fpath.resolve()).replace("\\", "/")

        node = CameraNodeConfig(
            camera_id=cid,
            name=_friendly_camera_name(stem),
            source=source_str,
            source_type=SourceType.VIDEO_FILE,
            enabled=True,
            position_x=pos_x,
            position_y=pos_y,
            description=f"Auto-loaded from {fpath.name}",
        )
        graph.add_node(node)
        node_configs.append(node)

    # Fully connect all pairs with bidirectional transitions
    if connect_all and len(node_configs) > 1:
        for i in range(len(node_configs)):
            for j in range(i + 1, len(node_configs)):
                edge = CameraEdgeConfig(
                    source_camera_id=node_configs[i].camera_id,
                    target_camera_id=node_configs[j].camera_id,
                    edge_type=EdgeType.ADJACENT,
                    direction=EdgeDirection.BIDIRECTIONAL,
                    enabled=True,
                    expected_min_transition_s=0.0,
                    expected_typical_transition_s=default_transit_time_s,
                    expected_max_transition_s=max_transit_time_s,
                )
                graph.add_edge(edge)

    logger.info(
        f"[DIR-LOADER] Successfully created CameraGraph with {len(node_configs)} "
        f"cameras and {graph.edge_count()} edges from '{dir_path}'"
    )
    return graph
