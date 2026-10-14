"""Unit tests for the directory loader module."""

import os
from pathlib import Path
import pytest

from src.camera.directory_loader import (
    auto_load_video_directory,
    discover_video_files,
    is_video_directory,
    _natural_sort_key,
    _sanitize_camera_id,
    _friendly_camera_name,
)
from src.core.multi_camera_types import EdgeDirection, EdgeType, SourceType
from src.multi_camera.camera_graph import CameraGraph


def test_sanitize_camera_id():
    assert _sanitize_camera_id("campus4-c0") == "campus4_c0"
    assert _sanitize_camera_id("01_entrance") == "cam_01_entrance"
    assert _sanitize_camera_id("LOBBY-CAM-North!") == "lobby_cam_north"
    assert _sanitize_camera_id("") == "cam"


def test_friendly_camera_name():
    assert _friendly_camera_name("campus4-c0") == "Campus4 C0"
    assert _friendly_camera_name("main_entrance_gate") == "Main Entrance Gate"
    assert _friendly_camera_name("") == "Camera"


def test_discover_nonexistent_directory(tmp_path):
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        discover_video_files(non_existent)


def test_discover_file_not_dir(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello")
    with pytest.raises(ValueError):
        discover_video_files(f)


def test_discover_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert discover_video_files(empty_dir) == []
    assert not is_video_directory(empty_dir)


def test_natural_sorting_and_extensions(tmp_path):
    dir_path = tmp_path / "videos"
    dir_path.mkdir()

    # Create dummy files
    (dir_path / "cam10.mp4").write_text("mock")
    (dir_path / "cam2.mp4").write_text("mock")
    (dir_path / "cam1.avi").write_text("mock")
    (dir_path / "cam20.mkv").write_text("mock")
    (dir_path / "notes.txt").write_text("ignore me")
    (dir_path / "image.png").write_text("ignore me")

    assert is_video_directory(dir_path) is True

    discovered = discover_video_files(dir_path)
    names = [f.name for f in discovered]
    assert names == ["cam1.avi", "cam2.mp4", "cam10.mp4", "cam20.mkv"]


def test_auto_load_video_directory_synthetic(tmp_path):
    dir_path = tmp_path / "cams"
    dir_path.mkdir()

    (dir_path / "c0.mp4").write_text("mock")
    (dir_path / "c1.mp4").write_text("mock")
    (dir_path / "c2.mp4").write_text("mock")

    graph = auto_load_video_directory(dir_path, default_transit_time_s=7.5)

    assert graph.node_count() == 3
    all_cids = sorted(list(graph.all_camera_ids()))
    assert all_cids == ["c0", "c1", "c2"]

    for cid in all_cids:
        node = graph.get_node(cid)
        assert node is not None
        assert node.source_type == SourceType.VIDEO_FILE
        assert str(dir_path).replace("\\", "/") in node.source.replace("\\", "/")
        assert node.enabled is True
        assert node.position_x > 0
        assert node.position_y > 0

    # 3 cameras fully connected = 3 pairs (c0-c1, c0-c2, c1-c2)
    assert graph.edge_count() == 3
    edges = list(graph._edges.values())
    assert len(edges) == 3
    for edge in edges:
        assert edge.edge_type == EdgeType.ADJACENT
        assert edge.direction == EdgeDirection.BIDIRECTIONAL
        assert edge.expected_typical_transition_s == 7.5

    # Check bidirectional neighbors
    assert set(graph.get_neighbors("c0")) == {"c1", "c2"}
    assert set(graph.get_neighbors("c1")) == {"c0", "c2"}
    assert set(graph.get_neighbors("c2")) == {"c0", "c1"}

    # Validation must pass cleanly
    errors = graph.validate()
    assert errors == []

    # Round-trip serialization
    d = graph.to_dict()
    restored = CameraGraph.from_dict(d)
    assert restored.node_count() == 3
    assert restored.edge_count() == 3


def test_auto_load_real_epfl_campus():
    real_path = Path("test_footage/EPFL campus")
    if not real_path.is_dir():
        pytest.skip("EPFL campus footage not present in workspace")

    graph = auto_load_video_directory(real_path)
    assert graph.node_count() == 3
    cids = sorted(list(graph.all_camera_ids()))
    assert cids == ["campus4_c0", "campus4_c1", "campus4_c2"]
    assert graph.edge_count() == 3
    assert graph.validate() == []
