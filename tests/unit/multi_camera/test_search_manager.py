"""Unit tests for SearchManager progressive search orchestration."""

import time
from unittest.mock import patch

import pytest

from src.core.config import SearchConfig
from src.core.multi_camera_types import (
    CameraEdgeConfig,
    CameraNodeConfig,
    EdgeDirection,
    EdgeType,
    SearchState,
    SourceType,
)
from src.multi_camera.camera_graph import CameraGraph
from src.multi_camera.search_manager import SearchManager


def _make_node(cam_id: str) -> CameraNodeConfig:
    return CameraNodeConfig(
        camera_id=cam_id, name=cam_id, source=0, source_type=SourceType.WEBCAM
    )


def _make_edge(src: str, tgt: str) -> CameraEdgeConfig:
    return CameraEdgeConfig(
        source_camera_id=src,
        target_camera_id=tgt,
        edge_type=EdgeType.ADJACENT,
        direction=EdgeDirection.BIDIRECTIONAL,
    )


def _build_graph_abc():
    """
    Build graph: A — B — C (linear chain)
    """
    g = CameraGraph()
    for c in "ABC":
        g.add_node(_make_node(c))
    g.add_edge(_make_edge("A", "B"))
    g.add_edge(_make_edge("B", "C"))
    return g


def _build_graph_star():
    """
    Build graph:
        B
       /
    A — C
       \
        D
    """
    g = CameraGraph()
    for c in "ABCD":
        g.add_node(_make_node(c))
    g.add_edge(_make_edge("A", "B"))
    g.add_edge(_make_edge("A", "C"))
    g.add_edge(_make_edge("A", "D"))
    return g


# --- Basic search lifecycle ---

def test_on_target_lost_activates_neighbors():
    graph = _build_graph_abc()
    config = SearchConfig(initial_radius=1, max_radius=3)
    sm = SearchManager(graph, config)

    cameras = sm.on_target_lost("A")
    camera_ids = [c[0] for c in cameras]

    assert "B" in camera_ids
    assert "C" not in camera_ids  # C is 2 hops away
    assert sm.is_searching


def test_on_target_lost_star_graph():
    graph = _build_graph_star()
    config = SearchConfig(initial_radius=1)
    sm = SearchManager(graph, config)

    cameras = sm.on_target_lost("A")
    camera_ids = [c[0] for c in cameras]

    assert sorted(camera_ids) == ["B", "C", "D"]


def test_search_state_idle_initially():
    graph = _build_graph_abc()
    sm = SearchManager(graph, SearchConfig())
    assert not sm.is_searching
    assert sm.search_state == SearchState.IDLE


# --- Candidate confirmation ---

def test_candidate_confirmation_requires_multiple_frames():
    graph = _build_graph_abc()
    config = SearchConfig(confirmation_frames=3)
    sm = SearchManager(graph, config)
    sm.on_target_lost("A")

    assert not sm.on_candidate_found("B", 0.85)
    assert not sm.on_candidate_found("B", 0.87)
    assert sm.on_candidate_found("B", 0.90)  # 3rd frame confirms

    assert sm.search_state == SearchState.TARGET_FOUND


def test_candidate_reset_on_different_camera():
    graph = _build_graph_abc()
    config = SearchConfig(confirmation_frames=3)
    sm = SearchManager(graph, config)
    sm.on_target_lost("A")

    sm.on_candidate_found("B", 0.85)
    sm.on_candidate_found("B", 0.87)
    # Switch to different camera resets count
    sm.on_candidate_found("C", 0.80)
    assert sm.search_state != SearchState.TARGET_FOUND


def test_candidate_lost_resets():
    graph = _build_graph_abc()
    config = SearchConfig(confirmation_frames=3)
    sm = SearchManager(graph, config)
    sm.on_target_lost("A")

    sm.on_candidate_found("B", 0.85)
    sm.on_candidate_lost("B")
    # After reset, need all 3 frames again
    assert not sm.on_candidate_found("B", 0.86)
    assert not sm.on_candidate_found("B", 0.87)
    assert sm.on_candidate_found("B", 0.88)


# --- Radius expansion ---

def test_tick_returns_none_when_no_expansion_needed():
    graph = _build_graph_abc()
    config = SearchConfig(per_radius_timeout_s=10.0)
    sm = SearchManager(graph, config)
    sm.on_target_lost("A")

    result = sm.tick(0.5)  # Only 0.5s elapsed
    assert result is None


def test_tick_expands_radius():
    graph = _build_graph_abc()
    config = SearchConfig(
        initial_radius=1,
        radius_increment=1,
        per_radius_timeout_s=0.01,  # Very short for testing
        max_radius=2,
        total_recovery_timeout_s=30.0,
    )
    sm = SearchManager(graph, config)
    sm.on_target_lost("A")

    # Wait for per-radius timeout
    time.sleep(0.02)

    result = sm.tick(1.0)
    assert result is not None
    new_camera_ids = [c[0] for c in result]
    assert "C" in new_camera_ids  # C is 2 hops from A


def test_tick_returns_empty_on_total_timeout():
    graph = _build_graph_abc()
    config = SearchConfig(
        per_radius_timeout_s=100.0,
        total_recovery_timeout_s=0.01,  # Very short
    )
    sm = SearchManager(graph, config)
    sm.on_target_lost("A")

    time.sleep(0.02)
    result = sm.tick(1.0)
    assert result == []  # Timed out
    assert sm.search_state == SearchState.TIMEOUT


# --- Reset ---

def test_reset_returns_to_idle():
    graph = _build_graph_abc()
    sm = SearchManager(graph, SearchConfig())
    sm.on_target_lost("A")
    assert sm.is_searching

    sm.reset()
    assert not sm.is_searching
    assert sm.search_state == SearchState.IDLE


# --- Progress reporting ---

def test_get_progress_during_search():
    graph = _build_graph_abc()
    config = SearchConfig(initial_radius=1)
    sm = SearchManager(graph, config)
    sm.on_target_lost("A")

    progress = sm.get_progress()
    assert progress.current_camera_id == "A"
    assert progress.search_radius == 1
    assert progress.state == SearchState.SEARCHING
    assert "B" in progress.active_cameras


def test_get_progress_shows_standby():
    graph = _build_graph_abc()
    config = SearchConfig(initial_radius=1, max_radius=2)
    sm = SearchManager(graph, config)
    sm.on_target_lost("A")

    progress = sm.get_progress()
    assert "C" in progress.standby_cameras  # C is at radius 2, not yet active
