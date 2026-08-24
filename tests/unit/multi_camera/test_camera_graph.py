"""Unit tests for CameraGraph topology operations."""

import pytest

from src.core.multi_camera_types import (
    CameraEdgeConfig,
    CameraNodeConfig,
    EdgeDirection,
    EdgeType,
    SourceType,
)
from src.multi_camera.camera_graph import CameraGraph


def _make_node(cam_id: str, name: str = "") -> CameraNodeConfig:
    return CameraNodeConfig(
        camera_id=cam_id,
        name=name or cam_id,
        source=0,
        source_type=SourceType.WEBCAM,
    )


def _make_edge(
    src: str,
    tgt: str,
    edge_type: EdgeType = EdgeType.ADJACENT,
    direction: EdgeDirection = EdgeDirection.BIDIRECTIONAL,
) -> CameraEdgeConfig:
    return CameraEdgeConfig(
        source_camera_id=src,
        target_camera_id=tgt,
        edge_type=edge_type,
        direction=direction,
    )


# --- Node tests ---

def test_add_and_get_node():
    g = CameraGraph()
    g.add_node(_make_node("A"))
    assert g.get_node("A") is not None
    assert g.get_node("A").camera_id == "A"
    assert g.node_count() == 1


def test_remove_node_removes_edges():
    g = CameraGraph()
    g.add_node(_make_node("A"))
    g.add_node(_make_node("B"))
    g.add_node(_make_node("C"))
    g.add_edge(_make_edge("A", "B"))
    g.add_edge(_make_edge("A", "C"))
    assert g.edge_count() == 2

    g.remove_node("A")
    assert g.get_node("A") is None
    assert g.edge_count() == 0
    assert g.get_neighbors("B") == []


def test_all_camera_ids():
    g = CameraGraph()
    g.add_node(_make_node("X"))
    g.add_node(_make_node("Y"))
    assert sorted(g.all_camera_ids()) == ["X", "Y"]


# --- Edge tests ---

def test_add_bidirectional_edge():
    g = CameraGraph()
    g.add_node(_make_node("A"))
    g.add_node(_make_node("B"))
    g.add_edge(_make_edge("A", "B", direction=EdgeDirection.BIDIRECTIONAL))

    assert "B" in g.get_neighbors("A")
    assert "A" in g.get_neighbors("B")


def test_add_directed_edge_a_to_b():
    g = CameraGraph()
    g.add_node(_make_node("A"))
    g.add_node(_make_node("B"))
    g.add_edge(_make_edge("A", "B", direction=EdgeDirection.A_TO_B))

    assert "B" in g.get_neighbors("A")
    assert "A" not in g.get_neighbors("B")


def test_add_directed_edge_b_to_a():
    g = CameraGraph()
    g.add_node(_make_node("A"))
    g.add_node(_make_node("B"))
    g.add_edge(_make_edge("A", "B", direction=EdgeDirection.B_TO_A))

    assert "B" not in g.get_neighbors("A")
    assert "A" in g.get_neighbors("B")


def test_self_edge_raises():
    g = CameraGraph()
    g.add_node(_make_node("A"))
    with pytest.raises(ValueError, match="self-edge"):
        g.add_edge(_make_edge("A", "A"))


def test_edge_to_missing_node_raises():
    g = CameraGraph()
    g.add_node(_make_node("A"))
    with pytest.raises(ValueError, match="not in graph"):
        g.add_edge(_make_edge("A", "Z"))


def test_remove_edge():
    g = CameraGraph()
    g.add_node(_make_node("A"))
    g.add_node(_make_node("B"))
    g.add_edge(_make_edge("A", "B"))
    g.remove_edge("A", "B")
    assert g.get_neighbors("A") == []
    assert g.edge_count() == 0


# --- Topology queries ---

def test_get_neighbors_radius_1():
    """
    Graph: A — B — C
    Neighbors of A at radius 1 = [B]
    """
    g = CameraGraph()
    g.add_node(_make_node("A"))
    g.add_node(_make_node("B"))
    g.add_node(_make_node("C"))
    g.add_edge(_make_edge("A", "B"))
    g.add_edge(_make_edge("B", "C"))

    n = g.get_neighbors("A", radius=1)
    assert n == ["B"]


def test_get_neighbors_radius_2():
    """
    Graph: A — B — C
    Neighbors of A at radius 2 = [B, C]
    """
    g = CameraGraph()
    g.add_node(_make_node("A"))
    g.add_node(_make_node("B"))
    g.add_node(_make_node("C"))
    g.add_edge(_make_edge("A", "B"))
    g.add_edge(_make_edge("B", "C"))

    n = g.get_neighbors("A", radius=2)
    assert sorted(n) == ["B", "C"]


def test_get_neighbors_excludes_disabled():
    g = CameraGraph()
    g.add_node(_make_node("A"))
    node_b = _make_node("B")
    node_b.enabled = False
    g.add_node(node_b)
    g.add_edge(_make_edge("A", "B"))

    assert g.get_neighbors("A") == []


def test_get_neighbors_by_radius():
    """
    Graph: A — B — C — D
    From A: radius 1 = [B], radius 2 = [C], radius 3 = [D]
    """
    g = CameraGraph()
    for c in "ABCD":
        g.add_node(_make_node(c))
    g.add_edge(_make_edge("A", "B"))
    g.add_edge(_make_edge("B", "C"))
    g.add_edge(_make_edge("C", "D"))

    by_r = g.get_neighbors_by_radius("A", max_radius=3)
    assert by_r[1] == ["B"]
    assert by_r[2] == ["C"]
    assert by_r[3] == ["D"]


def test_get_neighbors_by_radius_branching():
    """
    Graph:
        A — B
        A — C
        B — D
    From A: radius 1 = [B, C], radius 2 = [D]
    """
    g = CameraGraph()
    for c in "ABCD":
        g.add_node(_make_node(c))
    g.add_edge(_make_edge("A", "B"))
    g.add_edge(_make_edge("A", "C"))
    g.add_edge(_make_edge("B", "D"))

    by_r = g.get_neighbors_by_radius("A", max_radius=2)
    assert sorted(by_r[1]) == ["B", "C"]
    assert by_r[2] == ["D"]


def test_shortest_path_distance():
    g = CameraGraph()
    for c in "ABCDE":
        g.add_node(_make_node(c))
    g.add_edge(_make_edge("A", "B"))
    g.add_edge(_make_edge("B", "C"))
    g.add_edge(_make_edge("C", "D"))
    g.add_edge(_make_edge("D", "E"))

    assert g.shortest_path_distance("A", "A") == 0
    assert g.shortest_path_distance("A", "B") == 1
    assert g.shortest_path_distance("A", "C") == 2
    assert g.shortest_path_distance("A", "E") == 4


def test_shortest_path_no_path():
    g = CameraGraph()
    g.add_node(_make_node("A"))
    g.add_node(_make_node("B"))
    # No edge
    assert g.shortest_path_distance("A", "B") == -1


# --- Validation ---

def test_validate_empty_graph():
    g = CameraGraph()
    assert g.validate() == []


def test_validate_duplicate_names():
    g = CameraGraph()
    g.add_node(CameraNodeConfig(camera_id="A", name="Lobby", source=0))
    g.add_node(CameraNodeConfig(camera_id="B", name="Lobby", source=1))
    errors = g.validate()
    assert any("Duplicate camera name" in e for e in errors)


def test_validate_negative_transition_time():
    g = CameraGraph()
    g.add_node(_make_node("A"))
    g.add_node(_make_node("B"))
    g.add_edge(CameraEdgeConfig(
        source_camera_id="A",
        target_camera_id="B",
        expected_min_transition_s=-1.0,
    ))
    errors = g.validate()
    assert any("Negative" in e for e in errors)


# --- Serialization ---

def test_to_dict_and_from_dict_roundtrip():
    g = CameraGraph()
    g.add_node(_make_node("A"))
    g.add_node(_make_node("B"))
    g.add_edge(_make_edge("A", "B", edge_type=EdgeType.OVERLAP))
    g.background_map = "floor1.png"

    data = g.to_dict()
    g2 = CameraGraph.from_dict(data)

    assert g2.node_count() == 2
    assert g2.edge_count() == 1
    assert g2.get_node("A") is not None
    assert g2.get_edge("A", "B").edge_type == EdgeType.OVERLAP
    assert g2.background_map == "floor1.png"


def test_save_and_load(tmp_path):
    g = CameraGraph()
    g.add_node(_make_node("cam1"))
    g.add_node(_make_node("cam2"))
    g.add_edge(_make_edge("cam1", "cam2"))

    path = tmp_path / "test_graph.json"
    g.save(path)

    g2 = CameraGraph.load(path)
    assert g2.node_count() == 2
    assert g2.edge_count() == 1
    assert "cam2" in g2.get_neighbors("cam1")


def test_load_missing_file(tmp_path):
    g = CameraGraph.load(tmp_path / "nonexistent.json")
    assert g.node_count() == 0
