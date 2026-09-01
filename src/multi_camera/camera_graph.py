"""Camera topology graph: nodes, edges, adjacency queries, and distance calculations."""

from __future__ import annotations

import json
import logging
import heapq
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.core.multi_camera_types import (
    CameraEdgeConfig,
    CameraNodeConfig,
    EdgeDirection,
)

logger = logging.getLogger(__name__)


class CameraGraph:
    """
    Represents the surveillance camera network as a directed graph.

    Nodes are cameras. Edges represent physical/operational relationships
    (overlap, adjacent, travel) with optional transition timing constraints.

    Provides topology queries (neighbors at radius N, shortest path distance)
    used by the search manager to determine which cameras to activate.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, CameraNodeConfig] = {}
        # Adjacency list: camera_id -> set of neighbor camera_ids
        # Stores the effective directed edges after resolving bidirectional edges.
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        # Edge configs keyed by (source_id, target_id)
        self._edges: Dict[Tuple[str, str], CameraEdgeConfig] = {}
        # Optional background map image path
        self.background_map: Optional[str] = None

    # --- Node operations ---

    def add_node(self, config: CameraNodeConfig) -> None:
        """Add or update a camera node."""
        self._nodes[config.camera_id] = config
        if config.camera_id not in self._adj:
            self._adj[config.camera_id] = set()

    def remove_node(self, camera_id: str) -> None:
        """Remove a camera node and all its edges."""
        if camera_id not in self._nodes:
            return

        # Remove all edges involving this camera
        edges_to_remove = [
            key for key in self._edges
            if camera_id in key
        ]
        for key in edges_to_remove:
            self._remove_directed_edge(key[0], key[1])

        # Remove from adjacency list
        self._adj.pop(camera_id, None)
        for neighbors in self._adj.values():
            neighbors.discard(camera_id)

        del self._nodes[camera_id]

    def get_node(self, camera_id: str) -> Optional[CameraNodeConfig]:
        return self._nodes.get(camera_id)

    def all_camera_ids(self) -> List[str]:
        return list(self._nodes.keys())

    def node_count(self) -> int:
        return len(self._nodes)

    # --- Edge operations ---

    def add_edge(self, config: CameraEdgeConfig) -> None:
        """
        Add a connection between two cameras.

        Resolves EdgeDirection into the appropriate directed edges in the adjacency list.
        """
        src = config.source_camera_id
        tgt = config.target_camera_id

        if src not in self._nodes:
            raise ValueError(f"Source camera '{src}' not in graph")
        if tgt not in self._nodes:
            raise ValueError(f"Target camera '{tgt}' not in graph")
        if src == tgt:
            raise ValueError(f"Cannot create self-edge for camera '{src}'")

        # Store the canonical edge config
        self._edges[(src, tgt)] = config

        # Build directed adjacency based on direction
        if config.direction == EdgeDirection.BIDIRECTIONAL:
            self._adj[src].add(tgt)
            self._adj[tgt].add(src)
        elif config.direction == EdgeDirection.A_TO_B:
            self._adj[src].add(tgt)
            self._adj[tgt].discard(src)
        elif config.direction == EdgeDirection.B_TO_A:
            self._adj[tgt].add(src)
            self._adj[src].discard(tgt)

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Remove the edge between two cameras (both directions if bidirectional)."""
        self._remove_directed_edge(source_id, target_id)

    def _remove_directed_edge(self, source_id: str, target_id: str) -> None:
        config = self._edges.pop((source_id, target_id), None)
        # Also check reversed key for bidirectional edges stored with swapped IDs
        if config is None:
            config = self._edges.pop((target_id, source_id), None)

        if config is not None:
            self._adj.get(source_id, set()).discard(target_id)
            self._adj.get(target_id, set()).discard(source_id)

    def get_edge(self, source_id: str, target_id: str) -> Optional[CameraEdgeConfig]:
        """Get edge config between two cameras (checks both directions)."""
        edge = self._edges.get((source_id, target_id))
        if edge is None:
            edge = self._edges.get((target_id, source_id))
        return edge

    def get_edges_for_camera(self, camera_id: str) -> List[CameraEdgeConfig]:
        """Get all edge configs involving a specific camera."""
        return [
            config for key, config in self._edges.items()
            if camera_id in key
        ]

    def edge_count(self) -> int:
        return len(self._edges)

    # --- Topology queries ---

    def get_neighbors(self, camera_id: str, radius: int = 1) -> List[str]:
        """
        Get all camera IDs reachable within `radius` hops from `camera_id`.

        Uses BFS. Returns cameras at exactly radius=1..radius, excluding the source.
        Only traverses enabled edges and enabled cameras.
        """
        if camera_id not in self._nodes:
            return []

        visited: Set[str] = {camera_id}
        result: List[str] = []
        frontier: List[str] = [camera_id]

        for _ in range(radius):
            next_frontier: List[str] = []
            for node_id in frontier:
                for neighbor_id in self._adj.get(node_id, set()):
                    if neighbor_id in visited:
                        continue
                    # Check that the neighbor camera is enabled
                    neighbor_node = self._nodes.get(neighbor_id)
                    if neighbor_node is None or not neighbor_node.enabled:
                        continue
                    # Check that the edge is enabled
                    edge = self.get_edge(node_id, neighbor_id)
                    if edge is not None and not edge.enabled:
                        continue
                    visited.add(neighbor_id)
                    result.append(neighbor_id)
                    next_frontier.append(neighbor_id)
            frontier = next_frontier

        return result

    def get_neighbors_by_radius(
        self, camera_id: str, max_radius: int
    ) -> Dict[int, List[str]]:
        """
        Get neighbors grouped by hop distance from camera_id.

        Returns: {1: [cam_b, cam_c], 2: [cam_d, cam_e], ...}
        """
        if camera_id not in self._nodes:
            return {}

        visited: Set[str] = {camera_id}
        result: Dict[int, List[str]] = {}
        frontier: List[str] = [camera_id]

        for r in range(1, max_radius + 1):
            next_frontier: List[str] = []
            level_cameras: List[str] = []
            for node_id in frontier:
                for neighbor_id in self._adj.get(node_id, set()):
                    if neighbor_id in visited:
                        continue
                    neighbor_node = self._nodes.get(neighbor_id)
                    if neighbor_node is None or not neighbor_node.enabled:
                        continue
                    edge = self.get_edge(node_id, neighbor_id)
                    if edge is not None and not edge.enabled:
                        continue
                    visited.add(neighbor_id)
                    level_cameras.append(neighbor_id)
                    next_frontier.append(neighbor_id)
            if level_cameras:
                result[r] = level_cameras
            frontier = next_frontier

        return result

    def shortest_path_distance(self, source_id: str, target_id: str) -> int:
        """
        Compute shortest hop distance between two cameras via BFS.

        Returns -1 if no path exists.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return -1
        if source_id == target_id:
            return 0

        visited: Set[str] = {source_id}
        queue: deque[Tuple[str, int]] = deque([(source_id, 0)])

        while queue:
            node_id, dist = queue.popleft()
            for neighbor_id in self._adj.get(node_id, set()):
                if neighbor_id == target_id:
                    return dist + 1
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, dist + 1))

        return -1

    def shortest_path_min_time(self, source_id: str, target_id: str) -> float:
        """
        Compute the minimum physical transit time between two cameras using Dijkstra's algorithm.
        Returns -1.0 if no path exists.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return -1.0
        if source_id == target_id:
            return 0.0

        # Dijkstra priority queue: (cumulative_time, node_id)
        pq: List[Tuple[float, str]] = [(0.0, source_id)]
        min_times: Dict[str, float] = {source_id: 0.0}

        while pq:
            current_time, node_id = heapq.heappop(pq)

            if current_time > min_times.get(node_id, float('inf')):
                continue

            if node_id == target_id:
                return current_time

            for neighbor_id in self._adj.get(node_id, set()):
                edge = self.get_edge(node_id, neighbor_id)
                if not edge or not edge.enabled:
                    continue
                neighbor_node = self._nodes.get(neighbor_id)
                if not neighbor_node or not neighbor_node.enabled:
                    continue

                transit_s = edge.expected_min_transition_s or 0.0
                new_time = current_time + transit_s

                if new_time < min_times.get(neighbor_id, float('inf')):
                    min_times[neighbor_id] = new_time
                    heapq.heappush(pq, (new_time, neighbor_id))

        return -1.0

    # --- Validation ---

    def validate(self) -> List[str]:
        """
        Validate the graph configuration.

        Returns a list of error messages. Empty list means valid.
        """
        errors: List[str] = []

        # Check for unique camera IDs (enforced by dict, but check names)
        names = [n.name for n in self._nodes.values()]
        seen_names: Set[str] = set()
        for name in names:
            if name in seen_names:
                errors.append(f"Duplicate camera name: '{name}'")
            seen_names.add(name)

        # Check that all edge endpoints reference existing cameras
        for (src, tgt), config in self._edges.items():
            if src not in self._nodes:
                errors.append(f"Edge references non-existent camera: '{src}'")
            if tgt not in self._nodes:
                errors.append(f"Edge references non-existent camera: '{tgt}'")
            if src == tgt:
                errors.append(f"Self-edge on camera: '{src}'")

            # Validate transition times are sane
            if config.expected_min_transition_s is not None and config.expected_min_transition_s < 0:
                errors.append(f"Negative min transition time on edge {src}->{tgt}")
            if (config.expected_min_transition_s is not None
                    and config.expected_typical_transition_s is not None
                    and config.expected_min_transition_s > config.expected_typical_transition_s):
                errors.append(
                    f"Min transition > typical transition on edge {src}->{tgt}"
                )
            if (config.expected_typical_transition_s is not None
                    and config.expected_max_transition_s is not None
                    and config.expected_typical_transition_s > config.expected_max_transition_s):
                errors.append(
                    f"Typical transition > max transition on edge {src}->{tgt}"
                )

        # Check that enabled cameras have valid sources
        for cam in self._nodes.values():
            if cam.enabled and cam.source is None:
                errors.append(f"Enabled camera '{cam.camera_id}' has no source")

        # Check that at least one camera exists when edges exist
        if self._edges and not self._nodes:
            errors.append("Edges exist but no cameras defined")

        return errors

    # --- Serialization ---

    def to_dict(self) -> dict:
        """Serialize the entire graph to a JSON-compatible dictionary."""
        return {
            "version": 1,
            "cameras": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges.values()],
            "background_map": self.background_map,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraGraph:
        """Deserialize a graph from a dictionary."""
        graph = cls()
        graph.background_map = data.get("background_map")

        for cam_data in data.get("cameras", []):
            graph.add_node(CameraNodeConfig.from_dict(cam_data))

        for edge_data in data.get("edges", []):
            try:
                graph.add_edge(CameraEdgeConfig.from_dict(edge_data))
            except ValueError as e:
                logger.warning(f"Skipping invalid edge during load: {e}")

        return graph

    def save(self, path: Union[str, Path]) -> None:
        """Save the graph configuration to a JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Camera graph saved to {file_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> CameraGraph:
        """Load a graph configuration from a JSON file."""
        file_path = Path(path)
        if not file_path.is_file():
            logger.info(f"No graph file at {file_path}, returning empty graph")
            return cls()

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = cls.from_dict(data)
        logger.info(
            f"Camera graph loaded from {file_path}: "
            f"{graph.node_count()} cameras, {graph.edge_count()} edges"
        )
        return graph
