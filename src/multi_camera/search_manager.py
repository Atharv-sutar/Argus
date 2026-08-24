"""Search orchestration: expand/contract search radius, manage timeouts, activate/deactivate camera AI."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.core.config import SearchConfig
from src.core.multi_camera_types import SearchProgress, SearchState
from src.multi_camera.camera_graph import CameraGraph
from src.multi_camera.camera_priority import CameraPrioritizer
from src.multi_camera.search_state import SearchStateManager

logger = logging.getLogger(__name__)


class SearchManager:
    """
    Orchestrates the progressive multi-camera search for a lost target.

    Responsibilities:
    - Determines which cameras to activate when a target is lost
    - Manages search radius expansion after configurable timeouts
    - Tracks candidate confirmations across multiple frames
    - Reports search progress for UI visualization

    Does NOT:
    - Run detection/tracking/ReID itself (that's the pipeline's job)
    - Directly manage camera connections (that's the CameraNode's job)
    - Make identity decisions (that's the IdentityManager's job)
    """

    def __init__(self, graph: CameraGraph, config: SearchConfig) -> None:
        self._graph = graph
        self._config = config
        self._state = SearchStateManager(config)

    @property
    def is_searching(self) -> bool:
        return self._state.is_searching

    @property
    def search_state(self) -> SearchState:
        return self._state.state

    def on_target_lost(self, camera_id: str) -> List[Tuple[str, float]]:
        """
        Called when the target is lost on a camera.

        Starts a new search and returns the list of cameras to activate,
        sorted by priority (highest first).

        Args:
            camera_id: The camera where the target was lost.

        Returns:
            List of (camera_id, priority_score) to activate for search.
        """
        self._state.start_search(camera_id)

        # Get initial neighbors at configured initial radius
        neighbors = self._graph.get_neighbors(camera_id, self._config.initial_radius)

        # Score and rank neighbors
        ranked = self._rank_cameras(camera_id, neighbors, elapsed_s=0.0)

        # Register them as active
        for cam_id, _ in ranked:
            self._state.add_active_camera(cam_id)

        logger.info(
            f"Search initiated from '{camera_id}': "
            f"activating {len(ranked)} cameras at radius {self._config.initial_radius}"
        )
        return ranked

    def tick(self, elapsed_since_loss_s: float) -> Optional[List[Tuple[str, float]]]:
        """
        Called periodically during search to check for radius expansion or timeout.

        Args:
            elapsed_since_loss_s: Total seconds since the target was lost.

        Returns:
            - List of new (camera_id, priority_score) if radius expanded
            - Empty list if timed out (search should stop)
            - None if no change needed
        """
        if not self._state.is_searching:
            return None

        # Check total timeout
        if self._state.is_timed_out():
            self._state.mark_timeout()
            logger.info("Search timeout reached — stopping search")
            return []

        # Check per-radius timeout for expansion
        if self._state.should_expand():
            old_radius = self._state.search_radius
            new_radius = self._state.expand_radius()

            if new_radius > self._config.max_radius:
                self._state.mark_timeout()
                return []

            origin = self._state.origin_camera
            if origin is None:
                return None

            # Get cameras at the new radius level that haven't been searched yet
            by_radius = self._graph.get_neighbors_by_radius(origin, new_radius)
            new_cameras: List[str] = []
            for r in range(old_radius + 1, new_radius + 1):
                for cam_id in by_radius.get(r, []):
                    if cam_id not in self._state._searched_cameras and cam_id not in self._state._active_cameras:
                        new_cameras.append(cam_id)

            ranked = self._rank_cameras(origin, new_cameras, elapsed_since_loss_s)

            for cam_id, _ in ranked:
                self._state.add_active_camera(cam_id)

            logger.info(
                f"Search expanded to radius {new_radius}: "
                f"activating {len(ranked)} new cameras"
            )
            return ranked

        return None

    def on_candidate_found(
        self, camera_id: str, similarity: float
    ) -> bool:
        """
        Called when a ReID candidate is found on a camera.

        Tracks confirmations and returns True if the candidate is confirmed
        (similarity above threshold for N consecutive frames).

        Args:
            camera_id: Camera where the candidate was found.
            similarity: ReID similarity score.

        Returns:
            True if the candidate is confirmed (handoff should proceed).
        """
        count = self._state.record_candidate_confirmation(camera_id)
        logger.debug(
            f"Candidate on '{camera_id}': similarity={similarity:.3f}, "
            f"confirmation {count}/{self._config.confirmation_frames}"
        )
        if count >= self._config.confirmation_frames:
            self._state.mark_found(camera_id)
            return True
        return False

    def on_candidate_lost(self, camera_id: str) -> None:
        """Called when a candidate fails ReID on a subsequent frame."""
        if self._state._candidate_camera == camera_id:
            logger.debug(f"Candidate on '{camera_id}' lost — resetting confirmations")
            self._state.reset_candidate()

    def on_search_complete(self, camera_id: str) -> None:
        """
        Called when a camera finishes its search (either found target or exhausted).

        Deactivates the camera from active search.
        """
        self._state.remove_active_camera(camera_id)

    def reset(self) -> None:
        """Reset search state to idle."""
        self._state.reset()

    def get_progress(self) -> SearchProgress:
        """Get the current search progress snapshot for UI display."""
        return self._state.get_progress(self._graph)

    def _rank_cameras(
        self,
        origin_camera: str,
        camera_ids: List[str],
        elapsed_s: float,
    ) -> List[Tuple[str, float]]:
        """Rank candidate cameras by priority score, descending."""
        scored: List[Tuple[str, float]] = []
        for cam_id in camera_ids:
            distance = self._graph.shortest_path_distance(origin_camera, cam_id)
            if distance < 0:
                continue
            edge = self._graph.get_edge(origin_camera, cam_id)
            score = CameraPrioritizer.score(distance, elapsed_s, edge)
            scored.append((cam_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
