"""Observable search state for the multi-camera progressive search system."""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Set

from src.core.config import SearchConfig
from src.core.multi_camera_types import SearchProgress, SearchState
from src.multi_camera.camera_graph import CameraGraph

logger = logging.getLogger(__name__)


class SearchStateManager:
    """
    Manages the observable state of a multi-camera target search.

    Tracks:
    - Current active camera
    - Search radius
    - Search start time
    - Which cameras are actively being searched
    - Which cameras have already been searched
    - Search lifecycle state

    Does NOT own camera activation/deactivation — that is the SearchManager's job.
    """

    def __init__(self, config: SearchConfig) -> None:
        self._config = config
        self._state: SearchState = SearchState.IDLE
        self._origin_camera: Optional[str] = None
        self._search_radius: int = 0
        self._search_start_time: float = 0.0
        self._last_expansion_time: float = 0.0
        self._active_cameras: Set[str] = set()
        self._pending_cameras: Dict[str, float] = {}
        self._searched_cameras: Set[str] = set()
        self._candidate_camera: Optional[str] = None
        self._candidate_confirmations: int = 0

    @property
    def state(self) -> SearchState:
        return self._state

    @property
    def is_searching(self) -> bool:
        return self._state in (SearchState.SEARCHING, SearchState.EXPANDED)

    @property
    def search_radius(self) -> int:
        return self._search_radius

    @property
    def origin_camera(self) -> Optional[str]:
        return self._origin_camera

    def start_search(self, from_camera: str) -> None:
        """Begin a new search from the camera where target was lost."""
        self._state = SearchState.SEARCHING
        self._origin_camera = from_camera
        self._search_radius = self._config.initial_radius
        self._search_start_time = time.time()
        self._last_expansion_time = self._search_start_time
        self._active_cameras.clear()
        self._pending_cameras.clear()
        self._searched_cameras.clear()
        self._candidate_camera = None
        self._candidate_confirmations = 0
        logger.info(
            f"Search started from camera '{from_camera}' at radius {self._search_radius}"
        )

    def expand_radius(self) -> int:
        """
        Expand the search radius by the configured increment.

        Returns the new radius.
        """
        self._search_radius += self._config.radius_increment
        self._last_expansion_time = time.time()
        if self._search_radius > self._config.initial_radius:
            self._state = SearchState.EXPANDED
        logger.info(f"Search radius expanded to {self._search_radius}")
        return self._search_radius

    def should_expand(self) -> bool:
        """Check if the per-radius timeout has elapsed and expansion is allowed."""
        if not self.is_searching:
            return False
        if self._search_radius >= self._config.max_radius:
            return False
        
        elapsed = time.time() - self._last_expansion_time
        timeout_s = self._config.per_radius_timeout_s
        
        # Heuristic: if we still have pending cameras for the current radius,
        # we should give them a chance to activate and be searched, unless we've
        # waited for an excessively long time (e.g. 2x the normal timeout).
        if self._pending_cameras:
            if elapsed < timeout_s * 2.0:
                return False
                
        return elapsed >= timeout_s

    def add_pending_camera(self, camera_id: str, activation_delay_s: float) -> None:
        """Register a camera to be activated after a physical travel delay."""
        activation_time = self._search_start_time + activation_delay_s
        self._pending_cameras[camera_id] = activation_time

    def pop_ready_pending_cameras(self, current_time: float) -> List[str]:
        """Return and remove pending cameras that are now ready to be activated."""
        ready = []
        for cam_id, activation_time in list(self._pending_cameras.items()):
            if current_time >= activation_time:
                ready.append(cam_id)
                del self._pending_cameras[cam_id]
        return ready

    def is_timed_out(self) -> bool:
        """Check if the total recovery timeout has been exceeded."""
        if not self.is_searching:
            return False
        elapsed = time.time() - self._search_start_time
        return elapsed >= self._config.total_recovery_timeout_s

    def elapsed_s(self) -> float:
        """Seconds since search started."""
        if self._search_start_time <= 0:
            return 0.0
        return time.time() - self._search_start_time

    def add_active_camera(self, camera_id: str) -> None:
        self._active_cameras.add(camera_id)

    def remove_active_camera(self, camera_id: str) -> None:
        self._active_cameras.discard(camera_id)
        self._searched_cameras.add(camera_id)

    def mark_found(self, camera_id: str) -> None:
        """Mark that the target has been confirmed on a camera."""
        self._state = SearchState.TARGET_FOUND
        self._candidate_camera = camera_id
        logger.info(f"Target found on camera '{camera_id}'")

    def mark_timeout(self) -> None:
        """Mark the search as timed out."""
        self._state = SearchState.TIMEOUT
        logger.info("Search timed out — target not found")

    def reset(self) -> None:
        """Reset to idle state."""
        self._state = SearchState.IDLE
        self._origin_camera = None
        self._search_radius = 0
        self._search_start_time = 0.0
        self._last_expansion_time = 0.0
        self._active_cameras.clear()
        self._searched_cameras.clear()
        self._candidate_camera = None
        self._candidate_confirmations = 0

    def record_candidate_confirmation(self, camera_id: str) -> int:
        """
        Record a confirmation frame for a candidate camera.

        Returns the total confirmation count.
        """
        if self._candidate_camera != camera_id:
            self._candidate_camera = camera_id
            self._candidate_confirmations = 0
        self._candidate_confirmations += 1
        return self._candidate_confirmations

    def reset_candidate(self) -> None:
        """Reset candidate tracking (candidate failed confirmation)."""
        self._candidate_camera = None
        self._candidate_confirmations = 0

    def get_progress(self, graph: Optional[CameraGraph] = None) -> SearchProgress:
        """Build an observable progress snapshot."""
        standby: List[str] = []
        if graph is not None and self._origin_camera is not None and self.is_searching:
            # Cameras beyond current radius that might be activated later
            all_reachable = set(
                graph.get_neighbors(self._origin_camera, self._config.max_radius)
            )
            standby = [
                c for c in all_reachable
                if c not in self._active_cameras and c not in self._searched_cameras
            ]

        return SearchProgress(
            current_camera_id=self._origin_camera,
            search_radius=self._search_radius,
            elapsed_s=self.elapsed_s(),
            active_cameras=sorted(self._active_cameras),
            standby_cameras=sorted(standby),
            searched_cameras=sorted(self._searched_cameras),
            state=self._state,
            candidate_camera_id=self._candidate_camera,
            candidate_confirmation_count=self._candidate_confirmations,
        )
