"""Camera priority scoring for ranking search candidates."""

from __future__ import annotations

from typing import Optional

from src.core.multi_camera_types import CameraEdgeConfig, EdgeType
from src.multi_camera.transition import TransitionValidator


class CameraPrioritizer:
    """
    Scores candidate cameras for search priority.

    Initial implementation uses:
    - Graph distance (closer = higher priority)
    - Edge type (OVERLAP > ADJACENT > TRAVEL)
    - Transition timing plausibility

    Extensible to incorporate:
    - Target movement direction
    - Historical transition data
    - Expected travel time
    - Camera reliability
    - ReID similarity from previous sightings
    """

    @staticmethod
    def score(
        graph_distance: int,
        elapsed_since_loss_s: float,
        edge_config: Optional[CameraEdgeConfig] = None,
    ) -> float:
        """
        Compute a priority score for a candidate camera.

        Higher score = higher priority for search activation.

        Args:
            graph_distance: Hop count from the camera where target was lost.
            elapsed_since_loss_s: Seconds since target was lost.
            edge_config: Edge configuration (if direct neighbor).

        Returns:
            float: Priority score (higher = more urgent to search).
        """
        # Base score: inversely proportional to distance
        if graph_distance <= 0:
            return 0.0
        distance_score = 1.0 / graph_distance

        # Edge type bonus (only for direct neighbors)
        type_bonus = 0.0
        if edge_config is not None:
            if edge_config.edge_type == EdgeType.OVERLAP:
                type_bonus = 0.3  # Highest priority — person may already be visible
            elif edge_config.edge_type == EdgeType.ADJACENT:
                type_bonus = 0.15
            elif edge_config.edge_type == EdgeType.TRAVEL:
                type_bonus = 0.0

        # Transition timing plausibility
        timing_score = 1.0
        if edge_config is not None:
            timing_score = TransitionValidator.is_plausible(edge_config, elapsed_since_loss_s)

        return (distance_score + type_bonus) * timing_score
