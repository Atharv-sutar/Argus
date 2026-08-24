"""Transition timing validation between cameras."""

from __future__ import annotations

from src.core.multi_camera_types import CameraEdgeConfig


class TransitionValidator:
    """
    Evaluates temporal plausibility of a target appearing in a neighbor camera
    based on configured transition timing constraints on the edge.

    Returns a plausibility score [0.0, 1.0]:
    - 1.0 = perfect timing match
    - 0.0 = implausible (too early or far too late)
    - intermediate values for edge cases

    When no timing constraints are configured, returns 1.0 (always plausible).
    """

    @staticmethod
    def is_plausible(edge: CameraEdgeConfig, elapsed_since_loss_s: float) -> float:
        """
        Compute plausibility of a transition given elapsed time since target loss.

        Args:
            edge: The edge configuration with optional timing constraints.
            elapsed_since_loss_s: Seconds since target was lost on the source camera.

        Returns:
            float: Plausibility score in [0.0, 1.0].
        """
        min_t = edge.expected_min_transition_s
        typical_t = edge.expected_typical_transition_s
        max_t = edge.expected_max_transition_s

        # If no timing constraints are configured, always plausible
        if min_t is None and typical_t is None and max_t is None:
            return 1.0

        t = elapsed_since_loss_s

        # Too early — physically impossible
        if min_t is not None and t < min_t:
            # Allow small tolerance (50% of min) for detection lag
            if t < min_t * 0.5:
                return 0.0
            return 0.5  # Marginal — possible with fast movement

        # Within expected range
        if typical_t is not None:
            if min_t is not None and t <= typical_t:
                return 1.0
            elif t <= typical_t:
                return 1.0

        # Past typical but within max
        if max_t is not None:
            if t <= max_t:
                if typical_t is not None:
                    # Linear decay from 1.0 at typical to 0.5 at max
                    range_t = max_t - typical_t
                    if range_t > 0:
                        decay = (t - typical_t) / range_t
                        return max(0.5, 1.0 - 0.5 * decay)
                return 0.8
            else:
                # Beyond max — unlikely but not impossible
                overshoot = t - max_t
                if overshoot > max_t:
                    return 0.1  # Very unlikely
                return 0.3  # Unlikely

        # Only min was specified and we're past it
        return 1.0
