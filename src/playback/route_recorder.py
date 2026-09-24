"""Records the chronological route of a target across multiple cameras."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RouteEvent:
    """A discrete event in the target's journey."""
    camera_id: str
    track_id: int
    timestamp_ms: float          # video timestamp
    event_type: str              # "enter" | "exit" | "correction" | "lost" | "handoff"
    bbox: Tuple[int, int, int, int] # [x, y, w, h]
    confidence: float            # ReID confidence at handoff
    was_human_corrected: bool    # True if human clicked to correct


class RouteRecorder:
    """
    Maintains the chronological log of target movements for Recorded Mode.
    """

    def __init__(self) -> None:
        self.events: List[RouteEvent] = []
        self._current_camera: str = ""
        self._current_track: int = -1

    def log_event(self,
                  camera_id: str,
                  track_id: int,
                  timestamp_ms: float,
                  event_type: str,
                  bbox: Tuple[int, int, int, int],
                  confidence: float = 1.0,
                  was_human_corrected: bool = False) -> None:
        """Logs a new route event."""
        event = RouteEvent(
            camera_id=camera_id,
            track_id=track_id,
            timestamp_ms=timestamp_ms,
            event_type=event_type,
            bbox=bbox,
            confidence=confidence,
            was_human_corrected=was_human_corrected
        )
        self.events.append(event)
        logger.debug(f"[ROUTE] {event_type.upper()} on '{camera_id}' at {timestamp_ms:.1f}ms")

        if event_type in ("enter", "correction", "handoff"):
            self._current_camera = camera_id
            self._current_track = track_id
        elif event_type in ("exit", "lost"):
            self._current_camera = ""
            self._current_track = -1

    def get_events(self) -> List[RouteEvent]:
        """Returns the chronological list of events."""
        return sorted(self.events, key=lambda e: e.timestamp_ms)

    def clear(self) -> None:
        """Clears the recorded route."""
        self.events.clear()
        self._current_camera = ""
        self._current_track = -1
        logger.info("[ROUTE] Cleared route history.")
        
    def save_json(self, path: str) -> None:
        """Saves the route log to a JSON file."""
        import json
        out = []
        for e in self.events:
            out.append({
                "camera_id": e.camera_id,
                "track_id": e.track_id,
                "timestamp_ms": e.timestamp_ms,
                "event_type": e.event_type,
                "bbox": e.bbox,
                "confidence": e.confidence,
                "was_human_corrected": e.was_human_corrected
            })
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"[ROUTE] Saved {len(out)} events to {path}")
