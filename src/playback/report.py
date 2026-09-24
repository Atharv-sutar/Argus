"""Generates a text-based route report."""

import logging
from typing import List

from src.playback.route_recorder import RouteEvent

logger = logging.getLogger(__name__)

class RouteReporter:
    """Generates a human-readable text report of the target's journey."""

    def __init__(self, output_path: str = "report.txt"):
        self.output_path = output_path

    def generate_report(self, events: List[RouteEvent]) -> bool:
        if not events:
            logger.warning("[REPORTER] No events to report.")
            return False

        try:
            with open(self.output_path, "w") as f:
                f.write("TARGET ROUTE REPORT\n")
                f.write("===================\n\n")

                last_camera = None
                last_exit = None

                for event in sorted(events, key=lambda e: e.timestamp_ms):
                    ts_sec = event.timestamp_ms / 1000.0
                    mins = int(ts_sec // 60)
                    secs = int(ts_sec % 60)
                    ts_str = f"{mins:02d}:{secs:02d}"

                    if event.event_type in ("enter", "handoff", "correction"):
                        if last_exit is not None and last_camera is not None and last_camera != event.camera_id:
                            gap = (event.timestamp_ms - last_exit) / 1000.0
                            f.write(f"    [IN TRANSIT] {gap:.1f}s gap between {last_camera} and {event.camera_id}\n")
                        
                        action = "Entered"
                        if event.event_type == "handoff":
                            action = "Handoff to"
                        elif event.event_type == "correction":
                            action = "Human Corrected on"

                        f.write(f"[{ts_str}] {action} {event.camera_id} (Track #{event.track_id})\n")
                        last_camera = event.camera_id

                    elif event.event_type in ("exit", "lost"):
                        f.write(f"[{ts_str}] Exited/Lost on {event.camera_id}\n")
                        last_exit = event.timestamp_ms

            logger.info(f"[REPORTER] Saved route report to {self.output_path}")
            return True
        except Exception as e:
            logger.error(f"[REPORTER] Failed to write report: {e}")
            return False
