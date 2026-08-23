"""Visualization module for rendering bounding boxes, track IDs, and metrics."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.core.types import TrackResult


def _get_color(track_id: int) -> Tuple[int, int, int]:
    """Generates a stable, visually distinct BGR color from track ID."""
    np.random.seed(track_id * 101 % 2**32)
    color = np.random.randint(64, 235, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))


class FrameAnnotator:
    """Renders overlays (boxes, IDs, FPS, camera metadata) on image frames."""

    def __init__(
        self,
        draw_fps: bool = True,
        draw_boxes: bool = True,
        draw_ids: bool = True,
        box_thickness: int = 2,
        font_scale: float = 0.6,
    ) -> None:
        self.draw_fps = draw_fps
        self.draw_boxes = draw_boxes
        self.draw_ids = draw_ids
        self.box_thickness = box_thickness
        self.font_scale = font_scale

    def annotate(
        self,
        frame: np.ndarray,
        track_result: Optional[TrackResult] = None,
        fps: Optional[float] = None,
        camera_id: str = "camera_0",
    ) -> np.ndarray:
        """
        Draw annotations directly onto a copy of the input frame.

        Args:
            frame: Input BGR image array.
            track_result: Tracks to draw.
            fps: Current processing frames-per-second.
            camera_id: Camera identifier label.

        Returns:
            np.ndarray: Annotated frame.
        """
        if cv2 is None or frame is None:
            return frame

        canvas = frame.copy()

        # 1. Draw track bounding boxes and IDs
        if track_result is not None and self.draw_boxes:
            for track in track_result.tracks:
                x1, y1, x2, y2 = map(int, track.box.as_xyxy())
                color = _get_color(track.track_id)

                # Draw bounding box rectangle
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.box_thickness)

                # Draw track label tag
                if self.draw_ids:
                    label = f"ID: {track.track_id} ({track.confidence:.2f})"
                    (tw, th), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                    )
                    tag_y1 = max(0, y1 - th - baseline - 4)
                    tag_y2 = y1
                    tag_x2 = min(canvas.shape[1], x1 + tw + 6)

                    # Background rectangle for text
                    cv2.rectangle(
                        canvas,
                        (x1, tag_y1),
                        (tag_x2, tag_y2),
                        color,
                        -1
                    )
                    # Text string
                    cv2.putText(
                        canvas,
                        label,
                        (x1 + 3, tag_y2 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # 2. Draw HUD Info (Camera label, Track count, FPS)
        hud_lines = [f"Cam: {camera_id}"]
        if track_result is not None:
            hud_lines.append(f"Targets: {track_result.count}")
        if self.draw_fps and fps is not None:
            hud_lines.append(f"FPS: {fps:.1f}")

        y_offset = 25
        for line in hud_lines:
            cv2.putText(
                canvas,
                line,
                (12, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                line,
                (12, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 128),
                1,
                cv2.LINE_AA,
            )
            y_offset += 25

        return canvas
