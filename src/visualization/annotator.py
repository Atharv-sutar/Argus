"""Visualization module for rendering bounding boxes, track IDs, target locks, and metrics."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.core.types import Target, TargetState, TrackResult


def _get_color(track_id: int) -> Tuple[int, int, int]:
    """Generates a stable, visually distinct BGR color from track ID."""
    np.random.seed(track_id * 101 % 2**32)
    color = np.random.randint(64, 235, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))


class FrameAnnotator:
    """Renders overlays (boxes, IDs, target highlight, FPS, camera metadata) on image frames."""

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

    def _draw_target_brackets(
        self,
        canvas: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        length: int = 15,
        thickness: int = 3,
    ) -> None:
        """Draws corner brackets around a target bounding box for a tactical lock appearance."""
        # Top-left
        cv2.line(canvas, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(canvas, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-right
        cv2.line(canvas, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(canvas, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-left
        cv2.line(canvas, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(canvas, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-right
        cv2.line(canvas, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(canvas, (x2, y2), (x2, y2 - length), color, thickness)

    def annotate(
        self,
        frame: np.ndarray,
        track_result: Optional[TrackResult] = None,
        target: Optional[Target] = None,
        fps: Optional[float] = None,
        camera_id: str = "camera_0",
    ) -> np.ndarray:
        """
        Draw annotations directly onto a copy of the input frame.

        Args:
            frame: Input BGR image array.
            track_result: Tracks to draw.
            target: Currently selected target state (if any).
            fps: Current processing frames-per-second.
            camera_id: Camera identifier label.

        Returns:
            np.ndarray: Annotated frame.
        """
        if cv2 is None or frame is None:
            return frame

        canvas = frame.copy()
        target_track_id = target.track_id if (target and target.state != TargetState.UNSELECTED) else None

        # 1. Draw regular track bounding boxes and IDs
        if track_result is not None and self.draw_boxes:
            for track in track_result.tracks:
                is_target = (target_track_id is not None and track.track_id == target_track_id)
                if is_target:
                    continue  # Will be drawn with special target styling below

                x1, y1, x2, y2 = map(int, track.box.as_xyxy())
                color = _get_color(track.track_id)

                # Draw standard bounding box
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

                    cv2.rectangle(canvas, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
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

        # 2. Draw Target Highlight (if active)
        if target is not None and target.state != TargetState.UNSELECTED:
            target_color = (0, 215, 255)  # Vibrant Gold / Yellow
            if target.state == TargetState.LOST:
                target_color = (0, 100, 255)  # Amber / Warning Orange

            # Find target track or use last known box
            target_box = None
            if track_result is not None and target.track_id is not None:
                for track in track_result.tracks:
                    if track.track_id == target.track_id:
                        target_box = track.box
                        break

            if target_box is None and target.last_known_box is not None:
                target_box = target.last_known_box

            if target_box is not None:
                tx1, ty1, tx2, ty2 = map(int, target_box.as_xyxy())
                cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), target_color, self.box_thickness + 1)
                self._draw_target_brackets(canvas, tx1, ty1, tx2, ty2, target_color)

                status_text = f"[TARGET ID: {target.track_id}] {target.state.value}"
                if target.state == TargetState.LOST and target.lost_duration_ms > 0:
                    status_text += f" ({target.lost_duration_ms / 1000.0:.1f}s)"

                (stw, sth), sbase = cv2.getTextSize(
                    status_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale + 0.1, 2
                )
                stag_y1 = max(0, ty1 - sth - sbase - 6)
                stag_y2 = ty1
                stag_x2 = min(canvas.shape[1], tx1 + stw + 8)

                cv2.rectangle(canvas, (tx1, stag_y1), (stag_x2, stag_y2), target_color, -1)
                cv2.putText(
                    canvas,
                    status_text,
                    (tx1 + 4, stag_y2 - sbase - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale + 0.1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        # 3. Draw HUD Info (Camera label, Track count, Target status, FPS)
        hud_lines = [f"Cam: {camera_id}"]
        if track_result is not None:
            hud_lines.append(f"Visible: {track_result.count}")
        if target is not None and target.state != TargetState.UNSELECTED:
            hud_lines.append(f"Target: ID {target.track_id} [{target.state.value}]")
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
                (0, 255, 128) if "Target:" not in line else (0, 215, 255),
                1,
                cv2.LINE_AA,
            )
            y_offset += 25

        return canvas
