"""Generates a stitched chronological journey video from a completed route."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.playback.route_recorder import RouteEvent

logger = logging.getLogger(__name__)


class JourneyStitcher:
    """
    Creates a single contiguous video file detailing a target's journey
    across multiple cameras based on recorded route events.
    """

    def __init__(self,
                 video_sources: Dict[str, str],
                 output_path: str = "journey_video.mp4",
                 target_fps: float = 30.0,
                 resolution: Tuple[int, int] = (1280, 720)):
        """
        Args:
            video_sources: Dictionary mapping camera_id to its source video file path.
            output_path: Path where the stitched MP4 will be saved.
            target_fps: Frames per second of the output video.
            resolution: Resolution (width, height) of the output video.
        """
        self.video_sources = video_sources
        self.output_path = output_path
        self.target_fps = target_fps
        self.resolution = resolution
        self.w, self.h = resolution

    def stitch_journey(self, events: List[RouteEvent], final_time_ms: Optional[float] = None) -> bool:
        """
        Parses the sequence of RouteEvents and generates the stitched video.
        Returns True on success, False otherwise.
        """
        if not events:
            logger.warning("[STITCHER] No route events provided. Cannot stitch journey.")
            return False

        logger.info(f"[STITCHER] Starting journey stitch. Output: {self.output_path}")

        # Group events into continuous segments per camera
        segments = self._build_segments(events, final_time_ms=final_time_ms)

        if not segments:
            logger.warning("[STITCHER] No valid continuous segments found in route events.")
            return False

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.target_fps, self.resolution)

        if not out.isOpened():
            logger.error(f"[STITCHER] Failed to open VideoWriter at {self.output_path}")
            return False

        try:
            last_exit_time = None
            last_camera = None

            for seg in segments:
                cam_id = seg["camera_id"]
                start_ms = seg["start_ms"]
                end_ms = seg["end_ms"]

                # 1. Handle gap / IN TRANSIT
                if last_exit_time is not None and last_camera != cam_id:
                    gap_ms = start_ms - last_exit_time
                    if gap_ms > 1000:  # Only show IN TRANSIT for gaps > 1 second
                        self._render_transit_card(out, last_camera, cam_id, gap_ms)

                # 2. Extract frames for the segment
                success = self._extract_segment(out, cam_id, start_ms, end_ms)
                if not success:
                    logger.warning(f"[STITCHER] Failed to extract segment for {cam_id}")

                last_exit_time = end_ms
                last_camera = cam_id

            out.release()
            logger.info(f"[STITCHER] Journey video successfully saved to {self.output_path}")
            return True

        except Exception as e:
            logger.error(f"[STITCHER] Error during stitching: {e}")
            if out.isOpened():
                out.release()
            return False

    def _build_segments(self, events: List[RouteEvent], final_time_ms: Optional[float] = None) -> List[Dict]:
        """Converts discrete events into continuous time segments per camera."""
        segments = []
        current_seg = None

        for event in sorted(events, key=lambda e: e.timestamp_ms):
            if event.event_type in ("enter", "correction", "handoff"):
                if current_seg is not None:
                    # Close previous segment
                    current_seg["end_ms"] = max(event.timestamp_ms, current_seg["start_ms"] + 1500)
                    segments.append(current_seg)

                current_seg = {
                    "camera_id": event.camera_id,
                    "start_ms": event.timestamp_ms,
                    "end_ms": event.timestamp_ms + 5000  # initial estimate
                }
            elif event.event_type in ("exit", "lost"):
                if current_seg is not None and current_seg["camera_id"] == event.camera_id:
                    current_seg["end_ms"] = max(event.timestamp_ms, current_seg["start_ms"] + 1500)
                    segments.append(current_seg)
                    current_seg = None

        # Flush final segment if open
        if current_seg is not None:
            if final_time_ms is not None and final_time_ms > current_seg["start_ms"]:
                current_seg["end_ms"] = max(final_time_ms, current_seg["start_ms"] + 1500)
            segments.append(current_seg)

        # Merge adjacent segments for the same camera if contiguous
        merged = []
        for s in segments:
            if merged and merged[-1]["camera_id"] == s["camera_id"] and abs(s["start_ms"] - merged[-1]["end_ms"]) < 1000:
                merged[-1]["end_ms"] = max(merged[-1]["end_ms"], s["end_ms"])
            else:
                merged.append(s)

        return merged

    def _render_transit_card(self, out: cv2.VideoWriter, from_cam: str, to_cam: str, duration_ms: float) -> None:
        """Renders an 'IN TRANSIT' title card and writes it to the video."""
        card = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        card[:] = (20, 26, 38)  # Dark slate blue background

        # Render text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(card, "IN TRANSIT", (self.w // 2 - 120, self.h // 2 - 40), 
                    font, 1.5, (0, 215, 255), 2, cv2.LINE_AA)
        
        duration_sec = duration_ms / 1000.0
        text2 = f"From: {from_cam}  ->  To: {to_cam}"
        cv2.putText(card, text2, (self.w // 2 - 250, self.h // 2 + 20), 
                    font, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
        
        text3 = f"Elapsed Time: {duration_sec:.1f}s"
        cv2.putText(card, text3, (self.w // 2 - 150, self.h // 2 + 70), 
                    font, 0.8, (150, 150, 150), 1, cv2.LINE_AA)

        # Write card for 2 seconds (visual duration)
        frames_to_write = int(self.target_fps * 2.0)
        for _ in range(frames_to_write):
            out.write(card)

    def _extract_segment(self, out: cv2.VideoWriter, cam_id: str, start_ms: float, end_ms: float) -> bool:
        """Reads frames from the source video and writes them with overlay."""
        source_path = self.video_sources.get(cam_id)
        if not source_path or not Path(source_path).exists():
            logger.error(f"[STITCHER] Source file not found for {cam_id}: {source_path}")
            return False

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            logger.error(f"[STITCHER] Failed to open source video: {source_path}")
            return False

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Calculate start and end frames for robust seeking across all codecs
        start_frame = max(0, int((start_ms / 1000.0) * source_fps))
        end_frame = int((end_ms / 1000.0) * source_fps)
        if total_frames > 0:
            end_frame = min(end_frame, total_frames)
        if end_frame <= start_frame:
            end_frame = start_frame + int(source_fps * 3.0)  # at least 3 seconds

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)

        current_frame_idx = start_frame
        frames_written = 0

        while current_frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms <= 0.0 or pos_ms < start_ms:
                pos_ms = (current_frame_idx / source_fps) * 1000.0

            resized = cv2.resize(frame, self.resolution)
            self._apply_overlay(resized, cam_id, pos_ms)
            out.write(resized)
            frames_written += 1
            current_frame_idx += 1

        cap.release()
        logger.info(f"[STITCHER] Extracted {frames_written} frames for camera '{cam_id}' ({start_ms:.1f}ms -> {end_ms:.1f}ms)")
        return frames_written > 0

    def _apply_overlay(self, frame: np.ndarray, cam_id: str, pos_ms: float) -> None:
        """Applies timestamp and camera ID overlay to the frame."""
        # Top banner
        cv2.rectangle(frame, (0, 0), (self.w, 40), (10, 14, 22), -1)
        
        # Format timestamp
        secs = int(pos_ms / 1000)
        mins = secs // 60
        secs = secs % 60
        ms = int(pos_ms % 1000)
        ts_str = f"{mins:02d}:{secs:02d}.{ms:03d}"

        cv2.putText(frame, f"CAM: {cam_id} | TIME: {ts_str}", (20, 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)
