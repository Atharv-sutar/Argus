"""
Real-time forensic clip recorder for target surveillance evidence.

Purpose:
    Buffers annotated frames of the tracked target in real-time from target
    selection until the user triggers export. Produces a single continuous
    MP4 video of the target's complete journey across cameras.

Input:
    - Annotated BGR frames (np.ndarray) from the pipeline's active camera
    - Camera ID, timestamp, and target state metadata per frame

Output:
    - A single MP4 file containing the chronological target journey

Dependencies:
    - cv2 (VideoWriter for MP4 encoding)
    - numpy (frame manipulation)

Callers:
    - MultiCameraPipeline.step() feeds frames
    - MultiCameraPipeline.select_target_* starts recording
    - MultiCameraPipeline.export_forensic_clip() triggers export
    - UI server /api/forensic/* endpoints

State:
    - _is_recording: whether capture is active
    - _frames_buffer: in-memory frame ring buffer
    - _spill_writer: on-disk overflow for long recordings

GPU usage: None (CPU-only frame buffering and encoding)
CPU considerations: Frame resize + overlay is ~0.3ms per frame
Error conditions: Disk full during spill/export, codec unavailable
Testing: Feed synthetic frames, verify MP4 output frame count and duration
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ForensicClipRecorder:
    """
    Buffers annotated frames of the tracked target in real-time.
    On export, flushes all buffered frames into a single MP4 file.

    Lifecycle:
        1. start_recording(target_track_id, camera_id) — begins capture
        2. feed_frame(frame, camera_id, timestamp_ms, target_state) — called every pipeline step
        3. stop_and_export(output_path) -> str — finalizes and writes the MP4
        4. discard() — cancels recording without export

    Memory Management:
        Frames are held in-memory up to _max_memory_frames. Beyond that,
        oldest frames are flushed to a temporary on-disk video file to
        prevent unbounded RAM growth.
    """

    def __init__(
        self,
        target_fps: float = 15.0,
        resolution: Tuple[int, int] = (1280, 720),
        max_memory_frames: int = 9000,
    ) -> None:
        """
        Args:
            target_fps: Output video frame rate.
            resolution: Output video resolution (width, height).
            max_memory_frames: Max frames to keep in RAM before spilling to disk.
        """
        self._target_fps = target_fps
        self._resolution = resolution
        self._max_memory_frames = max_memory_frames

        self._is_recording: bool = False
        self._frames_buffer: List[np.ndarray] = []
        self._recording_start_time: float = 0.0
        self._target_track_id: Optional[int] = None
        self._current_camera_id: str = ""
        self._frame_count: int = 0
        self._last_feed_time: float = 0.0

        # Spill-to-disk state
        self._spill_writer: Optional[cv2.VideoWriter] = None
        self._spill_path: Optional[str] = None
        self._spill_frame_count: int = 0

        # Export state
        self._export_in_progress: bool = False
        self._last_export_path: Optional[str] = None

        self._lock = threading.Lock()

    # ─── Properties ────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def duration_seconds(self) -> float:
        if not self._is_recording:
            return 0.0
        return time.time() - self._recording_start_time

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def status(self) -> Dict:
        """Returns a serializable status dict for the REST API."""
        return {
            "recording": self._is_recording,
            "duration_s": round(self.duration_seconds, 1),
            "frames": self._frame_count,
            "camera_id": self._current_camera_id,
            "export_in_progress": self._export_in_progress,
            "last_export_path": self._last_export_path,
        }

    # ─── Lifecycle ─────────────────────────────────────────────────────

    def start_recording(self, target_track_id: int, camera_id: str) -> None:
        """Begin capturing frames for a new target investigation."""
        with self._lock:
            # If already recording, discard the old one silently
            if self._is_recording:
                self._cleanup_spill()

            self._is_recording = True
            self._frames_buffer.clear()
            self._recording_start_time = time.time()
            self._target_track_id = target_track_id
            self._current_camera_id = camera_id
            self._frame_count = 0
            self._spill_frame_count = 0
            self._last_feed_time = 0.0
            self._last_export_path = None

            logger.info(
                f"[FORENSIC] Recording started for target #{target_track_id} "
                f"on camera '{camera_id}'"
            )

    def feed_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        timestamp_ms: float,
        target_state: str,
    ) -> None:
        """
        Feed a single annotated frame into the recording buffer.
        Called from the pipeline's step() method for the active camera.
        """
        if not self._is_recording:
            return

        # Throttle to target FPS to avoid excessive frame accumulation
        now = time.time()
        min_interval = 1.0 / self._target_fps
        if self._last_feed_time > 0.0 and (now - self._last_feed_time) < min_interval:
            return
        self._last_feed_time = now

        try:
            # Resize if needed
            h, w = frame.shape[:2]
            target_w, target_h = self._resolution
            if w != target_w or h != target_h:
                out_frame = cv2.resize(frame, self._resolution, interpolation=cv2.INTER_LINEAR)
            else:
                out_frame = frame.copy()

            # Apply forensic overlay
            self._apply_forensic_overlay(out_frame, camera_id, timestamp_ms, target_state)

            with self._lock:
                self._frames_buffer.append(out_frame)
                self._frame_count += 1
                self._current_camera_id = camera_id

                # Spill to disk if buffer exceeds memory limit
                if len(self._frames_buffer) >= self._max_memory_frames:
                    self._spill_to_disk()

        except Exception as e:
            logger.debug(f"[FORENSIC] Frame feed error: {e}")

    def on_target_state_changed(
        self, old_state: str, new_state: str, camera_id: str
    ) -> None:
        """Insert visual state-transition cards into the recording."""
        if not self._is_recording:
            return

        cards: List[np.ndarray] = []

        if new_state in ("LOST", "LOST_PERMANENTLY"):
            cards = self._render_state_card(
                "TARGET LOST",
                f"Last seen on {self._current_camera_id}",
                duration_frames=int(self._target_fps * 1.5),
            )
        elif old_state in ("LOST", "SEARCHING", "RECOVERING") and new_state == "TRACKING":
            if camera_id != self._current_camera_id:
                cards = self._render_state_card(
                    "TARGET RECOVERED",
                    f"{self._current_camera_id}  →  {camera_id}",
                    duration_frames=int(self._target_fps * 1.5),
                )
            else:
                cards = self._render_state_card(
                    "TARGET RECOVERED",
                    f"On {camera_id}",
                    duration_frames=int(self._target_fps * 1.0),
                )
        elif new_state == "TRANSIT":
            cards = self._render_state_card(
                "IN TRANSIT",
                f"{self._current_camera_id}  →  {camera_id}",
                duration_frames=int(self._target_fps * 2.0),
            )

        if cards:
            with self._lock:
                self._frames_buffer.extend(cards)
                self._frame_count += len(cards)
                if camera_id and new_state != "LOST":
                    self._current_camera_id = camera_id

    def stop_and_export(self, output_path: str) -> str:
        """
        Stop recording and write the accumulated frames to an MP4 file.
        Returns the output file path.
        """
        self._export_in_progress = True
        try:
            with self._lock:
                self._is_recording = False

                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    output_path, fourcc, self._target_fps, self._resolution
                )

                if not writer.isOpened():
                    logger.error(f"[FORENSIC] Failed to open VideoWriter at {output_path}")
                    self._export_in_progress = False
                    return ""

                total_written = 0

                # 1. If there was a spill file, read it first
                if self._spill_path and os.path.exists(self._spill_path):
                    self._close_spill_writer()
                    cap = cv2.VideoCapture(self._spill_path)
                    if cap.isOpened():
                        while True:
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                break
                            # Ensure correct resolution
                            h, w = frame.shape[:2]
                            if (w, h) != self._resolution:
                                frame = cv2.resize(frame, self._resolution)
                            writer.write(frame)
                            total_written += 1
                        cap.release()

                # 2. Write in-memory buffer
                for frame in self._frames_buffer:
                    writer.write(frame)
                    total_written += 1

                writer.release()
                self._cleanup_spill()
                self._frames_buffer.clear()
                self._frame_count = 0
                self._last_export_path = output_path

                logger.info(
                    f"[FORENSIC] Exported {total_written} frames to {output_path} "
                    f"({total_written / self._target_fps:.1f}s video)"
                )
                return output_path

        except Exception as e:
            logger.error(f"[FORENSIC] Export failed: {e}")
            return ""
        finally:
            self._export_in_progress = False

    def discard(self) -> None:
        """Cancel recording and free all resources."""
        with self._lock:
            self._is_recording = False
            self._frames_buffer.clear()
            self._frame_count = 0
            self._spill_frame_count = 0
            self._cleanup_spill()
            logger.info("[FORENSIC] Recording discarded.")

    # ─── Internal Helpers ──────────────────────────────────────────────

    def _apply_forensic_overlay(
        self,
        frame: np.ndarray,
        camera_id: str,
        timestamp_ms: float,
        target_state: str,
    ) -> None:
        """Applies forensic metadata overlay to a frame in-place."""
        h, w = frame.shape[:2]

        # ── Top bar: Camera ID + elapsed time + REC indicator ──
        cv2.rectangle(frame, (0, 0), (w, 32), (10, 14, 22), -1)

        # Elapsed time since recording start
        elapsed = time.time() - self._recording_start_time
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        elapsed_str = f"{mins:02d}:{secs:02d}"

        # REC indicator (pulsing red dot)
        dot_color = (0, 0, 220) if int(time.time() * 2) % 2 == 0 else (0, 0, 160)
        cv2.circle(frame, (16, 16), 6, dot_color, -1)
        cv2.putText(
            frame, "REC", (28, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA,
        )

        # Camera + time
        info_text = f"CAM: {camera_id}  |  {elapsed_str}"
        cv2.putText(
            frame, info_text, (80, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA,
        )

        # ── Bottom bar: Target state badge ──
        cv2.rectangle(frame, (0, h - 28), (w, h), (10, 14, 22), -1)

        # State color mapping
        state_colors = {
            "TRACKING": (0, 200, 100),
            "LOCKED": (0, 255, 200),
            "OCCLUDED": (0, 180, 255),
            "LOST": (0, 80, 255),
            "UNCERTAIN": (0, 180, 255),
            "SEARCHING": (255, 180, 0),
            "RECOVERING": (255, 200, 0),
        }
        color = state_colors.get(target_state, (150, 150, 150))
        badge_text = f"TARGET: {target_state}"
        cv2.putText(
            frame, badge_text, (12, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

        # Frame counter on the right
        frame_text = f"F:{self._frame_count}"
        cv2.putText(
            frame, frame_text, (w - 100, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 140), 1, cv2.LINE_AA,
        )

    def _render_state_card(
        self, title: str, subtitle: str, duration_frames: int
    ) -> List[np.ndarray]:
        """Generates dark title card frames for state transitions."""
        target_w, target_h = self._resolution
        card = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        card[:] = (20, 26, 38)

        # Title
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _), _ = cv2.getTextSize(title, font, 1.5, 2)
        tx = (target_w - tw) // 2
        cv2.putText(
            card, title, (tx, target_h // 2 - 30),
            font, 1.5, (0, 215, 255), 2, cv2.LINE_AA,
        )

        # Subtitle
        (sw, _), _ = cv2.getTextSize(subtitle, font, 0.9, 2)
        sx = (target_w - sw) // 2
        cv2.putText(
            card, subtitle, (sx, target_h // 2 + 30),
            font, 0.9, (180, 180, 180), 2, cv2.LINE_AA,
        )

        # Elapsed time
        elapsed = time.time() - self._recording_start_time
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        time_text = f"Recording: {mins:02d}:{secs:02d}"
        (etw, _), _ = cv2.getTextSize(time_text, font, 0.6, 1)
        cv2.putText(
            card, time_text, ((target_w - etw) // 2, target_h // 2 + 80),
            font, 0.6, (100, 110, 120), 1, cv2.LINE_AA,
        )

        return [card.copy() for _ in range(max(1, duration_frames))]

    def _spill_to_disk(self) -> None:
        """Flush oldest frames from memory to a temporary video file."""
        if self._spill_writer is None:
            self._spill_path = os.path.join(
                tempfile.gettempdir(), f"argus_forensic_spill_{int(time.time())}.avi"
            )
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._spill_writer = cv2.VideoWriter(
                self._spill_path, fourcc, self._target_fps, self._resolution
            )
            if not self._spill_writer.isOpened():
                logger.error("[FORENSIC] Failed to create spill file")
                self._spill_writer = None
                return

        # Write half the buffer to disk, keep the recent half in memory
        half = len(self._frames_buffer) // 2
        for frame in self._frames_buffer[:half]:
            self._spill_writer.write(frame)
            self._spill_frame_count += 1

        self._frames_buffer = self._frames_buffer[half:]
        logger.debug(
            f"[FORENSIC] Spilled {half} frames to disk "
            f"(total spilled: {self._spill_frame_count}, in-memory: {len(self._frames_buffer)})"
        )

    def _close_spill_writer(self) -> None:
        """Release the spill VideoWriter if open."""
        if self._spill_writer is not None:
            try:
                self._spill_writer.release()
            except Exception:
                pass
            self._spill_writer = None

    def _cleanup_spill(self) -> None:
        """Close spill writer and remove temporary spill file."""
        self._close_spill_writer()
        if self._spill_path and os.path.exists(self._spill_path):
            try:
                os.remove(self._spill_path)
            except Exception:
                pass
        self._spill_path = None
        self._spill_frame_count = 0
