"""Orchestrates playback across multiple VideoFileCameras."""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List

from src.camera.video_file import VideoFileCamera

logger = logging.getLogger(__name__)


class PlaybackController:
    """
    Manages synchronized playback for multiple video files.
    Maintains a global clock, playback state, and speed.
    """

    def __init__(self) -> None:
        self.cameras: Dict[str, VideoFileCamera] = {}
        
        self.current_time_ms: float = 0.0
        self.max_duration_ms: float = 0.0
        self.playback_speed: float = 1.0
        
        self.is_playing: bool = False
        
        self._lock = threading.Lock()
        self._last_tick_time: float = 0.0

    def add_camera(self, camera_id: str, camera: VideoFileCamera) -> None:
        """Adds a camera to be managed by the controller."""
        with self._lock:
            self.cameras[camera_id] = camera
            if camera.duration_ms > self.max_duration_ms:
                self.max_duration_ms = camera.duration_ms

    def play(self) -> None:
        """Starts or resumes playback."""
        with self._lock:
            if not self.is_playing:
                self.is_playing = True
                self._last_tick_time = time.time()
                logger.info("Playback started.")

    def pause(self) -> None:
        """Pauses playback."""
        with self._lock:
            if self.is_playing:
                self.is_playing = False
                logger.info(f"Playback paused at {self.current_time_ms:.1f}ms.")

    def toggle_play_pause(self) -> None:
        """Toggles between play and pause."""
        with self._lock:
            if self.is_playing:
                self.is_playing = False
                logger.info(f"Playback paused at {self.current_time_ms:.1f}ms.")
            else:
                self.is_playing = True
                self._last_tick_time = time.time()
                logger.info("Playback resumed.")

    def seek(self, timestamp_ms: float) -> None:
        """Seeks all cameras to a specific timestamp."""
        with self._lock:
            target_ms = max(0.0, min(timestamp_ms, self.max_duration_ms))
            self.current_time_ms = target_ms
            logger.info(f"Seeked to {self.current_time_ms:.1f}ms.")
            for cam in self.cameras.values():
                cam.seek(self.current_time_ms)
            if self.is_playing:
                self._last_tick_time = time.time()

    def set_speed(self, speed: float) -> None:
        """Sets the playback speed multiplier."""
        with self._lock:
            self.playback_speed = max(0.1, speed)
            if self.is_playing:
                self._last_tick_time = time.time()
            logger.info(f"Playback speed set to {self.playback_speed}x.")

    def step(self) -> None:
        """
        Advances the global clock if playing.
        Should be called frequently by the pipeline's main loop.
        """
        with self._lock:
            if not self.is_playing:
                return

            now = time.time()
            delta_s = now - self._last_tick_time
            self._last_tick_time = now

            # Advance time according to speed
            delta_ms = delta_s * 1000.0 * self.playback_speed
            self.current_time_ms += delta_ms

            if self.current_time_ms >= self.max_duration_ms:
                self.current_time_ms = self.max_duration_ms
                self.is_playing = False
                logger.info("Playback reached end of videos.")

    def get_time(self) -> float:
        """Returns the current playback timestamp in milliseconds."""
        with self._lock:
            return self.current_time_ms

    def release_all(self) -> None:
        """Releases all managed video files."""
        with self._lock:
            self.is_playing = False
            for cam in self.cameras.values():
                cam.release()
            self.cameras.clear()
            self.current_time_ms = 0.0
            self.max_duration_ms = 0.0
