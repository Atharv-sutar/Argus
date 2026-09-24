"""Camera acquisition module for recorded video files."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.core.interfaces import BaseCamera

logger = logging.getLogger(__name__)


class VideoFileCamera(BaseCamera):
    """
    Acquires video frames from a recorded video file.
    Does NOT run a background thread. Driven synchronously by a PlaybackController.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._cap: Optional[Any] = None
        self._is_closed = False
        self._lock = threading.Lock()
        
        self.fps = 30.0
        self.total_frames = 0
        self.duration_ms = 0.0

        self._open_stream()

    def _open_stream(self) -> None:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required.")

        with self._lock:
            try:
                self._cap = cv2.VideoCapture(self.file_path)
                if not self._cap or not self._cap.isOpened():
                    logger.error(f"Failed to open video file: {self.file_path}")
                    self._is_closed = True
                    return

                self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
                self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if self.fps > 0:
                    self.duration_ms = (self.total_frames / self.fps) * 1000.0

                logger.info(f"Opened video file: {self.file_path} (FPS: {self.fps}, Duration: {self.duration_ms:.1f}ms)")
            except Exception as e:
                logger.error(f"Exception opening video file {self.file_path}: {e}")
                self._is_closed = True

    def is_opened(self) -> bool:
        with self._lock:
            return not self._is_closed and self._cap is not None and self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Standard BaseCamera read(). Reads the next sequential frame.
        """
        if not self.is_opened():
            return False, None, 0.0

        with self._lock:
            if self._cap is None:
                return False, None, 0.0
            
            # Simple throttle to prevent processing video at max CPU speed
            import time
            now = time.time()
            if not hasattr(self, '_last_read_time'):
                self._last_read_time = now
            else:
                elapsed = now - self._last_read_time
                target_interval = 1.0 / (self.fps or 30.0)
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
            self._last_read_time = time.time()
            
            try:
                pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            except Exception:
                pos_msec = 0.0
                
            success, frame = self._cap.read()
            if not success or frame is None:
                return False, None, pos_msec
            
            return True, frame, pos_msec

    def seek(self, timestamp_ms: float) -> bool:
        """
        Seeks to a specific timestamp in the video file.
        """
        if not self.is_opened():
            return False

        with self._lock:
            if self._cap is None:
                return False
            
            # Ensure within bounds
            target_ms = max(0.0, min(timestamp_ms, self.duration_ms))
            try:
                self._cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
                return True
            except Exception as e:
                logger.debug(f"Seek failed for {self.file_path}: {e}")
                return False

    def catch_up(self, target_ms: float) -> None:
        """Fast-forwards sequentially until the camera reaches the target timestamp, safely under lock."""
        if not self.is_opened():
            return
        with self._lock:
            if self._cap is None or self.fps <= 0:
                return
            try:
                pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                frame_duration = 1000.0 / self.fps
                while target_ms - pos_msec > frame_duration * 1.5:
                    self._cap.grab()
                    new_pos = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                    if new_pos == pos_msec or new_pos == 0.0:
                        break
                    pos_msec = new_pos
            except Exception:
                pass

    def read_at(self, timestamp_ms: float) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Seeks to the given timestamp and reads the frame.
        """
        self.seek(timestamp_ms)
        return self.read()

    def release(self) -> None:
        self._is_closed = True
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
        logger.info(f"Released video file: {self.file_path}")
