"""Camera acquisition module implementing BaseCamera using OpenCV."""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any, Optional, Tuple, Union
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.core.interfaces import BaseCamera

logger = logging.getLogger(__name__)


class OpenCVCamera(BaseCamera):
    """
    Acquires video frames from a webcam, video file, or RTSP stream using OpenCV.
    Supports asynchronous threaded reading to prevent USB I/O blocking and maximize FPS.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        use_thread: bool = True,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.use_thread = use_thread

        self._cap: Optional[Any] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None

        # Threaded capture state
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._is_closed = False
        self._consecutive_failures = 0
        self._lock: threading.Lock = threading.Lock()
        self._cap_lock: threading.Lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp_ms: float = 0.0
        self._has_new_frame = False

        self._open_stream()

    def _open_stream(self) -> None:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for OpenCVCamera but is not installed.")

        # Convert string digits to int if given e.g. "0"
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        with self._cap_lock:
            try:
                # On Windows, try DirectShow (cv2.CAP_DSHOW) first for webcam indices
                if isinstance(src, int) and sys.platform.startswith("win"):
                    self._cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                    if not self._cap or not self._cap.isOpened():
                        self._cap = cv2.VideoCapture(src, cv2.CAP_ANY)
                else:
                    self._cap = cv2.VideoCapture(src)

                if not self._cap or not self._cap.isOpened():
                    # Fallback to default if initial open failed
                    self._cap = cv2.VideoCapture(src, cv2.CAP_ANY)

                if not self._cap or not self._cap.isOpened():
                    logger.error(f"Failed to open video source: {self.source}")
                    self._is_closed = True
                    return

                try:
                    # Fix P-11: Set buffer size to 1 to prevent RTSP latency drift
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if self.width is not None and isinstance(src, int):
                        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    if self.height is not None and isinstance(src, int):
                        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    if self.fps is not None and isinstance(src, int):
                        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
                except Exception as e:
                    logger.debug(f"Could not set camera properties for {self.source}: {e}")
            except Exception as e:
                logger.error(f"Exception opening video source {self.source}: {e}")
                self._is_closed = True
                return

        self._start_time = time.time()
        logger.info(f"Successfully opened video source: {self.source}")

        # Start background reader thread if enabled
        if self.use_thread:
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

            # Non-blocking short wait for initial frame arrival (max 0.2s)
            deadline = time.time() + 0.2
            while self._latest_frame is None and time.time() < deadline and not self._is_closed:
                time.sleep(0.005)

    def _capture_loop(self) -> None:
        """Background thread continuously pulling frames to prevent driver buffer buildup."""
        while self._running and not self._is_closed:
            with self._cap_lock:
                if not self._running or self._is_closed:
                    break
                if self._cap is None or not self._cap.isOpened():
                    cap_is_open = False
                else:
                    cap_is_open = True
                    try:
                        success, frame = self._cap.read()
                    except Exception as e:
                        logger.debug(f"Exception in _capture_loop read for {self.source}: {e}")
                        success, frame = False, None

            if not cap_is_open:
                time.sleep(0.1)
                if self._running and not self._is_closed:
                    self._reopen_stream()
                continue

            if not self._running or self._is_closed:
                break

            if not success or frame is None:
                self._consecutive_failures += 1

                # If this is a video file, loop back to the beginning on EOF
                if isinstance(self.source, str) and not self.source.startswith(("rtsp://", "http://", "https://")):
                    with self._cap_lock:
                        if self._cap is not None:
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._consecutive_failures = 0
                    time.sleep(0.01)
                    continue

                if self._consecutive_failures >= 30:
                    if self._running and not self._is_closed:
                        logger.warning(f"Camera '{self.source}' stream read failure count={self._consecutive_failures}. Reconnecting...")
                        self._reopen_stream()
                    time.sleep(0.2)
                    continue

                time.sleep(0.01)
                continue

            self._consecutive_failures = 0
            self._frame_count += 1
            pos_msec = 0.0
            if not isinstance(self.source, int) and cv2 is not None:
                with self._cap_lock:
                    if self._cap is not None:
                        try:
                            pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                        except Exception:
                            pos_msec = 0.0
            now_ms = float(pos_msec) if pos_msec > 0.0 else (time.time() - (self._start_time or time.time())) * 1000.0

            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp_ms = now_ms
                self._has_new_frame = True

    def _reopen_stream(self) -> None:
        """Attempts to reopen the VideoCapture device on failure."""
        if cv2 is None or not self._running or self._is_closed:
            return
        with self._cap_lock:
            if not self._running or self._is_closed:
                return
            try:
                if self._cap is not None:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                src = self.source
                if isinstance(src, str) and src.isdigit():
                    src = int(src)

                if isinstance(src, int) and sys.platform.startswith("win"):
                    self._cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                    if not self._cap or not self._cap.isOpened():
                        self._cap = cv2.VideoCapture(src, cv2.CAP_ANY)
                else:
                    self._cap = cv2.VideoCapture(src)

                if self._cap and self._cap.isOpened():
                    try:
                        # Fix P-11: Set buffer size to 1 to prevent RTSP latency drift
                        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        if self.width is not None and isinstance(src, int):
                            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        if self.height is not None and isinstance(src, int):
                            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        if self.fps is not None and isinstance(src, int):
                            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
                    except Exception as e:
                        logger.debug(f"Could not set camera properties on reconnect for {self.source}: {e}")
                    self._consecutive_failures = 0
                    logger.info(f"Successfully reconnected to camera: {self.source}")
            except Exception as e:
                logger.debug(f"Reconnection attempt to '{self.source}' failed: {e}")

    def is_opened(self) -> bool:
        return not self._is_closed and (self._running or (self._cap is not None and self._cap.isOpened()))

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self.is_opened():
            return False, None, 0.0

        if self.use_thread:
            # Non-blocking instant read from latest frame buffer
            with self._lock:
                if self._latest_frame is not None:
                    return True, self._latest_frame.copy(), self._latest_timestamp_ms
                return False, None, 0.0

        with self._cap_lock:
            if self._cap is None:
                return False, None, 0.0

            success, frame = self._cap.read()
            if not success or frame is None:
                if isinstance(self.source, str) and not self.source.startswith(("rtsp://", "http://", "https://")):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = self._cap.read()

                if not success or frame is None:
                    return False, None, 0.0

            self._frame_count += 1
            pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC) if (cv2 is not None and not isinstance(self.source, int)) else 0.0
            now_ms = float(pos_msec) if pos_msec > 0.0 else (time.time() - (self._start_time or time.time())) * 1000.0

            return True, frame, now_ms

    def release(self) -> None:
        self._running = False
        self._is_closed = True
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.3)

        with self._cap_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as e:
                    logger.debug(f"Error releasing cap for {self.source}: {e}")
                self._cap = None
                logger.info(f"Released video source: {self.source}")



class SyntheticCamera(BaseCamera):
    """
    Generates synthetic frames with realistic moving targets for headless testing without a physical camera.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        max_frames: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.max_frames = max_frames
        self._frame_count = 0
        self._is_opened = True
        self._start_time = time.time()

    def is_opened(self) -> bool:
        if not self._is_opened:
            return False
        if self.max_frames is not None:
            return self._frame_count < self.max_frames
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self.is_opened():
            return False, None, 0.0

        # Generate a synthetic frame with surveillance camera visual texture
        frame = np.full((self.height, self.width, 3), 24, dtype=np.uint8)

        # Draw subtle floor grid pattern
        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (32, 36, 42), 1)
        for x in range(0, self.width, 40):
            cv2.line(frame, (0, x), (self.width, x), (32, 36, 42), 1)

        # Draw simulated moving person silhouettes
        t = self._frame_count
        cx = int(80 + ((t * 3) % (self.width - 160)))
        cy = int(self.height // 2 + 40 * np.sin(t * 0.05))
        w, h = 64, 150

        x1 = max(0, cx - w // 2)
        y1 = max(0, cy - h // 2)
        x2 = min(self.width, cx + w // 2)
        y2 = min(self.height, cy + h // 2)

        # Person body (simulated RGB appearance)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 160, 220), -1)
        # Head
        head_r = 18
        cv2.circle(frame, (cx, max(head_r, y1 - 10)), head_r, (210, 180, 160), -1)

        self._frame_count += 1
        timestamp_ms = (self._frame_count / self.fps) * 1000.0
        return True, frame, timestamp_ms

    def release(self) -> None:
        self._is_opened = False
