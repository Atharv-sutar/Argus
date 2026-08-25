"""Camera acquisition module implementing BaseCamera using OpenCV."""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple, Union
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

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None

        # Threaded capture state
        self._thread = None
        self._running = False
        self._lock = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp_ms: float = 0.0
        self._has_new_frame = False

        self._open_stream()

    def _open_stream(self) -> None:
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for OpenCVCamera but is not installed.")

        import threading
        self._lock = threading.Lock()

        # Convert string digits to int if given e.g. "0"
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        # Use standard VideoCapture backend (MSMF on Windows)
        self._cap = cv2.VideoCapture(src)

        if not self._cap or not self._cap.isOpened():
            # Fallback to default if initial open failed
            self._cap = cv2.VideoCapture(src, cv2.CAP_ANY)

        if not self._cap or not self._cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return

        if self.width is not None and isinstance(src, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None and isinstance(src, int):
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps is not None and isinstance(src, int):
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)


        self._start_time = time.time()
        logger.info(f"Successfully opened video source: {self.source}")

        # Start background reader thread if enabled
        if self.use_thread:
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

            # Block until the first frame arrives so read() never returns None
            # on a healthy camera immediately after construction.
            deadline = time.time() + 2.0
            while self._latest_frame is None and time.time() < deadline:
                time.sleep(0.01)
            if self._latest_frame is None:
                logger.warning(f"Camera '{self.source}' thread started but no frame received within 2s")


    def _capture_loop(self) -> None:
        """Background thread continuously pulling frames to prevent driver buffer buildup."""
        while self._running and self._cap is not None and self._cap.isOpened():
            success, frame = self._cap.read()
            if not success or frame is None:
                time.sleep(0.005)
                continue

            self._frame_count += 1
            pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC) if not isinstance(self.source, int) else 0.0
            now_ms = float(pos_msec) if pos_msec > 0.0 else (time.time() - (self._start_time or time.time())) * 1000.0

            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp_ms = now_ms
                self._has_new_frame = True

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self.is_opened() or self._cap is None:
            return False, None, 0.0

        if self.use_thread:
            # Non-blocking instant read from latest frame buffer
            with self._lock:
                if self._latest_frame is not None:
                    return True, self._latest_frame.copy(), self._latest_timestamp_ms
                return False, None, 0.0

        success, frame = self._cap.read()
        if not success or frame is None:
            return False, None, 0.0

        self._frame_count += 1
        pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC) if (cv2 is not None and not isinstance(self.source, int)) else 0.0
        now_ms = float(pos_msec) if pos_msec > 0.0 else (time.time() - (self._start_time or time.time())) * 1000.0

        return True, frame, now_ms

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(f"Released video source: {self.source}")



class SyntheticCamera(BaseCamera):
    """
    Generates synthetic frames with moving rectangular targets for headless testing without a physical camera.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        max_frames: int = 300,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.max_frames = max_frames
        self._frame_count = 0
        self._is_opened = True
        self._start_time = time.time()

    def is_opened(self) -> bool:
        return self._is_opened and (self._frame_count < self.max_frames)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self.is_opened():
            return False, None, 0.0

        # Generate a synthetic frame with dark background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw a simulated moving person rectangle
        t = self._frame_count
        cx = int(100 + (t * 2) % (self.width - 200))
        cy = int(150 + int(30 * np.sin(t * 0.1)))
        w, h = 60, 140

        x1 = max(0, cx - w // 2)
        y1 = max(0, cy - h // 2)
        x2 = min(self.width, cx + w // 2)
        y2 = min(self.height, cy + h // 2)

        # Draw box in synthetic frame
        frame[y1:y2, x1:x2] = [180, 180, 180]

        self._frame_count += 1
        timestamp_ms = (self._frame_count / self.fps) * 1000.0
        return True, frame, timestamp_ms

    def release(self) -> None:
        self._is_opened = False
