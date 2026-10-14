"""Per-camera continuous video segment writer."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.recording.segment_index import SegmentIndex

logger = logging.getLogger(__name__)


class SegmentWriter:
    """
    Asynchronously writes continuous video segments for a single camera feed.
    Rotates files every segment_duration_s and records metadata into SegmentIndex.
    """

    def __init__(
        self,
        camera_id: str,
        output_dir: str = "recordings",
        segment_duration_s: float = 300.0,
        fps: float = 20.0,
        resolution: Tuple[int, int] = (1280, 720),
        segment_index: Optional[SegmentIndex] = None,
        max_queue_size: int = 60,
        queue_size: Optional[int] = None,
    ) -> None:
        self.camera_id = camera_id
        self.output_dir = Path(output_dir)
        self.segment_duration_s = max(0.1, segment_duration_s)
        self.fps = max(1.0, fps)
        self.resolution = resolution
        self.segment_index = segment_index
        q_size = queue_size if queue_size is not None else max_queue_size
        self._max_queue_size = q_size

        self._queue: queue.Queue[Optional[Tuple[np.ndarray, float]]] = queue.Queue(maxsize=q_size)
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

        # Current segment state
        self._writer: Optional[cv2.VideoWriter] = None
        self._current_segment_id: Optional[str] = None
        self._current_file_path: Optional[Path] = None
        self._current_start_ms: float = 0.0
        self._current_start_epoch: float = 0.0
        self._current_end_ms: float = 0.0
        self._current_frame_count: int = 0

    def start(self) -> None:
        """Starts the background segment writer thread."""
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name=f"rec_writer_{self.camera_id}",
        )
        self._thread.start()
        logger.info(f"[RECORDING] Started continuous segment writer for '{self.camera_id}' (duration={self.segment_duration_s}s)")

    def feed_frame(self, frame: np.ndarray, timestamp_ms: float) -> bool:
        """
        Feeds a live frame into the writer queue via single-stream tap.
        Non-blocking: drops oldest frame if queue is full to prevent memory buildup.
        """
        if not self._is_running:
            return False

        try:
            self._queue.put_nowait((frame, timestamp_ms))
            return True
        except queue.Full:
            try:
                # Evict oldest frame to keep recording near real-time
                self._queue.get_nowait()
                self._queue.put_nowait((frame, timestamp_ms))
                return True
            except (queue.Empty, queue.Full):
                return False

    def stop(self) -> None:
        """Stops the writer thread and finalizes the active segment."""
        if not self._is_running:
            return
        self._is_running = False
        try:
            self._queue.put_nowait(None)  # Sentinel
        except queue.Full:
            pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        self._finalize_current_segment()
        logger.info(f"[RECORDING] Stopped continuous segment writer for '{self.camera_id}'")

    def _get_segment_path(self, now: datetime, seg_id: str) -> Path:
        """Generates hierarchical directory tree: recordings/<cam_id>/YYYY-MM-DD/HH/<cam_id>_...mp4"""
        date_dir = now.strftime("%Y-%m-%d")
        hour_dir = now.strftime("%H")
        dest_dir = self.output_dir / self.camera_id / date_dir / hour_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.camera_id}_{now.strftime('%Y%m%d_%H%M%S')}_{seg_id[:8]}.mp4"
        return dest_dir / filename

    def _open_new_segment(self, first_timestamp_ms: float) -> bool:
        """Initializes a new segment file with cv2.VideoWriter."""
        self._finalize_current_segment()

        now = datetime.now(timezone.utc)
        seg_id = str(uuid.uuid4())
        fpath = self._get_segment_path(now, seg_id)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(fpath), fourcc, self.fps, self.resolution)
        if not writer.isOpened():
            # Fallback codec
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(fpath), fourcc, self.fps, self.resolution)

        if not writer.isOpened():
            logger.error(f"[RECORDING] Failed to open VideoWriter for {fpath}")
            return False

        self._writer = writer
        self._current_segment_id = seg_id
        self._current_file_path = fpath
        self._current_start_ms = first_timestamp_ms
        self._current_start_epoch = time.time()
        self._current_end_ms = first_timestamp_ms
        self._current_frame_count = 0
        return True

    def _finalize_current_segment(self) -> None:
        """Closes the current VideoWriter and registers the segment in SQLite."""
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as e:
                logger.warning(f"[RECORDING] Error closing writer: {e}")
            self._writer = None

            if (
                self._current_file_path is not None
                and self._current_file_path.is_file()
                and self._current_frame_count > 0
            ):
                size_bytes = self._current_file_path.stat().st_size
                if self.segment_index is not None and self._current_segment_id:
                    self.segment_index.add_segment(
                        segment_id=self._current_segment_id,
                        camera_id=self.camera_id,
                        start_time_ms=self._current_start_ms,
                        end_time_ms=self._current_end_ms,
                        file_path=str(self._current_file_path.resolve()).replace("\\", "/"),
                        file_size_bytes=size_bytes,
                        frame_count=self._current_frame_count,
                    )
                logger.debug(
                    f"[RECORDING] Finalized segment for '{self.camera_id}': "
                    f"{self._current_file_path.name} ({self._current_frame_count} frames, {size_bytes / 1024 / 1024:.2f} MB)"
                )
            elif self._current_file_path is not None and self._current_file_path.is_file():
                # Remove empty segment file
                try:
                    self._current_file_path.unlink()
                except OSError:
                    pass

        self._current_segment_id = None
        self._current_file_path = None
        self._current_frame_count = 0

    def _writer_loop(self) -> None:
        """Worker loop reading frames from queue and writing to disk."""
        while self._is_running:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            frame, ts_ms = item

            # Check if current segment needs to rotate
            now_epoch = time.time()
            if self._writer is None or (now_epoch - self._current_start_epoch >= self.segment_duration_s):
                ok = self._open_new_segment(ts_ms)
                if not ok:
                    continue

            # Ensure frame matches resolution
            h, w = frame.shape[:2]
            if (w, h) != self.resolution:
                frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LINEAR)

            if self._writer is not None:
                self._writer.write(frame)
                self._current_frame_count += 1
                self._current_end_ms = ts_ms
