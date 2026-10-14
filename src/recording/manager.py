"""Continuous recording manager orchestrating per-camera writers and retention policies."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from src.core.multi_camera_types import SourceType
from src.recording.retention_manager import RetentionManager
from src.recording.segment_index import SegmentIndex
from src.recording.segment_writer import SegmentWriter

logger = logging.getLogger(__name__)


class ContinuousRecordingManager:
    """
    Orchestrates continuous background NVR recording for live surveillance cameras.
    Enforces the single-stream tap policy and strictly excludes offline video files.
    """

    def __init__(
        self,
        enabled: bool = True,
        output_dir: str = "recordings",
        segment_duration_s: float = 300.0,
        max_storage_gb: float = 50.0,
        max_age_days: float = 7.0,
        cleanup_interval_s: float = 300.0,
    ) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.segment_duration_s = segment_duration_s

        self.index = SegmentIndex(db_path=str(self.output_dir / "segments.db"))
        self.retention = RetentionManager(
            segment_index=self.index,
            output_dir=str(self.output_dir),
            max_storage_gb=max_storage_gb,
            max_age_days=max_age_days,
            cleanup_interval_s=cleanup_interval_s,
        )

        self._writers: Dict[str, SegmentWriter] = {}
        self._lock = threading.Lock()

        if self.enabled:
            self.retention.start()

    def is_eligible_source(self, source_type: Union[SourceType, str]) -> bool:
        """
        Determines if a camera source is eligible for continuous recording.
        Strict rule: Continuous recording applies strictly to live sources (RTSP/webcam),
        never offline video files or synthetic generators.
        """
        st_val = source_type.value if hasattr(source_type, "value") else str(source_type).lower().strip()
        return st_val in ("webcam", "rtsp", "local", "usb", "stream")

    def start_camera_recording(
        self,
        camera_id: str,
        source_type: Union[SourceType, str],
        resolution: Tuple[int, int] = (1280, 720),
        fps: float = 20.0,
    ) -> bool:
        """
        Starts continuous recording for a camera if enabled and source is a live feed.
        """
        if not self.enabled:
            return False

        if not self.is_eligible_source(source_type):
            logger.debug(
                f"[RECORDING] Skipping continuous recording for non-live source type '{source_type}' on '{camera_id}'"
            )
            return False

        with self._lock:
            if camera_id in self._writers:
                return True

            writer = SegmentWriter(
                camera_id=camera_id,
                output_dir=str(self.output_dir),
                segment_duration_s=self.segment_duration_s,
                fps=fps,
                resolution=resolution,
                segment_index=self.index,
            )
            writer.start()
            self._writers[camera_id] = writer
            logger.info(f"[RECORDING] Initialized continuous recording tap for live camera '{camera_id}' ({source_type})")
            return True

    def feed_frame(self, camera_id: str, frame: np.ndarray, timestamp_ms: float) -> bool:
        """
        Single-stream tap: feeds raw captured frame into the recording writer.
        Does not perform redundant decoding or open duplicate streams.
        """
        if not self.enabled:
            return False

        writer = self._writers.get(camera_id)
        if writer is not None:
            return writer.feed_frame(frame, timestamp_ms)
        return False

    def stop_camera_recording(self, camera_id: str) -> None:
        """Stops continuous recording for a single camera."""
        with self._lock:
            writer = self._writers.pop(camera_id, None)
            if writer is not None:
                writer.stop()

    def stop_all(self) -> None:
        """Stops all active camera writers and the retention manager."""
        with self._lock:
            for writer in self._writers.values():
                writer.stop()
            self._writers.clear()

        self.retention.stop()
        logger.info("[RECORDING] All continuous recording writers stopped")

    def get_storage_stats(self) -> Dict[str, Any]:
        """Returns runtime storage and recording telemetry."""
        total_bytes = self.index.get_total_storage_bytes()
        return {
            "enabled": self.enabled,
            "active_cameras": list(self._writers.keys()),
            "total_segments": self.index.get_segment_count(),
            "storage_used_bytes": total_bytes,
            "storage_used_mb": round(total_bytes / (1024 * 1024), 2),
            "storage_used_gb": round(total_bytes / (1024 * 1024 * 1024), 3),
        }
