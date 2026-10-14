"""Retention policy manager for purging aged and over-quota video segments."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.recording.segment_index import SegmentIndex

logger = logging.getLogger(__name__)


class RetentionManager:
    """
    Monitors storage usage and enforces age and quota retention policies.
    Runs periodically in a background daemon thread.
    """

    def __init__(
        self,
        segment_index: SegmentIndex,
        output_dir: str = "recordings",
        max_storage_gb: float = 50.0,
        max_age_days: float = 7.0,
        cleanup_interval_s: float = 300.0,
    ) -> None:
        self.segment_index = segment_index
        self.output_dir = Path(output_dir)
        self.max_storage_bytes = int(max(0.0001, max_storage_gb) * 1024 * 1024 * 1024)
        self.max_age_seconds = max(1.0, max_age_days * 86400.0)
        self.cleanup_interval_s = max(1.0, cleanup_interval_s)

        self._is_running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Starts the background retention cleanup daemon."""
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(
            target=self._retention_loop,
            daemon=True,
            name="rec_retention_mgr",
        )
        self._thread.start()
        logger.info(
            f"[RETENTION] Started retention manager: max_storage={self.max_storage_bytes / 1024 / 1024 / 1024:.1f}GB, "
            f"max_age={self.max_age_seconds / 86400.0:.1f}d, interval={self.cleanup_interval_s}s"
        )

    def stop(self) -> None:
        """Stops the background retention daemon."""
        self._is_running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("[RETENTION] Stopped retention manager")

    def _delete_segment_file_and_record(self, segment: Dict[str, Any]) -> bool:
        """Safely unlinks segment file from disk and deletes record from SQLite index."""
        seg_id = segment.get("segment_id")
        fpath_str = segment.get("file_path")
        if not seg_id:
            return False

        if fpath_str:
            fpath = Path(fpath_str)
            if fpath.is_file():
                try:
                    fpath.unlink()
                    logger.debug(f"[RETENTION] Deleted segment file: {fpath.name}")
                    # Prune empty parent directory
                    parent = fpath.parent
                    while parent != self.output_dir and parent.exists():
                        if not any(parent.iterdir()):
                            parent.rmdir()
                            parent = parent.parent
                        else:
                            break
                except OSError as e:
                    logger.warning(f"[RETENTION] Could not delete segment file {fpath}: {e}")

        self.segment_index.delete_segment(seg_id)
        return True

    def run_cleanup_now(self) -> Dict[str, int]:
        """
        Executes an immediate pass of retention cleanup.
        Returns a summary of deleted segments and freed bytes.
        """
        deleted_count = 0
        freed_bytes = 0

        # 1. Age-based cleanup
        cutoff_dt = datetime.now(timezone.utc) - timedelta(seconds=self.max_age_seconds)
        cutoff_iso = cutoff_dt.isoformat()
        aged_segments = self.segment_index.get_segments_older_than(cutoff_iso)

        for seg in aged_segments:
            size = seg.get("file_size_bytes", 0)
            if self._delete_segment_file_and_record(seg):
                deleted_count += 1
                freed_bytes += size

        # 2. Quota-based cleanup (delete oldest until total storage <= max_storage_bytes)
        current_total = self.segment_index.get_total_storage_bytes()
        while current_total > self.max_storage_bytes:
            oldest = self.segment_index.get_oldest_segments(limit=10)
            if not oldest:
                break
            for seg in oldest:
                if current_total <= self.max_storage_bytes:
                    break
                size = seg.get("file_size_bytes", 0)
                if self._delete_segment_file_and_record(seg):
                    deleted_count += 1
                    freed_bytes += size
                    current_total -= size

        if deleted_count > 0:
            logger.info(
                f"[RETENTION] Cleanup pass finished: deleted {deleted_count} segment(s), "
                f"freed {freed_bytes / 1024 / 1024:.2f} MB"
            )
        return {"deleted_count": deleted_count, "freed_bytes": freed_bytes}

    def enforce_retention(self) -> int:
        """Alias executing immediate retention cleanup and returning count of deleted segments."""
        return self.run_cleanup_now()["deleted_count"]

    def _retention_loop(self) -> None:
        """Background thread executing periodic cleanup."""
        while self._is_running:
            try:
                self.run_cleanup_now()
            except Exception as e:
                logger.error(f"[RETENTION] Error during retention cleanup pass: {e}")

            # Sleep in short increments for responsive shutdown
            slept = 0.0
            while self._is_running and slept < self.cleanup_interval_s:
                time.sleep(1.0)
                slept += 1.0
