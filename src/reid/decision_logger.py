"""Asynchronous non-blocking decision log harness for ReID tracking diagnostics."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DecisionLogger:
    """
    Non-blocking background JSONL writer for ReID decisions.
    Uses an in-memory bounded queue and daemon worker thread.
    Writing never blocks the real-time processing pipeline.
    """

    def __init__(
        self,
        log_path: str = "logs/decision_log.jsonl",
        enabled: bool = False,
        max_queue_size: int = 10000,
    ) -> None:
        self.log_path = log_path
        self.enabled = enabled
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._file = None

        if self.enabled:
            self._start()

    def _start(self) -> None:
        try:
            log_dir = os.path.dirname(self.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self._file = open(self.log_path, "a", encoding="utf-8")
        except Exception as e:
            logger.error(f"[DECISION-LOG] Failed to open log file '{self.log_path}': {e}")
            self.enabled = False
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._writer_loop, name="DecisionLogWriter", daemon=True
        )
        self._worker_thread.start()
        logger.info(f"[DECISION-LOG] Decision logging enabled -> '{self.log_path}'")

    def log(self, entry: Dict[str, Any]) -> None:
        """Enqueue decision log record non-blockingly."""
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            # Drop under extreme backpressure to guarantee pipeline thread never blocks
            pass

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                entry = self._queue.get(timeout=0.2)
                if self._file:
                    self._file.write(json.dumps(entry) + "\n")
                    self._file.flush()
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.debug(f"[DECISION-LOG] Write error: {e}")

    def close(self) -> None:
        """Flushes remaining items and cleanly stops writer."""
        if not self.enabled:
            return
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
        if self._file:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None
