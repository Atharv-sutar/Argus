"""SQLite index for continuous recording video segments."""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SegmentIndex:
    """
    Manages metadata for continuous recording segments in a local SQLite database.
    Thread-safe and persistent across application restarts.
    """

    def __init__(self, db_path: str = "recordings/segments.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    start_time_ms REAL NOT NULL,
                    end_time_ms REAL NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    frame_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_cam ON segments(camera_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_start ON segments(start_time_ms);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_created ON segments(created_at);")
            conn.commit()

    def add_segment(
        self,
        segment_id: str,
        camera_id: str,
        start_time_ms: float,
        end_time_ms: float,
        file_path: str,
        file_size_bytes: int,
        frame_count: int,
        created_at: Optional[Union[str, float]] = None,
    ) -> None:
        """Records a finalized video segment."""
        if created_at is None:
            created_at_str = datetime.now(timezone.utc).isoformat()
        elif isinstance(created_at, (int, float)):
            created_at_str = datetime.fromtimestamp(created_at, timezone.utc).isoformat()
        else:
            created_at_str = str(created_at)

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO segments (
                    segment_id, camera_id, start_time_ms, end_time_ms,
                    file_path, file_size_bytes, frame_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    segment_id,
                    camera_id,
                    start_time_ms,
                    end_time_ms,
                    file_path,
                    file_size_bytes,
                    frame_count,
                    created_at_str,
                ),
            )
            conn.commit()

    # Convenience alias
    insert_segment = add_segment

    def delete_segment(self, segment_id: str) -> bool:
        """Removes a segment from the database."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM segments WHERE segment_id = ?;", (segment_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_segments(
        self,
        camera_id: Optional[str] = None,
        start_time_ms: Optional[float] = None,
        end_time_ms: Optional[float] = None,
        start_ms: Optional[float] = None,
        end_ms: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Queries segments with optional camera and time range filters."""
        s_ms = start_time_ms if start_time_ms is not None else start_ms
        e_ms = end_time_ms if end_time_ms is not None else end_ms

        query = "SELECT * FROM segments WHERE 1=1"
        params: List[Any] = []

        if camera_id is not None:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if s_ms is not None:
            query += " AND end_time_ms >= ?"
            params.append(s_ms)
        if e_ms is not None:
            query += " AND start_time_ms <= ?"
            params.append(e_ms)

        query += " ORDER BY start_time_ms ASC LIMIT ?;"
        params.append(limit)

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # Convenience alias
    query_segments = get_segments

    def get_total_storage_bytes(self) -> int:
        """Returns total bytes across all recorded segments in the index."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT SUM(file_size_bytes) FROM segments;")
            val = cursor.fetchone()[0]
            return int(val) if val is not None else 0

    def get_segment_count(self) -> int:
        """Returns total number of recorded segments."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM segments;")
            val = cursor.fetchone()[0]
            return int(val) if val is not None else 0

    def get_oldest_segments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the oldest segments ordered by creation time."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM segments ORDER BY start_time_ms ASC, created_at ASC LIMIT ?;",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_segments_older_than(
        self,
        cutoff: Optional[Union[str, float]] = None,
        created_at_cutoff: Optional[Union[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Returns segments created before the specified timestamp (ISO string or epoch seconds)."""
        c = cutoff if cutoff is not None else created_at_cutoff
        if c is None:
            return []
        if isinstance(c, (int, float)):
            cutoff_iso = datetime.fromtimestamp(c, timezone.utc).isoformat()
        else:
            cutoff_iso = str(c)

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM segments WHERE created_at < ? ORDER BY created_at ASC;",
                (cutoff_iso,),
            ).fetchall()
            return [dict(row) for row in rows]
