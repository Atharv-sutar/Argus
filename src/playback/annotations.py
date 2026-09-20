import logging
import sqlite3
import uuid
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class Annotation:
    annotation_id: str
    timestamp_ms: float
    camera_id: str
    text: str
    created_at: str
    bbox: Optional[Tuple[int, int, int, int]]  # [x, y, w, h]
    annotation_type: str  # "note" | "flag" | "bookmark"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnnotationStore:
    """Manages annotations for the current investigation. Persisted to SQLite."""

    def __init__(self, db_path: str = "case_db.sqlite"):
        self.db_path = db_path
        self._init_db()


    def set_db_path(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS annotations (
                        annotation_id TEXT PRIMARY KEY,
                        timestamp_ms REAL NOT NULL,
                        camera_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        bbox_json TEXT,
                        annotation_type TEXT NOT NULL DEFAULT 'note'
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_anno_cam_time ON annotations(camera_id, timestamp_ms)")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize AnnotationStore DB: {e}")

    def add(self, annotation: Annotation) -> None:
        try:
            bbox_str = json.dumps(annotation.bbox) if annotation.bbox else None
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO annotations (annotation_id, timestamp_ms, camera_id, text, created_at, bbox_json, annotation_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    annotation.annotation_id,
                    annotation.timestamp_ms,
                    annotation.camera_id,
                    annotation.text,
                    annotation.created_at,
                    bbox_str,
                    annotation.annotation_type
                ))
        except sqlite3.Error as e:
            logger.error(f"Failed to add annotation: {e}")

    def update(self, annotation_id: str, text: str, annotation_type: Optional[str] = None) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                if annotation_type:
                    cursor = conn.execute("""
                        UPDATE annotations 
                        SET text = ?, annotation_type = ?
                        WHERE annotation_id = ?
                    """, (text, annotation_type, annotation_id))
                else:
                    cursor = conn.execute("""
                        UPDATE annotations 
                        SET text = ?
                        WHERE annotation_id = ?
                    """, (text, annotation_id))
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Failed to update annotation: {e}")
            return False

    def remove(self, annotation_id: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM annotations WHERE annotation_id = ?", (annotation_id,))
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Failed to remove annotation: {e}")
            return False

    def get_all(self) -> List[Annotation]:
        return self._query("SELECT * FROM annotations ORDER BY timestamp_ms ASC", ())

    def get_by_camera(self, camera_id: str) -> List[Annotation]:
        return self._query("SELECT * FROM annotations WHERE camera_id = ? ORDER BY timestamp_ms ASC", (camera_id,))

    def get_in_range(self, start_ms: float, end_ms: float) -> List[Annotation]:
        return self._query("SELECT * FROM annotations WHERE timestamp_ms >= ? AND timestamp_ms <= ? ORDER BY timestamp_ms ASC", (start_ms, end_ms))

    def clear_all(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM annotations")
        except sqlite3.Error as e:
            logger.error(f"Failed to clear annotations: {e}")

    def _query(self, sql: str, params: tuple) -> List[Annotation]:
        results = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)
                for row in cursor:
                    bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else None
                    if bbox and len(bbox) == 4:
                        bbox = tuple(bbox)
                    results.append(Annotation(
                        annotation_id=row["annotation_id"],
                        timestamp_ms=row["timestamp_ms"],
                        camera_id=row["camera_id"],
                        text=row["text"],
                        created_at=row["created_at"],
                        bbox=bbox,
                        annotation_type=row["annotation_type"]
                    ))
        except sqlite3.Error as e:
            logger.error(f"Database query error: {e}")
        return results
