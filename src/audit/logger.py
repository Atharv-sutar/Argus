"""
Forensic Audit Logger module for immutable chain-of-custody tracking.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

logger = logging.getLogger(__name__)

class AuditEventType(str, Enum):
    INVESTIGATION_START = "INVESTIGATION_START"
    INVESTIGATION_END = "INVESTIGATION_END"
    TARGET_SELECTED = "TARGET_SELECTED"
    TARGET_LOCKED = "TARGET_LOCKED"
    TARGET_LOST = "TARGET_LOST"
    SEARCH_START = "SEARCH_START"
    SEARCH_EXPAND = "SEARCH_EXPAND"
    REID_MATCH = "REID_MATCH"
    HANDOFF_PROPOSED = "HANDOFF_PROPOSED"
    HANDOFF_ACCEPTED = "HANDOFF_ACCEPTED"
    HANDOFF_REJECTED = "HANDOFF_REJECTED"
    HUMAN_CORRECTION = "HUMAN_CORRECTION"
    GALLERY_ADD = "GALLERY_ADD"
    GALLERY_REMOVE = "GALLERY_REMOVE"
    GALLERY_CLEAR = "GALLERY_CLEAR"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    ANNOTATION_ADD = "ANNOTATION_ADD"
    UNDO = "UNDO"
    REDO = "REDO"


class AuditLogger:
    """
    Appends events to a SQLite database. Implements a hash chain where
    each row's hash is derived from the previous row's hash to prevent tampering.
    """
    
    GENESIS_HASH = hashlib.sha256(b"AUDIT_LOG_GENESIS").hexdigest()

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp_utc TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        camera_id TEXT,
                        track_id INTEGER,
                        detail_json TEXT NOT NULL,
                        prev_hash TEXT NOT NULL,
                        row_hash TEXT NOT NULL
                    )
                """)
                conn.commit()

    def _compute_hash(self, timestamp: str, event_type: str, detail_json: str, prev_hash: str) -> str:
        raw = f"{timestamp}|{event_type}|{detail_json}|{prev_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_last_hash(self, cursor: sqlite3.Cursor) -> str:
        cursor.execute("SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else self.GENESIS_HASH

    def log(
        self,
        event_type: AuditEventType,
        detail: Dict[str, Any],
        camera_id: Optional[str] = None,
        track_id: Optional[int] = None
    ) -> None:
        """Log an event securely with hash-chaining."""
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        detail_json = json.dumps(detail, sort_keys=True)
        
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    prev_hash = self._get_last_hash(cursor)
                    row_hash = self._compute_hash(timestamp_utc, event_type.value, detail_json, prev_hash)
                    
                    cursor.execute("""
                        INSERT INTO audit_log 
                        (timestamp_utc, event_type, camera_id, track_id, detail_json, prev_hash, row_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (timestamp_utc, event_type.value, camera_id, track_id, detail_json, prev_hash, row_hash))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to write to audit log: {e}")

    def get_logs(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve recent logs, primarily for UI viewing."""
        logs = []
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT * FROM audit_log 
                        ORDER BY id DESC LIMIT ? OFFSET ?
                    """, (limit, offset))
                    
                    for row in cursor.fetchall():
                        logs.append({
                            "id": row["id"],
                            "timestamp_utc": row["timestamp_utc"],
                            "event_type": row["event_type"],
                            "camera_id": row["camera_id"],
                            "track_id": row["track_id"],
                            "detail": json.loads(row["detail_json"]),
                            "valid": True # Verification is separate
                        })
            except Exception as e:
                logger.error(f"Failed to read audit log: {e}")
        return logs

    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Verify the entire audit log integrity by recalculating all hashes.
        Returns (is_valid, error_message)
        """
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM audit_log ORDER BY id ASC")
                    
                    expected_prev = self.GENESIS_HASH
                    for row in cursor.fetchall():
                        if row["prev_hash"] != expected_prev:
                            return False, f"Broken chain at row {row['id']}: prev_hash mismatch"
                            
                        computed_hash = self._compute_hash(
                            row["timestamp_utc"], 
                            row["event_type"], 
                            row["detail_json"], 
                            row["prev_hash"]
                        )
                        
                        if row["row_hash"] != computed_hash:
                            return False, f"Tampered data at row {row['id']}: row_hash mismatch"
                            
                        expected_prev = computed_hash
                        
                return True, None
            except Exception as e:
                logger.error(f"Failed to verify audit log: {e}")
                return False, str(e)
