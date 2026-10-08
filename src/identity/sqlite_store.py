"""SQLite-backed persistent vector storage implementing BaseVectorStore."""

from __future__ import annotations

import sqlite3
import json
import base64
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from contextlib import closing

from src.core.interfaces import BaseVectorStore
from src.core.types import Embedding


class SQLiteVectorStore(BaseVectorStore):
    """
    SQLite-backed vector store for identity embeddings.
    Computes exact cosine similarity and persists data across restarts.
    """

    def __init__(self, db_path: str = "data/identities.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
        # Initialize FAISS backend for fast search
        from src.identity.faiss_store import FaissVectorStore
        self._faiss = FaissVectorStore()
        self._load_all_to_faiss()

    def _load_all_to_faiss(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, identity_id, vector_b64, model_name, version, 
                       crop_type, quality_score, camera_id, timestamp_ms
                FROM embeddings
            ''')
            rows = cursor.fetchall()
            for row in rows:
                ident_id = row[1]
                emb = self._deserialize_emb(row)
                self._faiss.add(emb, ident_id)

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_id TEXT NOT NULL,
                    vector_b64 TEXT NOT NULL,
                    model_name TEXT,
                    version TEXT,
                    crop_type TEXT,
                    quality_score REAL,
                    camera_id TEXT,
                    timestamp_ms REAL
                )
            ''')
            # Create an index to quickly remove or lookup by identity
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_identity_id ON embeddings (identity_id)')
            
            # Create a table for Identity metadata (JSON blobs)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS identities_metadata (
                    identity_id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL
                )
            ''')
            conn.commit()

    def save_identity_metadata(self, identity_id: str, data: dict) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO identities_metadata (identity_id, data_json)
                VALUES (?, ?)
            ''', (identity_id, json.dumps(data)))
            conn.commit()

    def load_all_identity_metadata(self) -> Dict[str, dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT identity_id, data_json FROM identities_metadata')
            rows = cursor.fetchall()
            return {row[0]: json.loads(row[1]) for row in rows}



    def _serialize_emb(self, emb: Embedding) -> str:
        # Base64 encode the flattened float32 array
        return base64.b64encode(emb.vector.tobytes()).decode("ascii")

    def _deserialize_emb(self, row: tuple) -> Embedding:
        (
            id_, identity_id, vector_b64, model_name, version,
            crop_type, quality_score, camera_id, timestamp_ms
        ) = row
        
        vector = np.frombuffer(base64.b64decode(vector_b64), dtype=np.float32)
        emb = Embedding(
            vector=vector,
            model_name=model_name,
            version=version,
            crop_type=crop_type,
            quality_score=quality_score,
            camera_id=camera_id,
            timestamp_ms=timestamp_ms
        )
        return emb

    def add(self, embedding: Embedding, identity_id: str) -> None:
        # Add to FAISS memory
        self._faiss.add(embedding, identity_id)
        # Persist to SQLite
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO embeddings (
                    identity_id, vector_b64, model_name, version, 
                    crop_type, quality_score, camera_id, timestamp_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                identity_id,
                self._serialize_emb(embedding),
                embedding.model_name,
                embedding.version,
                embedding.crop_type,
                embedding.quality_score,
                embedding.camera_id,
                embedding.timestamp_ms
            ))
            conn.commit()

    def search(self, embedding: Embedding, top_k: int = 1) -> List[Tuple[str, float]]:
        # Lightning fast FAISS search instead of SQLite full table scan
        return self._faiss.search(embedding, top_k)

    def get_embeddings_for_identity(self, identity_id: str) -> List[Embedding]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, identity_id, vector_b64, model_name, version, 
                       crop_type, quality_score, camera_id, timestamp_ms
                FROM embeddings
                WHERE identity_id = ?
                ORDER BY id ASC
            ''', (identity_id,))
            rows = cursor.fetchall()
            return [self._deserialize_emb(row) for row in rows]

    def count(self) -> int:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM embeddings')
            return cursor.fetchone()[0]

    def remove_identity(self, identity_id: str) -> None:
        self._faiss.remove_identity(identity_id)
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM embeddings WHERE identity_id = ?', (identity_id,))
            cursor.execute('DELETE FROM identities_metadata WHERE identity_id = ?', (identity_id,))
            conn.commit()

    def clear(self) -> None:
        self._faiss.clear()
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM embeddings')
            cursor.execute('DELETE FROM identities_metadata')
            conn.commit()
