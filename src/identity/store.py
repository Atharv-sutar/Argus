"""In-memory vector storage implementing BaseVectorStore with cosine similarity."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from src.core.interfaces import BaseVectorStore
from src.core.types import Embedding


class InMemoryVectorStore(BaseVectorStore):
    """
    Lightweight, fast in-memory vector store for identity embeddings.
    Computes exact cosine similarity without external database dependencies.
    """

    def __init__(self) -> None:
        self._entries: List[Tuple[Embedding, str]] = []

    def add(self, embedding: Embedding, identity_id: str) -> None:
        self._entries.append((embedding, identity_id))

    def search(self, embedding: Embedding, top_k: int = 1) -> List[Tuple[str, float]]:
        if not self._entries:
            return []

        # Track max similarity per identity ID
        best_per_identity: Dict[str, float] = {}
        for emb, ident_id in self._entries:
            sim = emb.cosine_similarity(embedding)
            if ident_id not in best_per_identity or sim > best_per_identity[ident_id]:
                best_per_identity[ident_id] = sim

        # Sort descending by similarity score
        ranked = sorted(best_per_identity.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def count(self) -> int:
        return len(self._entries)

    def remove_identity(self, identity_id: str) -> None:
        """Removes all stored embeddings for a given identity."""
        self._entries = [(emb, i_id) for emb, i_id in self._entries if i_id != identity_id]

    def clear(self) -> None:
        self._entries.clear()
