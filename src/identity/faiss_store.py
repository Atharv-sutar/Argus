"""FAISS-backed vector storage implementing BaseVectorStore."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from src.core.interfaces import BaseVectorStore
from src.core.types import Embedding

logger = logging.getLogger(__name__)

class FaissVectorStore(BaseVectorStore):
    """
    High-performance vector store using FAISS for identity embeddings.
    Provides scalable dense vector similarity search.
    """

    def __str__(self) -> str:
        return "FAISS Vector Store"

    def __init__(self, feature_dim: int = 512) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed. FaissVectorStore requires faiss-cpu or faiss-gpu.")
            
        self.feature_dim = feature_dim
        # IndexFlatIP calculates inner product. 
        # Equivalent to cosine similarity when vectors are L2-normalized.
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.feature_dim))
        
        self._current_faiss_id = 0
        self._faiss_id_to_identity: Dict[int, str] = {}
        self._identity_to_faiss_ids: Dict[str, List[int]] = {}

    def add(self, embedding: Embedding, identity_id: str) -> None:
        vec = embedding.vector
        if vec.shape[0] != self.feature_dim:
            logger.warning(f"Embedding dim {vec.shape[0]} doesn't match FAISS dim {self.feature_dim}. Ignored.")
            return

        vec_2d = np.expand_dims(vec, axis=0).astype(np.float32)
        
        faiss_id = self._current_faiss_id
        self._current_faiss_id += 1
        
        faiss_id_array = np.array([faiss_id], dtype=np.int64)
        self.index.add_with_ids(vec_2d, faiss_id_array)
        
        self._faiss_id_to_identity[faiss_id] = identity_id
        if identity_id not in self._identity_to_faiss_ids:
            self._identity_to_faiss_ids[identity_id] = []
        self._identity_to_faiss_ids[identity_id].append(faiss_id)

    def search(self, embedding: Embedding, top_k: int = 1) -> List[Tuple[str, float]]:
        if self.count() == 0:
            return []

        vec = embedding.vector
        if vec.shape[0] != self.feature_dim:
            return []

        vec_2d = np.expand_dims(vec, axis=0).astype(np.float32)
        
        # Search deeper because multiple vectors map to same identity
        search_k = min(self.count(), top_k * 10)
        distances, indices = self.index.search(vec_2d, search_k)
        
        best_per_identity: Dict[str, float] = {}
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            ident_id = self._faiss_id_to_identity.get(idx)
            if ident_id:
                sim = float(dist)
                if ident_id not in best_per_identity or sim > best_per_identity[ident_id]:
                    best_per_identity[ident_id] = sim

        ranked = sorted(best_per_identity.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def count(self) -> int:
        return self.index.ntotal

    def remove_identity(self, identity_id: str) -> None:
        faiss_ids = self._identity_to_faiss_ids.get(identity_id, [])
        if not faiss_ids:
            return
            
        faiss_ids_array = np.array(faiss_ids, dtype=np.int64)
        self.index.remove_ids(faiss_ids_array)
        
        for fid in faiss_ids:
            self._faiss_id_to_identity.pop(fid, None)
            
        self._identity_to_faiss_ids.pop(identity_id, None)

    def clear(self) -> None:
        self.index.reset()
        self._faiss_id_to_identity.clear()
        self._identity_to_faiss_ids.clear()
        self._current_faiss_id = 0
