"""Identity management and vector storage subsystem."""

from src.identity.manager import IdentityManager
from src.identity.store import InMemoryVectorStore
from src.identity.faiss_store import FaissVectorStore

__all__ = ["IdentityManager", "InMemoryVectorStore", "FaissVectorStore"]
