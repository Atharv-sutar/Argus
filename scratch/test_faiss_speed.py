import time
import numpy as np
from src.core.types import Embedding
from src.identity.faiss_store import FaissVectorStore

def test_faiss_performance():
    store = FaissVectorStore(feature_dim=512)
    
    print("--- FAISS Vector Storage Performance Test ---")
    print(f"Feature Dimension: {store.feature_dim}")
    
    # 1. Test insertion time for 10,000 identities
    n_identities = 10000
    print(f"\nGenerating {n_identities} random embeddings...")
    embeddings = [Embedding(np.random.randn(512).astype(np.float32)) for _ in range(n_identities)]
    
    print("Inserting into FAISS...")
    start = time.time()
    for i, emb in enumerate(embeddings):
        store.add(emb, f"identity_{i}")
    insert_time = time.time() - start
    
    print(f"Insertion of {n_identities} vectors took: {insert_time:.4f} seconds")
    print(f"Average insertion time: {(insert_time/n_identities)*1000:.4f} ms per vector")
    
    # 2. Test search time
    query_emb = embeddings[n_identities // 2]  # Query one of the existing embeddings
    
    print("\nSearching in FAISS...")
    start = time.time()
    results = store.search(query_emb, top_k=5)
    search_time = time.time() - start
    
    print(f"Search operation took: {search_time * 1000:.4f} ms")
    print(f"Top 5 Search Results: {results}")
    
    assert results[0][0] == f"identity_{n_identities // 2}", "FAISS did not return the exact match as top result!"
    print("\n✅ FAISS storage and retrieval verified successfully.")

if __name__ == "__main__":
    test_faiss_performance()
