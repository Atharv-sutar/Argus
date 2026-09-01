import pytest
import os
import sqlite3
import numpy as np
from pathlib import Path

from src.core.types import Embedding, Identity, TargetIdentityAnchor, ViewCluster
from src.identity.sqlite_store import SQLiteVectorStore
from src.identity.serialization import serialize_identity, deserialize_identity
from src.identity.manager import IdentityManager
from src.reid.extractor import PyTorchReIDExtractor

@pytest.fixture
def temp_db_path(tmp_path):
    db_path = tmp_path / "test_identities.db"
    yield str(db_path)
    if db_path.exists():
        os.remove(db_path)

def test_sqlite_vector_store_basic_operations(temp_db_path):
    store = SQLiteVectorStore(db_path=temp_db_path)
    
    # Add a mock embedding
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb = Embedding(vector=vec, model_name="test_model", version="1.0", crop_type="full")
    
    store.add(emb, "target_1")
    assert store.count() == 1
    
    # Test search (exact match)
    results = store.search(emb, top_k=1)
    assert len(results) == 1
    assert results[0][0] == "target_1"
    assert results[0][1] > 0.99
    
    # Test remove
    store.remove_identity("target_1")
    assert store.count() == 0
    
    # Test metadata save/load
    test_data = {"key": "value", "num": 42}
    store.save_identity_metadata("target_2", test_data)
    
    loaded = store.load_all_identity_metadata()
    assert "target_2" in loaded
    assert loaded["target_2"]["key"] == "value"
    assert loaded["target_2"]["num"] == 42
    
    # Test clear
    store.clear()
    assert len(store.load_all_identity_metadata()) == 0
    assert store.count() == 0

def test_identity_serialization_and_manager_persistence(temp_db_path):
    store = SQLiteVectorStore(db_path=temp_db_path)
    
    # Create mock identity manager
    class MockReID:
        def extract(self, crop):
            pass
            
    manager = IdentityManager(reid_extractor=MockReID(), vector_store=store)
    
    # Construct a rich Identity object
    ident = Identity(identity_id="target_0", label="Test Target")
    
    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    emb1 = Embedding(vector=vec1)
    emb2 = Embedding(vector=vec2)
    
    ident.trusted_gallery.append(emb1)
    ident.provisional_gallery.append(emb2)
    ident.confidence = 0.95
    ident.last_seen_timestamp_ms = 12345.6
    
    # Add an anchor
    anchor = TargetIdentityAnchor(
        identity_id="target_0", 
        label="Test Anchor",
        model_name="osnet_x0_25",
        feature_dim=512
    )
    cluster = ViewCluster(cluster_id="front")
    cluster.exemplars.append(emb1)
    anchor.clusters.append(cluster)
    ident.anchor = anchor
    
    # Inject it directly into manager for saving
    manager._identities["target_0"] = ident
    
    # Save to DB
    manager.save_to_db(temp_db_path)
    
    # Recreate manager and load
    new_manager = IdentityManager(reid_extractor=MockReID(), vector_store=store)
    new_manager.load_from_db(temp_db_path)
    
    # Verify loaded identity
    loaded_ident = new_manager.get_identity("target_0")
    assert loaded_ident is not None
    assert loaded_ident.label == "Test Target"
    assert loaded_ident.confidence == 0.95
    assert loaded_ident.last_seen_timestamp_ms == 12345.6
    
    # Verify galleries
    assert len(loaded_ident.trusted_gallery) == 1
    assert np.array_equal(loaded_ident.trusted_gallery[0].vector, vec1)
    
    assert len(loaded_ident.provisional_gallery) == 1
    assert np.array_equal(loaded_ident.provisional_gallery[0].vector, vec2)
    
    # Verify anchor
    assert loaded_ident.anchor is not None
    assert loaded_ident.anchor.label == "Test Anchor"
    assert len(loaded_ident.anchor.clusters) == 1
    assert loaded_ident.anchor.clusters[0].cluster_id == "front"
    assert len(loaded_ident.anchor.clusters[0].exemplars) == 1
    assert np.array_equal(loaded_ident.anchor.clusters[0].exemplars[0].vector, vec1)
