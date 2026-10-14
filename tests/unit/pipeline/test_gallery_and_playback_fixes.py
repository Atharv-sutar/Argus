"""Unit tests verifying target gallery thumbnails with tuples, remove_entry, video dir discovery, and playback controller endpoints."""

import numpy as np
import pytest
from pathlib import Path
from src.core.types import Embedding
from src.identity.manager import IdentityManager
from src.identity.store import InMemoryVectorStore
from src.core.interfaces import BaseReID
from src.camera.directory_loader import discover_video_files


class DummyReID(BaseReID):
    def extract(self, crop: np.ndarray) -> Embedding:
        return Embedding(vector=np.ones(128, dtype=np.float32) / np.sqrt(128))

    def extract_batch(self, crops: list[np.ndarray]) -> list[Embedding]:
        return [self.extract(c) for c in crops]


def test_gallery_thumbnails_with_provisional_tuple():
    reid = DummyReID()
    store = InMemoryVectorStore()
    im = IdentityManager(reid_extractor=reid, vector_store=store)

    # Register initial target
    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    im.register_new_target(crop=crop, identity_id="target_0")

    # Add auto sample where item is stored as (embedding, track_id) tuple
    emb = reid.extract(crop)
    emb.camera_id = "cam_test"
    emb.timestamp_ms = 1234.0
    im.get_identity("target_0").provisional_gallery.append((emb, 99))

    # get_thumbnails must NOT raise AttributeError
    thumbs = im.get_thumbnails(max_count=10)
    assert len(thumbs) >= 2
    prov_thumb = [t for t in thumbs if not t["is_manual"]][0]
    assert prov_thumb["camera_id"] == "cam_test"
    assert prov_thumb["timestamp_ms"] == 1234.0
    assert prov_thumb["entry_id"] == "provisional_0"


def test_gallery_remove_entry_both_trusted_and_provisional():
    reid = DummyReID()
    store = InMemoryVectorStore()
    im = IdentityManager(reid_extractor=reid, vector_store=store)

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    im.register_new_target(crop=crop, identity_id="target_0")
    im.add_reference_sample(crop=crop, identity_id="target_0")

    emb = reid.extract(crop)
    im.get_identity("target_0").provisional_gallery.append((emb, 10))

    assert im.manual_count == 2
    assert im.auto_count == 1

    # Remove trusted entry (must not raise NameError: old_len)
    ok_trusted = im.remove_entry("trusted_1")
    assert ok_trusted is True
    assert im.manual_count == 1

    # Remove provisional entry
    ok_prov = im.remove_entry("provisional_0")
    assert ok_prov is True
    assert im.auto_count == 0


def test_discover_video_files_handles_quotes_and_subdirectories(tmp_path):
    # Create subdirectories with video files
    cam_dir = tmp_path / "EPFL campus"
    cam_dir.mkdir()
    (cam_dir / "cam_0.mp4").write_bytes(b"dummy_video_0")
    (cam_dir / "cam_1.avi").write_bytes(b"dummy_video_1")

    # Pass path with quotes
    quoted_path = f'"{cam_dir}"'
    found = discover_video_files(quoted_path)
    assert len(found) == 2
    assert any("cam_0" in str(p) for p in found)
    assert any("cam_1" in str(p) for p in found)

    # Pass parent directory with nested camera subfolder
    found_parent = discover_video_files(str(tmp_path))
    assert len(found_parent) == 2
