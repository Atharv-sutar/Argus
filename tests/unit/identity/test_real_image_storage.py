"""Tests verifying that real stored images on disk participate in ReID and gallery storage."""

import os
import shutil
import tempfile
import numpy as np
import pytest
import cv2

from src.core.interfaces import BaseReID
from src.core.types import Embedding
from src.identity.manager import IdentityManager
from src.identity.sqlite_store import SQLiteVectorStore


class MockDeterministicReID(BaseReID):
    """Deterministic ReID mock that extracts color-based embeddings from real image arrays."""

    def __init__(self, dim: int = 128):
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def extract_batch(self, crops):
        return [self.extract(c) for c in crops]

    def extract(self, crop: np.ndarray) -> Embedding:
        assert isinstance(crop, np.ndarray), "Crop must be real numpy array"
        assert crop.size > 0, "Crop must not be empty"
        # Deterministic feature from image color and spatial moments
        h, w = crop.shape[:2]
        mean_colors = crop.mean(axis=(0, 1)) if crop.ndim == 3 else [float(crop.mean())] * 3
        # Generate stable vector
        vec = np.zeros(self._dim, dtype=np.float32)
        for i in range(self._dim):
            c_idx = i % len(mean_colors)
            vec[i] = float(mean_colors[c_idx]) * (1.0 + float(i) / self._dim) + float(h * w % 17)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return Embedding(vector=vec)


@pytest.fixture
def temp_gallery_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_gallery_crops_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_real_stored_images_saved_and_verified_on_disk(temp_gallery_dir):
    """
    Priority 0 & 1 Verification:
    1. Verify actual image files are saved to disk when target is registered.
    2. Verify image files contain real RGB data, not placeholders.
    3. Verify verify_gallery_from_disk() loads from disk and matches embeddings (>0.98).
    4. Verify rebuild_gallery_from_disk() reconstructs the gallery from disk.
    """
    reid = MockDeterministicReID()
    manager = IdentityManager(
        reid_extractor=reid,
        crops_dir=temp_gallery_dir,
        max_reference_samples=50,
        max_gallery_size=100,
    )

    # Create real synthetic crops with distinct visual patterns
    crop_person_a = np.zeros((120, 60, 3), dtype=np.uint8)
    crop_person_a[:, :, 0] = 180  # Blue shirt
    crop_person_a[:, :, 1] = 50
    crop_person_a[:, :, 2] = 40

    # Register target Person A
    emb = manager.register_new_target(crop_person_a, identity_id="target_0", label="Person_A")
    assert emb is not None

    # 1. Verify image file exists on disk
    disk_paths = manager.get_gallery_image_paths("target_0")
    assert len(disk_paths) == 1
    assert os.path.isfile(disk_paths[0])
    assert "target_0" in disk_paths[0]
    assert disk_paths[0].endswith(".jpg")

    # 2. Verify stored image file is a valid image with matching shape and data
    loaded_img = cv2.imread(disk_paths[0])
    assert loaded_img is not None
    assert loaded_img.shape == crop_person_a.shape
    # Check that colors match reasonably (within JPEG compression tolerance)
    assert np.allclose(loaded_img.mean(axis=(0, 1)), crop_person_a.mean(axis=(0, 1)), atol=3.0)

    # 3. Verify verify_gallery_from_disk reads from disk and validates re-extracted embedding
    report = manager.verify_gallery_from_disk("target_0")
    assert report["verified"] is True
    assert report["verified_count"] == 1
    assert report["total_entries"] == 1
    assert report["details"][0]["status"] == "VERIFIED_REAL_IMAGE"
    assert report["details"][0]["reextracted_similarity"] >= 0.98

    # 4. Verify rebuild_gallery_from_disk reconstructs gallery directly from disk
    rebuilt_count = manager.rebuild_gallery_from_disk("target_0")
    assert rebuilt_count == 1
    ident = manager.get_identity("target_0")
    assert ident is not None
    assert len(ident.trusted_gallery) == 1
    assert getattr(ident.trusted_gallery[0], "image_path", None) == disk_paths[0]


def test_gallery_storage_limit_surpasses_seven_images(temp_gallery_dir):
    """
    Priority 1 Verification:
    Verify that the image gallery is NOT limited to 7 images.
    Add 12 distinct reference crops; all 12 must be stored, retrieved, and verified.
    """
    reid = MockDeterministicReID()
    manager = IdentityManager(
        reid_extractor=reid,
        crops_dir=temp_gallery_dir,
        max_reference_samples=50,
        max_gallery_size=100,
    )

    # Initial seed
    seed_crop = np.full((100, 50, 3), 50, dtype=np.uint8)
    manager.register_new_target(seed_crop, identity_id="target_0", label="Person_A")

    # Add 11 more diverse samples (total 12)
    for i in range(1, 12):
        crop = np.full((100, 50, 3), 50 + i * 15, dtype=np.uint8)
        # Add visual texture to ensure diversity
        crop[i * 4:(i * 4) + 10, :, 1] = 220
        ok = manager.add_reference_sample(crop, identity_id="target_0", timestamp_ms=float(i * 1000), force=True)
        assert ok is True

    ident = manager.get_identity("target_0")
    assert ident is not None
    # Verify gallery size surpasses 7
    assert len(ident.trusted_gallery) == 12, f"Expected 12 images, but got {len(ident.trusted_gallery)}"
    assert manager.size == 12

    # Verify thumbnails expose all 12 items with stored disk image indicators
    thumbnails = manager.get_thumbnails(max_count=100)
    assert len(thumbnails) == 12
    for thumb in thumbnails:
        assert thumb["has_stored_image"] is True
        assert os.path.isfile(thumb["image_path"])

    # Verify all 12 disk image files pass verification
    report = manager.verify_gallery_from_disk("target_0")
    assert report["verified"] is True
    assert report["verified_count"] == 12


def test_gallery_persistence_with_disk_images(temp_gallery_dir):
    """
    Priority 1 Verification:
    Verify that gallery with on-disk images persists across database save and reload.
    """
    db_path = os.path.join(temp_gallery_dir, "test_gallery.db")
    store = SQLiteVectorStore(db_path=db_path)
    reid = MockDeterministicReID()

    # Create manager and enroll target
    manager1 = IdentityManager(
        reid_extractor=reid,
        vector_store=store,
        crops_dir=temp_gallery_dir,
        db_path=db_path,
        max_reference_samples=50,
    )

    crop = np.full((100, 50, 3), 120, dtype=np.uint8)
    crop[:, :, 2] = 200
    manager1.register_new_target(crop, identity_id="target_0", label="Target_Persist")

    # Add 2 more samples
    for i in range(1, 3):
        c = np.full((100, 50, 3), 120 + i * 20, dtype=np.uint8)
        manager1.add_reference_sample(c, identity_id="target_0", force=True)

    manager1.save_to_db(db_path)

    # Now create a fresh manager instance and load from SQLite
    store2 = SQLiteVectorStore(db_path=db_path)
    manager2 = IdentityManager(
        reid_extractor=reid,
        vector_store=store2,
        crops_dir=temp_gallery_dir,
        db_path=db_path,
        max_reference_samples=50,
    )
    manager2.load_from_db(db_path)

    ident2 = manager2.get_identity("target_0")
    assert ident2 is not None
    assert len(ident2.trusted_gallery) == 3

    # Verify all disk images are intact and pass verification in new manager
    report = manager2.verify_gallery_from_disk("target_0")
    assert report["verified"] is True
    assert report["verified_count"] == 3
