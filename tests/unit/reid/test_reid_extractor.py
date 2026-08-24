"""Unit tests for ReID feature extractor."""

import numpy as np
import pytest
from src.core.types import Embedding
from src.reid.extractor import PyTorchReIDExtractor


def test_embedding_normalization_and_similarity():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    emb1 = Embedding(vector=v1)
    emb2 = Embedding(vector=v2)
    emb3 = Embedding(vector=v3)

    assert pytest.approx(emb1.cosine_similarity(emb2), 0.001) == 1.0
    assert pytest.approx(emb1.cosine_similarity(emb3), 0.001) == 0.0


def test_reid_extractor_single_crop():
    extractor = PyTorchReIDExtractor(device="cpu")
    # Synthetic RGB person crop: 128 height, 64 width
    crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
    emb = extractor.extract(crop)

    assert isinstance(emb, Embedding)
    assert emb.dim > 0
    # Check L2 normalized
    norm = float(np.linalg.norm(emb.vector))
    assert pytest.approx(norm, 0.01) == 1.0


def test_reid_extractor_empty_crop():
    extractor = PyTorchReIDExtractor(device="cpu")
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    emb = extractor.extract(empty_crop)

    assert isinstance(emb, Embedding)
    assert emb.dim > 0


def test_reid_extractor_batch():
    extractor = PyTorchReIDExtractor(device="cpu")
    crop1 = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
    crop2 = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)

    embs = extractor.extract_batch([crop1, crop2])
    assert len(embs) == 2
    assert embs[0].dim == embs[1].dim
