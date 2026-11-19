"""Unit tests for embedding_quality discrimination evaluation script."""

import numpy as np
import pytest

from scripts.embedding_quality import compute_roc_auc, evaluate_embeddings


def test_compute_roc_auc_perfect():
    genuine = np.array([0.9, 0.85, 0.88, 0.92], dtype=np.float32)
    impostor = np.array([0.2, 0.3, 0.25, 0.35], dtype=np.float32)
    auc = compute_roc_auc(genuine, impostor)
    assert auc == 1.0


def test_compute_roc_auc_overlapping():
    genuine = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    impostor = np.array([0.4, 0.6, 0.8], dtype=np.float32)
    auc = compute_roc_auc(genuine, impostor)
    assert 0.0 <= auc <= 1.0


def test_evaluate_embeddings_synthetic_good_separation():
    # 2 persons, 5 vectors each
    # Person A vectors clustered around e1
    # Person B vectors clustered around e2
    np.random.seed(42)
    dim = 512
    base_a = np.zeros(dim, dtype=np.float32)
    base_a[0] = 1.0
    base_b = np.zeros(dim, dtype=np.float32)
    base_b[1] = 1.0

    labeled = []
    for _ in range(5):
        vec = base_a + np.random.normal(0, 0.05, dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        labeled.append(("person_A", vec))

    for _ in range(5):
        vec = base_b + np.random.normal(0, 0.05, dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        labeled.append(("person_B", vec))

    res = evaluate_embeddings(labeled)
    assert res["num_identities"] == 2
    assert res["total_crops"] == 10
    assert res["genuine_pairs"] > 0
    assert res["impostor_pairs"] > 0
    assert res["genuine"]["mean"] > res["impostor"]["mean"]
    assert res["rank1_rate_percent"] == 100.0
    assert res["roc_auc"] >= 0.95
    assert "SEPARATES PEOPLE" in res["verdict"]


def test_evaluate_embeddings_insufficient_identities():
    labeled = [("person_A", np.ones(512, dtype=np.float32)), ("person_A", np.ones(512, dtype=np.float32))]
    res = evaluate_embeddings(labeled)
    assert "error" in res
