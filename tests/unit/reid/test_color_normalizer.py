import numpy as np
import pytest

from src.reid.color_normalizer import ColorNormalizer


def test_color_normalizer_empty_and_none():
    normalizer = ColorNormalizer()
    assert normalizer.normalize(None) is None
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert normalizer.normalize(empty).size == 0


def test_color_normalizer_shape_and_type():
    normalizer = ColorNormalizer(apply_gray_world=True, apply_clahe=True)
    img = np.random.randint(0, 256, (128, 64, 3), dtype=np.uint8)
    out = normalizer.normalize(img)
    assert out.shape == (128, 64, 3)
    assert out.dtype == np.uint8


def test_color_normalizer_gaussian_mask():
    mask = ColorNormalizer.get_gaussian_center_mask(100, 50)
    assert mask.shape == (100, 50)
    assert mask.dtype == np.float32
    # Center should have higher weight than boundary
    center_val = mask[50, 25]
    corner_val = mask[0, 0]
    assert center_val > corner_val
    assert np.all(mask >= 0.15)
    assert np.all(mask <= 1.0)
