"""Re-Identification (ReID) feature extraction subsystem."""

from src.reid.extractor import (
    REID_MODEL_REGISTRY,
    PyTorchReIDExtractor,
    create_reid_extractor,
    get_reid_model_catalog,
    is_reid_model_supported,
    register_reid_model,
)

__all__ = [
    "REID_MODEL_REGISTRY",
    "PyTorchReIDExtractor",
    "create_reid_extractor",
    "get_reid_model_catalog",
    "is_reid_model_supported",
    "register_reid_model",
]