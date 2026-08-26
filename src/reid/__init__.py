"""Re-Identification (ReID) feature extraction subsystem."""

from src.reid.extractor import PyTorchReIDExtractor
from src.reid.gallery import TargetGallery
from src.core.types import GalleryEntry

__all__ = ["PyTorchReIDExtractor", "TargetGallery", "GalleryEntry"]
