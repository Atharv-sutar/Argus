"""Crop quality evaluation for ReID feature extraction and gallery management."""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.core.types import BoundingBox

logger = logging.getLogger(__name__)


class CropQualityEvaluator:
    """
    Evaluates person crop quality prior to ReID embedding extraction or gallery insertion.
    
    Checks:
    - Crop pixel validity and non-emptiness
    - Bounding-box minimum width and height
    - Aspect ratio constraints (rejects abnormally flat or ultra-narrow slices)
    - Image sharpness via Laplacian variance (rejects severe motion blur)
    - Detection confidence
    """

    def __init__(
        self,
        min_width: int = 16,
        min_height: int = 35,
        min_aspect_ratio: float = 0.15,
        max_aspect_ratio: float = 1.5,
        min_sharpness: float = 0.0,
        min_confidence: float = 0.0,
    ) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_sharpness = min_sharpness
        self.min_confidence = min_confidence


    def compute_sharpness(self, crop: np.ndarray) -> float:
        """
        Computes the Laplacian variance as a proxy for image sharpness.
        Higher value means sharper image, lower value means blurred.
        """
        if crop is None or crop.size == 0 or cv2 is None:
            return 0.0
        try:
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception as e:
            logger.debug(f"Error computing sharpness: {e}")
            return 0.0

    def evaluate(
        self,
        crop: Optional[np.ndarray],
        box: Optional[BoundingBox] = None,
        confidence: Optional[float] = None,
    ) -> Tuple[bool, float, str]:
        """
        Evaluates a candidate crop.

        Args:
            crop: BGR person crop image.
            box: Optional bounding box.
            confidence: Optional detector/tracker confidence.

        Returns:
            Tuple[is_valid, quality_score, reason]:
                - is_valid: True if crop passes all quality filters.
                - quality_score: Normalized quality score [0.0, 1.0].
                - reason: Explanation if rejected or accepted.
        """
        if crop is None or crop.size == 0:
            return False, 0.0, "EMPTY_CROP"

        h, w = crop.shape[:2]

        # 1. Size checks
        if w < self.min_width:
            return False, 0.0, f"WIDTH_TOO_SMALL_{w}px"
        if h < self.min_height:
            return False, 0.0, f"HEIGHT_TOO_SMALL_{h}px"

        # 2. Aspect ratio check (width / height)
        aspect_ratio = float(w) / float(h)
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False, 0.0, f"INVALID_ASPECT_RATIO_{aspect_ratio:.2f}"

        # 3. Detection / track confidence
        conf = confidence if confidence is not None else (box.confidence if box else 1.0)
        if conf < self.min_confidence:
            return False, 0.0, f"LOW_CONFIDENCE_{conf:.2f}"

        # 4. Sharpness evaluation
        sharpness = self.compute_sharpness(crop)
        if sharpness < self.min_sharpness:
            return False, float(sharpness / max(1.0, self.min_sharpness) * 0.5), f"BLURRY_SHARPNESS_{sharpness:.1f}"

        # Calculate normalized composite quality score [0.0, 1.0]
        # Sharpness above min_sharpness scales up to 100.0
        sharp_factor = min(1.0, sharpness / max(1.0, self.min_sharpness * 2.5))
        size_factor = min(1.0, (w * h) / (60 * 160))
        conf_factor = conf

        quality_score = float(0.4 * sharp_factor + 0.3 * size_factor + 0.3 * conf_factor)
        return True, quality_score, "PASSED"


# Alias for backward/forward compatibility
ReIDCropQuality = CropQualityEvaluator

