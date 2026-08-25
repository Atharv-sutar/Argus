"""
Color Normalization & Illumination Invariance Module.

Provides:
- Gray-World white balance normalization for cross-camera sensor color calibration.
- Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel in Lab space.
- Soft Gaussian center-weighted spatial masking to suppress background edge pixels.
"""

from __future__ import annotations

import cv2
import numpy as np


class ColorNormalizer:
    """
    Normalizes person crop imagery to be invariant across disparate camera sensors,
    exposure levels, and white-balance temperatures.
    """

    def __init__(
        self,
        apply_gray_world: bool = True,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        self.apply_gray_world = apply_gray_world
        self.apply_clahe = apply_clahe
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size,
        )

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Applies white-balance and contrast normalization to an input BGR image.

        Args:
            image: BGR uint8 image array (H, W, 3).

        Returns:
            Normalized BGR uint8 image array (H, W, 3).
        """
        if image is None or image.size == 0:
            return image

        img = image.copy()

        # 1. Gray-World White Balance Normalization
        if self.apply_gray_world:
            img = self._gray_world_balance(img)

        # 2. CLAHE Lightness Normalization (in Lab space)
        if self.apply_clahe:
            img = self._apply_clahe_lab(img)

        return img

    def _gray_world_balance(self, bgr_img: np.ndarray) -> np.ndarray:
        """Scales B, G, R channels so their mean values match the overall gray average."""
        b, g, r = cv2.split(bgr_img.astype(np.float32))
        mean_b = np.mean(b) + 1e-6
        mean_g = np.mean(g) + 1e-6
        mean_r = np.mean(r) + 1e-6

        mean_gray = (mean_b + mean_g + mean_r) / 3.0

        b = np.clip(b * (mean_gray / mean_b), 0, 255)
        g = np.clip(g * (mean_gray / mean_g), 0, 255)
        r = np.clip(r * (mean_gray / mean_r), 0, 255)

        return cv2.merge([b, g, r]).astype(np.uint8)

    def _apply_clahe_lab(self, bgr_img: np.ndarray) -> np.ndarray:
        """Applies CLAHE on the L channel of the Lab color space."""
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab_clahe = cv2.merge([l, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    @staticmethod
    def get_gaussian_center_mask(height: int, width: int) -> np.ndarray:
        """
        Generates an elliptical Gaussian soft mask with higher weights at the center
        and lower weights at the outer rectangular boundaries to suppress background pixels.

        Args:
            height: Image height.
            width: Image width.

        Returns:
            2D float32 array of shape (height, width) with values in (0, 1].
        """
        if height <= 0 or width <= 0:
            return np.ones((1, 1), dtype=np.float32)

        # Gaussian parameters
        y, x = np.ogrid[:height, :width]
        cy, cx = height / 2.0, width / 2.0
        # 2-sigma boundary at width/2 and height/2
        sigma_x = width / 2.5
        sigma_y = height / 2.2

        mask = np.exp(-(((x - cx) ** 2) / (2 * (sigma_x ** 2)) + ((y - cy) ** 2) / (2 * (sigma_y ** 2))))
        # Scale to range [0.15, 1.0] so edge pixels still contribute slightly
        mask = 0.15 + 0.85 * mask
        return mask.astype(np.float32)
