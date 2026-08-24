"""Person Re-Identification (ReID) feature extraction module."""

from __future__ import annotations

import logging
from typing import List, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.core.interfaces import BaseReID
from src.core.types import Embedding
from src.inference.device import resolve_inference_device

logger = logging.getLogger(__name__)


class PyTorchReIDExtractor(BaseReID):
    """
    Extracts appearance feature embeddings from cropped person observations
    using a hybrid deep convolutional backbone and multi-region spatial color representation.
    """

    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        device: str = "auto",
        input_size: tuple[int, int] = (256, 128),  # Height, Width
    ) -> None:
        self.model_name = model_name
        self.device_str = resolve_inference_device(device)
        self.input_size = input_size
        self._model = None
        self._device = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            import torch
            import torchvision.models as models

            self._device = torch.device(self.device_str)
            logger.info(f"Initializing ReID model '{self.model_name}' on device '{self.device_str}'...")

            if self.model_name == "mobilenet_v3_small":
                backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                backbone.classifier = torch.nn.Identity()
                self._model = backbone.to(self._device).eval()
            elif self.model_name == "resnet50":
                backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                backbone.fc = torch.nn.Identity()
                self._model = backbone.to(self._device).eval()
            else:
                backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                backbone.classifier = torch.nn.Identity()
                self._model = backbone.to(self._device).eval()

            logger.info("ReID model initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not load PyTorch ReID model ({e}). Using color-spatial histogram fallback.")
            self._model = None

    def _preprocess(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Preprocesses crop into (C, H, W) normalized float array for PyTorch."""
        if crop is None or crop.size == 0 or cv2 is None:
            return None

        h, w = self.input_size
        resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std

        return np.transpose(normalized, (2, 0, 1))

    def _extract_spatial_color_descriptor(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts multi-region vertical color-spatial histograms (HSV and Lab)
        across 4 vertical body segments (head, upper torso, lower torso, legs).
        """
        if crop is None or crop.size == 0 or cv2 is None:
            return np.zeros(192, dtype=np.float32)

        h, w = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        h_strips = 4
        strip_h = max(1, h // h_strips)
        strip_features = []

        # Focus horizontally on the central 70% of the bounding box to sample person clothing and exclude background
        x_start = int(0.15 * w)
        x_end = max(x_start + 1, int(0.85 * w))

        for i in range(h_strips):
            y_start = i * strip_h
            y_end = (i + 1) * strip_h if i < h_strips - 1 else h
            s_hsv = hsv[y_start:y_end, x_start:x_end]
            s_lab = lab[y_start:y_end, x_start:x_end]

            # Joint Hue-Saturation histogram (16 hue bins, 8 sat bins = 128 bins)
            hist_hs = cv2.calcHist([s_hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
            # a-b color histogram in Lab space (8 a bins, 8 b bins = 64 bins)
            hist_ab = cv2.calcHist([s_lab], [1, 2], None, [8, 8], [0, 256, 0, 256]).flatten()

            s_vec = np.concatenate([hist_hs, hist_ab])
            norm = np.linalg.norm(s_vec)
            if norm > 0:
                s_vec = s_vec / norm
            strip_features.append(s_vec)

        combined = np.concatenate(strip_features)
        norm = np.linalg.norm(combined)
        return (combined / norm) if norm > 0 else combined

    def extract(self, crop: np.ndarray) -> Embedding:
        if crop is None or crop.size == 0:
            return Embedding(vector=np.zeros(128, dtype=np.float32))

        # 1. Color-spatial descriptor
        color_vec = self._extract_spatial_color_descriptor(crop)

        # 2. Deep semantic feature
        if self._model is not None:
            try:
                import torch

                prep = self._preprocess(crop)
                if prep is not None:
                    tensor = torch.from_numpy(prep).unsqueeze(0).to(self._device)
                    with torch.no_grad():
                        feat = self._model(tensor).squeeze().cpu().numpy()
                    # Zero-center deep features to remove positive activation bias
                    feat = feat - np.mean(feat)
                    norm = np.linalg.norm(feat)
                    if norm > 0:
                        feat = feat / norm
                    # Fuse deep features (80%) with spatial color features (20%)
                    fused = np.concatenate([feat * 0.8944, color_vec * 0.4472])
                    return Embedding(vector=fused)
            except Exception as e:
                logger.error(f"Error in deep ReID extraction: {e}")

        return Embedding(vector=color_vec)

    def extract_batch(self, crops: List[np.ndarray]) -> List[Embedding]:
        if not crops:
            return []

        # If PyTorch model is loaded and multiple valid crops exist, batch the forward pass
        if self._model is not None and len(crops) > 1:
            try:
                import torch

                valid_preps = []
                indices = []
                for i, crop in enumerate(crops):
                    p = self._preprocess(crop)
                    if p is not None:
                        valid_preps.append(p)
                        indices.append(i)

                if valid_preps:
                    batch_tensor = torch.from_numpy(np.stack(valid_preps)).to(self._device)
                    with torch.no_grad():
                        feats = self._model(batch_tensor).cpu().numpy()

                    if len(feats.shape) > 2:
                        feats = feats.reshape(feats.shape[0], -1)

                    embeddings: List[Optional[Embedding]] = [None] * len(crops)
                    for idx, orig_i in enumerate(indices):
                        feat = feats[idx]
                        c_vec = self._extract_spatial_color_descriptor(crops[orig_i])
                        feat = feat - np.mean(feat)
                        norm = np.linalg.norm(feat)
                        if norm > 0:
                            feat = feat / norm
                        fused = np.concatenate([feat * 0.8944, c_vec * 0.4472])
                        embeddings[orig_i] = Embedding(vector=fused)

                    # Fallback for any invalid crops in the list
                    return [
                        emb if emb is not None else self.extract(crops[i])
                        for i, emb in enumerate(embeddings)
                    ]
            except Exception as e:
                logger.error(f"Error in deep ReID batch extraction: {e}")

        return [self.extract(c) for c in crops]
