"""Person Re-Identification (ReID) multi-region and texture-enriched feature extraction module."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.core.interfaces import BaseReID
from src.core.types import Embedding
from src.reid.color_normalizer import ColorNormalizer
from src.inference.device import resolve_inference_device

logger = logging.getLogger(__name__)


class PyTorchReIDExtractor(BaseReID):
    """
    High-capacity Person Re-Identification feature extractor supporting:
    1. Omni-Scale Network (OSNet-x0.25 / OSNet-x1.0, 512D) - Primary ReID backbone
    2. Multi-crop representations (Full Body, Upper Torso, Lower Body)
    3. Fast batched GPU/CPU inference with rich metadata
    """

    def __init__(
        self,
        model_name: str = "osnet_x0_25",
        device: str = "auto",
        input_size: Tuple[int, int] = (256, 128),
    ) -> None:
        self.model_name = model_name.lower()
        self.device_str = resolve_inference_device(device)
        self.input_size = input_size
        self.feature_dim = 512 if "dino" not in self.model_name else 384
        self._model = None
        self._osnet_model = None
        self._device = None
        self.color_normalizer = ColorNormalizer(apply_gray_world=True, apply_clahe=True)
        self._init_model()

    def _init_model(self) -> None:
        try:
            import torch
            self._device = torch.device(self.device_str)
            logger.info(f"Initializing ReID model '{self.model_name}' on device '{self.device_str}'...")

            if "osnet" in self.model_name or self.model_name in ("default", "reid"):
                from src.reid.backbones.osnet import build_osnet
                self._model = build_osnet(self.model_name if "x1_0" in self.model_name else "osnet_x0_25", pretrained=True).to(self._device).eval()
                self.feature_dim = 512
            elif "dino" in self.model_name:
                self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self._device).eval()
                self.feature_dim = 384
            elif "mobilenet" in self.model_name:
                import torchvision.models as models
                backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                backbone.classifier = torch.nn.Identity()
                self._model = backbone.to(self._device).eval()
                self.feature_dim = 576
            else:
                from src.reid.backbones.osnet import build_osnet
                self._model = build_osnet("osnet_x0_25", pretrained=True).to(self._device).eval()
                self.feature_dim = 512

            logger.info(f"ReID model '{self.model_name}' (dim={self.feature_dim}) initialized successfully on {self._device}.")
        except Exception as e:
            logger.warning(f"Could not load Person-ReID model ({e}). Attempting fallback...")
            try:
                from src.reid.backbones.osnet import build_osnet
                self._model = build_osnet("osnet_x0_25", pretrained=True).to(self._device).eval()
                self.feature_dim = 512
            except Exception as e2:
                logger.error(f"Fallback ReID initialization failed: {e2}")
                self._model = None

    def _apply_foreground_isolation(self, crop: np.ndarray) -> np.ndarray:
        """Suppresses outer border background leakage using smooth center attenuation."""
        if crop is None or crop.size == 0:
            return crop
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return crop

        # Create smooth center weighting
        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # Elliptical person-focused mask (emphasize central 80% width)
        mask = np.clip(1.25 - (0.9 * xx**2 + 0.25 * yy**2), 0.3, 1.0)
        mask = np.expand_dims(mask, axis=-1)

        mean_col = np.mean(crop, axis=(0, 1), keepdims=True)
        weighted = (crop.astype(np.float32) * mask + mean_col * (1.0 - mask)).astype(np.uint8)
        return weighted

    def _preprocess(self, crop: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Preprocesses crop into (C, H, W) normalized float32 array with background suppression."""
        if crop is None or crop.size == 0 or cv2 is None:
            return None

        isolated = self._apply_foreground_isolation(crop)
        h, w = target_size or self.input_size
        resized = cv2.resize(isolated, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std

        return np.transpose(normalized, (2, 0, 1)).astype(np.float32)

    def _extract_model_feature(self, crop: np.ndarray, model_override=None) -> np.ndarray:
        """Runs forward pass through the primary model to extract a single L2-normalized vector."""
        if crop is None or crop.size == 0 or cv2 is None:
            return np.zeros(self.feature_dim, dtype=np.float32)

        model = model_override or self._model
        if model is None:
            return np.zeros(self.feature_dim, dtype=np.float32)

        try:
            import torch
            prep = self._preprocess(crop)
            if prep is not None:
                tensor = torch.from_numpy(prep).unsqueeze(0).to(self._device)
                with torch.inference_mode():
                    feat = model(tensor).squeeze().cpu().numpy().astype(np.float32)
                if len(feat.shape) > 1:
                    feat = feat.flatten()
                norm = float(np.linalg.norm(feat))
                return (feat / norm) if norm > 0 else feat
        except Exception as e:
            logger.debug(f"Error in deep feature extraction: {e}")

        return np.zeros(self.feature_dim, dtype=np.float32)

    def extract(self, crop: np.ndarray) -> Embedding:
        """Extracts primary full-body embedding."""
        fused, _, _, _, _ = self.extract_decomposed(crop)
        return fused

    def extract_decomposed(
        self, crop: np.ndarray
    ) -> Tuple[Embedding, Embedding, Embedding, Embedding, Embedding]:
        """
        Extracts multi-crop representations:
        1. fused: Combined multi-scale representation
        2. deep: Full-body embedding
        3. global_view: Full crop embedding with metadata
        4. upper: Upper-torso crop embedding (head to waist)
        5. lower: Lower-body crop embedding (waist to feet)
        """
        if crop is None or crop.size == 0:
            zero_vec = np.zeros(self.feature_dim, dtype=np.float32)
            z = Embedding(vector=zero_vec, model_name=self.model_name, version="2.0")
            return z, z, z, z, z

        h, w = crop.shape[:2]
        upper_crop = crop[: max(1, int(0.55 * h)), :]
        lower_crop = crop[int(0.45 * h) :, :]

        # Batch forward pass for full, upper, and lower crops simultaneously
        batch_embs = self.extract_batch([crop, upper_crop, lower_crop])
        deep_vec = batch_embs[0].vector
        upper_vec = batch_embs[1].vector
        lower_vec = batch_embs[2].vector

        deep_emb = Embedding(
            vector=deep_vec,
            model_name=self.model_name,
            version="2.0",
            crop_type="full",
        )
        upper_emb = Embedding(
            vector=upper_vec,
            model_name=self.model_name,
            version="2.0",
            crop_type="upper",
        )
        lower_emb = Embedding(
            vector=lower_vec,
            model_name=self.model_name,
            version="2.0",
            crop_type="lower",
        )

        # 4. Fused Multi-Scale Representation
        if upper_vec.shape == deep_vec.shape and lower_vec.shape == deep_vec.shape:
            fused_vec = 0.60 * deep_vec + 0.25 * upper_vec + 0.15 * lower_vec
            fn = float(np.linalg.norm(fused_vec))
            fused_vec = (fused_vec / fn) if fn > 0 else fused_vec
        else:
            fused_vec = deep_vec

        fused_emb = Embedding(
            vector=fused_vec,
            model_name=self.model_name,
            version="2.0",
            crop_type="fused",
        )

        return fused_emb, deep_emb, deep_emb, upper_emb, lower_emb

    def extract_batch(self, crops: List[np.ndarray]) -> List[Embedding]:
        if not crops:
            return []

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
                    with torch.inference_mode():
                        feats = self._model(batch_tensor).cpu().numpy().astype(np.float32)

                    if len(feats.shape) > 2:
                        feats = feats.reshape(feats.shape[0], -1)

                    embeddings: List[Optional[Embedding]] = [None] * len(crops)
                    for idx, orig_i in enumerate(indices):
                        feat = feats[idx]
                        norm = float(np.linalg.norm(feat))
                        vec = (feat / norm) if norm > 0 else feat
                        embeddings[orig_i] = Embedding(
                            vector=vec,
                            model_name=self.model_name,
                            version="2.0",
                            crop_type="full",
                        )

                    return [
                        emb if emb is not None else self.extract(crops[i])
                        for i, emb in enumerate(embeddings)
                    ]
            except Exception as e:
                logger.error(f"Error in batch ReID extraction: {e}")

        return [self.extract(c) for c in crops]
