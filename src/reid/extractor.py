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
        self.feature_dim = 512
        self._model = None
        self._osnet_model = None
        self._device = None
        self.color_normalizer = ColorNormalizer(apply_gray_world=False, apply_clahe=False)
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
            elif "mobilenet" in self.model_name:
                import torchvision.models as models
                backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                backbone.classifier = torch.nn.Sequential(torch.nn.Identity())
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

    def _preprocess(self, crop: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Preprocesses crop into (C, H, W) normalized float32 tensor with aspect-ratio preservation."""
        if crop is None or crop.size == 0 or cv2 is None:
            return None

        target_h, target_w = target_size or self.input_size  # (256, 128)
        
        # Letterbox to maintain 1:2 human aspect ratio without squash distortion
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return None
            
        target_aspect = target_w / float(target_h)  # 0.5
        current_aspect = w / float(h)

        if current_aspect > target_aspect:
            # Crop is wider than 1:2 -> pad top and bottom
            new_h = int(w / target_aspect)
            pad_total = max(0, new_h - h)
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            padded = cv2.copyMakeBorder(crop, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[114, 114, 114])
        elif current_aspect < target_aspect:
            # Crop is narrower than 1:2 -> pad left and right
            new_w = int(h * target_aspect)
            pad_total = max(0, new_w - w)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded = cv2.copyMakeBorder(crop, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[114, 114, 114])
        else:
            padded = crop

        # OSNet expects input format (H=256, W=128)
        resized = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
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
        """Extracts primary full-body embedding with high-speed single forward pass."""
        if crop is None or crop.size == 0:
            return Embedding(
                vector=np.zeros(self.feature_dim, dtype=np.float32),
                model_name=self.model_name,
                version="2.0",
                crop_type="full",
            )
        embs = self.extract_batch([crop])
        return embs[0]

    def extract_decomposed(
        self, crop: np.ndarray
    ) -> Tuple[Embedding, Embedding, Embedding, Embedding, Embedding]:
        """
        Extracts multi-crop representations (Full Body, Upper Torso, Lower Body, Fused).
        Used when deep decomposed viewpoint representation is requested.
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

        # Fused Multi-Scale Representation
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
        """
        Fast GPU batched feature extraction for multiple candidate crops in a single tensor forward pass.
        """
        if not crops:
            return []

        zero_emb = lambda: Embedding(
            vector=np.zeros(self.feature_dim, dtype=np.float32),
            model_name=self.model_name,
            version="2.0",
            crop_type="full",
        )

        if self._model is not None:
            try:
                import torch

                valid_preps = []
                indices = []
                for i, crop in enumerate(crops):
                    if crop is not None and crop.size > 0:
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

                    embeddings: List[Embedding] = [zero_emb() for _ in range(len(crops))]
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

                    return embeddings
            except Exception as e:
                logger.error(f"Error in batch ReID extraction: {e}")

        return [zero_emb() for _ in crops]
