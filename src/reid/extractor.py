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
from src.inference.device import resolve_inference_device

logger = logging.getLogger(__name__)


class PyTorchReIDExtractor(BaseReID):
    """
    Extracts appearance feature embeddings from cropped person observations
    using a person-ReID trained deep convolutional backbone, multi-region upper/lower body
    spatial color representations, and gradient texture descriptors.
    """

    def __init__(
        self,
        model_name: str = "osnet_x0_25",
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
            from src.reid.backbones.osnet import build_osnet

            self._device = torch.device(self.device_str)
            logger.info(f"Initializing ReID model '{self.model_name}' on device '{self.device_str}'...")

            if "osnet" in self.model_name or "reid" in self.model_name:
                self._model = build_osnet(self.model_name, pretrained=True).to(self._device).eval()
            elif self.model_name == "mobilenet_v3_small":
                import torchvision.models as models
                backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                backbone.classifier = torch.nn.Identity()
                self._model = backbone.to(self._device).eval()
            else:
                self._model = build_osnet("osnet_x0_25", pretrained=True).to(self._device).eval()

            logger.info(f"ReID model '{self.model_name}' initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not load Person-ReID model ({e}). Deep features unavailable.")
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

    def _extract_region_color(self, region: np.ndarray) -> np.ndarray:
        """Extracts HSV (128 bins) + Lab (64 bins) color histogram for an image patch."""
        if region is None or region.size == 0 or cv2 is None:
            return np.zeros(192, dtype=np.float32)

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)

        hist_hs = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
        hist_ab = cv2.calcHist([lab], [1, 2], None, [8, 8], [0, 256, 0, 256]).flatten()

        vec = np.concatenate([hist_hs, hist_ab])
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec

    def _extract_texture_gradient(self, patch: np.ndarray) -> np.ndarray:
        """Extracts gradient orientation & magnitude texture distribution (16 bins)."""
        if patch is None or patch.size == 0 or cv2 is None:
            return np.zeros(16, dtype=np.float32)

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        hist_ang = cv2.calcHist([angle], [0], None, [8], [0, 360]).flatten()
        hist_mag = cv2.calcHist([mag], [0], None, [8], [0, 256]).flatten()

        vec = np.concatenate([hist_ang, hist_mag])
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec

    def _extract_spatial_color_descriptor(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts 4 vertical strips of color (HSV+Lab) and texture histograms
        focusing on central body region.
        """
        if crop is None or crop.size == 0 or cv2 is None:
            return np.zeros(832, dtype=np.float32)

        h, w = crop.shape[:2]
        h_strips = 4
        strip_h = max(1, h // h_strips)
        strip_features = []

        x_start = int(0.15 * w)
        x_end = max(x_start + 1, int(0.85 * w))

        for i in range(h_strips):
            y_start = i * strip_h
            y_end = (i + 1) * strip_h if i < h_strips - 1 else h
            patch = crop[y_start:y_end, x_start:x_end]

            color_vec = self._extract_region_color(patch)
            texture_vec = self._extract_texture_gradient(patch)

            s_vec = np.concatenate([color_vec, texture_vec])
            norm = np.linalg.norm(s_vec)
            if norm > 0:
                s_vec = s_vec / norm
            strip_features.append(s_vec)

        combined = np.concatenate(strip_features)
        norm = np.linalg.norm(combined)
        return (combined / norm) if norm > 0 else combined

    def extract(self, crop: np.ndarray) -> Embedding:
        fused, _, _, _, _ = self.extract_decomposed(crop)
        return fused

    def extract_decomposed(
        self, crop: np.ndarray
    ) -> Tuple[Embedding, Embedding, Embedding, Embedding, Embedding]:
        """
        Extracts decomposed embeddings:
        (fused, deep, global_color_texture, upper_body_color, lower_body_color).
        """
        if crop is None or crop.size == 0:
            zero_vec = np.zeros(128, dtype=np.float32)
            z = Embedding(vector=zero_vec)
            return z, z, z, z, z

        h, w = crop.shape[:2]
        x_start = int(0.15 * w)
        x_end = max(x_start + 1, int(0.85 * w))

        # 1. Upper body patch (shirt/torso: 15% to 55% height)
        upper_patch = crop[int(0.15 * h) : int(0.55 * h), x_start:x_end]
        upper_color = self._extract_region_color(upper_patch)
        upper_emb = Embedding(vector=upper_color)

        # 2. Lower body patch (pants/legs: 50% to 95% height)
        lower_patch = crop[int(0.50 * h) : int(0.95 * h), x_start:x_end]
        lower_color = self._extract_region_color(lower_patch)
        lower_emb = Embedding(vector=lower_color)

        # 3. 4-strip spatial color + texture descriptor
        color_vec = self._extract_spatial_color_descriptor(crop)
        color_emb = Embedding(vector=color_vec)

        # 4. Deep semantic Person-ReID feature (512D from OSNet)
        deep_vec = None
        if self._model is not None:
            try:
                import torch

                prep = self._preprocess(crop)
                if prep is not None:
                    tensor = torch.from_numpy(prep).unsqueeze(0).to(self._device)
                    with torch.no_grad():
                        feat = self._model(tensor).squeeze().cpu().numpy()
                    norm = np.linalg.norm(feat)
                    if norm > 0:
                        deep_vec = feat / norm
            except Exception as e:
                logger.error(f"Error in deep ReID extraction: {e}")

        if deep_vec is None:
            deep_vec = np.zeros(512, dtype=np.float32)

        deep_emb = Embedding(vector=deep_vec)

        # 5. Composite Fused Embedding (Dynamic concatenation without mutating shape on error)
        part_vec = np.concatenate([upper_color, lower_color])
        p_norm = np.linalg.norm(part_vec)
        if p_norm > 0:
            part_vec = part_vec / p_norm

        fused = np.concatenate([
            deep_vec * 0.7071,       # 50% energy
            color_vec * 0.5477,      # 30% energy
            part_vec * 0.4472,       # 20% energy
        ])
        f_norm = np.linalg.norm(fused)
        fused_emb = Embedding(vector=(fused / f_norm if f_norm > 0 else fused))

        return fused_emb, deep_emb, color_emb, upper_emb, lower_emb

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
                    with torch.no_grad():
                        feats = self._model(batch_tensor).cpu().numpy()

                    if len(feats.shape) > 2:
                        feats = feats.reshape(feats.shape[0], -1)

                    embeddings: List[Optional[Embedding]] = [None] * len(crops)
                    for idx, orig_i in enumerate(indices):
                        feat = feats[idx]
                        norm = np.linalg.norm(feat)
                        deep_vec = (feat / norm) if norm > 0 else feat

                        c_vec = self._extract_spatial_color_descriptor(crops[orig_i])

                        h_c, w_c = crops[orig_i].shape[:2]
                        xs = int(0.15 * w_c)
                        xe = max(xs + 1, int(0.85 * w_c))
                        up = crops[orig_i][int(0.15 * h_c) : int(0.55 * h_c), xs:xe]
                        lp = crops[orig_i][int(0.50 * h_c) : int(0.95 * h_c), xs:xe]
                        p_vec = np.concatenate([self._extract_region_color(up), self._extract_region_color(lp)])
                        pn = np.linalg.norm(p_vec)
                        if pn > 0:
                            p_vec = p_vec / pn

                        fused = np.concatenate([
                            deep_vec * 0.8367,
                            c_vec * 0.4690,
                            p_vec * 0.2828,
                        ])
                        fn = np.linalg.norm(fused)
                        if fn > 0:
                            fused = fused / fn
                        embeddings[orig_i] = Embedding(vector=fused)

                    return [
                        emb if emb is not None else self.extract(crops[i])
                        for i, emb in enumerate(embeddings)
                    ]
            except Exception as e:
                logger.error(f"Error in deep ReID batch extraction: {e}")

        return [self.extract(c) for c in crops]
