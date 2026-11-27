"""Person Re-Identification (ReID) multi-region and texture-enriched feature extraction module.

UNUSED:
- register_reid_model (lines 74-96): Modular ReID backbone extension hook kept for registering custom ReID models without editing core code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.core.interfaces import BaseReID
from src.core.types import Embedding

from src.inference.device import (
    HardwareProfile,
    configure_torch_execution,
    detect_hardware_profile,
)

logger = logging.getLogger(__name__)


# Modular ReID model registry: allows adding backbones in one place (used by factory, pipeline switcher, and UI API).
REID_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "person_reid_youtu": {
        "id": "person_reid_youtu",
        "name": "Person ReID Youtu (768D ONNX/TensorRT - Highest Accuracy)",
        "description": "High-discrimination 768D appearance embedding backbone for cross-camera target verification.",
        "feature_dim": 768,
        "min_vram_gb": 0.0,
        "onnx_candidates": (
            "src/reid/backbones/person_reid_youtu_2021nov.onnx",
            "models/person_reid_youtu_2021nov.onnx",
        ),
        "aliases": ("auto", "default", "reid", "youtu"),
    },
    "osnet_x1_0": {
        "id": "osnet_x1_0",
        "name": "OSNet x1.0 (512D High Capacity)",
        "description": "Full-width Omni-Scale network suited for mid/high-VRAM GPUs when PyTorch/OSNet is preferred.",
        "feature_dim": 512,
        "min_vram_gb": 6.0,
        "aliases": (),
    },
    "osnet_x0_25": {
        "id": "osnet_x0_25",
        "name": "OSNet x0.25 (512D Compact)",
        "description": "Ultra-lightweight Omni-Scale network suited for CPU or memory-constrained GPUs.",
        "feature_dim": 512,
        "min_vram_gb": 0.0,
        "aliases": (),
    },
    "mobilenet_v3_small": {
        "id": "mobilenet_v3_small",
        "name": "MobileNetV3-Small (576D PyTorch Fallback)",
        "description": "Compact torchvision feature extractor fallback.",
        "feature_dim": 576,
        "min_vram_gb": 0.0,
        "aliases": ("mobilenet",),
    },
}

_CUSTOM_REID_BUILDERS: Dict[str, Callable[..., BaseReID]] = {}


def register_reid_model(
    model_id: str,
    *,
    name: str,
    description: str = "",
    feature_dim: int = 512,
    min_vram_gb: float = 0.0,
    onnx_candidates: Tuple[str, ...] = (),
    builder: Optional[Callable[..., BaseReID]] = None,
) -> None:
    """Register an external ReID model specification and optional builder cleanly."""
    key = model_id.lower().strip()
    REID_MODEL_REGISTRY[key] = {
        "id": key,
        "name": name,
        "description": description,
        "feature_dim": feature_dim,
        "min_vram_gb": min_vram_gb,
        "onnx_candidates": tuple(onnx_candidates),
        "aliases": (),
    }
    if builder is not None:
        _CUSTOM_REID_BUILDERS[key] = builder


def is_reid_model_supported(model_name: str) -> bool:
    key = (model_name or "").lower().strip()
    if key in REID_MODEL_REGISTRY:
        return True
    for spec in REID_MODEL_REGISTRY.values():
        if key in spec.get("aliases", ()):
            return True
    return "osnet" in key or "mobilenet" in key or "youtu" in key


def get_reid_model_catalog(hardware_profile: Optional[HardwareProfile] = None) -> List[Dict[str, Any]]:
    """Return UI/API-ready metadata for available ReID models, dynamically marking recommendation."""
    profile = hardware_profile or detect_hardware_profile()
    recommended = profile.recommended_reid_model
    options: List[Dict[str, Any]] = []
    for spec in REID_MODEL_REGISTRY.values():
        mid = spec["id"]
        # Only advertise person_reid_youtu if its weights file is present locally
        cands = spec.get("onnx_candidates", ())
        if cands and not any(Path(c).is_file() for c in cands):
            continue
        options.append({
            "id": mid,
            "name": spec["name"],
            "description": spec.get("description", ""),
            "feature_dim": spec.get("feature_dim", 512),
            "min_vram_gb": spec.get("min_vram_gb", 0.0),
            "recommended": mid == recommended,
        })
    return options


class PyTorchReIDExtractor(BaseReID):
    """
    Hardware-adaptive Person Re-Identification feature extractor supporting:
    1. Youtu ReID ONNX/TensorRT (768D) and custom registered ONNX/TensorRT models
    2. Omni-Scale Network (OSNet-x0.25 / OSNet-x1.0, 512D)
    3. Multi-crop representations (Full Body, Upper Torso, Lower Body)
    4. Batched GPU/CPU inference with dynamic HardwareProfile scaling
    """

    def __init__(
        self,
        model_name: str = "auto",
        device: str = "auto",
        input_size: Tuple[int, int] = (256, 128),
        prefer_engine: bool = True,
        batch_size: Union[int, str] = "auto",
        require_pretrained_weights: bool = False,
        engine_type: str = "auto",
        precision: str = "auto",
        engine_cache_dir: str = "models/trt_cache",
        hardware_profile: Optional[HardwareProfile] = None,
    ) -> None:
        self.hardware_profile = hardware_profile or detect_hardware_profile(
            requested_device=device,
            engine_type=engine_type,
            precision=precision,
        )
        raw_name = (model_name or "auto").lower().strip()
        if raw_name in ("auto", "default", "reid"):
            self.model_name = self.hardware_profile.recommended_reid_model
        else:
            self.model_name = raw_name

        self.device_str = self.hardware_profile.device_str
        self.input_size = input_size
        self.feature_dim = 512
        self.prefer_engine = prefer_engine
        if batch_size is None or str(batch_size).lower() == "auto" or int(batch_size) <= 0:
            self.batch_size = max(1, self.hardware_profile.reid_batch_size)
        else:
            self.batch_size = max(1, int(batch_size))
        self.require_pretrained_weights = require_pretrained_weights
        self.engine_type = (engine_type or "auto").lower().strip()
        self.precision = (precision or "auto").lower().strip()
        self.engine_cache_dir = engine_cache_dir
        self._use_fp16 = bool(
            self.hardware_profile.device_type == "cuda"
            and self.hardware_profile.use_fp16
            and self.precision != "fp32"
        )
        self._model = None
        self._osnet_model = None
        self._device = None
        self._trt_session = None
        self._weights_report = {"loaded": 0, "expected": 0, "source": "none", "ok": False}
        self.active_backend = "PyTorch"

        self._init_model()

    @property
    def weights_report(self) -> dict:
        """Returns diagnostic report on loaded pretrained weights."""
        if hasattr(self._model, "weights_report"):
            return self._model.weights_report
        return self._weights_report

    def _init_model(self) -> None:
        configure_torch_execution(self.hardware_profile)
        if self.prefer_engine and self.engine_type != "pytorch":
            try:
                from src.inference.tensorrt_engine import TensorRTInferenceSession

                candidate_path = None
                dim = 512
                norm_name = self.model_name

                # 1. Registry-driven ONNX / TensorRT artifact lookup
                spec = REID_MODEL_REGISTRY.get(self.model_name)
                if spec and spec.get("onnx_candidates"):
                    for p_str in spec["onnx_candidates"]:
                        p = Path(p_str)
                        if p.is_file():
                            candidate_path = str(p)
                            dim = int(spec.get("feature_dim", 768))
                            norm_name = spec["id"]
                            break
                elif self.model_name in ("youtu", "person_reid_youtu"):
                    for p_str in REID_MODEL_REGISTRY["person_reid_youtu"]["onnx_candidates"]:
                        p = Path(p_str)
                        if p.is_file():
                            candidate_path = str(p)
                            dim = 768
                            norm_name = "person_reid_youtu"
                            break

                # 2. Check for explicit or fallback OSNet engine/onnx
                if candidate_path is None and "osnet" in self.model_name:
                    norm_name = "osnet_x1_0" if "x1_0" in self.model_name else "osnet_x0_25"
                    engine_path = Path("models") / f"{norm_name}.engine"
                    onnx_path = Path("models") / f"{norm_name}.onnx"
                    if engine_path.is_file() and self.hardware_profile.device_type == "cuda":
                        candidate_path = str(engine_path)
                        dim = 512
                    elif onnx_path.is_file():
                        candidate_path = str(onnx_path)
                        dim = 512

                if candidate_path:
                    logger.info(f"Loading accelerated ReID session from {candidate_path}...")
                    self._trt_session = TensorRTInferenceSession(
                        candidate_path,
                        engine_cache_dir=self.engine_cache_dir,
                        enable_fp16=self._use_fp16,
                        gpu_mem_limit=self.hardware_profile.ort_gpu_mem_limit_bytes,
                        trt_max_workspace_size=self.hardware_profile.trt_workspace_bytes,
                        device=self.device_str,
                        engine_type=self.engine_type,
                        hardware_profile=self.hardware_profile,
                    )
                    self.active_backend = self._trt_session.active_backend
                    self.feature_dim = dim
                    self.model_name = norm_name
                    self._weights_report = {"loaded": 1, "expected": 1, "source": candidate_path, "ok": True}
                    logger.info(f"ReID model '{norm_name}' loaded via {self.active_backend} (dim={self.feature_dim}).")
                    return
            except Exception as e:
                logger.warning(f"Could not load accelerated ReID session ({e}). Falling back to standard PyTorch.")
                self._trt_session = None

        try:
            import torch
            self._device = torch.device(self.device_str)
            logger.info(
                f"Initializing ReID model '{self.model_name}' on device '{self.device_str}' "
                f"(tier={self.hardware_profile.tier}, fp16={self._use_fp16})..."
            )

            if "osnet" in self.model_name or self.model_name in ("default", "reid", "person_reid_youtu"):
                from src.reid.backbones.osnet import build_osnet
                osnet_variant = "osnet_x1_0" if ("x1_0" in self.model_name or (self.model_name == "person_reid_youtu" and self.hardware_profile.vram_gb >= 6.0)) else "osnet_x0_25"
                self._model = build_osnet(
                    osnet_variant,
                    pretrained=True,
                    require_pretrained=self.require_pretrained_weights,
                ).to(self._device).eval()
                self.feature_dim = 512
                self.active_backend = f"PyTorch ({self._device}, {'fp16' if self._use_fp16 else 'fp32'})"
            elif "mobilenet" in self.model_name:
                import torchvision.models as models
                backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                backbone.classifier = torch.nn.Sequential(torch.nn.Identity())
                self._model = backbone.to(self._device).eval()
                self.feature_dim = 576
                self.active_backend = f"PyTorch ({self._device}, {'fp16' if self._use_fp16 else 'fp32'})"
                self._weights_report = {"loaded": len(backbone.state_dict()), "expected": len(backbone.state_dict()), "source": "torchvision", "ok": True}
            else:
                from src.reid.backbones.osnet import build_osnet
                self._model = build_osnet("osnet_x0_25", pretrained=True, require_pretrained=self.require_pretrained_weights).to(self._device).eval()
                self.feature_dim = 512
                self.active_backend = f"PyTorch ({self._device})"

            if hasattr(self._model, "weights_report"):
                self._weights_report = self._model.weights_report

            if self.require_pretrained_weights and not self._weights_report.get("ok", False):
                raise RuntimeError(
                    f"Pretrained weights required for ReID model '{self.model_name}', "
                    f"but failed to load. Report: {self._weights_report}"
                )

            logger.info(f"ReID model '{self.model_name}' (dim={self.feature_dim}) initialized successfully on {self._device}.")
        except Exception as e:
            logger.error(f"Could not load Person-ReID model '{self.model_name}': {e}")
            raise RuntimeError(f"Failed to initialize ReID model '{self.model_name}'") from e

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
            padded = cv2.copyMakeBorder(crop, pad_top, pad_bottom, 0, 0, cv2.BORDER_REPLICATE)
        elif current_aspect < target_aspect:
            # Crop is narrower than 1:2 -> pad left and right
            new_w = int(h * target_aspect)
            pad_total = max(0, new_w - w)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded = cv2.copyMakeBorder(crop, 0, 0, pad_left, pad_right, cv2.BORDER_REPLICATE)
        else:
            padded = crop

        # Standard ReID input format (H=256, W=128)
        resized = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std

        return np.transpose(normalized, (2, 0, 1)).astype(np.float32)

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
        Supports accelerated TensorRT/ONNX Runtime inference with seamless PyTorch fallback.
        """
        if not crops:
            return []

        zero_emb = lambda: Embedding(
            vector=np.zeros(self.feature_dim, dtype=np.float32),
            model_name=self.model_name,
            version="2.0",
            crop_type="full",
        )

        valid_preps = []
        indices = []
        for i, crop in enumerate(crops):
            if crop is not None and crop.size > 0:
                p = self._preprocess(crop)
                if p is not None:
                    valid_preps.append(p)
                    indices.append(i)

        if not valid_preps:
            return [zero_emb() for _ in crops]

        feats_list = []
        chunk_size = max(1, self.batch_size)

        for i in range(0, len(valid_preps), chunk_size):
            chunk_preps = valid_preps[i : i + chunk_size]
            chunk_feats = None

            # 1. Accelerated Engine Path (TensorRT / ONNX Runtime CUDA)
            if self._trt_session is not None:
                try:
                    batch_arr = np.stack(chunk_preps).astype(np.float32)
                    chunk_feats = self._trt_session.run(batch_arr)
                    del batch_arr
                except Exception as e:
                    logger.warning(f"Error in accelerated ReID batch inference ({e}). Falling back to PyTorch.")
                    chunk_feats = None

            # 2. PyTorch Fallback Path
            if chunk_feats is None and self._model is not None:
                try:
                    import torch
                    batch_tensor = torch.from_numpy(np.stack(chunk_preps)).to(self._device)
                    with torch.inference_mode():
                        if self._use_fp16 and self.hardware_profile.device_type == "cuda":
                            with torch.autocast("cuda", dtype=torch.float16):
                                out_tensor = self._model(batch_tensor)
                        else:
                            out_tensor = self._model(batch_tensor)
                        chunk_feats = out_tensor.float().cpu().numpy().astype(np.float32)
                    del batch_tensor
                except Exception as e:
                    logger.error(f"Error in batch PyTorch ReID extraction: {e}")
                    chunk_feats = np.zeros((len(chunk_preps), self.feature_dim), dtype=np.float32)

            if chunk_feats is not None:
                if len(chunk_feats.shape) > 2:
                    chunk_feats = chunk_feats.reshape(chunk_feats.shape[0], -1)
                feats_list.append(chunk_feats)

        if not feats_list:
            return [zero_emb() for _ in crops]

        feats = np.vstack(feats_list) if len(feats_list) > 1 else feats_list[0]

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


def create_reid_extractor(
    config: Any,
    hardware_profile: Optional[HardwareProfile] = None,
    model_name_override: Optional[str] = None,
) -> BaseReID:
    """Factory creating a configured BaseReID extractor matched to the detected HardwareProfile."""
    reid_cfg = getattr(config, "reid", None)
    inf_cfg = getattr(config, "inference", None)

    model_name = model_name_override or (getattr(reid_cfg, "model_name", "auto") if reid_cfg else "auto")
    key = (model_name or "auto").lower().strip()
    if key in _CUSTOM_REID_BUILDERS:
        return _CUSTOM_REID_BUILDERS[key](config=config, hardware_profile=hardware_profile)

    return PyTorchReIDExtractor(
        model_name=model_name,
        device=getattr(inf_cfg, "device", "auto") if inf_cfg else "auto",
        batch_size=getattr(reid_cfg, "batch_size", "auto") if reid_cfg else "auto",
        require_pretrained_weights=getattr(reid_cfg, "require_pretrained_weights", False) if reid_cfg else False,
        engine_type=getattr(inf_cfg, "engine_type", "auto") if inf_cfg else "auto",
        precision=getattr(inf_cfg, "precision", "auto") if inf_cfg else "auto",
        engine_cache_dir=getattr(inf_cfg, "engine_cache_dir", "models/trt_cache") if inf_cfg else "models/trt_cache",
        hardware_profile=hardware_profile,
    )
