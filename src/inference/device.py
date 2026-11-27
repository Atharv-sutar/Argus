"""Hardware capability detection, device selection, and adaptive GPU execution management.

UNUSED:
- resolve_inference_device (lines 473-488): Legacy public device helper kept for src.inference compatibility; internal code uses detect_hardware_profile().
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareProfile:
    """Resolved compute capabilities and adaptive execution parameters for the current host."""

    device_str: str
    device_type: str
    device_name: str
    tier: str
    tier_label: str
    vram_total_mb: float
    vram_free_mb: float
    compute_capability: Tuple[int, int]
    sm_count: int
    cpu_threads: int
    system_ram_gb: float
    has_tensor_cores: bool
    supports_fp16: bool
    supports_tf32: bool
    use_fp16: bool
    use_tf32: bool
    tensorrt_available: bool
    ort_providers: Tuple[str, ...]
    detection_backend_priority: Tuple[str, ...]
    recommended_detector: str
    detector_candidates: Tuple[str, ...]
    recommended_detection_imgsz: int
    detection_batch_size: int
    recommended_reid_model: str
    reid_batch_size: int
    reid_extract_interval: int
    ort_gpu_mem_limit_bytes: int
    trt_workspace_bytes: int

    @property
    def vram_gb(self) -> float:
        return self.vram_total_mb / 1024.0

    def to_telemetry_dict(
        self,
        *,
        vram_used_mb: Optional[float] = None,
        detector_backend: str = "",
        reid_model: str = "",
        reid_backend: str = "",
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "device_str": self.device_str,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "tier": self.tier,
            "tier_label": self.tier_label,
            "vram_total_mb": round(self.vram_total_mb, 1),
            "vram_used_mb": round(vram_used_mb, 1) if vram_used_mb is not None else 0.0,
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}"
            if self.device_type == "cuda"
            else "N/A",
            "sm_count": self.sm_count,
            "cpu_threads": self.cpu_threads,
            "system_ram_gb": round(self.system_ram_gb, 1),
            "has_tensor_cores": self.has_tensor_cores,
            "fp16_enabled": self.use_fp16,
            "tf32_enabled": self.use_tf32,
            "tensorrt_available": self.tensorrt_available,
            "detection_batch_size": self.detection_batch_size,
            "reid_batch_size": self.reid_batch_size,
            "reid_extract_interval": self.reid_extract_interval,
        }
        if detector_backend:
            data["detector_backend"] = detector_backend
        if reid_model:
            data["reid_model"] = reid_model
        if reid_backend:
            data["reid_backend"] = reid_backend
        return data

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _gpu_has_tensor_cores(device_name: str, cc: Tuple[int, int]) -> bool:
    major, minor = cc
    upper = (device_name or "").upper()
    if major >= 8:
        return True
    if major == 7 and minor >= 5:
        # GTX Turing cards (e.g., 16-series) share sm_75 without Tensor Cores; RTX / T4 / Quadro RTX include them.
        if "GTX" in upper and "RTX" not in upper:
            return False
        return True
    if major == 7 and minor == 0 and any(tok in upper for tok in ("V100", "TITAN V")):
        return True
    return False


def _gpu_supports_fp16(device_name: str, cc: Tuple[int, int], device_type: str) -> bool:
    if device_type == "mps":
        return True
    if device_type != "cuda":
        return False
    major, minor = cc
    # Turing (sm_75) and newer GPUs execute fast native FP16 (even non-Tensor-Core Turing runs 2x FP16 rate).
    if (major, minor) >= (7, 5):
        return True
    return _gpu_has_tensor_cores(device_name, cc)


def _get_system_ram_gb() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().total) / (1024.0 ** 3)
    except Exception:
        return 0.0


def _pick_default_reid_model(vram_gb: float, device_type: str) -> str:
    # Prefer the high-separation Youtu ONNX backbone when bundled locally, otherwise scale OSNet capacity.
    for path in (
        Path("src/reid/backbones/person_reid_youtu_2021nov.onnx"),
        Path("models/person_reid_youtu_2021nov.onnx"),
    ):
        if path.is_file():
            return "person_reid_youtu"
    if device_type == "cuda" and vram_gb >= 6.0:
        return "osnet_x1_0"
    return "osnet_x0_25"


def build_hardware_profile_from_specs(
    *,
    device_str: str,
    device_name: str,
    vram_total_mb: float = 0.0,
    vram_free_mb: float = 0.0,
    compute_capability: Tuple[int, int] = (0, 0),
    sm_count: int = 0,
    cpu_threads: Optional[int] = None,
    system_ram_gb: Optional[float] = None,
    tensorrt_available: bool = False,
    ort_providers: Sequence[str] = ("CPUExecutionProvider",),
    engine_type: str = "auto",
    precision: str = "auto",
) -> HardwareProfile:
    """Compute a deterministic HardwareProfile from observed hardware capabilities."""
    threads = cpu_threads if cpu_threads is not None else (os.cpu_count() or 4)
    ram_gb = system_ram_gb if system_ram_gb is not None else _get_system_ram_gb()

    if device_str.startswith("cuda"):
        device_type = "cuda"
    elif device_str == "mps":
        device_type = "mps"
    else:
        device_type = "cpu"

    vram_gb = max(0.0, float(vram_total_mb)) / 1024.0
    has_tc = _gpu_has_tensor_cores(device_name, compute_capability) if device_type == "cuda" else False
    supports_fp16 = _gpu_supports_fp16(device_name, compute_capability, device_type)
    supports_tf32 = device_type == "cuda" and compute_capability[0] >= 8

    req_prec = (precision or "auto").lower().strip()
    if req_prec == "fp16":
        use_fp16 = supports_fp16 or device_type == "cuda"
    elif req_prec == "fp32":
        use_fp16 = False
    else:
        use_fp16 = supports_fp16

    use_tf32 = supports_tf32 and req_prec != "fp32"
    req_engine = (engine_type or "auto").lower().strip()

    if device_type == "cpu":
        tier = "cpu"
        tier_label = f"CPU ({threads} threads)"
        det_priority: Tuple[str, ...] = ("onnx", "pt")
        det_candidates: Tuple[str, ...] = ("yolo11n.onnx", "yolo11n.pt")
        rec_det = "yolo11n.pt"
        det_batch = min(4, max(1, threads // 4))
        reid_batch = min(16, max(4, threads // 2))
        reid_interval = 4
        ort_limit = 0
        trt_ws = 0
        rec_reid = _pick_default_reid_model(0.0, "cpu")
    elif device_type == "mps":
        tier = "gpu_mid" if vram_gb >= 8.0 else "gpu_entry"
        tier_label = "Apple Metal GPU (MPS)"
        det_priority = ("pt", "onnx")
        det_candidates = ("yolo11s.pt", "yolo11n.pt") if vram_gb >= 8.0 else ("yolo11n.pt",)
        rec_det = det_candidates[0]
        det_batch = 6 if vram_gb >= 8.0 else 4
        reid_batch = 24 if vram_gb >= 8.0 else 16
        reid_interval = 2 if vram_gb >= 8.0 else 3
        ort_limit = 0
        trt_ws = 0
        rec_reid = _pick_default_reid_model(vram_gb, "mps")
    elif vram_gb >= 10.0:
        tier = "gpu_high"
        accel = "Tensor Cores" if has_tc else "CUDA"
        tier_label = f"High-VRAM {accel} GPU ({vram_gb:.1f} GB)"
        det_priority = _cuda_backend_priority(req_engine, tensorrt_available)
        det_candidates = ("yolo11m.engine", "yolo11m.pt", "yolo11s.engine", "yolo11s.pt", "yolo11n.engine", "yolo11n.pt")
        rec_det = "yolo11m.pt"
        det_batch = 16
        reid_batch = 64
        reid_interval = 1
        ort_limit = int(min(6144.0, max(2048.0, vram_total_mb * 0.35)) * 1024 * 1024)
        trt_ws = 4 * 1024 * 1024 * 1024
        rec_reid = _pick_default_reid_model(vram_gb, "cuda")
    elif vram_gb >= 6.0:
        tier = "gpu_mid"
        accel = "Tensor Cores" if has_tc else "CUDA"
        tier_label = f"Mid-Tier {accel} GPU ({vram_gb:.1f} GB)"
        det_priority = _cuda_backend_priority(req_engine, tensorrt_available)
        det_candidates = ("yolo11s.engine", "yolo11s.pt", "yolo11n.engine", "yolo11n.pt")
        rec_det = "yolo11s.pt"
        det_batch = 8
        reid_batch = 32
        reid_interval = 2
        ort_limit = int(min(3072.0, max(1024.0, vram_total_mb * 0.30)) * 1024 * 1024)
        trt_ws = 2 * 1024 * 1024 * 1024
        rec_reid = _pick_default_reid_model(vram_gb, "cuda")
    else:
        tier = "gpu_entry"
        accel = "Tensor Cores" if has_tc else "CUDA"
        tier_label = f"Compact {accel} GPU ({vram_gb:.1f} GB)"
        det_priority = _cuda_backend_priority(req_engine, tensorrt_available)
        det_candidates = ("yolo11n.engine", "yolo11n.pt")
        rec_det = "yolo11n.pt"
        det_batch = 4
        reid_batch = 16
        reid_interval = 3
        ort_limit = int(min(1536.0, max(512.0, vram_total_mb * 0.25)) * 1024 * 1024)
        trt_ws = int(min(1024.0, max(512.0, vram_total_mb * 0.25)) * 1024 * 1024)
        rec_reid = _pick_default_reid_model(vram_gb, "cuda")

    return HardwareProfile(
        device_str=device_str,
        device_type=device_type,
        device_name=device_name,
        tier=tier,
        tier_label=tier_label,
        vram_total_mb=float(vram_total_mb),
        vram_free_mb=float(vram_free_mb),
        compute_capability=compute_capability,
        sm_count=int(sm_count),
        cpu_threads=int(threads),
        system_ram_gb=float(ram_gb),
        has_tensor_cores=has_tc,
        supports_fp16=supports_fp16,
        supports_tf32=supports_tf32,
        use_fp16=use_fp16,
        use_tf32=use_tf32,
        tensorrt_available=bool(tensorrt_available),
        ort_providers=tuple(ort_providers),
        detection_backend_priority=det_priority,
        recommended_detector=rec_det,
        detector_candidates=det_candidates,
        recommended_detection_imgsz=640,
        detection_batch_size=det_batch,
        recommended_reid_model=rec_reid,
        reid_batch_size=reid_batch,
        reid_extract_interval=reid_interval,
        ort_gpu_mem_limit_bytes=ort_limit,
        trt_workspace_bytes=trt_ws,
    )


def _cuda_backend_priority(req_engine: str, tensorrt_available: bool) -> Tuple[str, ...]:
    if req_engine == "onnx":
        return ("onnx", "pt", "engine")
    if req_engine == "pytorch":
        return ("pt", "engine", "onnx")
    if req_engine == "tensorrt":
        return ("engine", "pt", "onnx")
    # Default "auto": prefer compiled TensorRT engines when present, then native CUDA .pt (avoids dynamic ONNX overhead).
    if tensorrt_available:
        return ("engine", "pt", "onnx")
    return ("engine", "pt", "onnx")


def _probe_ort_providers() -> Tuple[str, ...]:
    try:
        import onnxruntime as ort
        return tuple(ort.get_available_providers())
    except Exception:
        return ("CPUExecutionProvider",)


def _probe_tensorrt_available(ort_providers: Sequence[str]) -> bool:
    try:
        from src.inference.tensorrt_engine import is_tensorrt_available
        return bool(is_tensorrt_available() and "TensorrtExecutionProvider" in ort_providers)
    except Exception:
        return False


def get_gpu_memory_info(device_index: int = 0) -> Tuple[float, float, float]:
    """Return (used_mb, total_mb, free_mb) across all GPU runtimes via NVML with PyTorch fallback."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = float(info.used) / (1024.0 * 1024.0)
            total = float(info.total) / (1024.0 * 1024.0)
            free = float(info.free) / (1024.0 * 1024.0)
            return used, total, free
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_index)
            total = float(props.total_memory) / (1024.0 * 1024.0)
            used = float(torch.cuda.memory_allocated(device_index)) / (1024.0 * 1024.0)
            free = max(0.0, total - used)
            return used, total, free
    except Exception:
        pass
    return 0.0, 0.0, 0.0


def detect_hardware_profile(
    requested_device: str = "auto",
    engine_type: str = "auto",
    precision: str = "auto",
) -> HardwareProfile:
    """Detect host CPU/GPU capabilities and build an adaptive HardwareProfile."""
    req = (requested_device or "auto").lower().strip()
    ort_providers = _probe_ort_providers()
    trt_avail = _probe_tensorrt_available(ort_providers)

    if req == "cpu":
        import platform
        cpu_name = platform.processor() or "CPU"
        return build_hardware_profile_from_specs(
            device_str="cpu",
            device_name=cpu_name,
            tensorrt_available=False,
            ort_providers=ort_providers,
            engine_type=engine_type,
            precision=precision,
        )

    try:
        import torch
        if torch.cuda.is_available() and (req in ("auto", "cuda", "gpu") or req.startswith("cuda:")):
            idx = 0
            if req.startswith("cuda:"):
                try:
                    idx = int(req.split(":", 1)[1])
                except ValueError:
                    idx = 0
            props = torch.cuda.get_device_properties(idx)
            device_name = props.name
            cc = (int(props.major), int(props.minor))
            sm_count = int(getattr(props, "multi_processor_count", 0))
            _, nvml_total_mb, nvml_free_mb = get_gpu_memory_info(idx)
            total_mb = nvml_total_mb if nvml_total_mb > 0 else float(props.total_memory) / (1024.0 * 1024.0)
            free_mb = nvml_free_mb if nvml_free_mb > 0 else total_mb
            device_str = f"cuda:{idx}"
            return build_hardware_profile_from_specs(
                device_str=device_str,
                device_name=device_name,
                vram_total_mb=total_mb,
                vram_free_mb=free_mb,
                compute_capability=cc,
                sm_count=sm_count,
                tensorrt_available=trt_avail,
                ort_providers=ort_providers,
                engine_type=engine_type,
                precision=precision,
            )

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and req in ("auto", "mps", "gpu"):
            return build_hardware_profile_from_specs(
                device_str="mps",
                device_name="Apple Silicon GPU",
                vram_total_mb=_get_system_ram_gb() * 1024.0 * 0.5,
                ort_providers=ort_providers,
                engine_type=engine_type,
                precision=precision,
            )
    except ImportError:
        logger.warning("PyTorch not installed. Defaulting hardware profile to CPU.")
    except Exception as e:
        logger.warning(f"GPU capability probe failed ({e}). Falling back to CPU profile.")

    import platform
    if req.startswith("cuda"):
        logger.warning("CUDA requested but not available. Falling back to CPU.")
    return build_hardware_profile_from_specs(
        device_str="cpu",
        device_name=platform.processor() or "CPU",
        tensorrt_available=False,
        ort_providers=ort_providers,
        engine_type=engine_type,
        precision=precision,
    )


def configure_torch_execution(profile: HardwareProfile) -> None:
    """Apply backend precision and autotuner flags suited to the detected GPU."""
    if profile.device_type != "cuda":
        return
    try:
        import torch
        if not torch.cuda.is_available():
            return
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = bool(profile.use_tf32)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(profile.use_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if profile.use_tf32 else "highest")
    except Exception as e:
        logger.debug(f"Could not configure PyTorch CUDA execution flags: {e}")


def resolve_detector_model_for_profile(
    requested_model: str,
    profile: HardwareProfile,
    search_dirs: Optional[Sequence[Path]] = None,
) -> str:
    """Resolve 'auto' detector model selection cleanly without failing offline."""
    if requested_model and requested_model.lower().strip() != "auto":
        return requested_model

    dirs: List[Path] = [Path("."), Path("models")]
    if search_dirs:
        dirs.extend(search_dirs)
    try:
        from ultralytics.utils import SETTINGS
        wdir = SETTINGS.get("weights_dir")
        if wdir:
            dirs.append(Path(wdir))
    except Exception:
        pass

    for cand in profile.detector_candidates:
        cand_path = Path(cand)
        if cand_path.is_file():
            return str(cand_path)
        for d in dirs:
            p = d / cand_path.name
            if p.is_file():
                return str(p)

    # Safe baseline fallback present in repository root
    return "yolo11n.pt"


def resolve_inference_device(requested_device: str = "auto") -> str:
    """
    Resolve device string ('cuda', 'cpu', 'auto', 'mps') to an available execution device.

    Returns:
        str: Resolved device identifier ('cuda:0', 'mps', or 'cpu').
    """
    profile = detect_hardware_profile(requested_device=requested_device)
    if profile.device_type == "cuda":
        logger.info(
            f"Using {profile.device_str}: {profile.device_name} "
            f"({profile.vram_gb:.1f} GB VRAM, cc {profile.compute_capability[0]}.{profile.compute_capability[1]}, tier={profile.tier})"
        )
    elif profile.device_type == "mps":
        logger.info("Using Apple MPS device")
    return profile.device_str


def trim_gpu_memory() -> None:
    """Safely triggers CUDA cache clearing and garbage collection to avoid VRAM fragmentation."""
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception as e:
        logger.debug(f"[MEMORY] GPU trim exception: {e}")
