"""Dynamic dependency resolver based on configured technologies."""

from importlib import metadata
import logging
import subprocess
import sys
from typing import Optional, Tuple

from src.core.config import AppConfig

logger = logging.getLogger(__name__)


def _parse_version_tuple(version_str: str) -> Tuple[int, ...]:
    """Parses numeric version components (e.g. '2.5.1+cu121' -> (2, 5, 1))."""
    base = version_str.split("+", 1)[0].split("-", 1)[0]
    parts = []
    for segment in base.split("."):
        digits = "".join(ch for ch in segment if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _has_cuda_hardware(config: AppConfig) -> bool:
    """Checks without importing torch whether CPU-only execution was explicitly requested or CUDA is usable."""
    req_dev = str(getattr(config.inference, "device", "auto")).lower().strip()
    if req_dev == "cpu":
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _ensure_package(package_name: str, min_version: Optional[str] = None) -> None:
    try:
        installed_version: Optional[str] = None
        try:
            installed_version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            installed_version = None

        if installed_version is not None:
            if not min_version or _parse_version_tuple(installed_version) >= _parse_version_tuple(min_version):
                return
            logger.info(
                f"Package '{package_name}' version {installed_version} < {min_version}; upgrading..."
            )
        else:
            logger.info(f"Package '{package_name}' not found. Installing...")

        install_cmd = f"{package_name}>={min_version}" if min_version else package_name
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", install_cmd
        ])
        logger.info(f"Successfully installed '{install_cmd}'.")

    except Exception as e:
        logger.error(f"Failed to verify/install dependency '{package_name}': {e}")


def ensure_technologies(config: AppConfig) -> None:
    """Dynamically checks configured dependencies and installs only missing/outdated packages."""
    logger.info("Verifying dynamic dependencies based on configuration...")

    # 1. Object Detection Dependencies
    det_model = config.detection.model_name.lower()
    if "yolo11" in det_model:
        _ensure_package("ultralytics", "8.3.0")
    elif "yolo" in det_model:
        _ensure_package("ultralytics", "8.0.0")
    else:
        _ensure_package("ultralytics", "8.0.0")

    # 2. ReID Dependencies
    reid_model = config.reid.model_name.lower()
    if "osnet" in reid_model or "mobilenet" in reid_model or reid_model == "auto":
        _ensure_package("torch", "2.0.0")
        _ensure_package("torchvision", "0.15.0")

    # 3. Tracker Dependencies
    tracker_type = getattr(config.tracking, "tracker_type", "botsort").lower()
    if "bot" in tracker_type or "sort" in tracker_type or "byte" in tracker_type:
        _ensure_package("scipy")

    # 4. Vector Database Dependencies
    _ensure_package("faiss-cpu")

    # 5. Accelerated Inference & Hardware Telemetry Dependencies
    engine_type = getattr(config.inference, "engine_type", "auto").lower()
    if engine_type in ("tensorrt", "onnx", "auto"):
        _ensure_package("onnx")
        if "youtu" in reid_model or reid_model == "auto":
            has_gpu_ort = True
            try:
                metadata.version("onnxruntime-gpu")
            except metadata.PackageNotFoundError:
                has_gpu_ort = False
            if not has_gpu_ort:
                if _has_cuda_hardware(config):
                    _ensure_package("onnxruntime-gpu")
                else:
                    _ensure_package("onnxruntime")
    _ensure_package("nvidia-ml-py")

    logger.info("Dynamic dependencies verified.")
