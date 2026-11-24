"""Inference execution and hardware-adaptive device management layer."""

from src.inference.device import (
    HardwareProfile,
    build_hardware_profile_from_specs,
    configure_torch_execution,
    detect_hardware_profile,
    get_gpu_memory_info,
    resolve_detector_model_for_profile,
    resolve_inference_device,
    trim_gpu_memory,
)

__all__ = [
    "HardwareProfile",
    "build_hardware_profile_from_specs",
    "configure_torch_execution",
    "detect_hardware_profile",
    "get_gpu_memory_info",
    "resolve_detector_model_for_profile",
    "resolve_inference_device",
    "trim_gpu_memory",
]