"""Device selection and execution management for AI inference."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_inference_device(requested_device: str = "auto") -> str:
    """
    Resolve device string ('cuda', 'cpu', 'auto') to an available PyTorch device.

    Args:
        requested_device: Device configuration string ('auto', 'cuda', 'cpu', etc.)

    Returns:
        str: Resolved device identifier ('cuda:0' or 'cpu').
    """
    requested_device = (requested_device or "auto").lower().strip()

    if requested_device == "cpu":
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            if requested_device in ("auto", "cuda", "gpu"):
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Using CUDA device 0: {device_name}")
                return "cuda:0"
            elif requested_device.startswith("cuda:"):
                return requested_device
        else:
            if requested_device.startswith("cuda"):
                logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    except ImportError:
        logger.warning("PyTorch not installed. Defaulting inference device to CPU.")
        return "cpu"

    return "cpu"


def trim_gpu_memory() -> None:
    """
    Safely triggers PyTorch CUDA cache clearing and garbage collection.
    Prevents progressive VRAM fragmentation on memory-constrained GPUs (e.g. 4GB GTX 1650).
    """
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

