"""NVIDIA TensorRT and GPU Accelerated Inference Engine.

Provides high-performance model compilation, caching, and execution:
1. Auto-resolves Windows CUDA/cuDNN DLL paths.
2. TensorRT engine compilation and ONNX graph optimization.
3. ONNX Runtime TensorRT / CUDA execution provider management.
4. Dynamic batch size support for real-time surveillance pipelines.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


def setup_cuda_dll_path() -> None:
    """Configures Windows DLL search path to resolve PyTorch's CUDA and cuDNN libraries."""
    if sys.platform != "win32":
        return

    try:
        import torch
        torch_dir = Path(torch.__file__).resolve().parent
        torch_lib = torch_dir / "lib"
        if torch_lib.is_dir():
            torch_lib_str = str(torch_lib)
            if torch_lib_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib_str + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(torch_lib_str)
                except Exception:
                    pass
    except Exception as e:
        logger.debug("Failed to register torch CUDA DLL directory: %s", e)


# Ensure DLLs are registered at module load
setup_cuda_dll_path()


def is_tensorrt_available() -> bool:
    """Returns True if TensorRT Python API or TensorRT runtime DLLs are available."""
    try:
        import tensorrt
        return True
    except ImportError:
        pass

    import ctypes
    for dll_name in ("nvinfer.dll", "nvinfer_10.dll", "nvinfer_8.dll"):
        try:
            ctypes.CDLL(dll_name)
            return True
        except Exception:
            continue
    return False


def get_available_execution_providers() -> List[str]:
    """Returns list of available execution providers for ONNX Runtime."""
    try:
        setup_cuda_dll_path()
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]


def export_yolo_to_tensorrt(
    weights_path: str = "yolo11n.pt",
    imgsz: int = 640,
    half: bool = False,
    device: str = "0",
) -> str:
    """
    Exports a YOLO model to TensorRT (.engine) or optimized ONNX (.onnx).
    Returns the path to the exported engine file.
    """
    weights_file = Path(weights_path)
    if not weights_file.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights_file}")

    engine_candidate = weights_file.with_suffix(".engine")
    if engine_candidate.is_file():
        logger.info(f"Existing TensorRT engine found: {engine_candidate}")
        return str(engine_candidate)

    onnx_candidate = weights_file.with_suffix(".onnx")
    if onnx_candidate.is_file():
        logger.info(f"Existing optimized ONNX model found: {onnx_candidate}")
        return str(onnx_candidate)

    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_file))

        # Try native TensorRT engine export first if tensorrt is installed
        if is_tensorrt_available():
            try:
                logger.info(f"Attempting TensorRT engine export for '{weights_file}'...")
                path = model.export(
                    format="engine",
                    imgsz=imgsz,
                    half=half,
                    dynamic=True,
                    device=device,
                )
                if path and Path(path).exists():
                    logger.info(f"Successfully exported TensorRT engine: {path}")
                    return str(path)
            except Exception as e:
                logger.warning(f"Native TensorRT engine export failed ({e}). Falling back to optimized ONNX.")

        # Fallback to high-speed ONNX export with slimming
        logger.info(f"Exporting '{weights_file}' to optimized ONNX...")
        path = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=True,
            half=half,
        )
        logger.info(f"Successfully exported model to ONNX: {path}")
        return str(path)

    except Exception as e:
        logger.error(f"Failed to export YOLO model to TensorRT/ONNX: {e}")
        return str(weights_file)


def export_reid_to_onnx(
    model_name: str = "osnet_x1_0",
    output_path: Optional[str] = None,
    device: str = "cuda:0",
) -> str:
    """
    Exports OSNet ReID PyTorch backbone to an ONNX model with dynamic batch size.
    Returns path to exported ONNX file.
    """
    out_path = Path(output_path) if output_path else Path("models") / f"{model_name}.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.is_file():
        logger.info(f"Using existing ReID model at: {out_path}")
        return str(out_path)

    logger.info(f"Exporting ReID model '{model_name}' to ONNX at '{out_path}'...")
    import torch
    from src.reid.backbones.osnet import build_osnet

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = build_osnet(model_name if "x1_0" in model_name else "osnet_x0_25", pretrained=True).to(dev).eval()

    dummy_input = torch.randn(1, 3, 256, 128, device=dev)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )

    logger.info(f"Successfully exported ReID model to: {out_path}")
    return str(out_path)


class TensorRTInferenceSession:
    """
    Wrapper for ONNX Runtime executing with TensorrtExecutionProvider or CUDAExecutionProvider.
    Maintains engine caching and memory limits tailored for 4GB GPUs.
    """

    def __init__(
        self,
        model_path: str,
        engine_cache_dir: str = "models/trt_cache",
        device_id: int = 0,
        enable_fp16: bool = False,
    ) -> None:
        setup_cuda_dll_path()
        import onnxruntime as ort

        self.model_path = model_path
        self.engine_cache_dir = Path(engine_cache_dir)
        self.engine_cache_dir.mkdir(parents=True, exist_ok=True)

        session_opts = ort.SessionOptions()
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        trt_provider_options = {
            "device_id": device_id,
            "trt_max_workspace_size": 1073741824,  # 1 GB
            "trt_fp16_enable": enable_fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(self.engine_cache_dir),
        }

        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 1073741824,  # 1 GB
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": True,
        }

        # Build prioritized provider hierarchy: TensorRT -> CUDA -> CPU
        providers: List[Union[str, Tuple[str, Dict[str, Any]]]] = []
        avail = ort.get_available_providers()

        if is_tensorrt_available() and "TensorrtExecutionProvider" in avail:
            providers.append(("TensorrtExecutionProvider", trt_provider_options))
        if "CUDAExecutionProvider" in avail:
            providers.append(("CUDAExecutionProvider", cuda_provider_options))
        providers.append("CPUExecutionProvider")

        logger.info(f"Initializing TensorRT/CUDA InferenceSession for '{model_path}'...")
        try:
            self.session = ort.InferenceSession(str(model_path), session_opts, providers=providers)
            active_providers = self.session.get_providers()
            self.active_backend = active_providers[0] if active_providers else "Unknown"
            logger.info(f"InferenceSession initialized successfully. Active Provider: {self.active_backend}")
        except Exception as e:
            logger.warning(f"Primary provider initialization failed ({e}). Falling back to CPU.")
            self.session = ort.InferenceSession(str(model_path), session_opts, providers=["CPUExecutionProvider"])
            self.active_backend = "CPUExecutionProvider"

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run(self, input_array: np.ndarray) -> np.ndarray:
        """Run batched forward inference pass."""
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        return outputs[0]
