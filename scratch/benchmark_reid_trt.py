"""Benchmark PyTorch vs ONNX Runtime (TensorRT/CUDA) for OSNet."""

import time
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reid.backbones.osnet import build_osnet

def benchmark():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. PyTorch Model
    model = build_osnet("osnet_x1_0", pretrained=True).to(device).eval()
    dummy_input = torch.randn(4, 3, 256, 128, device=device)

    # Warmup
    for _ in range(5):
        _ = model(dummy_input)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(50):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - t0) / 50 * 1000
    print(f"PyTorch CUDA inference (batch=4): {pytorch_time:.2f} ms")

    # 2. Export to ONNX
    onnx_path = "models/osnet_x1_0.onnx"
    Path("models").mkdir(exist_ok=True)
    
    print(f"Exporting OSNet to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print("Export complete.")

    # 3. ONNX Runtime with TensorRT / CUDA
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    # Check which are supported
    avail = ort.get_available_providers()
    selected_providers = [p for p in providers if p in avail]
    print(f"Selected ORT Providers: {selected_providers}")

    session_opts = ort.SessionOptions()
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, session_opts, providers=selected_providers)

    dummy_np = np.random.randn(4, 3, 256, 128).astype(np.float32)

    # Warmup
    for _ in range(5):
        _ = session.run(["embedding"], {"input": dummy_np})

    t0 = time.time()
    for _ in range(50):
        _ = session.run(["embedding"], {"input": dummy_np})
    ort_time = (time.time() - t0) / 50 * 1000
    print(f"ONNX Runtime TensorRT/CUDA inference (batch=4): {ort_time:.2f} ms")
    print(f"Speedup: {pytorch_time / ort_time:.2f}x")

if __name__ == "__main__":
    benchmark()
