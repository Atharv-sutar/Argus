"""Benchmark PyTorch FP16 on GTX 1650."""

import time
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reid.backbones.osnet import build_osnet

def benchmark_half():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # FP32
    model_fp32 = build_osnet("osnet_x1_0", pretrained=True).to(device).eval()
    inp_fp32 = torch.randn(4, 3, 256, 128, device=device)
    for _ in range(10): _ = model_fp32(inp_fp32)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100): _ = model_fp32(inp_fp32)
    torch.cuda.synchronize()
    fp32_ms = (time.time() - t0) / 100 * 1000
    print(f"OSNet FP32 CUDA (batch=4): {fp32_ms:.2f} ms")

    # FP16 (Half Precision)
    model_fp16 = build_osnet("osnet_x1_0", pretrained=True).to(device).half().eval()
    inp_fp16 = torch.randn(4, 3, 256, 128, device=device).half()
    for _ in range(10): _ = model_fp16(inp_fp16)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100): _ = model_fp16(inp_fp16)
    torch.cuda.synchronize()
    fp16_ms = (time.time() - t0) / 100 * 1000
    print(f"OSNet FP16 CUDA (batch=4): {fp16_ms:.2f} ms")
    print(f"FP16 Speedup: {fp32_ms / fp16_ms:.2f}x | Memory reduction: 50%")

if __name__ == "__main__":
    benchmark_half()
