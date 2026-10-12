"""Benchmark YOLO11 on GTX 1650."""

import time
import torch
import numpy as np
from ultralytics import YOLO

def benchmark_yolo():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load YOLO11
    model = YOLO("yolo11n.pt")
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        _ = model.predict(dummy_frame, device=device, verbose=False)

    # Measure single frame
    t0 = time.time()
    for _ in range(30):
        _ = model.predict(dummy_frame, device=device, verbose=False)
    fps_single = 30 / (time.time() - t0)
    latency_single = (time.time() - t0) / 30 * 1000
    print(f"YOLO11n PyTorch Single Frame: {latency_single:.2f} ms ({fps_single:.1f} FPS)")

if __name__ == "__main__":
    benchmark_yolo()
