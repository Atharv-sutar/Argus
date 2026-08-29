import os
import sys
sys.path.insert(0, ".")
import time
import torch
import numpy as np
from src.reid.backbones.osnet import build_osnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Benchmarking on device:", device)

for name in ["osnet_x0_25", "osnet_x1_0"]:
    try:
        model = build_osnet(name, pretrained=True).to(device).eval()
        dummy_input = torch.randn(8, 3, 256, 128, device=device)
        
        # Warmup
        with torch.inference_mode():
            for _ in range(10):
                _ = model(dummy_input)
                
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(50):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 50.0
        print(f"Model {name}: batch 8 inference time = {dt*1000:.2f} ms ({dt/8*1000:.2f} ms per crop)")
    except Exception as e:
        print(f"Model {name} failed: {e}")
