import os
import sys
sys.path.insert(0, ".")
import torch
import numpy as np
from src.reid.backbones.osnet import build_osnet
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_osnet("osnet_x1_0", pretrained=True).to(device).eval()

print("Model parameters count:", sum(p.numel() for p in model.parameters()))

# Let's inspect the weights in model.fc and conv5 to check if pretrained weights loaded!
for name, param in model.named_parameters():
    print(f"Layer {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    if "fc" in name:
        print(f"  Sample values: {param.flatten()[:5].tolist()}")
    if "conv1" in name:
        print(f"  Sample conv1: {param.flatten()[:5].tolist()}")
    if "conv2.0.conv1" in name:
        break
