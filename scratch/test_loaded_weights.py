import os
import sys
sys.path.insert(0, ".")
import torch
from src.reid.backbones.osnet import build_osnet, osnet_x1_0, osnet_x0_25

for name in ["osnet_x0_25", "osnet_x1_0"]:
    model = build_osnet(name, pretrained=True)
    sd = model.state_dict()
    print(f"\nModel {name}: total params in state_dict={len(sd)}")
    # Check fc layer
    for k in sd:
        if "fc" in k:
            print(f"  {k}: shape={sd[k].shape}, norm={torch.norm(sd[k].float()).item():.4f}")
