import os
import sys
import torch

models_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
for name in ["osnet_x0_25_msmt17.pt", "osnet_x1_0_msmt17.pt", "osnet_ain_x1_0_msmt17.pt"]:
    p = os.path.join(models_dir, name)
    if os.path.exists(p):
        d = torch.load(p, map_location="cpu", weights_only=False)
        print(f"\n--- {name} ---")
        print("Type:", type(d))
        if isinstance(d, dict):
            print("Keys:", list(d.keys())[:5])
            sd = d.get("state_dict", d)
            print("State dict keys count:", len(sd.keys()))
            print("First 5 state dict keys:", list(sd.keys())[:5])
            print("Last 5 state dict keys:", list(sd.keys())[-5:])
