import os
import torch
import torch.hub

def inspect():
    models_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    p1 = os.path.join(models_dir, "osnet_x1_0_msmt17.pt")
    p2 = os.path.join(models_dir, "osnet_ain_x1_0_msmt17.pt")
    
    sd1 = torch.load(p1, map_location="cpu")
    if "state_dict" in sd1: sd1 = sd1["state_dict"]
    print(f"OSNet-x1.0 tensors: {len(sd1)}")
    
    sd2 = torch.load(p2, map_location="cpu")
    if "state_dict" in sd2: sd2 = sd2["state_dict"]
    print(f"OSNet-AIN-x1.0 tensors: {len(sd2)}")
    
    ain_keys = [k for k in sd2.keys() if "ain" in k or "in" in k or "gate" in k]
    print("Sample AIN specific keys:", ain_keys[:10])

if __name__ == "__main__":
    inspect()
