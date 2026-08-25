import os
import torch
import torch.hub

def test_keys():
    models_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    p2 = os.path.join(models_dir, "osnet_ain_x1_0_msmt17.pt")
    sd2 = torch.load(p2, map_location="cpu", weights_only=False)
    if "state_dict" in sd2: sd2 = sd2["state_dict"]
    
    for k in sorted(sd2.keys()):
        if "conv2.0" in k:
            print(k)

if __name__ == "__main__":
    test_keys()
