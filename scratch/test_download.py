import gdown
import os
import torch
import torch.hub

def test_download():
    models_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. osnet_x1_0_msmt17
    p1 = os.path.join(models_dir, "osnet_x1_0_msmt17.pt")
    if not os.path.exists(p1):
        print("Downloading osnet_x1_0_msmt17...")
        gdown.download(id="112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M", output=p1, quiet=False)
    
    # 2. osnet_ain_x1_0_msmt17
    p2 = os.path.join(models_dir, "osnet_ain_x1_0_msmt17.pt")
    if not os.path.exists(p2):
        print("Downloading osnet_ain_x1_0_msmt17...")
        gdown.download(id="1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal", output=p2, quiet=False)
        
    print(f"p1 exists: {os.path.exists(p1)} (size: {os.path.getsize(p1) if os.path.exists(p1) else 0})")
    print(f"p2 exists: {os.path.exists(p2)} (size: {os.path.getsize(p2) if os.path.exists(p2) else 0})")

if __name__ == "__main__":
    test_download()
