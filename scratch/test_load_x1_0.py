import sys
import os
sys.path.insert(0, '.')
import torch
import torch.hub
from src.reid.backbones.osnet import OSNet, _load_pretrained_weights

def test_load():
    models_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    p1 = os.path.join(models_dir, "osnet_x1_0_msmt17.pt")
    
    model = OSNet(blocks=[2, 2, 2], channels=[64, 256, 384, 512], feature_dim=512)
    
    state_dict = torch.load(p1, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
    
    model_dict = model.state_dict()
    filtered = {k.replace("module.", ""): v for k, v in state_dict.items() if k.replace("module.", "") in model_dict and model_dict[k.replace("module.", "")].shape == v.shape}
    print(f"Matched {len(filtered)} / {len(model_dict)} tensors for OSNet-x1.0")
    
    model.load_state_dict(filtered, strict=False)
    model.eval()
    x = torch.randn(2, 3, 256, 128)
    out = model(x)
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    test_load()
