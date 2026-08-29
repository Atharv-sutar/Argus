import os
import sys
sys.path.insert(0, ".")
import torch
from src.reid.backbones.osnet import OSNet

model = OSNet(blocks=[2, 2, 2], channels=[64, 256, 384, 512], feature_dim=512)
model_keys = set(model.state_dict().keys())

models_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
ckpt = torch.load(os.path.join(models_dir, "osnet_x1_0_msmt17.pt"), map_location="cpu", weights_only=False)
if "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]

ckpt_keys = set(k.replace("module.", "") for k in ckpt.keys())

missing_in_model = ckpt_keys - model_keys
missing_in_ckpt = model_keys - ckpt_keys
print("Missing in model from ckpt:", len(missing_in_model), list(missing_in_model)[:10])
print("Missing in ckpt from model:", len(missing_in_ckpt), list(missing_in_ckpt)[:10])

# Check shape mismatches
shape_mismatches = []
for k in model_keys.intersection(ckpt_keys):
    if model.state_dict()[k].shape != ckpt[k].shape and ('module.' + k not in ckpt or model.state_dict()[k].shape != ckpt['module.' + k].shape):
        shape_mismatches.append((k, model.state_dict()[k].shape, ckpt.get(k, ckpt.get('module.' + k)).shape))

print("Shape mismatches:", len(shape_mismatches), shape_mismatches)
