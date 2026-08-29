import os
import sys
sys.path.insert(0, ".")
import torch
import numpy as np

print("Torch version:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("Torch hub dir:", torch.hub.get_dir())

checkpoints_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
print("Checkpoints dir:", checkpoints_dir)
if os.path.exists(checkpoints_dir):
    print("Files in checkpoints dir:", os.listdir(checkpoints_dir))
else:
    print("Checkpoints dir does not exist.")

from src.reid.backbones.osnet import build_osnet

print("Building osnet_x0_25...")
try:
    model = build_osnet("osnet_x0_25", pretrained=True)
    print("osnet_x0_25 built successfully.")
except Exception as e:
    print("osnet_x0_25 build failed:", e)

from src.reid.extractor import PyTorchReIDExtractor

extractor = PyTorchReIDExtractor(model_name="osnet_x0_25", device="cuda" if torch.cuda.is_available() else "cpu")

# Test similarity with random noise images vs distinct color images
img_red = np.zeros((200, 100, 3), dtype=np.uint8)
img_red[:, :, 2] = 255 # Pure red (in BGR)

img_blue = np.zeros((200, 100, 3), dtype=np.uint8)
img_blue[:, :, 0] = 255 # Pure blue (in BGR)

img_green = np.zeros((200, 100, 3), dtype=np.uint8)
img_green[:, :, 1] = 255 # Pure green (in BGR)

emb_red = extractor.extract(img_red)
emb_blue = extractor.extract(img_blue)
emb_green = extractor.extract(img_green)

print("Dim:", emb_red.dim)
print("Red norm:", np.linalg.norm(emb_red.vector))
print("Red vector[:10]:", emb_red.vector[:10])
print("Sim(Red, Red):", float(np.dot(emb_red.vector, emb_red.vector)))
print("Sim(Red, Blue):", float(np.dot(emb_red.vector, emb_blue.vector)))
print("Sim(Red, Green):", float(np.dot(emb_red.vector, emb_green.vector)))
print("Sim(Blue, Green):", float(np.dot(emb_blue.vector, emb_green.vector)))

# Test with random noise images
np.random.seed(42)
img_rand1 = np.random.randint(0, 256, (200, 100, 3), dtype=np.uint8)
img_rand2 = np.random.randint(0, 256, (200, 100, 3), dtype=np.uint8)

emb_rand1 = extractor.extract(img_rand1)
emb_rand2 = extractor.extract(img_rand2)
print("Sim(Rand1, Rand2):", float(np.dot(emb_rand1.vector, emb_rand2.vector)))
