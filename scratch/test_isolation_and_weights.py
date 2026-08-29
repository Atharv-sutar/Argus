import os
import sys
sys.path.insert(0, ".")
import cv2
import torch
import numpy as np
from src.reid.backbones.osnet import build_osnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_osnet("osnet_x0_25", pretrained=True).to(device).eval()

def preprocess_raw(crop):
    resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb - mean) / std
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)

def extract_feat(crop, with_isolation=False):
    if with_isolation:
        h, w = crop.shape[:2]
        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        mask = np.clip(1.25 - (0.9 * xx**2 + 0.25 * yy**2), 0.3, 1.0)
        mask = np.expand_dims(mask, axis=-1)
        mean_col = np.mean(crop, axis=(0, 1), keepdims=True)
        crop = (crop.astype(np.float32) * mask + mean_col * (1.0 - mask)).astype(np.uint8)
    
    prep = preprocess_raw(crop)
    t = torch.from_numpy(prep).unsqueeze(0).to(device)
    with torch.inference_mode():
        feat = model(t).squeeze().cpu().numpy()
    feat = feat / np.linalg.norm(feat)
    return feat

np.random.seed(42)
img1 = np.random.randint(0, 256, (200, 100, 3), dtype=np.uint8)
img2 = np.random.randint(0, 256, (200, 100, 3), dtype=np.uint8)

f1_iso = extract_feat(img1, with_isolation=True)
f2_iso = extract_feat(img2, with_isolation=True)
print("Rand1 vs Rand2 (WITH isolation):", float(np.dot(f1_iso, f2_iso)))

f1_raw = extract_feat(img1, with_isolation=False)
f2_raw = extract_feat(img2, with_isolation=False)
print("Rand1 vs Rand2 (WITHOUT isolation):", float(np.dot(f1_raw, f2_raw)))

# Test on simulated real person features: Upper torso shirt vs different shirt colors, trousers, etc.
# Person A: White shirt (top half), Blue jeans (bottom half)
person_A = np.zeros((256, 128, 3), dtype=np.uint8)
person_A[:128, :] = (240, 240, 240) # White shirt
person_A[128:, :] = (180, 50, 20)  # Blue jeans (BGR)

# Person B: Black shirt (top half), Khaki pants (bottom half)
person_B = np.zeros((256, 128, 3), dtype=np.uint8)
person_B[:128, :] = (20, 20, 20)   # Black shirt
person_B[128:, :] = (60, 140, 180) # Khaki pants (BGR)

# Person C: Red shirt (top half), Black pants (bottom half)
person_C = np.zeros((256, 128, 3), dtype=np.uint8)
person_C[:128, :] = (20, 20, 220)  # Red shirt
person_C[128:, :] = (30, 30, 30)   # Black pants

# Person A slightly moved / shifted
person_A_var = person_A.copy()
noise = np.random.normal(0, 10, person_A.shape).astype(np.int16)
person_A_var = np.clip(person_A_var.astype(np.int16) + noise, 0, 255).astype(np.uint8)

fa_raw = extract_feat(person_A, False)
fa_var_raw = extract_feat(person_A_var, False)
fb_raw = extract_feat(person_B, False)
fc_raw = extract_feat(person_C, False)

print("\n--- RAW (Standard ReID Preprocessing) ---")
print("Person A vs Person A (noisy/different view):", float(np.dot(fa_raw, fa_var_raw)))
print("Person A vs Person B (Black shirt):", float(np.dot(fa_raw, fb_raw)))
print("Person A vs Person C (Red shirt):", float(np.dot(fa_raw, fc_raw)))
print("Person B vs Person C:", float(np.dot(fb_raw, fc_raw)))

fa_iso = extract_feat(person_A, True)
fa_var_iso = extract_feat(person_A_var, True)
fb_iso = extract_feat(person_B, True)
fc_iso = extract_feat(person_C, True)

print("\n--- WITH FOREGROUND ISOLATION (Current in project) ---")
print("Person A vs Person A (noisy):", float(np.dot(fa_iso, fa_var_iso)))
print("Person A vs Person B:", float(np.dot(fa_iso, fb_iso)))
print("Person A vs Person C:", float(np.dot(fa_iso, fc_iso)))
print("Person B vs Person C:", float(np.dot(fb_iso, fc_iso)))
