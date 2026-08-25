"""Benchmark test comparing direct 128x256 resizing vs aspect-ratio-preserving letterboxing."""

import glob
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.reid.extractor import PyTorchReIDExtractor



def preprocess_letterbox(crop: np.ndarray, target_size=(256, 128)) -> np.ndarray:
    """Letterboxes crop preserving aspect ratio and pads with mean grey value."""
    target_h, target_w = target_size
    h, w = crop.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    
    # Center image in canvas
    dx = (target_w - nw) // 2
    dy = (target_h - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas


def run_benchmark():
    scratch_dir = r'C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch'
    crop_files = glob.glob(os.path.join(scratch_dir, "reid_cand_*.png"))
    
    if len(crop_files) < 10:
        print(f"Not enough scratch crops found ({len(crop_files)}). Creating synthetic person crops...")
        crops = []
        for i in range(10):
            # Synthetic crops with different aspect ratios
            img = np.random.randint(50, 200, (np.random.randint(100, 300), np.random.randint(40, 150), 3), dtype=np.uint8)
            crops.append(img)
    else:
        crops = [cv2.imread(f) for f in crop_files[:30] if cv2.imread(f) is not None]

    print(f"Loaded {len(crops)} crops for preprocessing benchmark.")

    extractor = PyTorchReIDExtractor(model_name="mobilenet_v3_small", device="cpu")

    direct_embeddings = []
    letterbox_embeddings = []

    for crop in crops:
        # 1. Direct resize embedding (default extractor)
        emb_direct = extractor.extract(crop)
        direct_embeddings.append(emb_direct)

        # 2. Letterbox embedding
        letterboxed_crop = preprocess_letterbox(crop)
        emb_letterbox = extractor.extract(letterboxed_crop)
        letterbox_embeddings.append(emb_letterbox)

    # Compute pairwise similarities for both strategies
    n = len(crops)
    direct_sims = []
    letterbox_sims = []

    for i in range(n):
        for j in range(i + 1, n):
            sim_d = direct_embeddings[i].cosine_similarity(direct_embeddings[j])
            sim_l = letterbox_embeddings[i].cosine_similarity(letterbox_embeddings[j])
            direct_sims.append(sim_d)
            letterbox_sims.append(sim_l)

    print("\n--- Preprocessing Strategy Comparison Benchmark ---")
    print(f"Total pairwise comparisons: {len(direct_sims)}")
    print(f"Direct 128x256   -> Mean Sim: {np.mean(direct_sims):.4f}, Std: {np.std(direct_sims):.4f}, Min: {np.min(direct_sims):.4f}, Max: {np.max(direct_sims):.4f}")
    print(f"Letterbox 128x256 -> Mean Sim: {np.mean(letterbox_sims):.4f}, Std: {np.std(letterbox_sims):.4f}, Min: {np.min(letterbox_sims):.4f}, Max: {np.max(letterbox_sims):.4f}")
    print("---------------------------------------------------\n")


if __name__ == "__main__":
    run_benchmark()
