import glob
import os
import cv2
import numpy as np
import torch
from src.reid.extractor import PyTorchReIDExtractor

def main():
    scratch_dir = r"C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch"
    crops = sorted(glob.glob(os.path.join(scratch_dir, "reid_cand_*.png")))
    extractor = PyTorchReIDExtractor(model_name="osnet_x0_25", device="cuda")

    # Group by track
    by_track = {}
    for c in crops:
        parts = os.path.basename(c).replace(".png", "").split("_")
        if "track" in parts:
            t = parts[parts.index("track") + 1]
            by_track.setdefault(t, []).append(c)

    tracks = [t for t, l in by_track.items() if len(l) >= 6][:15]
    print(f"Testing {len(tracks)} tracks on OSNet with direct crops...")

    def extract_direct(crop):
        h, w = (256, 128)
        resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
        tensor = torch.from_numpy(np.transpose(norm, (2, 0, 1))).unsqueeze(0).to("cuda")
        with torch.inference_mode():
            feat = extractor._model(tensor).squeeze().cpu().numpy().astype(np.float32)
        l2 = np.linalg.norm(feat)
        return feat / l2 if l2 > 0 else feat

    embeddings = {}
    for t in tracks:
        track_crops = [cv2.imread(p) for p in by_track[t][:8]]
        track_crops = [c for c in track_crops if c is not None and c.size > 0]
        if len(track_crops) >= 2:
            embs = [extract_direct(c) for c in track_crops]
            embeddings[t] = embs

    genuine_sims = []
    for t, embs in embeddings.items():
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                genuine_sims.append(float(np.dot(embs[i], embs[j])))

    impostor_sims = []
    t_list = list(embeddings.keys())
    for i in range(len(t_list)):
        for j in range(i + 1, len(t_list)):
            for e1 in embeddings[t_list[i]][:2]:
                for e2 in embeddings[t_list[j]][:2]:
                    impostor_sims.append(float(np.dot(e1, e2)))

    print(f"Direct Genuine:  {len(genuine_sims)} | Mean: {np.mean(genuine_sims):.4f} | Min: {np.min(genuine_sims):.4f} | Max: {np.max(genuine_sims):.4f}")
    print(f"Direct Impostor: {len(impostor_sims)} | Mean: {np.mean(impostor_sims):.4f} | Min: {np.min(impostor_sims):.4f} | Max: {np.max(impostor_sims):.4f}")

if __name__ == "__main__":
    main()
