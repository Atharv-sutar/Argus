import sys
sys.path.insert(0, '.')
import os
import cv2
import torch
import numpy as np
import math
from src.reid.extractor import PyTorchReIDExtractor
from src.benchmark.dataset import BenchmarkDataset

SCRATCH_DIR = r"C:\Users\athar\.gemini\antigravity-ide\brain\75b2be4d-720e-4da1-82d4-60616d7160d8\scratch"

def binomial_prob_k_or_more(n, k, p):
    """P(X >= k) for X ~ Binomial(n, p)"""
    total = 0.0
    for i in range(k, n + 1):
        comb = math.comb(n, i)
        total += comb * (p ** i) * ((1.0 - p) ** (n - i))
    return total

def run_investigation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    
    # -------------------------------------------------------------
    # 1. Investigate the 0.0507 genuine-pair outlier
    # -------------------------------------------------------------
    print("=================== 1. INVESTIGATING 0.0507 GENUINE OUTLIER ===================")
    extractor = PyTorchReIDExtractor(model_name="osnet_x0_25", device=device, input_size=(256, 128))
    
    obs_by_track = {}
    for obs in dataset.observations:
        obs_by_track.setdefault(obs.identity_id, []).append(obs)
        
    track_embeds = {}
    track_crops = {}
    track_meta = {}
    for tid, obs_list in obs_by_track.items():
        sample_obs = obs_list[::max(1, len(obs_list)//6)][:6]
        crops = [o.get_crop() for o in sample_obs if o.get_crop() is not None]
        valid_obs = [o for o in sample_obs if o.get_crop() is not None]
        if crops:
            embeds = extractor.extract_batch(crops)
            track_embeds[tid] = [e.vector for e in embeds]
            track_crops[tid] = crops
            track_meta[tid] = [(o.frame_id, o.camera_id, o.track_id) for o in valid_obs]
            
    # Anchors for the 4 human subjects in this recording session
    subject_anchors = {
        "Subject_A_WhiteSleeveless": track_embeds.get("person_2", [np.zeros(512)])[0],
        "Subject_B_StripedShirt":    track_embeds.get("person_0", [np.zeros(512)])[0],
        "Subject_C_DarkShirt":       track_embeds.get("person_44", [np.zeros(512)])[0],
        "Subject_D_Bystander":       track_embeds.get("person_38", [np.zeros(512)])[0],
    }
    
    # Group tracks by subject anchor
    subject_tracks = {"Subject_A_WhiteSleeveless": [], "Subject_B_StripedShirt": [], "Subject_C_DarkShirt": [], "Subject_D_Bystander": []}
    for tid, vecs in track_embeds.items():
        proto = np.mean(vecs, axis=0)
        proto = proto / np.linalg.norm(proto)
        sims = {s: float(np.dot(proto, subject_anchors[s])) for s in subject_anchors}
        best_sub = max(sims.keys(), key=lambda s: sims[s])
        best_sim = sims[best_sub]
        subject_tracks[best_sub].append((tid, best_sim))
        
    print("Subject track assignments & anchor similarity distributions:")
    for sname, t_list in subject_tracks.items():
        min_anchor_sim = min(sim for tid, sim in t_list) if t_list else 0
        max_anchor_sim = max(sim for tid, sim in t_list) if t_list else 0
        print(f"  {sname}: {len(t_list)} tracks | anchor sim range: [{min_anchor_sim:.4f}, {max_anchor_sim:.4f}]")
        low_sim_tracks = [t for t in t_list if t[1] < 0.60]
        if low_sim_tracks:
            print(f"    WARNING: Tracks with low anchor similarity in {sname}: {low_sim_tracks}")

    # Now find the lowest genuine pairs
    genuine_pairs = []
    for sname, t_list in subject_tracks.items():
        tids = [t[0] for t in t_list]
        for i, t1 in enumerate(tids):
            vecs1 = track_embeds[t1]
            for j, t2 in enumerate(tids[i:]):
                vecs2 = track_embeds[t2]
                start_b = 0 if t1 != t2 else 0 # we check all
                for a in range(len(vecs1)):
                    b_range = range(a + 1, len(vecs2)) if t1 == t2 else range(len(vecs2))
                    for b in b_range:
                        sim = float(np.dot(vecs1[a], vecs2[b]))
                        genuine_pairs.append((sim, sname, t1, a, t2, b, track_meta[t1][a], track_meta[t2][b]))
                        
    genuine_pairs.sort(key=lambda x: x[0])
    print(f"\nTop 10 Lowest Genuine Pairs:")
    for rank, (sim, sname, t1, a, t2, b, meta1, meta2) in enumerate(genuine_pairs[:10]):
        print(f"  #{rank+1}: Sim={sim:.4f} | Subject={sname} | {t1} (crop {a}, meta={meta1}) vs {t2} (crop {b}, meta={meta2})")
        
    # Save the lowest genuine crops for visual inspection
    if genuine_pairs:
        lowest = genuine_pairs[0]
        sim, sname, t1, a, t2, b, meta1, meta2 = lowest
        crop1 = track_crops[t1][a]
        crop2 = track_crops[t2][b]
        cv2.imwrite(os.path.join(SCRATCH_DIR, "outlier_crop1.png"), crop1)
        cv2.imwrite(os.path.join(SCRATCH_DIR, "outlier_crop2.png"), crop2)
        print(f"Saved outlier_crop1.png ({meta1}) and outlier_crop2.png ({meta2}) with similarity {sim:.4f}")

if __name__ == "__main__":
    run_investigation()
