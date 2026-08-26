import sys
sys.path.insert(0, '.')
import os
import math
import cv2
import torch
import numpy as np
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

def run_analysis():
    print("=================== 1. FILTERING FLAT DUMMY CROPS & RECOMPUTING DISTRIBUTIONS ===================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    extractor = PyTorchReIDExtractor(model_name="osnet_x0_25", device=device, input_size=(256, 128))
    
    # Filter valid real crops (variance > 10, non-degenerate size)
    valid_obs_by_track = {}
    for obs in dataset.observations:
        crop = obs.get_crop()
        if crop is None or crop.size == 0:
            continue
        # Check if flat synthetic test crop (e.g. solid blue/green/red)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if np.std(gray) < 5.0 or crop.shape[0] < 30 or crop.shape[1] < 15:
            continue
        valid_obs_by_track.setdefault(obs.identity_id, []).append(obs)
        
    track_embeds = {}
    for tid, obs_list in valid_obs_by_track.items():
        sample_obs = obs_list[::max(1, len(obs_list)//6)][:6]
        crops = [o.get_crop() for o in sample_obs]
        if crops:
            embeds = extractor.extract_batch(crops)
            track_embeds[tid] = [e.vector for e in embeds]
            
    # Anchor assignment for the 4 real subjects
    subject_anchors = {
        "Subject_A_WhiteSleeveless": track_embeds.get("person_2", [np.zeros(512)])[0],
        "Subject_B_StripedShirt":    track_embeds.get("person_0", [np.zeros(512)])[0],
        "Subject_C_DarkShirt":       track_embeds.get("person_44", [np.zeros(512)])[0],
        "Subject_D_Bystander":       track_embeds.get("person_88", [np.zeros(512)])[0],
    }
    
    subject_tracks = {"Subject_A_WhiteSleeveless": [], "Subject_B_StripedShirt": [], "Subject_C_DarkShirt": [], "Subject_D_Bystander": []}
    for tid, vecs in track_embeds.items():
        proto = np.mean(vecs, axis=0)
        proto = proto / np.linalg.norm(proto)
        sims = {s: float(np.dot(proto, subject_anchors[s])) for s in subject_anchors}
        best_sub = max(sims.keys(), key=lambda s: sims[s])
        subject_tracks[best_sub].append(tid)
        
    # Genuine vs Impostor distributions
    genuine_sims = []
    impostor_sims = []
    
    s_keys = list(subject_tracks.keys())
    for s in s_keys:
        vecs = [v for tid in subject_tracks[s] for v in track_embeds[tid]]
        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                genuine_sims.append(float(np.dot(vecs[a], vecs[b])))
                
    for i, s1 in enumerate(s_keys):
        vecs1 = [v for tid in subject_tracks[s1] for v in track_embeds[tid]]
        for s2 in s_keys[i+1:]:
            vecs2 = [v for tid in subject_tracks[s2] for v in track_embeds[tid]]
            for a in range(len(vecs1)):
                for b in range(len(vecs2)):
                    impostor_sims.append(float(np.dot(vecs1[a], vecs2[b])))
                    
    g_arr = np.array(genuine_sims)
    i_arr = np.array(impostor_sims)
    
    print(f"Genuine Distribution (Clean Real Crops, N={len(g_arr)}):")
    print(f"  Min={np.min(g_arr):.4f}, p0.1={np.percentile(g_arr, 0.1):.4f}, p1={np.percentile(g_arr, 1.0):.4f}, p5={np.percentile(g_arr, 5.0):.4f}, p10={np.percentile(g_arr, 10.0):.4f}, p50={np.percentile(g_arr, 50.0):.4f}, Mean={np.mean(g_arr):.4f} (±{np.std(g_arr):.4f})")
    print(f"Impostor Distribution (Clean Real Crops, N={len(i_arr)}):")
    print(f"  Mean={np.mean(i_arr):.4f} (±{np.std(i_arr):.4f}), p90={np.percentile(i_arr, 90.0):.4f}, p95={np.percentile(i_arr, 95.0):.4f}, p99={np.percentile(i_arr, 99.0):.4f}, p99.9={np.percentile(i_arr, 99.9):.4f}, Max={np.max(i_arr):.4f}")

    print("\n=================== 2. EXACT BINOMIAL COMPOUND PROBABILITY MATH ===================")
    # Compare:
    # 1. Incorrect prior formula: p^4 (strict 4-of-4)
    # 2. Correct Binomial formula: P(>=3 of 4) for W=4, ConsistencyRatio >= 0.75
    # 3. Binomial formula for W=5 (P >= 4 of 5, ratio=0.80) and W=6 (P >= 4 of 6, ratio=0.667 or P >= 5 of 6, ratio=0.833)
    
    candidates = [0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96]
    print(f"{'Tau':>6} | {'Per-Frame FMR':>14} | {'Per-Frame TPR':>14} | {'Incorrect FMR (p^4)':>20} | {'Correct FMR (>=3/4)':>20} | {'Compound TPR (>=3/4)':>20} | {'Compound TPR (>=4/6)':>20}")
    print("-" * 125)
    
    for tau in candidates:
        p_fmr = float(np.mean(i_arr >= tau))
        p_tpr = float(np.mean(g_arr >= tau))
        
        incorrect_fmr = p_fmr ** 4
        correct_fmr_3_of_4 = binomial_prob_k_or_more(4, 3, p_fmr)
        correct_tpr_3_of_4 = binomial_prob_k_or_more(4, 3, p_tpr)
        correct_tpr_4_of_6 = binomial_prob_k_or_more(6, 4, p_tpr)
        correct_fmr_4_of_6 = binomial_prob_k_or_more(6, 4, p_fmr)
        
        print(f"{tau:6.2f} | {p_fmr*100:13.2f}% | {p_tpr*100:13.2f}% | {incorrect_fmr*100:19.4f}% | {correct_fmr_3_of_4*100:19.4f}% | {correct_tpr_3_of_4*100:19.2f}% | {correct_tpr_4_of_6*100:19.2f}%")

if __name__ == "__main__":
    run_analysis()
