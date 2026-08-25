import sys
sys.path.insert(0, '.')
import time
import os
import glob
import cv2
import torch
import numpy as np
from src.reid.extractor import PyTorchReIDExtractor
from src.benchmark.dataset import BenchmarkDataset

SCRATCH_DIR = r"C:\Users\athar\.gemini\antigravity-ide\brain\75b2be4d-720e-4da1-82d4-60616d7160d8\scratch"

def evaluate_variant(variant_name: str, device: str = "cuda"):
    print(f"\n{'='*25} EVALUATING: {variant_name} {'='*25}", flush=True)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    input_size = (224, 224) if "dino" in variant_name else (256, 128)
    extractor = PyTorchReIDExtractor(model_name=variant_name, device=device, input_size=input_size)
    
    # 1. Latency measurement
    dummy_single = np.random.randint(0, 256, (input_size[0], input_size[1], 3), dtype=np.uint8)
    dummy_batch = [dummy_single for _ in range(4)]
    
    for _ in range(10):
        _ = extractor.extract(dummy_single)
        _ = extractor.extract_batch(dummy_batch)
    
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        
    t0 = time.perf_counter()
    n_iters = 50
    for _ in range(n_iters):
        _ = extractor.extract(dummy_single)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    lat_single_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0
    
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = extractor.extract_batch(dummy_batch)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    lat_batch_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0
    
    vram_mb = 0.0
    if device == "cuda" and torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
    print(f"Latency Single (B=1): {lat_single_ms:.2f} ms", flush=True)
    print(f"Latency Batch (B=4):  {lat_batch_ms:.2f} ms ({lat_batch_ms/4:.2f} ms/person)", flush=True)
    print(f"Peak VRAM Usage:      {vram_mb:.2f} MB", flush=True)
    
    # 2. Score Separation on Real Surveillance Video Trajectories
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    print(f"Loaded {len(dataset.observations)} crops across {len(dataset.identities)} identities.", flush=True)
    
    subject_obs = {}
    for obs in dataset.observations:
        subject_obs.setdefault(obs.identity_id, []).append(obs)
            
    subject_embeds = {}
    for sid, obs_list in subject_obs.items():
        sample_obs = obs_list[::max(1, len(obs_list)//10)][:10]
        crops = [o.get_crop() for o in sample_obs if o.get_crop() is not None]
        if crops:
            embeds = extractor.extract_batch(crops)
            subject_embeds[sid] = [e.vector for e in embeds]
            
    genuine_sims = []
    impostor_sims = []
    
    subjects = list(subject_embeds.keys())
    for i, s1 in enumerate(subjects):
        em1 = subject_embeds[s1]
        for a in range(len(em1)):
            for b in range(a + 1, len(em1)):
                genuine_sims.append(float(np.dot(em1[a], em1[b])))
        for s2 in subjects[i+1:]:
            em2 = subject_embeds[s2]
            for a in range(len(em1)):
                for b in range(len(em2)):
                    impostor_sims.append(float(np.dot(em1[a], em2[b])))
                    
    g_mean = float(np.mean(genuine_sims)) if genuine_sims else 0.0
    g_min = float(np.min(genuine_sims)) if genuine_sims else 0.0
    g_std = float(np.std(genuine_sims)) if genuine_sims else 0.0
    
    i_mean = float(np.mean(impostor_sims)) if impostor_sims else 0.0
    i_max = float(np.max(impostor_sims)) if impostor_sims else 0.0
    i_std = float(np.std(impostor_sims)) if impostor_sims else 0.0
    
    gap = g_mean - i_mean
    margin_min_max = g_min - i_max
    
    print(f"Genuine Pairs:  N={len(genuine_sims)} | Mean={g_mean:.4f} (±{g_std:.4f}) | Min={g_min:.4f}", flush=True)
    print(f"Impostor Pairs: N={len(impostor_sims)} | Mean={i_mean:.4f} (±{i_std:.4f}) | Max={i_max:.4f}", flush=True)
    print(f"Score Gap (Mean Gap): {gap:.4f}", flush=True)
    print(f"Min Genuine - Max Impostor Margin: {margin_min_max:.4f}", flush=True)
    
    # 3. F2 & F3 Direct Pair Evaluations
    t_a_crops = [obs.get_crop() for obs in subject_obs.get("person_0", [])[:8] if obs.get_crop() is not None]
    t_b_crops = [obs.get_crop() for obs in subject_obs.get("person_1", [])[:8] if obs.get_crop() is not None]
    
    if t_a_crops and t_b_crops:
        emb_a = [e.vector for e in extractor.extract_batch(t_a_crops)]
        emb_b = [e.vector for e in extractor.extract_batch(t_b_crops)]
        
        # Target A vs Target A (F2 test)
        f2_same = [float(np.dot(emb_a[i], emb_a[j])) for i in range(len(emb_a)) for j in range(i+1, len(emb_a))]
        # Target A vs Bystander B (F3 test)
        f3_diff = [float(np.dot(emb_a[i], emb_b[j])) for i in range(len(emb_a)) for j in range(len(emb_b))]
        
        print(f"\n[F2/F3 Scenario Evaluation]", flush=True)
        print(f"  F2 Target-A vs Target-A (Same Person): Mean={np.mean(f2_same):.4f} | Min={np.min(f2_same):.4f}", flush=True)
        print(f"  F3 Target-A vs Bystander-B (Hard Impostor): Mean={np.mean(f3_diff):.4f} | Max={np.max(f3_diff):.4f}", flush=True)
        print(f"  F2/F3 Delta Margin: {np.mean(f2_same) - np.mean(f3_diff):.4f}", flush=True)
    
    return {
        "variant": variant_name,
        "lat_single_ms": lat_single_ms,
        "lat_batch_ms": lat_batch_ms,
        "vram_mb": vram_mb,
        "g_mean": g_mean,
        "g_min": g_min,
        "i_mean": i_mean,
        "i_max": i_max,
        "gap": gap,
    }

if __name__ == "__main__":
    results = {}
    for var in ["dinov2", "osnet_x0_25", "osnet_x1_0"]:
        try:
            results[var] = evaluate_variant(var)
        except Exception as e:
            print(f"Error evaluating {var}: {e}", flush=True)
