import sys
sys.path.insert(0, '.')
import os
import numpy as np
import torch
from src.reid.extractor import PyTorchReIDExtractor
from src.benchmark.dataset import BenchmarkDataset

SCRATCH_DIR = r"C:\Users\athar\.gemini\antigravity-ide\brain\75b2be4d-720e-4da1-82d4-60616d7160d8\scratch"

def investigate():
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    print(f"Loaded {len(dataset.observations)} crops across {len(dataset.identities)} identities.")
    
    subject_obs = {}
    for obs in dataset.observations:
        subject_obs.setdefault(obs.identity_id, []).append(obs)
        
    for var in ["osnet_x0_25", "osnet_x1_0"]:
        print(f"\n=================== Inspecting {var} ===================")
        extractor = PyTorchReIDExtractor(model_name=var, device="cuda" if torch.cuda.is_available() else "cpu")
        
        subject_embeds = {}
        subject_meta = {}
        for sid, obs_list in subject_obs.items():
            sample_obs = obs_list[::max(1, len(obs_list)//10)][:10]
            crops = [o.get_crop() for o in sample_obs if o.get_crop() is not None]
            crops_valid = [o for o in sample_obs if o.get_crop() is not None]
            if crops:
                embeds = extractor.extract_batch(crops)
                subject_embeds[sid] = [e.vector for e in embeds]
                subject_meta[sid] = [(o.frame_id, o.track_id, o.camera_id) for o in crops_valid]
                
        # Check L2 norms & feature stats
        all_vecs = [v for s_list in subject_embeds.values() for v in s_list]
        norms = [np.linalg.norm(v) for v in all_vecs]
        stds = [np.std(v) for v in all_vecs]
        print(f"Norms: min={np.min(norms):.5f}, max={np.max(norms):.5f}")
        print(f"Vector stds across 512 dimensions: min={np.min(stds):.5f}, mean={np.mean(stds):.5f}, max={np.max(stds):.5f}")
        
        # Check all impostor pairs
        subjects = list(subject_embeds.keys())
        impostor_pairs = []
        for i, s1 in enumerate(subjects):
            em1 = subject_embeds[s1]
            for s2 in subjects[i+1:]:
                em2 = subject_embeds[s2]
                for a in range(len(em1)):
                    for b in range(len(em2)):
                        sim = float(np.dot(em1[a], em2[b]))
                        impostor_pairs.append((sim, s1, a, s2, b, subject_meta[s1][a], subject_meta[s2][b]))
                        
        impostor_pairs.sort(key=lambda x: x[0], reverse=True)
        print(f"Top 10 highest impostor similarities for {var}:")
        for rank, (sim, s1, a, s2, b, meta1, meta2) in enumerate(impostor_pairs[:10]):
            print(f"  #{rank+1}: Sim={sim:.6f} | {s1} (crop {a}, meta={meta1}) vs {s2} (crop {b}, meta={meta2})")
            
        # Check bottom 5 lowest impostor similarities
        print(f"Bottom 5 lowest impostor similarities for {var}:")
        for rank, (sim, s1, a, s2, b, meta1, meta2) in enumerate(impostor_pairs[-5:]):
            print(f"  #{rank+1}: Sim={sim:.6f} | {s1} vs {s2}")

if __name__ == "__main__":
    investigate()
