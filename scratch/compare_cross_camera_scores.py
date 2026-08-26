import sys
sys.path.insert(0, '.')
import os
import cv2
import torch
import numpy as np
from src.reid.extractor import PyTorchReIDExtractor
from src.identity.manager import IdentityManager
from src.benchmark.dataset import BenchmarkDataset

SCRATCH_DIR = r"C:\Users\athar\.gemini\antigravity-ide\brain\75b2be4d-720e-4da1-82d4-60616d7160d8\scratch"

def compare_scores():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    extractor = PyTorchReIDExtractor(model_name="osnet_x0_25", device=device, input_size=(256, 128))
    
    obs_by_track = {}
    for obs in dataset.observations:
        crop = obs.get_crop()
        if crop is not None and crop.size > 0:
            obs_by_track.setdefault(obs.identity_id, []).append(obs)
            
    p24_crops = [o.get_crop() for o in obs_by_track.get("person_24", []) if o.get_crop() is not None] # Target on cam_0
    p75_crops = [o.get_crop() for o in obs_by_track.get("person_75", []) if o.get_crop() is not None] # Target on cam_1
    p83_crops = [o.get_crop() for o in obs_by_track.get("person_83", []) if o.get_crop() is not None] # Target on cam_2
    
    b_crops_18 = [o.get_crop() for o in obs_by_track.get("person_18", []) if o.get_crop() is not None] # Bystander B
    b_crops_44 = [o.get_crop() for o in obs_by_track.get("person_44", []) if o.get_crop() is not None] # Bystander C
    
    id_mgr = IdentityManager(
        reid_extractor=extractor,
        vector_store=None,
        similarity_threshold=0.75,
        reacquisition_threshold=0.78,
        reference_threshold=0.72,
        upper_threshold=0.70,
    )
    
    id_mgr.register_new_target(crop=p24_crops[0], identity_id="target_A")
    
    print("=== Target on Cam_1 (p75) vs Target Anchor (p24) ===")
    t_scores = []
    for c in p75_crops[:10]:
        res = id_mgr.evaluate_candidate_crop(c, "target_A")
        t_scores.append((res.candidate_score, res.proto_sim, res.upper_sim, res.is_match))
        print(f"  Target Cam1: CandScore={res.candidate_score:.4f}, Proto={res.proto_sim:.4f}, Upper={res.upper_sim:.4f}, is_match={res.is_match}")
        
    print("\n=== Bystander B (p18) vs Target Anchor (p24) ===")
    b18_scores = []
    for c in b_crops_18[:10]:
        res = id_mgr.evaluate_candidate_crop(c, "target_A")
        b18_scores.append((res.candidate_score, res.proto_sim, res.upper_sim, res.is_match))
        print(f"  Bystander B: CandScore={res.candidate_score:.4f}, Proto={res.proto_sim:.4f}, Upper={res.upper_sim:.4f}, is_match={res.is_match}")
        
    print("\n=== Bystander C (p44) vs Target Anchor (p24) ===")
    b44_scores = []
    for c in b_crops_44[:10]:
        res = id_mgr.evaluate_candidate_crop(c, "target_A")
        b44_scores.append((res.candidate_score, res.proto_sim, res.upper_sim, res.is_match))
        print(f"  Bystander C: CandScore={res.candidate_score:.4f}, Proto={res.proto_sim:.4f}, Upper={res.upper_sim:.4f}, is_match={res.is_match}")

if __name__ == "__main__":
    compare_scores()
