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

def diagnose():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    extractor = PyTorchReIDExtractor(model_name="osnet_x0_25", device=device, input_size=(256, 128))
    
    obs_by_track = {}
    for obs in dataset.observations:
        crop = obs.get_crop()
        if crop is not None and crop.size > 0:
            obs_by_track.setdefault(obs.identity_id, []).append(obs)
            
    p24_crops = [o.get_crop() for o in obs_by_track.get("person_24", []) if o.get_crop() is not None]
    p75_crops = [o.get_crop() for o in obs_by_track.get("person_75", []) if o.get_crop() is not None]
    
    id_mgr = IdentityManager(
        reid_extractor=extractor,
        vector_store=None,
        similarity_threshold=0.90,
        reacquisition_threshold=0.95,
        reference_threshold=0.92,
        upper_threshold=0.88,
    )
    
    anchor_crop = p24_crops[0]
    id_mgr.register_new_target(crop=anchor_crop, identity_id="target_A")
    
    print(f"Target registered with anchor crop shape: {anchor_crop.shape}")
    
    for i, crop in enumerate(p75_crops[:5]):
        eval_res = id_mgr.evaluate_candidate_crop(crop, "target_A")
        print(f"\nCrop {i}: Shape={crop.shape}")
        print(f"  is_match: {eval_res.is_match} | decision: {eval_res.decision}")
        print(f"  candidate_score: {eval_res.candidate_score:.4f} | proto_sim: {eval_res.proto_sim:.4f} | best_ref_sim: {eval_res.best_ref_sim:.4f}")
        print(f"  upper_sim: {eval_res.upper_sim:.4f} (th={id_mgr.upper_threshold}) | lower_sim: {eval_res.lower_sim:.4f}")
        print(f"  rejection_reasons: {eval_res.rejection_reasons}")

if __name__ == "__main__":
    diagnose()
