import sys
sys.path.insert(0, '.')
import os
import cv2
import torch
import numpy as np
from src.core.types import BoundingBox, Track, TrackState, MatchDecisionState, VerifiedIdentityDecision
from src.reid.extractor import PyTorchReIDExtractor
from src.identity.manager import IdentityManager
from src.benchmark.dataset import BenchmarkDataset

SCRATCH_DIR = r"C:\Users\athar\.gemini\antigravity-ide\brain\75b2be4d-720e-4da1-82d4-60616d7160d8\scratch"
TARGET_TRACKS = {'person_12', 'person_2', 'person_24', 'person_30', 'person_37', 'person_38', 'person_51', 'person_52', 'person_53', 'person_55', 'person_56', 'person_74', 'person_75', 'person_82', 'person_83'}
BYSTANDER_TRACKS = {'person_0', 'person_1', 'person_10', 'person_11', 'person_13', 'person_14', 'person_15', 'person_16', 'person_17', 'person_18', 'person_44', 'person_48'}

def find_safe_threshold():
    print("=================== LONE BYSTANDER SAFETY THRESHOLD CALIBRATION ===================", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    extractor = PyTorchReIDExtractor(model_name="osnet_x0_25", device=device, input_size=(256, 128))
    
    target_crops = []
    bystander_crops = []
    
    for obs in dataset.observations:
        crop = obs.get_crop()
        if crop is not None and crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if np.std(gray) >= 5.0 and crop.shape[0] >= 40 and crop.shape[1] >= 20:
                if obs.identity_id in TARGET_TRACKS:
                    target_crops.append(crop)
                elif obs.identity_id in BYSTANDER_TRACKS:
                    bystander_crops.append(crop)
                    
    # Initialize Multi-View Target Anchor (Cam 0 + Cam 1)
    id_mgr = IdentityManager(
        reid_extractor=extractor,
        vector_store=None,
        similarity_threshold=0.80,
        reacquisition_threshold=0.80,
        reference_threshold=0.80,
        upper_threshold=0.75,
    )
    id_mgr.register_new_target(target_crops[0], "target_A")
    for c in target_crops[1:4]:
        id_mgr.add_reference_sample(c, "target_A")
        
    fake_token = VerifiedIdentityDecision(
        target_identity_id="target_A",
        authorized_track_id=202,
        decision_state=MatchDecisionState.MATCH,
        confidence=0.85,
        margin=0.15,
        timestamp_ms=1000.0,
        reason="cross_camera_verified",
        source_camera_id="camera_1",
    )
    id_mgr.enroll_cross_camera_viewpoint(target_crops[len(target_crops)//2], "target_A", decision=fake_token, timestamp_ms=1000.0)
    
    # Measure breakdown for each bystander crop
    b_records = []
    for c in bystander_crops:
        ev = id_mgr.evaluate_candidate_crop(c, "target_A")
        b_records.append((ev.candidate_score, ev.proto_sim, ev.best_ref_sim, ev.upper_sim, ev.lower_sim))
        
    t_records = []
    for c in target_crops:
        ev = id_mgr.evaluate_candidate_crop(c, "target_A")
        t_records.append((ev.candidate_score, ev.proto_sim, ev.best_ref_sim, ev.upper_sim, ev.lower_sim))
        
    print(f"\n--- Target Decomposed Breakdown (N={len(t_records)}) ---")
    print(f"CandidateScore: Mean={np.mean([r[0] for r in t_records]):.4f}, Min={min([r[0] for r in t_records]):.4f}, p5={np.percentile([r[0] for r in t_records], 5):.4f}")
    print(f"ProtoSim:       Mean={np.mean([r[1] for r in t_records]):.4f}, Min={min([r[1] for r in t_records]):.4f}, p5={np.percentile([r[1] for r in t_records], 5):.4f}")
    print(f"BestRefSim:     Mean={np.mean([r[2] for r in t_records]):.4f}, Min={min([r[2] for r in t_records]):.4f}, p5={np.percentile([r[2] for r in t_records], 5):.4f}")
    print(f"UpperSim:       Mean={np.mean([r[3] for r in t_records]):.4f}, Min={min([r[3] for r in t_records]):.4f}, p5={np.percentile([r[3] for r in t_records], 5):.4f}")
    print(f"LowerSim:       Mean={np.mean([r[4] for r in t_records]):.4f}, Min={min([r[4] for r in t_records]):.4f}, p5={np.percentile([r[4] for r in t_records], 5):.4f}")
    
    print(f"\n--- Bystander Decomposed Breakdown (N={len(b_records)}) ---")
    print(f"CandidateScore: Mean={np.mean([r[0] for r in b_records]):.4f}, Max={max([r[0] for r in b_records]):.4f}, p95={np.percentile([r[0] for r in b_records], 95):.4f}")
    print(f"ProtoSim:       Mean={np.mean([r[1] for r in b_records]):.4f}, Max={max([r[1] for r in b_records]):.4f}, p95={np.percentile([r[1] for r in b_records], 95):.4f}")
    print(f"BestRefSim:     Mean={np.mean([r[2] for r in b_records]):.4f}, Max={max([r[2] for r in b_records]):.4f}, p95={np.percentile([r[2] for r in b_records], 95):.4f}")
    print(f"UpperSim:       Mean={np.mean([r[3] for r in b_records]):.4f}, Max={max([r[3] for r in b_records]):.4f}, p95={np.percentile([r[3] for r in b_records], 95):.4f}")
    print(f"LowerSim:       Mean={np.mean([r[4] for r in b_records]):.4f}, Max={max([r[4] for r in b_records]):.4f}, p95={np.percentile([r[4] for r in b_records], 95):.4f}")

if __name__ == "__main__":
    find_safe_threshold()
