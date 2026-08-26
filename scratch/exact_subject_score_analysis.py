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

def analyze_fast():
    print("=================== FAST GROUND TRUTH SCORE DISTRIBUTION ANALYSIS ===================", flush=True)
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
                    
    # Subsample 100 random crops to ensure fast execution
    np.random.seed(42)
    t_indices = np.random.choice(len(target_crops), size=min(120, len(target_crops)), replace=False)
    b_indices = np.random.choice(len(bystander_crops), size=min(120, len(bystander_crops)), replace=False)
    
    eval_target = [target_crops[i] for i in t_indices]
    eval_bystander = [bystander_crops[i] for i in b_indices]
    
    print(f"Evaluating: Target Crops = {len(eval_target)}, Bystander Crops = {len(eval_bystander)}")
    
    # 1. Baseline Anchor (Camera 0 alone)
    id_mgr_single = IdentityManager(
        reid_extractor=extractor,
        vector_store=None,
        similarity_threshold=0.80,
        reacquisition_threshold=0.85,
        reference_threshold=0.80,
        upper_threshold=0.75,
    )
    anchor_crop = eval_target[0]
    id_mgr_single.register_new_target(anchor_crop, "target_A")
    for c in eval_target[1:4]:
        id_mgr_single.add_reference_sample(c, "target_A")
        
    t_res_single = [id_mgr_single.evaluate_candidate_crop(c, "target_A") for c in eval_target]
    b_res_single = [id_mgr_single.evaluate_candidate_crop(c, "target_A") for c in eval_bystander]
    
    t_scores_single = [r.candidate_score for r in t_res_single]
    t_upper_single = [r.upper_sim for r in t_res_single]
    
    b_scores_single = [r.candidate_score for r in b_res_single]
    b_upper_single = [r.upper_sim for r in b_res_single]
    
    print(f"\n--- [1] SINGLE-VIEW ANCHOR (Camera 0 Alone) ---")
    print(f"Target (N={len(eval_target)}):")
    print(f"  Score:    Mean={np.mean(t_scores_single):.4f} +/- {np.std(t_scores_single):.4f} (Min={min(t_scores_single):.4f}, p5={np.percentile(t_scores_single, 5):.4f}, p10={np.percentile(t_scores_single, 10):.4f}, Median={np.median(t_scores_single):.4f})")
    print(f"  UpperSim: Mean={np.mean(t_upper_single):.4f} +/- {np.std(t_upper_single):.4f} (Min={min(t_upper_single):.4f}, p5={np.percentile(t_upper_single, 5):.4f})")
    
    print(f"Bystanders (N={len(eval_bystander)}):")
    print(f"  Score:    Mean={np.mean(b_scores_single):.4f} +/- {np.std(b_scores_single):.4f} (Max={max(b_scores_single):.4f}, p95={np.percentile(b_scores_single, 95):.4f}, p90={np.percentile(b_scores_single, 90):.4f}, Median={np.median(b_scores_single):.4f})")
    print(f"  UpperSim: Mean={np.mean(b_upper_single):.4f} +/- {np.std(b_upper_single):.4f} (Max={max(b_upper_single):.4f}, p95={np.percentile(b_upper_single, 95):.4f})")
    
    # 2. Multi-View Anchor (Enrolling Cam 1 Viewpoint)
    cam1_target = eval_target[len(eval_target)//2]
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
    id_mgr_single.enroll_cross_camera_viewpoint(cam1_target, "target_A", decision=fake_token, timestamp_ms=1000.0)
    
    t_res_multi = [id_mgr_single.evaluate_candidate_crop(c, "target_A") for c in eval_target]
    b_res_multi = [id_mgr_single.evaluate_candidate_crop(c, "target_A") for c in eval_bystander]
    
    t_scores_multi = [r.candidate_score for r in t_res_multi]
    t_upper_multi = [r.upper_sim for r in t_res_multi]
    
    b_scores_multi = [r.candidate_score for r in b_res_multi]
    b_upper_multi = [r.upper_sim for r in b_res_multi]
    
    print(f"\n--- [2] MULTI-VIEW ANCHOR (Post-Gallery Fix: Camera 0 + Camera 1 Enrolled) ---")
    print(f"Target (N={len(eval_target)}):")
    print(f"  Score:    Mean={np.mean(t_scores_multi):.4f} +/- {np.std(t_scores_multi):.4f} (Min={min(t_scores_multi):.4f}, p5={np.percentile(t_scores_multi, 5):.4f}, p10={np.percentile(t_scores_multi, 10):.4f}, Median={np.median(t_scores_multi):.4f})")
    print(f"  UpperSim: Mean={np.mean(t_upper_multi):.4f} +/- {np.std(t_upper_multi):.4f} (Min={min(t_upper_multi):.4f}, p5={np.percentile(t_upper_multi, 5):.4f})")
    
    print(f"Bystanders (N={len(eval_bystander)}):")
    print(f"  Score:    Mean={np.mean(b_scores_multi):.4f} +/- {np.std(b_scores_multi):.4f} (Max={max(b_scores_multi):.4f}, p95={np.percentile(b_scores_multi, 95):.4f}, p90={np.percentile(b_scores_multi, 90):.4f}, Median={np.median(b_scores_multi):.4f})")
    print(f"  UpperSim: Mean={np.mean(b_upper_multi):.4f} +/- {np.std(b_upper_multi):.4f} (Max={max(b_upper_multi):.4f}, p95={np.percentile(b_upper_multi, 95):.4f})")
    
    # 3. Derive Optimal Thresholds
    print("\n--- [3] THRESHOLD DERIVATION MATRIX ---")
    print(f"{'Threshold (Score/Upper)':<25} | {'Target TPR (Single-View)':<25} | {'Bystander FMR (Single)':<25} | {'Target TPR (Multi-View)':<25} | {'Bystander FMR (Multi)':<25}")
    for th, up_th in [(0.90, 0.85), (0.88, 0.82), (0.85, 0.80), (0.82, 0.78), (0.80, 0.75), (0.78, 0.72)]:
        tpr_s = sum(1 for s, u in zip(t_scores_single, t_upper_single) if s >= th and u >= up_th) / len(t_scores_single) * 100
        fmr_s = sum(1 for s, u in zip(b_scores_single, b_upper_single) if s >= th and u >= up_th) / len(b_scores_single) * 100
        tpr_m = sum(1 for s, u in zip(t_scores_multi, t_upper_multi) if s >= th and u >= up_th) / len(t_scores_multi) * 100
        fmr_m = sum(1 for s, u in zip(b_scores_multi, b_upper_multi) if s >= th and u >= up_th) / len(b_scores_multi) * 100
        print(f"tau={th:.2f}, upper={up_th:.2f}         | {tpr_s:6.2f}%                   | {fmr_s:6.2f}%                   | {tpr_m:6.2f}%                   | {fmr_m:6.2f}%")

if __name__ == "__main__":
    analyze_fast()
