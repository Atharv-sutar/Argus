import sys
sys.path.insert(0, '.')
import cv2
import torch
import numpy as np
from src.core.types import BoundingBox, Track, TrackState, MatchDecisionState, VerifiedIdentityDecision
from src.reid.extractor import PyTorchReIDExtractor
from src.identity.manager import IdentityManager
from src.benchmark.dataset import BenchmarkDataset

SCRATCH_DIR = r"C:\Users\athar\.gemini\antigravity-ide\brain\75b2be4d-720e-4da1-82d4-60616d7160d8\scratch"

def run_calibration_study():
    print("=================== STEP 2: DATA-DERIVED THRESHOLD CALIBRATION ===================", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    extractor = PyTorchReIDExtractor(model_name="osnet_x0_25", device=device, input_size=(256, 128))
    
    obs_by_track = {}
    for obs in dataset.observations:
        crop = obs.get_crop()
        if crop is not None and crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if np.std(gray) >= 5.0 and crop.shape[0] >= 30:
                obs_by_track.setdefault(obs.identity_id, []).append(obs)
                
    # Target Actor (Subject 1: White sleeveless + glasses)
    crops_cam0 = [o.get_crop() for o in obs_by_track.get("person_24", []) if o.get_crop() is not None]
    crops_cam1 = [o.get_crop() for o in obs_by_track.get("person_75", []) if o.get_crop() is not None]
    crops_cam2_desk = [o.get_crop() for o in obs_by_track.get("person_83", []) if o.get_crop() is not None]
    crops_cam2_stand = [o.get_crop() for o in obs_by_track.get("person_2", []) if o.get_crop() is not None]
    
    # Bystanders
    bystander_p18 = [o.get_crop() for o in obs_by_track.get("person_18", []) if o.get_crop() is not None]
    bystander_p15 = [o.get_crop() for o in obs_by_track.get("person_15", []) if o.get_crop() is not None]
    bystander_p0 = [o.get_crop() for o in obs_by_track.get("person_0", []) if o.get_crop() is not None]
    bystander_p10 = [o.get_crop() for o in obs_by_track.get("person_10", []) if o.get_crop() is not None]
    bystander_p48 = [o.get_crop() for o in obs_by_track.get("person_48", []) if o.get_crop() is not None]
    bystander_p51 = [o.get_crop() for o in obs_by_track.get("person_51", []) if o.get_crop() is not None]
    
    print(f"Target Crops: Cam0={len(crops_cam0)}, Cam1={len(crops_cam1)}, Cam2_desk={len(crops_cam2_desk)}, Cam2_stand={len(crops_cam2_stand)}")
    print(f"Bystanders: p18={len(bystander_p18)}, p15={len(bystander_p15)}, p0={len(bystander_p0)}, p10={len(bystander_p10)}, p48={len(bystander_p48)}, p51={len(bystander_p51)}")
    
    # Initialize Target on Cam 0
    id_mgr = IdentityManager(
        reid_extractor=extractor,
        vector_store=None,
        similarity_threshold=0.70, # evaluate raw score levels
        reacquisition_threshold=0.70,
        reference_threshold=0.70,
        upper_threshold=0.65,
    )
    id_mgr.register_new_target(crops_cam0[0], "target_A")
    for c in crops_cam0[1:4]:
        id_mgr.add_reference_sample(c, "target_A")
        
    print("\n--- Phase 1: Target Initial Transition (Cam 0 -> Cam 1) vs Bystanders on Cam 1 ---")
    t_cam1_scores = []
    t_cam1_upper = []
    for c in crops_cam1:
        ev = id_mgr.evaluate_candidate_crop(c, "target_A")
        t_cam1_scores.append(ev.candidate_score)
        t_cam1_upper.append(ev.upper_sim)
        
    b18_scores, b18_upper = [], []
    for c in bystander_p18:
        ev = id_mgr.evaluate_candidate_crop(c, "target_A")
        b18_scores.append(ev.candidate_score)
        b18_upper.append(ev.upper_sim)
        
    b15_scores, b15_upper = [], []
    for c in bystander_p15:
        ev = id_mgr.evaluate_candidate_crop(c, "target_A")
        b15_scores.append(ev.candidate_score)
        b15_upper.append(ev.upper_sim)
        
    b48_scores, b48_upper = [], []
    for c in bystander_p48:
        ev = id_mgr.evaluate_candidate_crop(c, "target_A")
        b48_scores.append(ev.candidate_score)
        b48_upper.append(ev.upper_sim)
        
    print(f"Target Cam 1:       CandidateScore = {np.mean(t_cam1_scores):.4f} +/- {np.std(t_cam1_scores):.4f} (Min={min(t_cam1_scores):.4f}, Max={max(t_cam1_scores):.4f}) | UpperSim = {np.mean(t_cam1_upper):.4f} (Min={min(t_cam1_upper):.4f})")
    print(f"Bystander p18:      CandidateScore = {np.mean(b18_scores):.4f} +/- {np.std(b18_scores):.4f} (Min={min(b18_scores):.4f}, Max={max(b18_scores):.4f}) | UpperSim = {np.mean(b18_upper):.4f} (Max={max(b18_upper):.4f})")
    print(f"Bystander p15:      CandidateScore = {np.mean(b15_scores):.4f} +/- {np.std(b15_scores):.4f} (Min={min(b15_scores):.4f}, Max={max(b15_scores):.4f}) | UpperSim = {np.mean(b15_upper):.4f} (Max={max(b15_upper):.4f})")
    print(f"Bystander p48:      CandidateScore = {np.mean(b48_scores):.4f} +/- {np.std(b48_scores):.4f} (Min={min(b48_scores):.4f}, Max={max(b48_scores):.4f}) | UpperSim = {np.mean(b48_upper):.4f} (Max={max(b48_upper):.4f})")
    
    # Enroll Cam 1 Viewpoint
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
    id_mgr.enroll_cross_camera_viewpoint(crops_cam1[0], "target_A", decision=fake_token, timestamp_ms=1000.0)
    
    print("\n--- Phase 2: Target Subsequent Transition (Cam 1 -> Cam 2) vs Bystanders on Cam 2 (Post-Gallery Fix) ---")
    t_cam2_desk_scores = []
    t_cam2_desk_upper = []
    for c in crops_cam2_desk:
        ev = id_mgr.evaluate_candidate_crop(c, "target_A")
        t_cam2_desk_scores.append(ev.candidate_score)
        t_cam2_desk_upper.append(ev.upper_sim)
        
    b10_scores, b10_upper = [], []
    for c in bystander_p10:
        ev = id_mgr.evaluate_candidate_crop(c, "target_A")
        b10_scores.append(ev.candidate_score)
        b10_upper.append(ev.upper_sim)
        
    b51_scores, b51_upper = [], []
    for c in bystander_p51:
        ev = id_mgr.evaluate_candidate_crop(c, "target_A")
        b51_scores.append(ev.candidate_score)
        b51_upper.append(ev.upper_sim)
        
    print(f"Target Cam 2 Desk:  CandidateScore = {np.mean(t_cam2_desk_scores):.4f} +/- {np.std(t_cam2_desk_scores):.4f} (Min={min(t_cam2_desk_scores):.4f}, Max={max(t_cam2_desk_scores):.4f}) | UpperSim = {np.mean(t_cam2_desk_upper):.4f} (Min={min(t_cam2_desk_upper):.4f})")
    print(f"Bystander p10:      CandidateScore = {np.mean(b10_scores):.4f} +/- {np.std(b10_scores):.4f} (Min={min(b10_scores):.4f}, Max={max(b10_scores):.4f}) | UpperSim = {np.mean(b10_upper):.4f} (Max={max(b10_upper):.4f})")
    print(f"Bystander p51:      CandidateScore = {np.mean(b51_scores):.4f} +/- {np.std(b51_scores):.4f} (Min={min(b51_scores):.4f}, Max={max(b51_scores):.4f}) | UpperSim = {np.mean(b51_upper):.4f} (Max={max(b51_upper):.4f})")
    
    # Summary of Separations
    print("\n--- Summary of Measured Separation Margins ---")
    all_bystander_scores = b18_scores + b15_scores + b48_scores + b10_scores + b51_scores
    max_bystander = max(all_bystander_scores)
    p95_bystander = np.percentile(all_bystander_scores, 95)
    
    print(f"All Bystanders (N={len(all_bystander_scores)}): Mean={np.mean(all_bystander_scores):.4f}, p95={p95_bystander:.4f}, Max={max_bystander:.4f}")
    print(f"Target Initial Cam 1 (N={len(t_cam1_scores)}):   Mean={np.mean(t_cam1_scores):.4f}, Min={min(t_cam1_scores):.4f}, Max={max(t_cam1_scores):.4f}")
    print(f"Target Post-Fix Cam 2 (N={len(t_cam2_desk_scores)}):  Mean={np.mean(t_cam2_desk_scores):.4f}, Min={min(t_cam2_desk_scores):.4f}, Max={max(t_cam2_desk_scores):.4f}")
    print(f"Separation Gap (Target Cam 1 Min vs Bystander Max): {min(t_cam1_scores) - max_bystander:+.4f}")
    print(f"Separation Gap (Target Cam 2 Min vs Bystander Max): {min(t_cam2_desk_scores) - max_bystander:+.4f}")

if __name__ == "__main__":
    run_calibration_study()
