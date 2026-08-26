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

def observe_gallery_fix_impact():
    print("=================== STEP 1: ISOLATED GALLERY FIX IMPACT OBSERVATION ===================", flush=True)
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
    # Cam 0 (Enrollment): person_24 (close-up portrait webcam)
    # Cam 1 (First Transition): person_75 (seated desk view)
    # Cam 2 (Second Transition): person_83 (seated desk view) & person_2 (standing wide view)
    
    crops_cam0 = [o.get_crop() for o in obs_by_track.get("person_24", []) if o.get_crop() is not None]
    crops_cam1 = [o.get_crop() for o in obs_by_track.get("person_75", []) if o.get_crop() is not None]
    crops_cam2_desk = [o.get_crop() for o in obs_by_track.get("person_83", []) if o.get_crop() is not None]
    crops_cam2_standing = [o.get_crop() for o in obs_by_track.get("person_2", []) if o.get_crop() is not None]
    
    print(f"Loaded Real Crops: Cam0={len(crops_cam0)}, Cam1={len(crops_cam1)}, Cam2_desk={len(crops_cam2_desk)}, Cam2_standing={len(crops_cam2_standing)}")
    
    # -------------------------------------------------------------
    # Experiment A: Baseline without Gallery Fix (Only Cam0 enrollment)
    # -------------------------------------------------------------
    print("\n--- Baseline: Target enrolled on Cam 0 ONLY (No cross-camera enrollment) ---", flush=True)
    id_mgr_base = IdentityManager(
        reid_extractor=extractor,
        vector_store=None,
        similarity_threshold=0.90,
        reacquisition_threshold=0.95,
        reference_threshold=0.92,
        upper_threshold=0.88,
    )
    id_mgr_base.register_new_target(crops_cam0[0], "target_A")
    for c in crops_cam0[1:4]:
        id_mgr_base.add_reference_sample(c, "target_A")
        
    print(f"Initial Anchor ViewClusters: {len(id_mgr_base.get_identity('target_A').anchor.clusters)}")
    
    # Evaluate Cam 1 crops against Cam 0 anchor
    cam1_scores_base = []
    for i, crop in enumerate(crops_cam1[:10]):
        ev = id_mgr_base.evaluate_candidate_crop(crop, "target_A")
        cam1_scores_base.append((ev.candidate_score, ev.proto_sim, ev.best_ref_sim, ev.upper_sim, ev.is_match))
        
    # Evaluate Cam 2 crops against Cam 0 anchor
    cam2_scores_base = []
    for i, crop in enumerate(crops_cam2_desk[:10]):
        ev = id_mgr_base.evaluate_candidate_crop(crop, "target_A")
        cam2_scores_base.append((ev.candidate_score, ev.proto_sim, ev.best_ref_sim, ev.upper_sim, ev.is_match))
        
    cam2_std_scores_base = []
    for i, crop in enumerate(crops_cam2_standing[:10]):
        ev = id_mgr_base.evaluate_candidate_crop(crop, "target_A")
        cam2_std_scores_base.append((ev.candidate_score, ev.proto_sim, ev.best_ref_sim, ev.upper_sim, ev.is_match))
        
    print(f"Cam 1 (Seated) vs Cam 0 Anchor alone: CandidateScore Mean = {np.mean([s[0] for s in cam1_scores_base]):.4f} (Max={max([s[0] for s in cam1_scores_base]):.4f}, Min={min([s[0] for s in cam1_scores_base]):.4f}) | Passed is_match={sum([1 for s in cam1_scores_base if s[4]])}/10")
    print(f"Cam 2 (Desk)   vs Cam 0 Anchor alone: CandidateScore Mean = {np.mean([s[0] for s in cam2_scores_base]):.4f} (Max={max([s[0] for s in cam2_scores_base]):.4f}, Min={min([s[0] for s in cam2_scores_base]):.4f}) | Passed is_match={sum([1 for s in cam2_scores_base if s[4]])}/10")
    print(f"Cam 2 (Stand)  vs Cam 0 Anchor alone: CandidateScore Mean = {np.mean([s[0] for s in cam2_std_scores_base]):.4f} (Max={max([s[0] for s in cam2_std_scores_base]):.4f}, Min={min([s[0] for s in cam2_std_scores_base]):.4f}) | Passed is_match={sum([1 for s in cam2_std_scores_base if s[4]])}/10")
    
    # -------------------------------------------------------------
    # Experiment B: Post-Gallery Fix (Cam 0 enrollment + Verified Cam 1 enrollment)
    # -------------------------------------------------------------
    print("\n--- Post-Gallery Fix: Enrolling verified Cam 1 viewpoint into TargetIdentityAnchor ---", flush=True)
    id_mgr_fixed = IdentityManager(
        reid_extractor=extractor,
        vector_store=None,
        similarity_threshold=0.90,
        reacquisition_threshold=0.95,
        reference_threshold=0.92,
        upper_threshold=0.88,
    )
    id_mgr_fixed.register_new_target(crops_cam0[0], "target_A")
    for c in crops_cam0[1:4]:
        id_mgr_fixed.add_reference_sample(c, "target_A")
        
    # Simulate verified cross-camera reacquisition on Cam 1 with verified decision token
    fake_verified_token = VerifiedIdentityDecision(
        target_identity_id="target_A",
        authorized_track_id=202,
        decision_state=MatchDecisionState.MATCH,
        confidence=0.85,
        margin=0.15,
        timestamp_ms=1000.0,
        reason="cross_camera_verified",
        source_camera_id="camera_1",
    )
    
    enrolled = id_mgr_fixed.enroll_cross_camera_viewpoint(crops_cam1[0], "target_A", decision=fake_verified_token, timestamp_ms=1000.0)
    print(f"Cam 1 viewpoint enrolled successfully: {enrolled}")
    print(f"Updated Anchor ViewClusters: {len(id_mgr_fixed.get_identity('target_A').anchor.clusters)}")
    
    # Re-evaluate Cam 2 crops against multi-view anchor (Cam 0 + Cam 1)
    cam2_scores_fixed = []
    for i, crop in enumerate(crops_cam2_desk[:10]):
        ev = id_mgr_fixed.evaluate_candidate_crop(crop, "target_A")
        cam2_scores_fixed.append((ev.candidate_score, ev.proto_sim, ev.best_ref_sim, ev.upper_sim, ev.is_match))
        
    cam2_std_scores_fixed = []
    for i, crop in enumerate(crops_cam2_standing[:10]):
        ev = id_mgr_fixed.evaluate_candidate_crop(crop, "target_A")
        cam2_std_scores_fixed.append((ev.candidate_score, ev.proto_sim, ev.best_ref_sim, ev.upper_sim, ev.is_match))
        
    print(f"\n[RESULTS AFTER GALLERY FIX]")
    print(f"Cam 2 (Desk)   vs Multi-View Anchor: CandidateScore Mean = {np.mean([s[0] for s in cam2_scores_fixed]):.4f} (Max={max([s[0] for s in cam2_scores_fixed]):.4f}, Min={min([s[0] for s in cam2_scores_fixed]):.4f}) | Passed is_match={sum([1 for s in cam2_scores_fixed if s[4]])}/10")
    print(f"Cam 2 (Stand)  vs Multi-View Anchor: CandidateScore Mean = {np.mean([s[0] for s in cam2_std_scores_fixed]):.4f} (Max={max([s[0] for s in cam2_std_scores_fixed]):.4f}, Min={min([s[0] for s in cam2_std_scores_fixed]):.4f}) | Passed is_match={sum([1 for s in cam2_std_scores_fixed if s[4]])}/10")
    
    print("\nDetailed Score Progression for Cam 2 Desk Crops:")
    for idx, (b, f) in enumerate(zip(cam2_scores_base, cam2_scores_fixed)):
        print(f"  Crop {idx}: BaseScore={b[0]:.4f} (Proto={b[1]:.4f}, Ref={b[2]:.4f}) -> FixedScore={f[0]:.4f} (Proto={f[1]:.4f}, Ref={f[2]:.4f}) | Gain = +{f[0]-b[0]:.4f}")

if __name__ == "__main__":
    observe_gallery_fix_impact()
