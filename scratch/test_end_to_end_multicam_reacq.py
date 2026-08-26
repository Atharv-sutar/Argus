import sys
sys.path.insert(0, '.')
import time
import os
import cv2
import torch
import numpy as np
from src.core.types import BoundingBox, Track, TrackState, MatchDecisionState
from src.reid.extractor import PyTorchReIDExtractor
from src.identity.manager import IdentityManager
from src.identity.evidence import EvidenceEngine
from src.benchmark.dataset import BenchmarkDataset

SCRATCH_DIR = r"C:\Users\athar\.gemini\antigravity-ide\brain\75b2be4d-720e-4da1-82d4-60616d7160d8\scratch"

def run_fast_end_to_end_multicam_simulation():
    print("=================== FAST END-TO-END MULTI-CAMERA REACQUISITION BENCHMARK ===================", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    extractor = PyTorchReIDExtractor(model_name="osnet_x0_25", device=device, input_size=(256, 128))
    
    # Load crops for our actors
    obs_by_track = {}
    for obs in dataset.observations:
        crop = obs.get_crop()
        if crop is not None and crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if np.std(gray) >= 5.0 and crop.shape[0] >= 30:
                obs_by_track.setdefault(obs.identity_id, []).append(obs)
                
    target_crops_cam0 = [o.get_crop() for o in obs_by_track.get("person_24", []) if o.get_crop() is not None]
    target_crops_cam1 = [o.get_crop() for o in obs_by_track.get("person_75", []) if o.get_crop() is not None]
    target_crops_cam2 = [o.get_crop() for o in obs_by_track.get("person_83", []) if o.get_crop() is not None]
    
    bystander_crops_b = [o.get_crop() for o in obs_by_track.get("person_18", []) if o.get_crop() is not None]
    bystander_crops_c = [o.get_crop() for o in obs_by_track.get("person_44", []) if o.get_crop() is not None]
    
    print(f"Loaded: Target Cam0 ({len(target_crops_cam0)}), Cam1 ({len(target_crops_cam1)}), Cam2 ({len(target_crops_cam2)})")
    print(f"Loaded: Bystanders ({len(bystander_crops_b)} + {len(bystander_crops_c)})")
    
    configs_to_test = [
        ("Current Config (reacq=0.95, sim=0.90, W=4, ratio=0.75)", 0.95, 0.90, 0.92, 0.88, 4, 0.75),
        ("Calibrated 4-Frame (reacq=0.90, sim=0.88, W=4, ratio=0.75)", 0.90, 0.88, 0.90, 0.85, 4, 0.75),
        ("Calibrated 5-Frame (reacq=0.90, sim=0.88, W=5, ratio=0.75)", 0.90, 0.88, 0.90, 0.85, 5, 0.75),
        ("Calibrated 6-Frame (reacq=0.90, sim=0.88, W=6, ratio=0.67)", 0.90, 0.88, 0.90, 0.85, 6, 0.67),
        ("Relaxed Threshold (reacq=0.88, sim=0.85, W=4, ratio=0.75)", 0.88, 0.85, 0.88, 0.82, 4, 0.75),
    ]
    
    for cfg_name, reacq_th, sim_th, ref_th, upper_th, win_size, ratio in configs_to_test:
        print(f"\n=================== Testing: {cfg_name} ===================", flush=True)
        
        n_trials = 20
        total_correct_reacquisitions = 0
        total_false_adoptions = 0
        total_correct_lost_holds = 0
        frames_to_reacquire = []
        
        for trial in range(n_trials):
            id_mgr = IdentityManager(
                reid_extractor=extractor,
                vector_store=None,
                similarity_threshold=sim_th,
                reacquisition_threshold=reacq_th,
                reference_threshold=ref_th,
                upper_threshold=upper_th,
                reacquisition_min_frames=win_size,
            )
            
            # Step 1: Initialize target on Camera 0
            anchor_crop = target_crops_cam0[trial % len(target_crops_cam0)]
            target_identity_id = "target_person_A"
            id_mgr.register_new_target(crop=anchor_crop, identity_id=target_identity_id)
            
            # Step 2: Target is lost on Camera 0 (Target enters LOST state)
            # Bystanders appear on Camera 1 while target is lost
            evidence_cam1 = EvidenceEngine(
                window_size=win_size,
                min_similarity_threshold=sim_th,
                reacquisition_threshold=reacq_th,
                reacquisition_min_frames=win_size,
                min_consistency_ratio=ratio,
                min_margin_threshold=0.05,
            )
            
            # 10 frames of lone bystanders on Camera 1
            for f in range(10):
                b_crop = bystander_crops_b[(trial + f) % len(bystander_crops_b)]
                eval_res = id_mgr.evaluate_candidate_crop(b_crop, target_identity_id)
                sim_full = eval_res.candidate_score
                is_match = eval_res.is_match
                
                track_bystander = Track(track_id=101, box=BoundingBox(10.0, 10.0, 100.0, 200.0), state=TrackState.TRACKED)
                evidence_cam1.register_observation(
                    track_id=101,
                    frame_id=f,
                    timestamp_ms=float(f * 33.3),
                    crop_quality=1.0,
                    similarity=sim_full,
                    margin=0.0,
                    is_match=is_match,
                    box=track_bystander.box,
                )
                
                decision = evidence_cam1.evaluate_all_candidates(
                    candidate_evaluations=[(track_bystander, sim_full, is_match, 1.0)],
                    target_identity_id=target_identity_id,
                    current_tracked_id=None,
                    is_reacquisition=True,
                )
                
                if decision.is_confirmed and decision.best_track_id == 101:
                    total_false_adoptions += 1
                else:
                    total_correct_lost_holds += 1
                    
            # Step 3: Target arrives on Camera 1 (alongside a bystander)
            target_reacquired = False
            for f in range(15):
                t_crop = target_crops_cam1[(trial + f) % len(target_crops_cam1)]
                eval_t = id_mgr.evaluate_candidate_crop(t_crop, target_identity_id)
                sim_t_full = eval_t.candidate_score
                is_t_match = eval_t.is_match
                
                b_crop = bystander_crops_c[(trial + f) % len(bystander_crops_c)]
                eval_b = id_mgr.evaluate_candidate_crop(b_crop, target_identity_id)
                sim_b_full = eval_b.candidate_score
                is_b_match = eval_b.is_match
                
                track_target = Track(track_id=202, box=BoundingBox(20.0 + f * 5.0, 20.0, 100.0 + f * 5.0, 200.0), state=TrackState.TRACKED)
                track_bystander = Track(track_id=303, box=BoundingBox(150.0 + f * 5.0, 20.0, 250.0 + f * 5.0, 200.0), state=TrackState.TRACKED)
                
                evidence_cam1.register_observation(
                    track_id=202,
                    frame_id=10 + f,
                    timestamp_ms=float((10 + f) * 33.3),
                    crop_quality=1.0,
                    similarity=sim_t_full,
                    margin=sim_t_full - sim_b_full,
                    is_match=is_t_match,
                    box=track_target.box,
                )
                evidence_cam1.register_observation(
                    track_id=303,
                    frame_id=10 + f,
                    timestamp_ms=float((10 + f) * 33.3),
                    crop_quality=1.0,
                    similarity=sim_b_full,
                    margin=sim_t_full - sim_b_full,
                    is_match=is_b_match,
                    box=track_bystander.box,
                )
                
                decision = evidence_cam1.evaluate_all_candidates(
                    candidate_evaluations=[
                        (track_target, sim_t_full, is_t_match, 1.0),
                        (track_bystander, sim_b_full, is_b_match, 1.0),
                    ],
                    target_identity_id=target_identity_id,
                    current_tracked_id=None,
                    is_reacquisition=True,
                )
                
                if decision.is_confirmed:
                    if decision.best_track_id == 202:
                        target_reacquired = True
                        frames_to_reacquire.append(f + 1)
                        total_correct_reacquisitions += 1
                        break
                    elif decision.best_track_id == 303:
                        total_false_adoptions += 1
                        break
                        
        print(f"Results across {n_trials} multi-camera trials:")
        print(f"  Correct Reacquisitions: {total_correct_reacquisitions}/{n_trials} ({total_correct_reacquisitions/n_trials*100:.1f}%)")
        print(f"  False Bystander Adoptions: {total_false_adoptions} (0 allowed)")
        print(f"  Correct LOST Holds: {total_correct_lost_holds}/{n_trials*10} ({total_correct_lost_holds/(n_trials*10)*100:.1f}%)")
        if frames_to_reacquire:
            print(f"  Average Frames to Reacquire: {np.mean(frames_to_reacquire):.1f} frames (Min={min(frames_to_reacquire)}, Max={max(frames_to_reacquire)})")

if __name__ == "__main__":
    run_fast_end_to_end_multicam_simulation()
