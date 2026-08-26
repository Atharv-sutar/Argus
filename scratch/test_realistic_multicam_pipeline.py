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

def run_fast_realistic_multicam():
    print("=================== FAST REALISTIC MULTI-CAMERA BENCHMARK ===================", flush=True)
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
                
    # Crops
    target_cam0 = [o.get_crop() for o in obs_by_track.get("person_24", []) if o.get_crop() is not None]
    target_cam1 = [o.get_crop() for o in obs_by_track.get("person_75", []) if o.get_crop() is not None]
    target_cam2 = [o.get_crop() for o in obs_by_track.get("person_83", []) if o.get_crop() is not None]
    target_standing = [o.get_crop() for o in obs_by_track.get("person_2", []) if o.get_crop() is not None]
    
    bystander_cam1 = [o.get_crop() for o in obs_by_track.get("person_18", []) if o.get_crop() is not None]
    bystander_cam2 = [o.get_crop() for o in obs_by_track.get("person_15", []) if o.get_crop() is not None]
    
    print(f"Target: Cam0={len(target_cam0)}, Cam1={len(target_cam1)}, Cam2={len(target_cam2)}, Standing={len(target_standing)}")
    print(f"Bystander: Cam1={len(bystander_cam1)}, Cam2={len(bystander_cam2)}")
    
    candidates = [
        ("Current Config (reacq=0.95, sim=0.90, ref=0.92, upper=0.88, W=4, ratio=0.75)", 0.95, 0.90, 0.92, 0.88, 4, 0.75),
        ("Calibrated Strict (reacq=0.85, sim=0.80, ref=0.78, upper=0.75, W=4, ratio=0.75)", 0.85, 0.80, 0.78, 0.75, 4, 0.75),
        ("Calibrated Optimal (reacq=0.78, sim=0.75, ref=0.72, upper=0.70, W=4, ratio=0.75)", 0.78, 0.75, 0.72, 0.70, 4, 0.75),
        ("Calibrated 5-Frame (reacq=0.78, sim=0.75, ref=0.72, upper=0.70, W=5, ratio=0.80)", 0.78, 0.75, 0.72, 0.70, 5, 0.80),
        ("Calibrated 6-Frame (reacq=0.78, sim=0.75, ref=0.72, upper=0.70, W=6, ratio=0.67)", 0.78, 0.75, 0.72, 0.70, 6, 0.67),
    ]
    
    for name, reacq_th, sim_th, ref_th, upper_th, win_size, ratio in candidates:
        print(f"\n=================== Testing: {name} ===================", flush=True)
        n_trials = 20
        reacq_cam1_success = 0
        reacq_cam2_success = 0
        false_adoptions = 0
        lost_holds = 0
        frames_cam1 = []
        frames_cam2 = []
        
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
            
            # Step 1: Register Target on Cam 0 + 1 reference sample
            anchor_crop = target_standing[trial % len(target_standing)]
            target_id = "target_A"
            id_mgr.register_new_target(crop=anchor_crop, identity_id=target_id)
            id_mgr.add_reference_sample(crop=target_cam0[trial % len(target_cam0)], identity_id=target_id)
            
            # Step 2: Target Lost. Bystanders cross Cam 1
            ev_cam1 = EvidenceEngine(
                window_size=win_size,
                min_similarity_threshold=sim_th,
                reacquisition_threshold=reacq_th,
                reacquisition_min_frames=win_size,
                min_consistency_ratio=ratio,
                min_margin_threshold=0.05,
            )
            
            for f in range(10):
                b_crop = bystander_cam1[(trial + f) % len(bystander_cam1)]
                eval_b = id_mgr.evaluate_candidate_crop(b_crop, target_id)
                tr_b = Track(track_id=101, box=BoundingBox(10.0 + f*2, 10.0, 100.0 + f*2, 200.0), state=TrackState.TRACKED)
                ev_cam1.register_observation(101, f, float(f * 33.3), 1.0, eval_b.candidate_score, 0.0, eval_b.is_match, tr_b.box)
                dec = ev_cam1.evaluate_all_candidates([(tr_b, eval_b.candidate_score, eval_b.is_match, 1.0)], target_id, is_reacquisition=True)
                if dec.is_confirmed and dec.best_track_id == 101:
                    false_adoptions += 1
                else:
                    lost_holds += 1
                    
            # Step 3: Target arrives on Cam 1 alongside Bystander
            reacquired_cam1 = False
            for f in range(15):
                t_crop = target_cam1[(trial + f) % len(target_cam1)]
                eval_t = id_mgr.evaluate_candidate_crop(t_crop, target_id)
                
                b_crop = bystander_cam1[(trial + f) % len(bystander_cam1)]
                eval_b = id_mgr.evaluate_candidate_crop(b_crop, target_id)
                
                tr_t = Track(track_id=202, box=BoundingBox(20.0 + f*5, 20.0, 100.0 + f*5, 200.0), state=TrackState.TRACKED)
                tr_b = Track(track_id=303, box=BoundingBox(150.0 + f*5, 20.0, 250.0 + f*5, 200.0), state=TrackState.TRACKED)
                
                margin = eval_t.candidate_score - eval_b.candidate_score
                ev_cam1.register_observation(202, 10 + f, float((10 + f) * 33.3), 1.0, eval_t.candidate_score, margin, eval_t.is_match, tr_t.box)
                ev_cam1.register_observation(303, 10 + f, float((10 + f) * 33.3), 1.0, eval_b.candidate_score, margin, eval_b.is_match, tr_b.box)
                
                dec = ev_cam1.evaluate_all_candidates(
                    [(tr_t, eval_t.candidate_score, eval_t.is_match, 1.0), (tr_b, eval_b.candidate_score, eval_b.is_match, 1.0)],
                    target_id,
                    is_reacquisition=True,
                )
                if dec.is_confirmed:
                    if dec.best_track_id == 202:
                        reacquired_cam1 = True
                        frames_cam1.append(f + 1)
                        reacq_cam1_success += 1
                        # Adaptive gallery update on successful confirmation
                        id_mgr.verified_update(t_crop, target_id, float((10 + f) * 33.3))
                        break
                    elif dec.best_track_id == 303:
                        false_adoptions += 1
                        break
                        
            # Step 4: Target Handoff from Cam 1 to Cam 2
            if reacquired_cam1:
                ev_cam2 = EvidenceEngine(
                    window_size=win_size,
                    min_similarity_threshold=sim_th,
                    reacquisition_threshold=reacq_th,
                    reacquisition_min_frames=win_size,
                    min_consistency_ratio=ratio,
                    min_margin_threshold=0.05,
                )
                for f in range(15):
                    t_c2 = target_cam2[(trial + f) % len(target_cam2)]
                    eval_t2 = id_mgr.evaluate_candidate_crop(t_c2, target_id)
                    
                    b_c2 = bystander_cam2[(trial + f) % len(bystander_cam2)]
                    eval_b2 = id_mgr.evaluate_candidate_crop(b_c2, target_id)
                    
                    tr_t2 = Track(track_id=404, box=BoundingBox(30.0 + f*5, 30.0, 110.0 + f*5, 210.0), state=TrackState.TRACKED)
                    tr_b2 = Track(track_id=505, box=BoundingBox(180.0 + f*5, 30.0, 280.0 + f*5, 210.0), state=TrackState.TRACKED)
                    
                    margin2 = eval_t2.candidate_score - eval_b2.candidate_score
                    ev_cam2.register_observation(404, 30 + f, float((30 + f) * 33.3), 1.0, eval_t2.candidate_score, margin2, eval_t2.is_match, tr_t2.box)
                    ev_cam2.register_observation(505, 30 + f, float((30 + f) * 33.3), 1.0, eval_b2.candidate_score, margin2, eval_b2.is_match, tr_b2.box)
                    
                    dec2 = ev_cam2.evaluate_all_candidates(
                        [(tr_t2, eval_t2.candidate_score, eval_t2.is_match, 1.0), (tr_b2, eval_b2.candidate_score, eval_b2.is_match, 1.0)],
                        target_id,
                        is_reacquisition=True,
                    )
                    if dec2.is_confirmed:
                        if dec2.best_track_id == 404:
                            frames_cam2.append(f + 1)
                            reacq_cam2_success += 1
                            break
                        elif dec2.best_track_id == 505:
                            false_adoptions += 1
                            break
                            
        print(f"Results across {n_trials} multi-camera trials:")
        print(f"  Cam 1 Reacquisition Rate: {reacq_cam1_success}/{n_trials} ({reacq_cam1_success/n_trials*100:.1f}%) | Latency: {np.mean(frames_cam1):.1f} frames (~{np.mean(frames_cam1)*33.3:.0f} ms)" if frames_cam1 else f"  Cam 1 Reacq Rate: 0%")
        print(f"  Cam 2 Handoff Rate:       {reacq_cam2_success}/{n_trials} ({reacq_cam2_success/n_trials*100:.1f}%) | Latency: {np.mean(frames_cam2):.1f} frames (~{np.mean(frames_cam2)*33.3:.0f} ms)" if frames_cam2 else f"  Cam 2 Handoff Rate: 0%")
        print(f"  False Bystander Adoptions: {false_adoptions} (0 allowed)")
        print(f"  Correct LOST Holds:       {lost_holds}/{n_trials*10} ({lost_holds/(n_trials*10)*100:.1f}%)")

if __name__ == "__main__":
    run_fast_realistic_multicam()
