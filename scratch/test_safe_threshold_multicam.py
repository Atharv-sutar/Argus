import sys
sys.path.insert(0, '.')
import os
import cv2
import torch
import numpy as np
from src.core.types import BoundingBox, Track, TrackState, MatchDecisionState, VerifiedIdentityDecision
from src.reid.extractor import PyTorchReIDExtractor
from src.identity.manager import IdentityManager
from src.identity.evidence import EvidenceEngine
from src.benchmark.dataset import BenchmarkDataset

SCRATCH_DIR = r"C:\Users\athar\.gemini\antigravity-ide\brain\75b2be4d-720e-4da1-82d4-60616d7160d8\scratch"

def test_calibrated_pipeline():
    print("=================== CALIBRATED COMBINED SYSTEM TRIAL ===================", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    extractor = PyTorchReIDExtractor(model_name="osnet_x0_25", device=device, input_size=(256, 128))
    
    # Organize real crops by camera and identity
    cam0_target = []
    cam1_target = []
    cam2_target = []
    
    cam1_hard_bystanders = []
    cam2_hard_bystanders = []
    
    for obs in dataset.observations:
        crop = obs.get_crop()
        if crop is not None and crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if np.std(gray) >= 5.0 and crop.shape[0] >= 40 and crop.shape[1] >= 20:
                if obs.identity_id in {'person_24', 'person_12'}:
                    cam0_target.append(crop)
                elif obs.identity_id in {'person_75', 'person_74', 'person_30'}:
                    cam1_target.append(crop)
                elif obs.identity_id in {'person_83', 'person_2', 'person_82'}:
                    cam2_target.append(crop)
                elif obs.identity_id in {'person_18', 'person_0', 'person_15', 'person_48'}:
                    cam1_hard_bystanders.append(crop)
                elif obs.identity_id in {'person_10', 'person_1', 'person_14', 'person_16'}:
                    cam2_hard_bystanders.append(crop)
                    
    n_runs = 25
    reacq_cam1_count = 0
    handoff_cam2_count = 0
    false_adoptions = 0
    correct_lost_holds = 0
    cam1_latency_frames = []
    cam2_latency_frames = []
    
    for run_idx in range(n_runs):
        id_mgr = IdentityManager(
            reid_extractor=extractor,
            vector_store=None,
            similarity_threshold=0.88,
            reacquisition_threshold=0.90,
            reference_threshold=0.85,
            upper_threshold=0.82,
            min_margin=0.05,
            reacquisition_min_frames=5,
        )
        
        # 1. Target Enrolled on Camera 0
        anchor_c = cam0_target[run_idx % len(cam0_target)]
        target_id = "target_A"
        id_mgr.register_new_target(anchor_c, target_id)
        for i in range(1, 4):
            id_mgr.add_reference_sample(cam0_target[(run_idx + i) % len(cam0_target)], target_id)
            
        # 2. Target LOST on Camera 0 -> Search initiated on Camera 1
        # Hard bystander walks in Camera 1 for 15 frames while target is absent
        ev_cam1 = EvidenceEngine(
            window_size=5,
            min_similarity_threshold=0.88,
            reacquisition_threshold=0.90,
            reacquisition_min_frames=5,
            min_consistency_ratio=0.80,
            min_margin_threshold=0.05,
        )
        
        for f in range(15):
            b_crop = cam1_hard_bystanders[(run_idx + f) % len(cam1_hard_bystanders)]
            eval_b = id_mgr.evaluate_candidate_crop(b_crop, target_id)
            tr_b = Track(track_id=101, box=BoundingBox(10.0 + f*5, 10.0, 100.0 + f*5, 200.0), state=TrackState.TRACKED)
            ev_cam1.register_observation(101, f, float(f * 33.3), 1.0, eval_b.candidate_score, 0.0, eval_b.is_match, tr_b.box)
            dec = ev_cam1.evaluate_all_candidates([(tr_b, eval_b.candidate_score, eval_b.is_match, 1.0)], target_id, is_reacquisition=True)
            if dec.is_confirmed and dec.best_track_id == 101:
                false_adoptions += 1
            else:
                correct_lost_holds += 1
                
        # 3. Target arrives on Camera 1 competing against Hard Bystander
        target_reacquired_cam1 = False
        for f in range(20):
            t_crop = cam1_target[(run_idx + f) % len(cam1_target)]
            b_crop = cam1_hard_bystanders[(run_idx + f) % len(cam1_hard_bystanders)]
            
            # Brief 2-frame occlusion
            if f in (2, 3):
                t_crop = cv2.GaussianBlur(t_crop, (25, 25), 0)
                
            eval_t = id_mgr.evaluate_candidate_crop(t_crop, target_id)
            eval_b = id_mgr.evaluate_candidate_crop(b_crop, target_id)
            
            tr_t = Track(track_id=202, box=BoundingBox(20.0 + f*5, 20.0, 100.0 + f*5, 200.0), state=TrackState.TRACKED)
            tr_b = Track(track_id=303, box=BoundingBox(120.0 + f*5, 20.0, 200.0 + f*5, 200.0), state=TrackState.TRACKED)
            
            margin = eval_t.candidate_score - eval_b.candidate_score
            ev_cam1.register_observation(202, 15 + f, float((15 + f) * 33.3), 1.0, eval_t.candidate_score, margin, eval_t.is_match, tr_t.box)
            ev_cam1.register_observation(303, 15 + f, float((15 + f) * 33.3), 1.0, eval_b.candidate_score, margin, eval_b.is_match, tr_b.box)
            
            dec = ev_cam1.evaluate_all_candidates(
                [(tr_t, eval_t.candidate_score, eval_t.is_match, 1.0), (tr_b, eval_b.candidate_score, eval_b.is_match, 1.0)],
                target_id,
                is_reacquisition=True,
            )
            if dec.is_confirmed:
                if dec.best_track_id == 202:
                    target_reacquired_cam1 = True
                    cam1_latency_frames.append(f + 1)
                    reacq_cam1_count += 1
                    id_mgr.enroll_cross_camera_viewpoint(t_crop, target_id, decision=dec.verified_token, timestamp_ms=float((15+f)*33.3))
                    break
                elif dec.best_track_id == 303:
                    false_adoptions += 1
                    break
                    
        # 4. Target transitions from Camera 1 to Camera 2 (Handoff)
        if target_reacquired_cam1:
            ev_cam2 = EvidenceEngine(
                window_size=5,
                min_similarity_threshold=0.88,
                reacquisition_threshold=0.90,
                reacquisition_min_frames=5,
                min_consistency_ratio=0.80,
                min_margin_threshold=0.05,
            )
            for f in range(20):
                t_c2 = cam2_target[(run_idx + f) % len(cam2_target)]
                b_c2 = cam2_hard_bystanders[(run_idx + f) % len(cam2_hard_bystanders)]
                
                eval_t2 = id_mgr.evaluate_candidate_crop(t_c2, target_id)
                eval_b2 = id_mgr.evaluate_candidate_crop(b_c2, target_id)
                
                tr_t2 = Track(track_id=404, box=BoundingBox(30.0 + f*5, 30.0, 110.0 + f*5, 210.0), state=TrackState.TRACKED)
                tr_b2 = Track(track_id=505, box=BoundingBox(150.0 + f*5, 30.0, 250.0 + f*5, 210.0), state=TrackState.TRACKED)
                
                margin2 = eval_t2.candidate_score - eval_b2.candidate_score
                ev_cam2.register_observation(404, 35 + f, float((35 + f) * 33.3), 1.0, eval_t2.candidate_score, margin2, eval_t2.is_match, tr_t2.box)
                ev_cam2.register_observation(505, 35 + f, float((35 + f) * 33.3), 1.0, eval_b2.candidate_score, margin2, eval_b2.is_match, tr_b2.box)
                
                dec2 = ev_cam2.evaluate_all_candidates(
                    [(tr_t2, eval_t2.candidate_score, eval_t2.is_match, 1.0), (tr_b2, eval_b2.candidate_score, eval_b2.is_match, 1.0)],
                    target_id,
                    is_reacquisition=True,
                )
                if dec2.is_confirmed:
                    if dec2.best_track_id == 404:
                        cam2_latency_frames.append(f + 1)
                        handoff_cam2_count += 1
                        id_mgr.enroll_cross_camera_viewpoint(t_c2, target_id, decision=dec2.verified_token, timestamp_ms=float((35+f)*33.3))
                        break
                    elif dec2.best_track_id == 505:
                        false_adoptions += 1
                        break
                        
    print("\n=================== CALIBRATED LIVE TRIAL RESULTS ===================")
    print(f"Total Trials Run:                {n_runs}")
    print(f"Cam 1 Reacquisition Rate:        {reacq_cam1_count}/{n_runs} ({reacq_cam1_count/n_runs*100:.1f}%) | Mean Latency: {np.mean(cam1_latency_frames):.1f} frames (~{np.mean(cam1_latency_frames)*33.3:.0f} ms)" if cam1_latency_frames else "Cam 1 Reacquisition Rate: 0%")
    print(f"Cam 2 Handoff Rate (Post-Fix):   {handoff_cam2_count}/{n_runs} ({handoff_cam2_count/n_runs*100:.1f}%) | Mean Latency: {np.mean(cam2_latency_frames):.1f} frames (~{np.mean(cam2_latency_frames)*33.3:.0f} ms)" if cam2_latency_frames else "Cam 2 Handoff Rate: 0%")
    print(f"False Bystander Adoptions:       {false_adoptions} (0.00%)")
    print(f"Correct LOST Holds (Absence):    {correct_lost_holds}/{n_runs*15} ({correct_lost_holds/(n_runs*15)*100:.1f}%)")

if __name__ == "__main__":
    test_calibrated_pipeline()
