"""Comprehensive Person Re-Identification (ReID) Benchmark and Evaluation Suite."""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.core.types import Embedding
from src.identity.manager import IdentityManager
from src.identity.evidence import EvidenceEngine
from src.reid.extractor import PyTorchReIDExtractor
from src.reid.quality import CropQualityEvaluator


def load_scratch_track_crops(scratch_dir: str) -> Dict[str, List[np.ndarray]]:
    """Loads all test crops from scratch directory, grouping them by track ID."""
    crop_paths = sorted(glob.glob(os.path.join(scratch_dir, "reid_cand_*.png")))
    tracks: Dict[str, List[np.ndarray]] = {}
    for p in crop_paths:
        parts = os.path.basename(p).replace(".png", "").split("_")
        if "track" in parts:
            t = parts[parts.index("track") + 1]
            img = cv2.imread(p)
            if img is not None and img.size > 0:
                tracks.setdefault(t, []).append(img)
    return tracks


def run_benchmark():
    print("=" * 80)
    print("      ARGUS RE-IDENTIFICATION (ReID) EVALUATION & BENCHMARK SUITE")
    print("=" * 80)

    # 1. Initialize ReID extractor and Identity Manager
    print("Initializing Foundation DINOv2 / Multi-Crop ReID Extractor with Foreground Isolation...")
    extractor = PyTorchReIDExtractor(model_name="dinov2", device="cpu")
    quality_eval = CropQualityEvaluator(min_width=20, min_height=50, min_sharpness=20.0)
    identity_mgr = IdentityManager(
        reid_extractor=extractor,
        similarity_threshold=0.78,
        reacquisition_threshold=0.82,
        reference_threshold=0.75,
        upper_threshold=0.60,
        min_margin=0.08,
        max_reference_samples=4,
        max_gallery_size=6,
        reacquisition_min_frames=4,
        quality_evaluator=quality_eval,
    )

    scratch_dir = r"C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch"
    tracks = load_scratch_track_crops(scratch_dir)

    print(f"Loaded {len(tracks)} tracks from validation archive.")
    if len(tracks) < 2:
        print("Creating synthetic multi-view and hard-negative evaluation tracks...")
        t1_base = np.zeros((160, 60, 3), dtype=np.uint8)
        t1_base[:80, :] = [180, 50, 50]   # Blue jacket
        t1_base[80:, :] = [40, 40, 40]    # Dark pants

        t2_base = np.zeros((160, 60, 3), dtype=np.uint8)
        t2_base[:80, :] = [50, 180, 50]   # Green jacket
        t2_base[80:, :] = [200, 200, 200] # Light pants

        tracks = {
            "target_track": [cv2.convertScaleAbs(t1_base, alpha=1.0 + 0.05 * i, beta=i * 2) for i in range(10)],
            "impostor_track_1": [cv2.convertScaleAbs(t2_base, alpha=1.0 + 0.05 * i, beta=i * 2) for i in range(10)],
        }

    sorted_tracks = sorted(tracks.items(), key=lambda x: len(x[1]), reverse=True)
    target_track_id, target_crops = sorted_tracks[0]
    impostor_track_id, impostor_crops = sorted_tracks[1]

    print(f"Primary Target Track: {target_track_id} ({len(target_crops)} views)")
    print(f"Primary Impostor Track: {impostor_track_id} ({len(impostor_crops)} views)")

    # 2. Multi-View Enrollment
    print("\n--- PHASE 1: Multi-View Enrollment ---")
    identity_mgr.register_new_target(target_crops[0], identity_id="target_0", label="Target VIP")
    for c in target_crops[1:4]:
        identity_mgr.add_reference_sample(c, identity_id="target_0")

    ident = identity_mgr.get_identity("target_0")
    print(f"Trusted Reference Gallery Size: {len(ident.trusted_gallery)} views")
    print(f"Trusted Centroid Prototype Created: {ident.trusted_prototype is not None} (dim={ident.trusted_prototype.dim})")

    # 3. Genuine Target Views Evaluation
    target_eval_crops = target_crops[:15] if len(target_crops) >= 15 else target_crops
    intra_scores = []
    for c in target_eval_crops:
        ev = identity_mgr.evaluate_candidate_crop(c, identity_id="target_0")
        intra_scores.append(ev.candidate_score)

    # 4. Impostor Views Evaluation
    impostor_eval_crops = impostor_crops[:15] if len(impostor_crops) >= 15 else impostor_crops
    inter_scores = []
    for c in impostor_eval_crops:
        ev = identity_mgr.evaluate_candidate_crop(c, identity_id="target_0")
        inter_scores.append(ev.candidate_score)

    intra_mean = float(np.mean(intra_scores))
    intra_min = float(np.min(intra_scores))
    intra_max = float(np.max(intra_scores))

    inter_mean = float(np.mean(inter_scores))
    inter_min = float(np.min(inter_scores))
    inter_max = float(np.max(inter_scores))

    separation_gap = intra_mean - inter_mean

    # 5. Metrics calculation
    thresh = identity_mgr.similarity_threshold
    true_positives = sum(1 for s in intra_scores if s >= thresh)
    false_positives = sum(1 for s in inter_scores if s >= thresh)

    tpr = (true_positives / len(intra_scores)) * 100.0
    fpr = (false_positives / len(inter_scores)) * 100.0

    print("\n" + "=" * 80)
    print("                      BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    print(f"Intra-Person (Target Genuine) Similarity: Mean = {intra_mean:.4f} [Min={intra_min:.4f}, Max={intra_max:.4f}]")
    print(f"Inter-Person (Impostor/Bystander) Sim:   Mean = {inter_mean:.4f} [Min={inter_min:.4f}, Max={inter_max:.4f}]")
    print(f"Score Separation Margin (Delta):          {separation_gap:+.4f}")
    print("-" * 80)
    print(f"True Positive Rate (TPR / Target Recall):  {tpr:6.2f}% (Threshold >= {thresh:.2f})")
    print(f"False Match Rate (FMR / False Accept):    {fpr:6.2f}% (Target < 1.0%)")
    print("=" * 80)

    # 6. Failure Mode Resilience Audit
    print("\n--- PHASE 2: Failure Mode Resilience Audit ---")

    # Audit 1: Scooper Proximity Rejection
    cand_list = [
        ("target_person", target_crops[1] if len(target_crops) > 1 else target_crops[0]),
        ("scooper_bystander", impostor_crops[0]),
    ]
    best_cand, best_s, sec_s, margin = identity_mgr.find_best_candidate(cand_list, "target_0")
    print(f"1. Competing Candidates (Target vs Scooper): Winner = '{best_cand}' (Score={best_s:.3f}, Margin={margin:.3f} >= {identity_mgr.min_margin:.2f}) -> {'PASS' if best_cand == 'target_person' else 'FAIL'}")

    # Audit 2: Target Left Scene -> Bystander Rejected
    only_impostor = [("bystander_only", impostor_crops[0])]
    best_imp, imp_s, _, _ = identity_mgr.find_best_candidate(only_impostor, "target_0")
    print(f"2. Target Left Scene (Only Bystander Present): Winner = {best_imp} (Score={imp_s:.3f}) -> {'PASS (Rejected)' if best_imp is None else 'FAIL (Falsely Adopted)'}")

    # Audit 3: Sole-Bystander 20-Frame Reacquisition Trap Audit
    from src.core.types import BoundingBox, Track
    bystander_tr = Track(track_id=99, box=BoundingBox(10, 10, 50, 100))
    bystander_adopted = False
    for f_idx in range(1, 21):
        ev_imp = identity_mgr.evaluate_candidate_crop(impostor_crops[0], "target_0")
        identity_mgr.evidence_engine.register_observation(
            track_id=99,
            frame_id=f_idx,
            timestamp_ms=f_idx * 33.3,
            crop_quality=ev_imp.crop_quality_score,
            similarity=ev_imp.candidate_score,
            margin=0.0,
            is_match=ev_imp.is_match,
        )
        dec = identity_mgr.evidence_engine.evaluate_all_candidates(
            candidate_evaluations=[(bystander_tr, ev_imp.candidate_score, ev_imp.is_match, ev_imp.crop_quality_score)],
            target_identity_id="target_0",
            current_tracked_id=None,
            is_reacquisition=True,
        )
        if dec.is_confirmed:
            bystander_adopted = True
            break
    print(f"3. Sole-Bystander 20-Frame Reacquisition Immunity: Bystander Adopted = {bystander_adopted} -> {'PASS (Immune)' if not bystander_adopted else 'FAIL (Adopted)'}")

    # Audit 4: Template Poisoning Protection
    accepted_poison = identity_mgr.verified_update(impostor_crops[0], "target_0")
    print(f"4. Template Poisoning Protection: Impostor Accepted into Gallery = {accepted_poison} -> {'PASS (Blocked)' if not accepted_poison else 'FAIL (Contaminated)'}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
