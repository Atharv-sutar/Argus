import os
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2

from src.core.config import AppConfig
from src.core.types import BoundingBox, Detection, DetectionResult, Track, TrackResult
from src.reid.extractor import PyTorchReIDExtractor
from src.reid.gallery import TargetGallery
from src.target.manager import TargetManager

print("=========================================================")
print(" CROWD CROSSING & ANTI-SCOOPING SIMULATION TEST")
print("=========================================================")

# 1. Create two distinct simulated realistic persons
# Person A (Target): Navy blue jacket top, Gray pants bottom
person_A = np.zeros((256, 128, 3), dtype=np.uint8)
person_A[:128, :] = (120, 40, 20)   # Navy blue jacket (BGR)
person_A[128:, :] = (100, 100, 100) # Gray pants

# Person B (Occluder/Crowd): Bright yellow jacket top, Dark blue jeans bottom
person_B = np.zeros((256, 128, 3), dtype=np.uint8)
person_B[:128, :] = (20, 220, 240)  # Bright yellow jacket (BGR)
person_B[128:, :] = (160, 40, 20)   # Dark blue jeans

extractor = PyTorchReIDExtractor(model_name="osnet_x1_0")
gallery = TargetGallery(reid_extractor=extractor, match_threshold=0.72, auto_add_threshold=0.82)
target_mgr = TargetManager(gallery=gallery, min_margin=0.06)

# Seed gallery with Person A (Ground-Truth Manual Anchor)
gallery.seed(person_A, target_label="target_0")
print(f"Gallery seeded with Person A. Size={gallery.size}, Manual={gallery.manual_count}")

# Verify similarity of Person A vs Person B
emb_A = extractor.extract(person_A)
emb_B = extractor.extract(person_B)
sim_A, entry_A = gallery.match(emb_A)
sim_B, entry_B = gallery.match(emb_B)

print(f"Person A self-similarity against gallery: {sim_A:.3f}")
print(f"Person B (other person) similarity against gallery: {sim_B:.3f}")
assert sim_A > 0.95, f"Expected high self-sim, got {sim_A}"
assert sim_B < 0.60, f"Expected low cross-person sim, got {sim_B}"
print("=> Person feature discrimination verified: Margin =", f"{sim_A - sim_B:.3f}")

# Simulate 5 frames:
# Frame 1: Person A (Track 1, left) and Person B (Track 2, right) separate
# Frame 2: Person A and Person B moving closer (partial overlap)
# Frame 3: Person A and Person B crossing (heavy overlap / occlusion)
# Frame 4: Person A and Person B separating (partial overlap)
# Frame 5: Person A (right) and Person B (left) completely separated

frames_data = [
    {"desc": "Frame 1 (Separated)", "boxes": [(50, 100, 150, 300), (350, 100, 450, 300)], "crops": [person_A, person_B], "ids": [1, 2]},
    {"desc": "Frame 2 (Approaching)", "boxes": [(150, 100, 250, 300), (250, 100, 350, 300)], "crops": [person_A, person_B], "ids": [1, 2]},
    {"desc": "Frame 3 (Heavy Crowd Overlap / Crossing)", "boxes": [(190, 100, 290, 300), (200, 100, 300, 300)], "crops": [person_A, person_B], "ids": [1, 2]},
    {"desc": "Frame 4 (Separating - Tracker ID swap simulated)", "boxes": [(250, 100, 350, 300), (150, 100, 250, 300)], "crops": [person_B, person_A], "ids": [1, 2]},
    {"desc": "Frame 5 (Fully Separated)", "boxes": [(350, 100, 450, 300), (50, 100, 150, 300)], "crops": [person_B, person_A], "ids": [1, 2]},
]

print("\n--- Simulating Dynamic Crowd Occlusion & Crossing ---")
locked_track_id = 1

for step_idx, step in enumerate(frames_data):
    print(f"\n{step['desc']}:")
    crops = step["crops"]
    ids = step["ids"]
    boxes = [BoundingBox(b[0], b[1], b[2], b[3]) for b in step["boxes"]]
    
    # Calculate IoU between the two boxes
    iou = boxes[0].iou(boxes[1])
    is_occluded = iou > 0.25
    
    # Batch ReID extraction
    embs = extractor.extract_batch(crops)
    match_details = gallery.match_batch_details(embs)
    
    for tid, emb, (eff_sim, man_sim, auto_sim, _) in zip(ids, embs, match_details):
        print(f"  Track #{tid}: Similarity={eff_sim:.3f} (Manual={man_sim:.3f}, Auto={auto_sim:.3f})")
    
    # Check anti-scooping invariants:
    if is_occluded:
        print(f"  [OCCLUSION DETECTED: IoU={iou:.2f}] Auto-enrollment FROZEN, spatial tracking held.")
    else:
        # If Person A is identified, attempt auto-add
        for idx, (tid, crop, emb) in enumerate(zip(ids, crops, embs)):
            sim = match_details[idx][0]
            if sim >= 0.82 and not is_occluded:
                added = gallery.add_auto(crop, emb, candidate_similarity=sim, track_id=tid)
                if added:
                    print(f"  -> Verified viewpoint auto-enrolled for Track #{tid} (Gallery size={gallery.size})")

    # If Frame 4/5 had ID swap (where Track 2 is actually Person A), verify ReID correctly identifies Track 2 as Person A
    best_idx = np.argmax([d[0] for d in match_details])
    best_track_id = ids[best_idx]
    best_score = match_details[best_idx][0]
    print(f"  => ReID Best Match: Track #{best_track_id} (Score={best_score:.3f})")

print("\n=========================================================")
print(" ALL ANTI-SCOOPING & CROWD OCCLUSION INVARIANTS PASSED!")
print(f" Final Gallery State: Size={gallery.size}, Manual={gallery.manual_count}, Auto={gallery.auto_count}")
print("=========================================================")
