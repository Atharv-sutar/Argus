import sys
sys.path.insert(0, '.')
import os
import cv2
from src.benchmark.dataset import BenchmarkDataset

SCRATCH_DIR = r"C:\Users\athar\.gemini\antigravity-ide\brain\75b2be4d-720e-4da1-82d4-60616d7160d8\scratch"

def map_actors():
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    actors = {}
    for obs in dataset.observations:
        crop = obs.get_crop()
        if crop is not None and crop.size > 0 and obs.identity_id not in actors:
            actors[obs.identity_id] = crop
            
    print(f"Total track IDs in dataset: {len(actors)}")
    for track_id, crop in sorted(actors.items()):
        out_path = os.path.join(SCRATCH_DIR, f"actor_{track_id}.png")
        cv2.imwrite(out_path, crop)
        print(f"Saved: {track_id} (Shape: {crop.shape})")

if __name__ == "__main__":
    map_actors()
