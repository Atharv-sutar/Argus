import sys
sys.path.insert(0, '.')
import os
import cv2
from src.benchmark.dataset import BenchmarkDataset

SCRATCH_DIR = r"C:\Users\athar\.gemini\antigravity-ide\brain\75b2be4d-720e-4da1-82d4-60616d7160d8\scratch"

def inspect_pairs():
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    
    obs_by_id = {}
    for obs in dataset.observations:
        obs_by_id.setdefault(obs.identity_id, []).append(obs)
        
    p75 = obs_by_id.get("person_75", [])
    p83 = obs_by_id.get("person_83", [])
    p2 = obs_by_id.get("person_2", [])
    p24 = obs_by_id.get("person_24", [])
    
    print(f"person_75: {len(p75)} crops. First crop: frame {p75[0].frame_id}, cam {p75[0].camera_id}")
    print(f"person_83: {len(p83)} crops. First crop: frame {p83[0].frame_id}, cam {p83[0].camera_id}")
    print(f"person_2:  {len(p2)} crops. First crop: frame {p2[0].frame_id}, cam {p2[0].camera_id}")
    print(f"person_24: {len(p24)} crops. First crop: frame {p24[0].frame_id}, cam {p24[0].camera_id}")
    
    # Save crops to scratch for visual comparison
    if p75 and p83:
        cv2.imwrite(os.path.join(SCRATCH_DIR, "inspect_p75_crop.png"), p75[len(p75)//2].get_crop())
        cv2.imwrite(os.path.join(SCRATCH_DIR, "inspect_p83_crop.png"), p83[len(p83)//2].get_crop())
        print("Saved inspect_p75_crop.png and inspect_p83_crop.png")
        
    if p2 and p24:
        cv2.imwrite(os.path.join(SCRATCH_DIR, "inspect_p2_crop.png"), p2[len(p2)//2].get_crop())
        cv2.imwrite(os.path.join(SCRATCH_DIR, "inspect_p24_crop.png"), p24[len(p24)//2].get_crop())
        print("Saved inspect_p2_crop.png and inspect_p24_crop.png")

if __name__ == "__main__":
    inspect_pairs()
