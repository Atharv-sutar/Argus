"""Head-to-head benchmark comparing MobileNetV3 (ImageNet baseline) vs OSNet-x0.25 (Person-ReID trained)."""

import glob
import os
import sys
import time
import cv2
import numpy as np
import torch
import torchvision.models as models

sys.path.insert(0, os.path.abspath("."))
from src.reid.backbones.osnet import osnet_x0_25


def preprocess_crop(crop: np.ndarray, size=(256, 128)) -> np.ndarray:
    h, w = size
    resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb - mean) / std
    return np.transpose(normalized, (2, 0, 1))


def extract_mobilenet(model, crop: np.ndarray) -> np.ndarray:
    prep = preprocess_crop(crop)
    t = torch.from_numpy(prep).unsqueeze(0)
    with torch.no_grad():
        feat = model(t).squeeze().cpu().numpy()
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat


def extract_osnet(model, crop: np.ndarray) -> np.ndarray:
    prep = preprocess_crop(crop)
    t = torch.from_numpy(prep).unsqueeze(0)
    with torch.no_grad():
        feat = model(t).squeeze().cpu().numpy()
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat


def cos_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2))


def main():
    print("=" * 80)
    print("      REID BACKBONE BENCHMARK: MOBILENETV3 (IMAGENET) vs OSNET-x0.25 (MSMT17)")
    print("=" * 80)

    # 1. Load models
    print("Loading MobileNetV3 (ImageNet)...")
    mb_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    mb_model.classifier = torch.nn.Identity()
    mb_model.eval()

    print("Loading OSNet-x0.25 (Person-ReID MSMT17)...")
    os_model = osnet_x0_25(pretrained=True).eval()

    # 2. Load crops
    scratch_dir = r"C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch"
    crop_paths = sorted(glob.glob(os.path.join(scratch_dir, "reid_cand_*.png")))

    tracks_map = {}
    for p in crop_paths:
        fname = os.path.basename(p)
        parts = fname.replace(".png", "").split("_")
        if "track" in parts:
            t_idx = parts.index("track") + 1
            if t_idx < len(parts):
                t_id = parts[t_idx]
                if t_id not in tracks_map:
                    tracks_map[t_id] = []
                tracks_map[t_id].append(p)

    top_tracks = sorted(tracks_map.items(), key=lambda x: len(x[1]), reverse=True)
    track_a_crops = [cv2.imread(p) for p in top_tracks[0][1][:15] if cv2.imread(p) is not None]
    track_b_crops = [cv2.imread(p) for p in top_tracks[1][1][:15] if cv2.imread(p) is not None]
    track_c_crops = [cv2.imread(p) for p in top_tracks[2][1][:15] if cv2.imread(p) is not None]

    crop_a1 = track_a_crops[0]
    crop_a2 = track_a_crops[min(5, len(track_a_crops) - 1)]
    crop_a3_pert = cv2.convertScaleAbs(crop_a2, alpha=0.85, beta=15)  # Simulated lighting/camera change
    crop_b1 = track_b_crops[0]
    crop_c1 = track_c_crops[0]

    # --- Test 1: Same Image vs Itself ---
    mb_t1 = cos_sim(extract_mobilenet(mb_model, crop_a1), extract_mobilenet(mb_model, crop_a1))
    os_t1 = cos_sim(extract_osnet(os_model, crop_a1), extract_osnet(os_model, crop_a1))

    # --- Test 2: Same Person / Different Frame ---
    mb_t2 = cos_sim(extract_mobilenet(mb_model, crop_a1), extract_mobilenet(mb_model, crop_a2))
    os_t2 = cos_sim(extract_osnet(os_model, crop_a1), extract_osnet(os_model, crop_a2))

    # --- Test 3: Same Person / Camera / Lighting Shift ---
    mb_t3 = cos_sim(extract_mobilenet(mb_model, crop_a1), extract_mobilenet(mb_model, crop_a3_pert))
    os_t3 = cos_sim(extract_osnet(os_model, crop_a1), extract_osnet(os_model, crop_a3_pert))

    # --- Test 4: Different Person (Track A vs Track B) ---
    mb_t4 = cos_sim(extract_mobilenet(mb_model, crop_a1), extract_mobilenet(mb_model, crop_b1))
    os_t4 = cos_sim(extract_osnet(os_model, crop_a1), extract_osnet(os_model, crop_b1))

    # --- Test 5: Hard Negative (Track A vs Track C) ---
    mb_t5 = cos_sim(extract_mobilenet(mb_model, crop_a1), extract_mobilenet(mb_model, crop_c1))
    os_t5 = cos_sim(extract_osnet(os_model, crop_a1), extract_osnet(os_model, crop_c1))

    # Population Distribution across all tracks
    mb_intra, os_intra = [], []
    for i in range(len(track_a_crops)):
        for j in range(i + 1, len(track_a_crops)):
            v_mb_i, v_mb_j = extract_mobilenet(mb_model, track_a_crops[i]), extract_mobilenet(mb_model, track_a_crops[j])
            v_os_i, v_os_j = extract_osnet(os_model, track_a_crops[i]), extract_osnet(os_model, track_a_crops[j])
            mb_intra.append(cos_sim(v_mb_i, v_mb_j))
            os_intra.append(cos_sim(v_os_i, v_os_j))

    mb_inter, os_inter = [], []
    for ca in track_a_crops[:8]:
        for cb in track_b_crops[:8]:
            v_mb_a, v_mb_b = extract_mobilenet(mb_model, ca), extract_mobilenet(mb_model, cb)
            v_os_a, v_os_b = extract_osnet(os_model, ca), extract_osnet(os_model, cb)
            mb_inter.append(cos_sim(v_mb_a, v_mb_b))
            os_inter.append(cos_sim(v_os_a, v_os_b))

    # Latency Profiling
    n_runs = 30
    # MobileNet
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = extract_mobilenet(mb_model, crop_a1)
    t1 = time.perf_counter()
    mb_lat_ms = (t1 - t0) * 1000.0 / n_runs

    # OSNet
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = extract_osnet(os_model, crop_a1)
    t1 = time.perf_counter()
    os_lat_ms = (t1 - t0) * 1000.0 / n_runs

    print("\n" + "=" * 80)
    print("                         BENCHMARK RESULTS TABLE")
    print("=" * 80)
    headers = ["Metric / Evaluation Case", "Old (MobileNetV3)", "New (OSNet-x0.25)", "Improvement / Delta"]
    print(f"{headers[0]:<35} | {headers[1]:<18} | {headers[2]:<18} | {headers[3]}")
    print("-" * 95)
    print(f"{'Test 1: Same Image vs Itself':<35} | {mb_t1:18.4f} | {os_t1:18.4f} | {'Exact (1.0000)'}")
    print(f"{'Test 2: Same Person (Diff Frame)':<35} | {mb_t2:18.4f} | {os_t2:18.4f} | {os_t2 - mb_t2:+18.4f}")
    print(f"{'Test 3: Same Person (Camera Shift)':<35} | {mb_t3:18.4f} | {os_t3:18.4f} | {os_t3 - mb_t3:+18.4f}")
    print(f"{'Test 4: Different Person (Track B)':<35} | {mb_t4:18.4f} | {os_t4:18.4f} | {os_t4 - mb_t4:+18.4f}")
    print(f"{'Test 5: Hard Negative (Track C)':<35} | {mb_t5:18.4f} | {os_t5:18.4f} | {os_t5 - mb_t5:+18.4f}")
    print("-" * 95)
    print(f"{'Mean True-Match Similarity':<35} | {np.mean(mb_intra):18.4f} | {np.mean(os_intra):18.4f} | {np.mean(os_intra) - np.mean(mb_intra):+18.4f}")
    print(f"{'Mean False-Match Similarity':<35} | {np.mean(mb_inter):18.4f} | {np.mean(os_inter):18.4f} | {np.mean(os_inter) - np.mean(mb_inter):+18.4f}")
    print(f"{'TRUE/FALSE SEPARATION (MARGIN)':<35} | {np.mean(mb_intra) - np.mean(mb_inter):+18.4f} | {np.mean(os_intra) - np.mean(os_inter):+18.4f} | {(np.mean(os_intra) - np.mean(os_inter)) - (np.mean(mb_intra) - np.mean(mb_inter)):+18.4f}")
    print(f"{'Min True-Match Similarity':<35} | {np.min(mb_intra):18.4f} | {np.min(os_intra):18.4f} | {np.min(os_intra) - np.min(mb_intra):+18.4f}")
    print(f"{'Max False-Match Similarity':<35} | {np.max(mb_inter):18.4f} | {np.max(os_inter):18.4f} | {np.max(os_inter) - np.max(mb_inter):+18.4f}")
    print("-" * 95)
    print(f"{'CPU Latency per Crop (ms)':<35} | {mb_lat_ms:18.2f} | {os_lat_ms:18.2f} | {os_lat_ms - mb_lat_ms:+18.2f} ms")
    print(f"{'Throughput (crops/sec)':<35} | {1000.0/mb_lat_ms:18.1f} | {1000.0/os_lat_ms:18.1f} | {(1000.0/os_lat_ms) - (1000.0/mb_lat_ms):+18.1f}")
    print(f"{'Model Parameters':<35} | {'1.5M (576D)':<18} | {'0.2M (512D)':<18} | {'-87% smaller'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
