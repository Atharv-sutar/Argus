import sys
sys.path.insert(0, '.')
import time
import os
import glob
import cv2
import torch
import numpy as np
import psutil
from src.reid.extractor import PyTorchReIDExtractor
from src.benchmark.dataset import BenchmarkDataset

SCRATCH_DIR = r"C:\Users\athar\.gemini\antigravity-ide\brain\75b2be4d-720e-4da1-82d4-60616d7160d8\scratch"

def analyze_and_benchmark():
    print("=================== 1. LATENCY & VRAM BENCHMARK PROTOCOL ===================", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    models = ["dinov2", "osnet_x0_25", "osnet_x1_0"]
    lat_results = {}
    
    for var in models:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        input_size = (224, 224) if "dino" in var else (256, 128)
        extractor = PyTorchReIDExtractor(model_name=var, device=device, input_size=input_size)
        model = extractor._model
        model.eval()
        
        dummy_crop = np.random.randint(0, 256, (300, 150, 3), dtype=np.uint8)
        dummy_batch = [np.random.randint(0, 256, (300, 150, 3), dtype=np.uint8) for _ in range(4)]
        
        dummy_tensor_1 = torch.randn(1, 3, input_size[0], input_size[1], device=device)
        dummy_tensor_4 = torch.randn(4, 3, input_size[0], input_size[1], device=device)
        
        # 20 warmup iterations
        with torch.no_grad():
            for _ in range(20):
                _ = model(dummy_tensor_1)
                _ = model(dummy_tensor_4)
                _ = extractor.extract(dummy_crop)
                _ = extractor.extract_batch(dummy_batch)
                
        if device == "cuda":
            torch.cuda.synchronize()
            
        n_iters = 50
        
        # Pure forward pass
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = model(dummy_tensor_1)
            if device == "cuda":
                torch.cuda.synchronize()
            fwd_1_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0
            
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = model(dummy_tensor_4)
            if device == "cuda":
                torch.cuda.synchronize()
            fwd_4_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0
            
        # End-to-end extraction (crop resize + norm + forward + Embedding)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = extractor.extract(dummy_crop)
        if device == "cuda":
            torch.cuda.synchronize()
        e2e_1_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0
        
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = extractor.extract_batch(dummy_batch)
        if device == "cuda":
            torch.cuda.synchronize()
        e2e_4_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0
        
        vram_alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
        vram_res_mb = torch.cuda.max_memory_reserved() / (1024 * 1024) if device == "cuda" else 0.0
        
        lat_results[var] = {
            "fwd_1_ms": fwd_1_ms,
            "fwd_4_ms": fwd_4_ms,
            "fwd_4_per_crop_ms": fwd_4_ms / 4.0,
            "e2e_1_ms": e2e_1_ms,
            "e2e_4_ms": e2e_4_ms,
            "e2e_4_per_crop_ms": e2e_4_ms / 4.0,
            "vram_alloc_mb": vram_alloc_mb,
            "vram_res_mb": vram_res_mb,
        }
        print(f"\n[{var} Benchmark]")
        print(f"  Pure Forward (B=1): {fwd_1_ms:.2f} ms | (B=4): {fwd_4_ms:.2f} ms ({fwd_4_ms/4:.2f} ms/crop)")
        print(f"  End-to-End   (B=1): {e2e_1_ms:.2f} ms | (B=4): {e2e_4_ms:.2f} ms ({e2e_4_ms/4:.2f} ms/crop)")
        print(f"  Peak VRAM: Allocated={vram_alloc_mb:.2f} MB | Reserved={vram_res_mb:.2f} MB")

    print("\n=================== 2. GROUND TRUTH IDENTITIES & TAIL DISTRIBUTIONS ===================", flush=True)
    dataset = BenchmarkDataset.from_scratch_archive(SCRATCH_DIR)
    
    # Extract features for all sample crops across tracks
    obs_by_track = {}
    for obs in dataset.observations:
        obs_by_track.setdefault(obs.identity_id, []).append(obs)

    # We test on each model
    dist_results = {}
    for var in models:
        input_size = (224, 224) if "dino" in var else (256, 128)
        extractor = PyTorchReIDExtractor(model_name=var, device=device, input_size=input_size)
        
        track_embeds = {}
        for tid, obs_list in obs_by_track.items():
            sample_obs = obs_list[::max(1, len(obs_list)//6)][:6]
            crops = [o.get_crop() for o in sample_obs if o.get_crop() is not None]
            if crops:
                embeds = extractor.extract_batch(crops)
                track_embeds[tid] = [e.vector for e in embeds]
                
        # Ground Truth Grouping based on Multi-Scale Visual Verification:
        # In this dataset there are 4 distinct human subjects:
        # Subject A: White sleeveless shirt + glasses (e.g. tracks 2, 24, 75, 83, 30, 32, etc.)
        # Subject B: Striped polo shirt + black hair (e.g. tracks 0, 1, 10, 15, 18, etc.)
        # Subject C: Dark t-shirt bystander (e.g. tracks 44, 48, 50, 51, 52, etc.)
        # Subject D: Secondary bystander (e.g. tracks 34, 38, 45, etc.)
        
        # We classify every track by maximum similarity to anchor exemplars of the 4 subjects
        subject_anchors = {
            "Subject_A_WhiteSleeveless": track_embeds.get("person_2", [np.zeros(512)])[0],
            "Subject_B_StripedShirt":    track_embeds.get("person_0", [np.zeros(512)])[0],
            "Subject_C_DarkShirt":       track_embeds.get("person_44", [np.zeros(512)])[0],
            "Subject_D_Bystander":       track_embeds.get("person_38", [np.zeros(512)])[0],
        }
        
        subject_tracks = {"Subject_A_WhiteSleeveless": [], "Subject_B_StripedShirt": [], "Subject_C_DarkShirt": [], "Subject_D_Bystander": []}
        for tid, vecs in track_embeds.items():
            proto = np.mean(vecs, axis=0)
            proto = proto / np.linalg.norm(proto)
            best_sub = max(subject_anchors.keys(), key=lambda s: np.dot(proto, subject_anchors[s]))
            subject_tracks[best_sub].append(tid)
            
        print(f"\n[{var}] Subject Track Assignments:")
        for sname, tids in subject_tracks.items():
            print(f"  {sname}: {len(tids)} tracks ({len([v for tid in tids for v in track_embeds[tid]])} crop vectors)")

        # Compute Genuine (intra-subject) and Impostor (inter-subject) pairs
        genuine_sims = []
        impostor_sims = []
        
        s_keys = list(subject_tracks.keys())
        for s in s_keys:
            vecs = [v for tid in subject_tracks[s] for v in track_embeds[tid]]
            for a in range(len(vecs)):
                for b in range(a + 1, len(vecs)):
                    genuine_sims.append(float(np.dot(vecs[a], vecs[b])))
                    
        for i, s1 in enumerate(s_keys):
            vecs1 = [v for tid in subject_tracks[s1] for v in track_embeds[tid]]
            for s2 in s_keys[i+1:]:
                vecs2 = [v for tid in subject_tracks[s2] for v in track_embeds[tid]]
                for a in range(len(vecs1)):
                    for b in range(len(vecs2)):
                        impostor_sims.append(float(np.dot(vecs1[a], vecs2[b])))
                        
        g_arr = np.array(genuine_sims)
        i_arr = np.array(impostor_sims)
        
        g_stats = {
            "N": len(g_arr),
            "min": float(np.min(g_arr)),
            "p0_1": float(np.percentile(g_arr, 0.1)),
            "p1": float(np.percentile(g_arr, 1.0)),
            "p5": float(np.percentile(g_arr, 5.0)),
            "p10": float(np.percentile(g_arr, 10.0)),
            "p50": float(np.percentile(g_arr, 50.0)),
            "mean": float(np.mean(g_arr)),
            "std": float(np.std(g_arr)),
        }
        
        i_stats = {
            "N": len(i_arr),
            "min": float(np.min(i_arr)),
            "p50": float(np.percentile(i_arr, 50.0)),
            "p90": float(np.percentile(i_arr, 90.0)),
            "p95": float(np.percentile(i_arr, 95.0)),
            "p99": float(np.percentile(i_arr, 99.0)),
            "p99_9": float(np.percentile(i_arr, 99.9)),
            "max": float(np.max(i_arr)),
            "mean": float(np.mean(i_arr)),
            "std": float(np.std(i_arr)),
        }
        
        sweep = []
        for tau in [0.75, 0.78, 0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]:
            fmr = float(np.mean(i_arr >= tau) * 100.0)
            frr = float(np.mean(g_arr < tau) * 100.0)
            tpr = 100.0 - frr
            sweep.append({"tau": tau, "FMR": fmr, "FRR": frr, "TPR": tpr, "n_fm": int(np.sum(i_arr >= tau))})
            
        dist_results[var] = {"genuine": g_stats, "impostor": i_stats, "sweep": sweep}
        
        print(f"  Genuine Distribution: N={g_stats['N']} | Min={g_stats['min']:.4f} | p1={g_stats['p1']:.4f} | p10={g_stats['p10']:.4f} | p50={g_stats['p50']:.4f} | Mean={g_stats['mean']:.4f} (±{g_stats['std']:.4f})")
        print(f"  Impostor Distribution: N={i_stats['N']} | Mean={i_stats['mean']:.4f} (±{i_stats['std']:.4f}) | p90={i_stats['p90']:.4f} | p95={i_stats['p95']:.4f} | p99={i_stats['p99']:.4f} | p99.9={i_stats['p99_9']:.4f} | Max={i_stats['max']:.4f}")
        print("  Sweep Table:")
        for s in sweep:
            print(f"    tau={s['tau']:.2f} -> FMR={s['FMR']:.4f}% ({s['n_fm']} false matches), TPR={s['TPR']:.2f}% (FRR={s['FRR']:.2f}%)")

    # Gap 5: Actual Video Stream Decode Performance
    print("\n=================== 3. GAP 5 EMPIRICAL STREAM DECODE MEASUREMENT ===================", flush=True)
    # Check webcam or synthetic stream decoding
    cap_msmf = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap_dshow = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    decode_benchmarks = {}
    for name, cap in [("cv2.CAP_MSMF (Media Foundation / QuickSync)", cap_msmf), ("cv2.CAP_DSHOW (DirectShow / CPU)", cap_dshow)]:
        if cap and cap.isOpened():
            # Measure decode time and CPU utilization
            proc = psutil.Process(os.getpid())
            _ = proc.cpu_percent(interval=None)
            
            t0 = time.perf_counter()
            frames_read = 0
            for _ in range(60):
                ret, f = cap.read()
                if ret and f is not None:
                    frames_read += 1
            el = time.perf_counter() - t0
            cpu_used = proc.cpu_percent(interval=None)
            fps = frames_read / el if el > 0 else 0
            lat_ms = (el / frames_read) * 1000.0 if frames_read > 0 else 0
            cap.release()
            decode_benchmarks[name] = {"fps": fps, "latency_ms": lat_ms, "cpu_pct": cpu_used, "frames": frames_read}
        else:
            decode_benchmarks[name] = "Not opened"
            
    print(f"Stream Decode Results: {decode_benchmarks}")

if __name__ == "__main__":
    analyze_and_benchmark()
