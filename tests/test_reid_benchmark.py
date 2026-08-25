"""Comprehensive accuracy, latency, and discrimination benchmark for Argus ReID system."""

import glob
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.core.types import Embedding, Identity
from src.identity.manager import IdentityManager
from src.identity.store import InMemoryVectorStore
from src.reid.extractor import PyTorchReIDExtractor
from src.reid.quality import CropQualityEvaluator


def run_benchmark():
    scratch_dir = r'C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch'
    crop_files = glob.glob(os.path.join(scratch_dir, "reid_cand_*.png"))

    if len(crop_files) < 10:
        print(f"Using synthetic person crops for benchmark...")
        crops = []
        for i in range(20):
            img = np.random.randint(40, 220, (180, 70, 3), dtype=np.uint8)
            crops.append(img)
    else:
        crops = [cv2.imread(f) for f in crop_files[:40] if cv2.imread(f) is not None]

    print(f"=== Argus ReID Recognition & Discrimination Benchmark ===")
    print(f"Dataset: {len(crops)} person observation crops loaded.\n")

    extractor = PyTorchReIDExtractor(model_name="mobilenet_v3_small", device="cpu")
    quality_eval = CropQualityEvaluator()
    identity_manager = IdentityManager(
        reid_extractor=extractor,
        vector_store=InMemoryVectorStore(),
        similarity_threshold=0.68,
        reference_threshold=0.60,
        min_margin=0.06,
        max_feature_disagreement=0.25,
    )

    # 1. Latency Profiling
    t0 = time.perf_counter()
    extracted_features = []
    for crop in crops:
        fused, deep, color, upper, lower = extractor.extract_decomposed(crop)
        extracted_features.append((fused, deep, color, upper, lower))
    t1 = time.perf_counter()
    total_time_ms = (t1 - t0) * 1000.0
    latency_per_crop = total_time_ms / len(crops)

    print("--- 1. Latency & Feature Extraction Profile ---")
    print(f"Total extraction time: {total_time_ms:.2f} ms for {len(crops)} crops")
    print(f"Per-crop latency (CPU MobileNetV3 + Multi-Region + Texture): {latency_per_crop:.2f} ms")
    print(f"Throughput: {1000.0 / latency_per_crop:.1f} crops/sec\n")

    # 2. Pairwise Similarity & Agreement Distributions
    n = len(crops)
    fused_sims = []
    deep_sims = []
    color_sims = []
    disagreements = []

    for i in range(n):
        for j in range(i + 1, n):
            f_i, d_i, c_i, _, _ = extracted_features[i]
            f_j, d_j, c_j, _, _ = extracted_features[j]

            sim_f = f_i.cosine_similarity(f_j)
            sim_d = d_i.cosine_similarity(d_j)
            sim_c = c_i.cosine_similarity(c_j)
            disagree = abs(sim_d - sim_c)

            fused_sims.append(sim_f)
            deep_sims.append(sim_d)
            color_sims.append(sim_c)
            disagreements.append(disagree)

    print("--- 2. Inter-Person Separation Distributions ---")
    print(f"Comparisons count: {len(fused_sims)}")
    print(f"Deep Feature Similarity     : Mean={np.mean(deep_sims):.3f}, Std={np.std(deep_sims):.3f}, Max={np.max(deep_sims):.3f}")
    print(f"Color & Texture Similarity  : Mean={np.mean(color_sims):.3f}, Std={np.std(color_sims):.3f}, Max={np.max(color_sims):.3f}")
    print(f"Fused Composite Similarity  : Mean={np.mean(fused_sims):.3f}, Std={np.std(fused_sims):.3f}, Max={np.max(fused_sims):.3f}")
    print(f"Mean Feature Disagreement   : {np.mean(disagreements):.3f} (Std: {np.std(disagreements):.3f})\n")

    # 3. Prototype Fusion & Multi-Component Matching Validation
    # Register Target A with crops 0..3
    identity_manager.register_new_target(crops[0], "target_A")
    for k in range(1, min(4, len(crops))):
        identity_manager.add_reference_sample(crops[k], "target_A")

    ident = identity_manager.get_identity("target_A")
    has_proto = ident.reference_prototype is not None
    ref_count = len(ident.reference_gallery)

    print("--- 3. Target Prototype Manifold Evaluation ---")
    print(f"Target 'target_A' Reference Gallery Size: {ref_count}")
    print(f"Reference Prototype Centroid Created: {has_proto}")

    # Evaluate candidate match and false positive rejection
    eval_target_self = identity_manager.evaluate_candidate_crop(crops[0], "target_A")
    print(f"\n[Self-Match Test (Same Person)]")
    print(f"Score: {eval_target_self.candidate_score:.3f} | ProtoSim: {eval_target_self.proto_sim:.3f} | Match: {eval_target_self.is_match}")
    print(f"Agreement Passed: {eval_target_self.feature_agreement_passed}")

    if len(crops) > 10:
        eval_non_target = identity_manager.evaluate_candidate_crop(crops[10], "target_A")
        print(f"\n[Different Person Impostor Test]")
        print(f"Score: {eval_non_target.candidate_score:.3f} | ProtoSim: {eval_non_target.proto_sim:.3f} | Match: {eval_non_target.is_match}")
        print(f"Agreement Passed: {eval_non_target.feature_agreement_passed}")
        print(f"Impostor Correctly Rejected: {not eval_non_target.is_match}")

    print("\n=======================================================")


if __name__ == "__main__":
    run_benchmark()
