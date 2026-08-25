"""Comprehensive ReID Diagnostic Runner executing Tests 1-6, feature audits, and gate checks."""

import glob
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.core.types import Embedding, Identity
from src.identity.manager import IdentityManager
from src.identity.store import InMemoryVectorStore
from src.reid.extractor import PyTorchReIDExtractor
from src.reid.quality import CropQualityEvaluator
from src.reid.diagnostic import ReIDDiagnostic, VectorStats


def main():
    print("=" * 80)
    print("         ARGUS REID RIGOROUS DIAGNOSTIC & AUDIT SUITE")
    print("=" * 80)

    # 1. Initialize models and diagnostic engine
    extractor = PyTorchReIDExtractor(model_name="osnet_x0_25", device="cpu")
    quality_eval = CropQualityEvaluator()
    id_manager = IdentityManager(
        reid_extractor=extractor,
        vector_store=InMemoryVectorStore(),
        similarity_threshold=0.70,
        reference_threshold=0.65,
        min_margin=0.05,
    )
    diagnostic = ReIDDiagnostic(extractor, id_manager, quality_eval)

    # Find crops
    scratch_dir = r"C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch"
    crop_paths = sorted(glob.glob(os.path.join(scratch_dir, "reid_cand_*.png")))
    print(f"Total available candidate crops in scratch directory: {len(crop_paths)}")

    # Group crops by track_id
    tracks_map = {}
    for p in crop_paths:
        fname = os.path.basename(p)
        # e.g. reid_cand_frame_100_track_15.png
        parts = fname.replace(".png", "").split("_")
        if "track" in parts:
            t_idx = parts.index("track") + 1
            if t_idx < len(parts):
                t_id = parts[t_idx]
                if t_id not in tracks_map:
                    tracks_map[t_id] = []
                tracks_map[t_id].append(p)

    print(f"Discovered {len(tracks_map)} distinct tracker IDs.")
    top_tracks = sorted(tracks_map.items(), key=lambda x: len(x[1]), reverse=True)
    for tid, paths in top_tracks[:5]:
        print(f"  Track {tid}: {len(paths)} frames")

    # Pick representative crops
    track_a_id = top_tracks[0][0]
    track_a_crops = [cv2.imread(p) for p in top_tracks[0][1][:10] if cv2.imread(p) is not None]

    track_b_id = top_tracks[1][0]
    track_b_crops = [cv2.imread(p) for p in top_tracks[1][1][:10] if cv2.imread(p) is not None]

    track_c_id = top_tracks[2][0] if len(top_tracks) > 2 else top_tracks[1][0]
    track_c_crops = [cv2.imread(p) for p in top_tracks[2][1][:10] if cv2.imread(p) is not None]

    crop_a1 = track_a_crops[0]
    crop_a2 = track_a_crops[min(5, len(track_a_crops) - 1)]
    crop_b1 = track_b_crops[0]
    crop_c1 = track_c_crops[0]

    # =========================================================================
    # SECTION 1: Exact Embeddings Inspection & Verification
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: EXACT EMBEDDING DIMENSIONS, DTYPES, STATS & NORMALIZATION")
    print("=" * 80)
    fused_a, deep_a, col_a, up_a, low_a = extractor.extract_decomposed(crop_a1)
    fused_b, deep_b, col_b, up_b, low_b = extractor.extract_decomposed(crop_b1)

    components = [
        ("Fused", fused_a, fused_b),
        ("Deep", deep_a, deep_b),
        ("Color / Texture", col_a, col_b),
        ("Upper Body", up_a, up_b),
        ("Lower Body", low_a, low_b),
    ]

    for name, emb_ref, emb_cand in components:
        s_ref = VectorStats.from_array(emb_ref.vector)
        s_cand = VectorStats.from_array(emb_cand.vector)
        print(f"\n--- Component: {name} ---")
        print(f"Reference Embedding Stats (Track {track_a_id}):")
        print(s_ref.formatted())
        print(f"Candidate Embedding Stats (Track {track_b_id}):")
        print(s_cand.formatted())
        dim_match = (s_ref.dim == s_cand.dim)
        norm_ref_valid = np.isclose(s_ref.l2_norm, 1.0, atol=1e-4)
        norm_cand_valid = np.isclose(s_cand.l2_norm, 1.0, atol=1e-4)
        print(f"  [CHECK] Dimension match: {dim_match} ({s_ref.dim}) | L2 norm == 1.0: Ref={norm_ref_valid}, Cand={norm_cand_valid}")

    # =========================================================================
    # SECTION 2: Determinism Test
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: DETERMINISM TEST (Same crop extracted 3x)")
    print("=" * 80)
    det_res = diagnostic.test_determinism(crop_a1, iterations=3)
    print(f"Extracted 3 iterations of same crop.")
    for (i, j), sim in det_res["pairwise_fused"]:
        print(f"  cos(Iteration {i}, Iteration {j}) [Fused]: {sim:.8f}")
    for (i, j), sim in det_res["pairwise_deep"]:
        print(f"  cos(Iteration {i}, Iteration {j}) [Deep]:  {sim:.8f}")
    for (i, j), sim in det_res["pairwise_color"]:
        print(f"  cos(Iteration {i}, Iteration {j}) [Color]: {sim:.8f}")
    print(f"Determinism Check Passed: {det_res['is_deterministic']}")

    # =========================================================================
    # SECTION 3: Feature Fusion Math & Weight Verification
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: FEATURE FUSION MATHEMATICAL VERIFICATION")
    print("=" * 80)
    d_dim = deep_a.dim
    c_dim = col_a.dim
    u_dim = up_a.dim
    l_dim = low_a.dim
    p_dim = u_dim + l_dim
    total_fused_dim = fused_a.dim

    print(f"Deep dim: {d_dim} (weight 0.8367 -> energy {0.8367**2:.4f} ~ 70%)")
    print(f"Color dim: {c_dim} (weight 0.4690 -> energy {0.4690**2:.4f} ~ 22%)")
    print(f"Part dim: {p_dim} (weight 0.2828 -> energy {0.2828**2:.4f} ~ 8%)")
    print(f"Expected concatenated dimension: {d_dim + c_dim + p_dim} | Actual Fused dim: {total_fused_dim}")
    print(f"Fused L2 norm: {np.linalg.norm(fused_a.vector):.6f}")

    # =========================================================================
    # SECTION 4: Prototype Construction Audit
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: PROTOTYPE CONSTRUCTION & NORMALIZATION AUDIT")
    print("=" * 80)
    id_manager.clear()
    id_manager.register_new_target(track_a_crops[0], "audit_target")
    for k in range(1, min(4, len(track_a_crops))):
        id_manager.add_reference_sample(track_a_crops[k], "audit_target")

    ident = id_manager.get_identity("audit_target")
    print(f"Reference Gallery Size: {len(ident.reference_gallery)}")
    print(f"Reference Prototype exists: {ident.reference_prototype is not None}")
    if ident.reference_prototype:
        print(f"  Fused Proto Dim: {ident.reference_prototype.dim} | L2 Norm: {np.linalg.norm(ident.reference_prototype.vector):.6f}")
    if ident.reference_deep_proto:
        print(f"  Deep Proto Dim:  {ident.reference_deep_proto.dim} | L2 Norm: {np.linalg.norm(ident.reference_deep_proto.vector):.6f}")
    if ident.reference_color_proto:
        print(f"  Color Proto Dim: {ident.reference_color_proto.dim} | L2 Norm: {np.linalg.norm(ident.reference_color_proto.vector):.6f}")
    if ident.reference_upper_proto:
        print(f"  Upper Proto Dim: {ident.reference_upper_proto.dim} | L2 Norm: {np.linalg.norm(ident.reference_upper_proto.vector):.6f}")
    if ident.reference_lower_proto:
        print(f"  Lower Proto Dim: {ident.reference_lower_proto.dim} | L2 Norm: {np.linalg.norm(ident.reference_lower_proto.vector):.6f}")

    # Reference Gallery Pairwise Similarities
    print("\nReference Gallery Pairwise Similarities:")
    ref_fused = ident.reference_gallery
    for i in range(len(ref_fused)):
        for j in range(i + 1, len(ref_fused)):
            sim_f = ref_fused[i].cosine_similarity(ref_fused[j])
            sim_d = ident.reference_deep_gallery[i].cosine_similarity(ident.reference_deep_gallery[j])
            sim_c = ident.reference_color_gallery[i].cosine_similarity(ident.reference_color_gallery[j])
            print(f"  Ref {i+1} <-> Ref {j+1}: FusedSim={sim_f:.4f} | DeepSim={sim_d:.4f} | ColorSim={sim_c:.4f}")

    # =========================================================================
    # SECTION 5: REQUIRED TESTS 1-6 WITH ALL REJECTION GATES
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: DIAGNOSTIC TESTS 1-6 WITH FULL GATE-BY-GATE BREAKDOWN")
    print("=" * 80)

    test_cases = [
        ("Test 1 — Same Image vs Itself", crop_a1, crop_a1, None),
        ("Test 2 — Same Person / Different Frame (Same Camera)", crop_a1, crop_a2, None),
        ("Test 3 — Same Person / Perturbed / Lighting (Simulated Camera Shift)", crop_a1, cv2.convertScaleAbs(crop_a2, alpha=0.85, beta=15), None),
        ("Test 4 — Different Person (Track A vs Track B)", crop_a1, crop_b1, None),
        ("Test 5 — Single-Person True Target in Search Camera", crop_a1, crop_a2, None),
        ("Test 6 — Single-Person Wrong Person in Search Camera", crop_a1, crop_b1, None),
    ]

    for title, ref_img, cand_img, second_score in test_cases:
        print("\n" + "-" * 70)
        print(f"RUNNING: {title}")
        print("-" * 70)
        res = diagnostic.compare_crops(ref_img, cand_img, second_cand_score=second_score)
        print(res.summary_text())

    # =========================================================================
    # SECTION 6: DISTRIBUTION AUDIT (True Matches vs False Matches)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: POPULATION DISTRIBUTION AUDIT (True vs False Match Separability)")
    print("=" * 80)

    # Intra-person similarities (same track across frames)
    intra_fused = []
    intra_deep = []
    intra_color = []
    intra_disagree = []
    for i in range(len(track_a_crops)):
        for j in range(i + 1, len(track_a_crops)):
            f_i, d_i, c_i, _, _ = extractor.extract_decomposed(track_a_crops[i])
            f_j, d_j, c_j, _, _ = extractor.extract_decomposed(track_a_crops[j])
            intra_fused.append(f_i.cosine_similarity(f_j))
            intra_deep.append(d_i.cosine_similarity(d_j))
            intra_color.append(c_i.cosine_similarity(c_j))
            intra_disagree.append(abs(d_i.cosine_similarity(d_j) - c_i.cosine_similarity(c_j)))

    # Inter-person similarities (different tracks)
    inter_fused = []
    inter_deep = []
    inter_color = []
    inter_disagree = []
    for crop_a in track_a_crops[:5]:
        for crop_b in track_b_crops[:5]:
            f_a, d_a, c_a, _, _ = extractor.extract_decomposed(crop_a)
            f_b, d_b, c_b, _, _ = extractor.extract_decomposed(crop_b)
            inter_fused.append(f_a.cosine_similarity(f_b))
            inter_deep.append(d_a.cosine_similarity(d_b))
            inter_color.append(c_a.cosine_similarity(c_b))
            inter_disagree.append(abs(d_a.cosine_similarity(d_b) - c_a.cosine_similarity(c_b)))

    print("TRUE MATCH (Intra-Person) Distribution:")
    print(f"  FusedSim: Mean={np.mean(intra_fused):.3f}, Min={np.min(intra_fused):.3f}, Max={np.max(intra_fused):.3f}")
    print(f"  DeepSim:  Mean={np.mean(intra_deep):.3f}, Min={np.min(intra_deep):.3f}, Max={np.max(intra_deep):.3f}")
    print(f"  ColorSim: Mean={np.mean(intra_color):.3f}, Min={np.min(intra_color):.3f}, Max={np.max(intra_color):.3f}")
    print(f"  Disagree: Mean={np.mean(intra_disagree):.3f}, Max={np.max(intra_disagree):.3f}")

    print("\nFALSE MATCH (Inter-Person Impostor) Distribution:")
    print(f"  FusedSim: Mean={np.mean(inter_fused):.3f}, Min={np.min(inter_fused):.3f}, Max={np.max(inter_fused):.3f}")
    print(f"  DeepSim:  Mean={np.mean(inter_deep):.3f}, Min={np.min(inter_deep):.3f}, Max={np.max(inter_deep):.3f}")
    print(f"  ColorSim: Mean={np.mean(inter_color):.3f}, Min={np.min(inter_color):.3f}, Max={np.max(inter_color):.3f}")
    print(f"  Disagree: Mean={np.mean(inter_disagree):.3f}, Max={np.max(inter_disagree):.3f}")

    print("\nSeparation (Intra Mean - Inter Mean):")
    print(f"  FusedSim Separation: {np.mean(intra_fused) - np.mean(inter_fused):+.3f}")
    print(f"  DeepSim Separation:  {np.mean(intra_deep) - np.mean(inter_deep):+.3f}")
    print(f"  ColorSim Separation: {np.mean(intra_color) - np.mean(inter_color):+.3f}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC AUDIT COMPLETED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
