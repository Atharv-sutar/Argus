"""ReID Decision Viewer: Forensic inspection tool for candidate rankings and authorization decisions."""

from __future__ import annotations

import argparse
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.core.types import MatchDecisionState
from src.identity.manager import IdentityManager
from src.reid.extractor import PyTorchReIDExtractor
from src.reid.quality import CropQualityEvaluator


def display_decision_inspection(
    target_crop_path: str,
    candidate_crop_paths: list[str],
    device: str = "cpu",
):
    print("=" * 80)
    print("           ARGUS ReID DECISION VIEWER & FORENSIC INSPECTOR")
    print("=" * 80)

    if not os.path.exists(target_crop_path):
        print(f"Error: Target crop path does not exist: {target_crop_path}")
        return

    target_img = cv2.imread(target_crop_path)
    if target_img is None:
        print(f"Error: Could not read target image {target_crop_path}")
        return

    print(f"Loading ReID extractor (DINOv2) on device '{device}'...")
    extractor = PyTorchReIDExtractor(model_name="dinov2", device=device)
    quality = CropQualityEvaluator()
    im = IdentityManager(reid_extractor=extractor, quality_evaluator=quality)

    print(f"\n[ENROLLMENT] Registering target from: {target_crop_path}")
    im.register_new_target(target_img, identity_id="target_0", label="Target Inspector")
    ident = im.get_identity("target_0")

    print(f"  Target Anchor Initialized: {ident.anchor is not None}")
    if ident.anchor:
        print(f"  Anchor Clusters: {len(ident.anchor.clusters)}")
        for idx, cl in enumerate(ident.anchor.clusters):
            print(f"    - Cluster {idx} ({cl.label}): {len(cl.exemplars)} exemplars")

    print("\n" + "-" * 80)
    print("                      CANDIDATE EVALUATION & RANKING")
    print("-" * 80)
    print(f"{'Rank':<5} | {'Candidate File':<35} | {'Score':<8} | {'Quality':<8} | {'Decision'}")
    print("-" * 80)

    evaluated_cands = []
    for p in candidate_crop_paths:
        if not os.path.exists(p):
            continue
        c_img = cv2.imread(p)
        if c_img is None:
            continue
        ev = im.evaluate_candidate_crop(c_img, "target_0")
        evaluated_cands.append((os.path.basename(p), ev))

    evaluated_cands.sort(key=lambda x: x[1].candidate_score, reverse=True)

    for rank, (name, ev) in enumerate(evaluated_cands, 1):
        status = "MATCH" if ev.is_match else ev.decision
        print(f"{rank:<5} | {name:<35} | {ev.candidate_score:<8.4f} | {ev.crop_quality_score:<8.2f} | {status}")
        cluster_info = {"proto": round(ev.proto_sim, 4), "best_ref": round(ev.best_ref_sim, 4), "upper": round(ev.upper_sim, 4)}
        print(f"      -> Cluster Breakdown: {cluster_info}")
        if ev.quality_reason:
            print(f"      -> Quality Reason: {ev.quality_reason}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Argus ReID Decision Viewer & Diagnostic Inspector")
    parser.add_argument("--target", type=str, required=False, help="Path to target reference image")
    parser.add_argument("--candidates", nargs="+", help="Paths to candidate crop images")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    if not args.target or not args.candidates:
        print("Usage: python scripts/reid_decision_viewer.py --target path/to/target.png --candidates path/to/cand1.png path/to/cand2.png")
        print("\nDemo Mode: Running self-inspection with synthetic validation samples...")
        t_img = np.full((120, 50, 3), [160, 60, 30], dtype=np.uint8)
        c1 = np.full((120, 50, 3), [155, 58, 30], dtype=np.uint8)
        c2 = np.full((120, 50, 3), [30, 30, 190], dtype=np.uint8)

        os.makedirs("diagnostics/demo", exist_ok=True)
        cv2.imwrite("diagnostics/demo/target.png", t_img)
        cv2.imwrite("diagnostics/demo/cand_similar.png", c1)
        cv2.imwrite("diagnostics/demo/cand_impostor.png", c2)

        display_decision_inspection(
            "diagnostics/demo/target.png",
            ["diagnostics/demo/cand_similar.png", "diagnostics/demo/cand_impostor.png"],
            device=args.device,
        )
        return

    display_decision_inspection(args.target, args.candidates, device=args.device)


if __name__ == "__main__":
    main()
