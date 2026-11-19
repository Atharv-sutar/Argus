"""Evaluation tool for ReID embedding discrimination and separation quality.

Computes:
1. Genuine vs Impostor cosine similarity distributions (mean, std, min, max, p5, p95).
2. Genuine p5 vs Impostor p95 margin.
3. Per-frame Rank-1 accuracy and top-1 margin.
4. Area Under ROC Curve (AUC).
5. Plain-language verdict on whether the model reliably separates people.

Usage:
  python scripts/embedding_quality.py --crops path/to/crops_dir
  python scripts/embedding_quality.py --clip path/to/video.mp4 --labels path/to/labels.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.types import Embedding
from src.reid.extractor import PyTorchReIDExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("embedding_quality")


def compute_roc_auc(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    """Computes exact Area Under the ROC Curve (AUC) via vectorized rank sums."""
    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)
    if n_gen == 0 or n_imp == 0:
        return 0.0

    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones(n_gen, dtype=np.int32), np.zeros(n_imp, dtype=np.int32)])

    order = np.argsort(scores)
    sorted_labels = labels[order]

    # Cumulative sum of negative samples (false positives)
    cum_negatives = np.cumsum(1 - sorted_labels)
    # Total sum of negatives before each positive sample
    auc = float(np.sum(cum_negatives[sorted_labels == 1])) / (n_gen * n_imp)
    return float(np.clip(auc, 0.0, 1.0))


def evaluate_embeddings(
    labeled_embeddings: List[Tuple[str, np.ndarray]],
) -> Dict[str, Any]:
    """
    Computes genuine vs impostor metrics from a list of (person_id, embedding_vector) tuples.
    """
    if len(labeled_embeddings) < 2:
        return {
            "error": "Not enough samples to evaluate (need at least 2 samples).",
            "verdict": "INSUFFICIENT_DATA",
        }

    person_ids = [pid for pid, _ in labeled_embeddings]
    vectors = np.stack([vec / (np.linalg.norm(vec) + 1e-8) for _, vec in labeled_embeddings])

    unique_persons = sorted(list(set(person_ids)))
    if len(unique_persons) < 2:
        return {
            "error": f"Only 1 unique identity found ('{unique_persons[0]}'). Impostor pairs cannot be formed.",
            "verdict": "SINGLE_IDENTITY_ONLY",
        }

    sim_matrix = np.dot(vectors, vectors.T)
    n = len(labeled_embeddings)

    genuine_scores: List[float] = []
    impostor_scores: List[float] = []

    # Rank-1 evaluation
    rank1_hits = 0
    rank1_margins: List[float] = []

    for i in range(n):
        probe_pid = person_ids[i]
        sims = sim_matrix[i].copy()
        sims[i] = -1.0  # Exclude self-match

        best_idx = int(np.argmax(sims))
        best_pid = person_ids[best_idx]

        # Find best genuine and best impostor for margin calculation
        gen_sims = [sim_matrix[i, j] for j in range(n) if j != i and person_ids[j] == probe_pid]
        imp_sims = [sim_matrix[i, j] for j in range(n) if person_ids[j] != probe_pid]

        if gen_sims:
            best_gen = max(gen_sims)
            best_imp = max(imp_sims) if imp_sims else 0.0
            rank1_margins.append(best_gen - best_imp)
            if best_pid == probe_pid:
                rank1_hits += 1

        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            if person_ids[i] == person_ids[j]:
                genuine_scores.append(s)
            else:
                impostor_scores.append(s)

    gen_arr = np.array(genuine_scores, dtype=np.float32)
    imp_arr = np.array(impostor_scores, dtype=np.float32)

    if len(gen_arr) == 0:
        return {
            "error": "No genuine pairs found (each person only has 1 crop). Need multiple crops per identity.",
            "verdict": "NO_GENUINE_PAIRS",
        }

    gen_mean = float(np.mean(gen_arr))
    gen_std = float(np.std(gen_arr))
    gen_p5 = float(np.percentile(gen_arr, 5))
    gen_median = float(np.median(gen_arr))

    imp_mean = float(np.mean(imp_arr))
    imp_std = float(np.std(imp_arr))
    imp_p95 = float(np.percentile(imp_arr, 95))
    imp_median = float(np.median(imp_arr))

    separation_margin = gen_p5 - imp_p95
    mean_margin = gen_mean - imp_mean

    rank1_rate = (rank1_hits / max(1, len(rank1_margins))) * 100.0
    avg_rank1_margin = float(np.mean(rank1_margins)) if rank1_margins else 0.0

    auc = compute_roc_auc(gen_arr, imp_arr)

    # Plain language verdict
    # Separation criteria:
    # 1. AUC >= 0.85
    # 2. Rank-1 >= 85%
    # 3. Impostor p95 strictly below genuine p5, or mean margin >= 0.15
    separates = (auc >= 0.85) and (rank1_rate >= 80.0) and (separation_margin > -0.05 or mean_margin >= 0.15)
    verdict_text = "MODEL SEPARATES PEOPLE (Clear discriminative margin confirmed)" if separates else "MODEL DOES NOT RELIABLY SEPARATE PEOPLE (Distribution overlap observed; risk of scooping/false match)"

    return {
        "num_identities": len(unique_persons),
        "total_crops": n,
        "genuine_pairs": len(gen_arr),
        "impostor_pairs": len(imp_arr),
        "genuine": {
            "mean": gen_mean,
            "std": gen_std,
            "median": gen_median,
            "p5": gen_p5,
            "min": float(np.min(gen_arr)),
            "max": float(np.max(gen_arr)),
        },
        "impostor": {
            "mean": imp_mean,
            "std": imp_std,
            "median": imp_median,
            "p95": imp_p95,
            "min": float(np.min(imp_arr)),
            "max": float(np.max(imp_arr)),
        },
        "separation_margin_p5_minus_p95": separation_margin,
        "mean_margin": mean_margin,
        "rank1_rate_percent": rank1_rate,
        "avg_rank1_margin": avg_rank1_margin,
        "roc_auc": auc,
        "verdict": verdict_text,
        "passes_quality": separates,
    }


def load_crops_from_directory(crops_dir: str) -> List[Tuple[str, np.ndarray]]:
    """Loads person crops organized in subfolders or named by ID."""
    labeled_crops: List[Tuple[str, np.ndarray]] = []
    p = Path(crops_dir)

    # 1. Subfolder structure: crops_dir/<person_id>/*.jpg
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if subdirs:
        for sdir in sorted(subdirs):
            person_id = sdir.name
            for img_path in sorted(list(sdir.glob("*.jpg")) + list(sdir.glob("*.png")) + list(sdir.glob("*.jpeg"))):
                img = cv2.imread(str(img_path))
                if img is not None and img.size > 0:
                    labeled_crops.append((person_id, img))
        return labeled_crops

    # 2. Flat structure: crops_dir/<person_id>_*.jpg
    for img_path in sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg"))):
        name = img_path.stem
        # Split on first underscore or dash
        parts = name.replace("-", "_").split("_")
        person_id = parts[0] if len(parts) > 1 else "person_1"
        img = cv2.imread(str(img_path))
        if img is not None and img.size > 0:
            labeled_crops.append((person_id, img))

    return labeled_crops


def load_crops_from_clip_and_labels(
    clip_path: str, labels_path: str, detector=None
) -> List[Tuple[str, np.ndarray]]:
    """
    Extracts person crops from a labeled video clip.
    Labels JSON format:
    {
      "intervals": [
        {"start_s": 0.0, "end_s": 10.0, "person": "person_A"},
        {"start_s": 10.0, "end_s": 20.0, "person": "person_B"}
      ]
    }
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    intervals = data.get("intervals", [])

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        logger.error(f"Cannot open clip '{clip_path}'")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_idx = 0
    labeled_crops = []

    if detector is None:
        from src.detection.yolo_detector import YOLODetector
        detector = YOLODetector(confidence_threshold=0.4)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        ts_s = frame_idx / fps

        # Find matching label
        current_person = None
        for item in intervals:
            if item.get("start_s", 0.0) <= ts_s <= item.get("end_s", float("inf")):
                current_person = item.get("person")
                break

        if current_person:
            # Sample every 5th frame
            if frame_idx % 5 == 0:
                det_res = detector.detect(frame)
                for det in det_res.detections:
                    b = det.box
                    crop = frame[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
                    if crop.size > 0 and crop.shape[0] >= 32 and crop.shape[1] >= 16:
                        labeled_crops.append((current_person, crop))

        frame_idx += 1

    cap.release()
    return labeled_crops


def print_report(res: Dict[str, Any]) -> None:
    """Prints a structured ASCII report of ReID quality metrics."""
    print("\n" + "=" * 65)
    print("        ARGUS ReID EMBEDDING DISCRIMINATION REPORT        ")
    print("=" * 65)

    if "error" in res:
        print(f"ERROR: {res['error']}")
        print(f"VERDICT: {res.get('verdict', 'FAILED')}")
        print("=" * 65 + "\n")
        return

    print(f"Identities Evaluated    : {res['num_identities']}")
    print(f"Total Crops Analyzed    : {res['total_crops']}")
    print(f"Genuine Pairs           : {res['genuine_pairs']}")
    print(f"Impostor Pairs          : {res['impostor_pairs']}")
    print("-" * 65)
    print("DISTRIBUTION METRICS:")
    g = res["genuine"]
    i = res["impostor"]
    print(f"  Genuine  Similarity   : Mean={g['mean']:.4f}  Std={g['std']:.4f}  p5={g['p5']:.4f}  [Min={g['min']:.4f}, Max={g['max']:.4f}]")
    print(f"  Impostor Similarity   : Mean={i['mean']:.4f}  Std={i['std']:.4f}  p95={i['p95']:.4f} [Min={i['min']:.4f}, Max={i['max']:.4f}]")
    print("-" * 65)
    print("SEPARATION & DISCRIMINATION:")
    print(f"  Mean Margin (Gen - Imp)       : {res['mean_margin']:+.4f}")
    print(f"  Safety Margin (Gen p5 - Imp p95): {res['separation_margin_p5_minus_p95']:+.4f}")
    print(f"  Rank-1 Identification Rate    : {res['rank1_rate_percent']:.2f}% (Avg Margin: {res['avg_rank1_margin']:+.4f})")
    print(f"  ROC Area Under Curve (AUC)    : {res['roc_auc']:.4f}")
    print("=" * 65)
    print(f"VERDICT: {res['verdict']}")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Argus ReID Embedding Discrimination Evaluator")
    parser.add_argument("--crops", type=str, default=None, help="Directory containing person crops (subfolders per ID)")
    parser.add_argument("--clip", type=str, default=None, help="Path to video clip (.mp4, .avi)")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels.json for clip intervals")
    parser.add_argument("--model", type=str, default="osnet_x0_25", help="ReID model name ('osnet_x0_25', etc.)")
    parser.add_argument("--device", type=str, default="auto", help="Inference device ('auto', 'cpu', 'cuda')")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path for automated testing")
    args = parser.parse_args()

    if not args.crops and not (args.clip and args.labels):
        print("Usage error: Provide either --crops <dir> OR --clip <file> --labels <json>")
        sys.exit(1)

    logger.info(f"Initializing ReID extractor (model='{args.model}', device='{args.device}')...")
    extractor = PyTorchReIDExtractor(model_name=args.model, device=args.device, prefer_engine=False)

    labeled_crops = []
    if args.crops:
        logger.info(f"Loading crops from directory: '{args.crops}'")
        labeled_crops = load_crops_from_directory(args.crops)
    elif args.clip and args.labels:
        logger.info(f"Extracting labeled crops from clip: '{args.clip}' using '{args.labels}'")
        labeled_crops = load_crops_from_clip_and_labels(args.clip, args.labels)

    if not labeled_crops:
        logger.error("No valid crops found.")
        sys.exit(1)

    logger.info(f"Extracting embeddings for {len(labeled_crops)} crops...")
    crops = [c for _, c in labeled_crops]
    embeddings = extractor.extract_batch(crops)

    labeled_embeddings = [(pid, emb.vector) for (pid, _), emb in zip(labeled_crops, embeddings)]
    results = evaluate_embeddings(labeled_embeddings)
    print_report(results)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
