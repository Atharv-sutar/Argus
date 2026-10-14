"""CLI Runner for Argus ReID and Multi-Camera Tracking Evaluation Harness."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.evaluator import ReIDEvaluationHarness
from src.benchmark.types import GroundTruthDataset
from src.core.config import AppConfig
from src.core.multi_camera_types import (
    CameraEdgeConfig,
    CameraNodeConfig,
    EdgeType,
    SourceType,
)
from src.detection.yolo_detector import YOLODetector
from src.multi_camera.camera_graph import CameraGraph
from src.reid.extractor import PyTorchReIDExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("argus.eval")


def build_camera_graph(footage_dir: Path) -> Tuple[CameraGraph, dict[str, str]]:
    """Builds camera graph topology and source mapping for EPFL Campus footage."""
    graph = CameraGraph()

    cam_files = {
        "cam_0": footage_dir / "campus4-c0.avi",
        "cam_1": footage_dir / "campus4-c1.avi",
        "cam_2": footage_dir / "campus4-c2.avi",
    }

    # Add nodes
    for cam_id, path in cam_files.items():
        if path.exists():
            graph.add_node(CameraNodeConfig(
                camera_id=cam_id,
                name=f"EPFL_{cam_id}",
                source=str(path),
                source_type=SourceType.VIDEO_FILE,
            ))
        else:
            logger.warning(f"Footage file not found for {cam_id}: {path}")

    # Add edges: cam_0 <-> cam_2, cam_0 <-> cam_1, cam_1 <-> cam_2
    graph.add_edge(CameraEdgeConfig(
        source_camera_id="cam_0",
        target_camera_id="cam_2",
        edge_type=EdgeType.ADJACENT,
        expected_min_transition_s=1.0,
        expected_max_transition_s=25.0,
    ))
    graph.add_edge(CameraEdgeConfig(
        source_camera_id="cam_0",
        target_camera_id="cam_1",
        edge_type=EdgeType.ADJACENT,
        expected_min_transition_s=1.0,
        expected_max_transition_s=25.0,
    ))
    graph.add_edge(CameraEdgeConfig(
        source_camera_id="cam_1",
        target_camera_id="cam_2",
        edge_type=EdgeType.ADJACENT,
        expected_min_transition_s=1.0,
        expected_max_transition_s=25.0,
    ))

    sources = {cam_id: str(path) for cam_id, path in cam_files.items() if path.exists()}
    return graph, sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Argus ReID & Tracking on Multi-Camera Footage.")
    parser.add_argument("--footage-dir", type=str, default="test_footage/EPFL campus", help="Path to footage folder")
    parser.add_argument("--gt-file", type=str, default="cases/epfl_campus4_gt.json", help="Path to ground truth JSON file")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to base AppConfig YAML")
    parser.add_argument("--model", type=str, default=None, help="Override ReID model (e.g. osnet_x1_0, osnet_x0_25)")
    parser.add_argument("--match-thresh", type=float, default=None, help="Override ReID match_threshold")
    parser.add_argument("--reacq-thresh", type=float, default=None, help="Override ReID reacquisition_threshold")
    parser.add_argument("--frames", type=int, default=400, help="Max frames to process (default: 400)")
    parser.add_argument("--seed-cam", type=str, default="cam_0", help="Camera to seed target on")
    parser.add_argument("--seed-frame", type=int, default=35, help="Frame to seed target on (default: 35 for EPFL C0)")
    parser.add_argument("--output-json", type=str, default="diagnostics/reid_eval_report.json", help="Path to save report")

    args = parser.parse_args()

    # 1. Load config
    cfg_path = Path(args.config)
    cfg = AppConfig.from_yaml(cfg_path) if cfg_path.exists() else AppConfig()

    if args.model:
        cfg.reid.model_name = args.model
    if args.match_thresh is not None:
        cfg.reid.match_threshold = args.match_thresh
    if args.reacq_thresh is not None:
        cfg.reid.reacquisition_threshold = args.reacq_thresh

    footage_dir = Path(args.footage_dir)
    if not footage_dir.exists():
        logger.error(f"Footage directory '{footage_dir}' does not exist.")
        sys.exit(1)

    gt_file = Path(args.gt_file)
    if not gt_file.exists():
        logger.error(f"Ground truth file '{gt_file}' not found. Run ground truth generator first.")
        sys.exit(1)

    print("=" * 70)
    print("      ARGUS RE-IDENTIFICATION & TRACKING EVALUATION HARNESS")
    print("=" * 70)
    print(f"Footage Directory    : {footage_dir}")
    print(f"Ground Truth Dataset : {gt_file}")
    print(f"ReID Model           : {cfg.reid.model_name}")
    print(f"Match Threshold      : {cfg.reid.match_threshold:.3f}")
    print(f"Reacq Threshold      : {cfg.reid.reacquisition_threshold:.3f}")
    print(f"Frames to Evaluate   : {args.frames}")
    print(f"Seed Camera & Frame  : {args.seed_cam} @ Frame {args.seed_frame}")
    print("=" * 70)

    # 2. Build graph and load dataset
    graph, sources = build_camera_graph(footage_dir)
    dataset = GroundTruthDataset.load_from_file(gt_file)

    # 3. Initialize Shared Components
    reid_extractor = PyTorchReIDExtractor(model_name=cfg.reid.model_name)
    shared_detector = YOLODetector(
        model_name=cfg.detection.model_name,
        confidence_threshold=cfg.detection.confidence_threshold,
        iou_threshold=cfg.detection.iou_threshold,
    )

    harness = ReIDEvaluationHarness(
        graph=graph,
        config=cfg,
        camera_sources=sources,
        dataset=dataset,
        target_person_id=dataset.target_person_id,
        reid_extractor=reid_extractor,
        shared_detector=shared_detector,
    )

    t0 = time.time()
    last_print = [0]

    def on_progress(cur: int, tot: int) -> None:
        if cur - last_print[0] >= 50 or cur == tot:
            pct = (cur / tot * 100.0) if tot > 0 else 0.0
            print(f"Progress: [{cur}/{tot}] frames evaluated ({pct:.1f}%)...")
            last_print[0] = cur

    metrics = harness.run(
        seed_camera_id=args.seed_cam,
        seed_frame_id=args.seed_frame,
        max_frames=args.frames,
        progress_callback=on_progress,
    )
    elapsed = time.time() - t0

    fps = metrics.total_frames / elapsed if elapsed > 0 else 0.0

    print("\n" + metrics.summary_table())
    print(f"\nExecution Time: {elapsed:.2f}s ({fps:.1f} FPS multi-camera throughput)\n")

    # 4. Save JSON report
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "reid_model": cfg.reid.model_name,
            "match_threshold": cfg.reid.match_threshold,
            "reacquisition_threshold": cfg.reid.reacquisition_threshold,
            "auto_add_threshold": cfg.reid.auto_add_threshold,
            "lock_switch_margin": cfg.reid.lock_switch_margin,
            "min_margin": cfg.reid.min_margin,
            "uncertain_band_min": cfg.reid.uncertain_band_min,
            "uncertain_band_max": cfg.reid.uncertain_band_max,
        },
        "metrics": metrics.to_dict(),
        "elapsed_seconds": round(elapsed, 2),
        "evaluation_fps": round(fps, 1),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"Detailed JSON evaluation report written to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
