"""Main application entry point for running the single-camera surveillance pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None

from src.camera.capture import OpenCVCamera, SyntheticCamera
from src.core.config import AppConfig
from src.detection.yolo_detector import YOLODetector
from src.pipeline.single_camera import SingleCameraPipeline
from src.tracking.byte_tracker import ByteTracker
from src.visualization.annotator import FrameAnnotator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("argus.app")


def build_pipeline(
    config: AppConfig,
    source_override: Optional[str] = None,
    device_override: Optional[str] = None,
    use_synthetic: bool = False,
) -> SingleCameraPipeline:
    """Constructs and wires all single-camera components according to configuration."""

    # 1. Camera
    if use_synthetic or str(source_override).lower() == "synthetic":
        logger.info("Initializing SyntheticCamera (headless simulation).")
        camera = SyntheticCamera(
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
        )
    else:
        src = source_override if source_override is not None else config.camera.source
        camera = OpenCVCamera(
            source=src,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
        )

    # 2. Detector
    dev = device_override if device_override is not None else config.inference.device
    detector = YOLODetector(
        model_name=config.detection.model_name,
        confidence_threshold=config.detection.confidence_threshold,
        iou_threshold=config.detection.iou_threshold,
        target_classes=config.detection.target_classes,
        device=dev,
        image_size=config.detection.image_size,
    )

    # 3. Tracker
    tracker = ByteTracker(
        track_thresh=config.tracking.track_thresh,
        match_thresh=config.tracking.match_thresh,
        track_buffer=config.tracking.track_buffer,
        min_box_area=config.tracking.min_box_area,
    )

    # 4. Annotator
    annotator = FrameAnnotator(
        draw_fps=config.visualization.draw_fps,
        draw_boxes=config.visualization.draw_boxes,
        draw_ids=config.visualization.draw_ids,
        box_thickness=config.visualization.box_thickness,
        font_scale=config.visualization.font_scale,
    )

    return SingleCameraPipeline(
        camera=camera,
        detector=detector,
        tracker=tracker,
        annotator=annotator,
        camera_id=config.camera.name,
    )


def run_app(
    config_path: str = "configs/default.yaml",
    source: Optional[str] = None,
    device: Optional[str] = None,
    no_gui: bool = False,
    synthetic: bool = False,
) -> None:
    """Loads configuration and runs the live pipeline loop."""
    config_file = Path(config_path)
    if config_file.is_file():
        config = AppConfig.from_yaml(config_file)
    else:
        logger.warning(f"Config file '{config_path}' not found. Using defaults.")
        config = AppConfig()

    pipeline = build_pipeline(
        config=config,
        source_override=source,
        device_override=device,
        use_synthetic=synthetic,
    )

    window_name = config.visualization.window_name
    show_window = config.visualization.show_window and not no_gui and (cv2 is not None)

    logger.info("Starting Argus single-camera surveillance pipeline. Press 'q' to quit.")

    try:
        for annotated_frame, track_result in pipeline.stream():
            if show_window:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or Esc
                    logger.info("Quit key pressed.")
                    break
            else:
                if track_result.count > 0:
                    logger.info(
                        f"Frame {track_result.frame_id}: Active Tracks = {track_result.count}"
                    )
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
    finally:
        pipeline.stop()
        if show_window:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Argus Real-Time Surveillance Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--source", type=str, default=None, help="Camera index, video file, or 'synthetic'")
    parser.add_argument("--device", type=str, default=None, help="Inference device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--no-gui", action="store_true", help="Run without graphical display window")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic frame generator")

    args = parser.parse_args()
    run_app(
        config_path=args.config,
        source=args.source,
        device=args.device,
        no_gui=args.no_gui,
        synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
