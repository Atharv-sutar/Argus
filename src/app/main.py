"""Main application entry point for running the single-camera surveillance pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None

from src.camera.capture import OpenCVCamera, SyntheticCamera
from src.core.config import AppConfig
from src.detection.yolo_detector import YOLODetector
from src.identity.manager import IdentityManager
from src.multi_camera.camera_graph import CameraGraph
from src.multi_camera.ui_server import run_ui_server
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.pipeline.single_camera import SingleCameraPipeline
from src.reid.extractor import PyTorchReIDExtractor
from src.target.manager import TargetManager
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
    initial_target_id: Optional[int] = None,
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

    # 4. Target Manager
    target_manager = TargetManager()
    if initial_target_id is not None:
        target_manager.select_by_track_id(initial_target_id)

    # 5. ReID & Identity Manager
    reid_extractor = PyTorchReIDExtractor(
        model_name=config.reid.model_name,
        device=dev,
    )
    identity_manager = IdentityManager(
        reid_extractor=reid_extractor,
        similarity_threshold=config.reid.similarity_threshold,
        min_margin=config.reid.min_margin,
        max_gallery_size=config.reid.gallery_size,
    )

    # 6. Annotator
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
        target_manager=target_manager,
        identity_manager=identity_manager,
        annotator=annotator,
        camera_id=config.camera.name,
        reid_interval=config.reid.extract_interval_frames,
        min_margin=config.reid.min_margin,
    )


def build_multi_camera_pipeline(
    config: AppConfig,
    graph_path: Optional[str] = None,
) -> MultiCameraPipeline:
    """Loads camera topology graph and constructs MultiCameraPipeline."""
    g_path = graph_path or config.multi_camera.graph_file
    graph = CameraGraph.load(g_path)
    if graph.node_count() == 0:
        logger.warning(
            f"Camera graph at '{g_path}' has 0 cameras configured. "
            f"Use '--map-ui' to configure cameras and connections first."
        )

    return MultiCameraPipeline(
        graph=graph,
        config=config,
    )


def run_multi_camera_app(
    config_path: str = "configs/default.yaml",
    graph_path: Optional[str] = None,
    no_gui: bool = False,
    serve_ui: bool = False,
    ui_port: int = 8765,
) -> None:
    """Runs the multi-camera surveillance system with topology-aware tracking."""
    config_file = Path(config_path)
    config = AppConfig.from_yaml(config_file) if config_file.is_file() else AppConfig()

    pipeline = build_multi_camera_pipeline(config, graph_path=graph_path)

    # Optionally run background UI server for live monitor
    if serve_ui:
        run_ui_server(port=ui_port, graph_file=graph_path or config.multi_camera.graph_file, pipeline=pipeline, block=False)
        logger.info(f"Live Monitor web UI available at http://127.0.0.1:{ui_port}")

    show_window = config.visualization.show_window and not no_gui and (cv2 is not None)
    window_name = "Argus Multi-Camera Surveillance"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and pipeline.active_camera_id:
            selected_id = pipeline.select_target_on_camera(pipeline.active_camera_id, float(x), float(y))
            if selected_id is not None:
                logger.info(f"[USER ACTION] Selected target on active camera '{pipeline.active_camera_id}' | Tracker={selected_id}")

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

    logger.info("Starting Multi-Camera Pipeline. Controls: Left-Click: Select Target | 'c': Clear Target | 'q': Quit")

    try:
        for camera_results in pipeline.stream():
            active_id = pipeline.active_camera_id
            active_data = camera_results.get(active_id) if active_id else None

            if show_window and active_data and active_data[0] is not None:
                frame_to_show = active_data[0]
                cv2.imshow(window_name, frame_to_show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("c"):
                    logger.info("[USER ACTION] 'c' key pressed: Deselecting target across all cameras.")
                    pipeline.clear_target()
            elif not show_window:
                time.sleep(0.03)
    except KeyboardInterrupt:
        logger.info("Multi-camera pipeline interrupted.")
    finally:
        pipeline.stop()
        if show_window:
            cv2.destroyAllWindows()


def run_map_ui(
    graph_path: str = "configs/camera_graph.json",
    port: int = 8765,
) -> None:
    """Launches the interactive Camera Mapping Web UI server."""
    logger.info(f"Opening Argus Camera Mapping UI at http://127.0.0.1:{port}")
    run_ui_server(port=port, graph_file=graph_path, pipeline=None, block=True)


def run_app(
    config_path: str = "configs/default.yaml",
    source: Optional[str] = None,
    device: Optional[str] = None,
    no_gui: bool = False,
    synthetic: bool = False,
    target_id: Optional[int] = None,
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
        initial_target_id=target_id,
    )

    window_name = config.visualization.window_name
    show_window = config.visualization.show_window and not no_gui and (cv2 is not None)

    # Mouse callback to allow clicking directly on targets in GUI window
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_id = pipeline.select_target_by_point(float(x), float(y))
            if selected_id is not None:
                logger.info(f"[USER ACTION] Clicked and locked onto Target ID: {selected_id}")

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

    logger.info("Starting Argus pipeline. Controls: Left-Click: Select Target | 'c': Clear Target | 'q': Quit")

    try:
        for annotated_frame, track_result, target in pipeline.stream():
            if show_window:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or Esc
                    logger.info("Quit key pressed.")
                    break
                elif key == ord("c"):
                    logger.info("[USER ACTION] 'c' key pressed: Deselecting target.")
                    pipeline.clear_target()
            else:
                if track_result.count > 0 or target.track_id is not None:
                    target_info = f", Target ID {target.track_id} [{target.state.value}]" if target.track_id else ""
                    logger.info(
                        f"Frame {track_result.frame_id}: Active Tracks = {track_result.count}{target_info}"
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
    parser.add_argument("--graph", type=str, default="configs/camera_graph.json", help="Path to camera topology graph JSON")
    parser.add_argument("--multi-camera", action="store_true", help="Run in multi-camera graph tracking mode")
    parser.add_argument("--map-ui", action="store_true", help="Launch interactive web camera mapping UI")
    parser.add_argument("--ui-port", type=int, default=8765, help="Port for the mapping web UI")
    parser.add_argument("--source", type=str, default=None, help="Single-camera index, video file, or 'synthetic'")
    parser.add_argument("--device", type=str, default=None, help="Inference device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--target-id", type=int, default=None, help="Pre-select specific Track ID as target")
    parser.add_argument("--no-gui", action="store_true", help="Run without graphical display window")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic frame generator")

    args = parser.parse_args()

    if args.map_ui:
        run_map_ui(graph_path=args.graph, port=args.ui_port)
    elif args.multi_camera:
        run_multi_camera_app(
            config_path=args.config,
            graph_path=args.graph,
            no_gui=args.no_gui,
            serve_ui=True,
            ui_port=args.ui_port,
        )
    else:
        run_app(
            config_path=args.config,
            source=args.source,
            device=args.device,
            no_gui=args.no_gui,
            synthetic=args.synthetic,
            target_id=args.target_id,
        )


if __name__ == "__main__":
    main()

