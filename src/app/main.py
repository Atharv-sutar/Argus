"""Main application entry point for running the unified Argus surveillance pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.core.types import Target, TargetState, TrackResult
from src.multi_camera.camera_graph import CameraGraph
from src.multi_camera.ui_server import run_ui_server
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("argus.app")


def _launch_browser_when_ready(port: int, delay: float = 0.8) -> None:
    """Opens the web browser to the given port in a background daemon thread."""
    def _open() -> None:
        time.sleep(delay)
        url = f"http://127.0.0.1:{port}"
        try:
            logger.info(f"Automatically opening browser at {url}")
            webbrowser.open(url)
        except Exception as e:
            logger.debug(f"Could not open browser automatically: {e}")

    threading.Thread(target=_open, daemon=True).start()


def _prompt_camera_source(
    graph_path: str = "configs/camera_graph.json",
    default_source: Any = 0,
    allow_interactive: bool = True,
) -> Any:
    """
    Prompts the user to select an available camera source if running interactively,
    or returns the configured default source without blocking.
    """
    if not allow_interactive or not sys.stdin.isatty():
        return default_source

    graph_cameras: List[Dict[str, Any]] = []
    try:
        g_file = Path(graph_path)
        if g_file.is_file():
            import json
            with open(g_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            graph_cameras = data.get("cameras", [])
    except Exception:
        pass

    options: List[Tuple[str, Any]] = []

    # Add cameras from graph
    for cam in graph_cameras:
        name = cam.get("name") or cam.get("camera_id")
        src = cam.get("source")
        cid = cam.get("camera_id")
        options.append((f"{name} ({cid}) [source: {src}]", src))

    # Add standard webcams 0 and 1 if not already present
    existing_srcs = {opt[1] for opt in options}
    if 0 not in existing_srcs and "0" not in existing_srcs:
        options.append(("Default Local Webcam (Device Index 0)", 0))
    if 1 not in existing_srcs and "1" not in existing_srcs:
        options.append(("Secondary Local Webcam (Device Index 1)", 1))

    # Add synthetic generator option
    options.append(("Synthetic Simulation Feed (Headless Test Generator)", "synthetic"))
    # Add custom path option
    options.append(("Custom Video File / RTSP Stream URL...", "CUSTOM"))

    print("\n" + "=" * 65)
    print(" ARGUS SURVEILLANCE — SELECT CAMERA SOURCE")
    print("=" * 65)
    for i, (label, _) in enumerate(options, 1):
        print(f"  [{i}] {label}")
    print("=" * 65)

    try:
        user_choice = input(f"Select camera [1-{len(options)}] (or press Enter for default '{default_source}'): ").strip()
        if not user_choice:
            return default_source

        if user_choice.isdigit():
            idx = int(user_choice) - 1
            if 0 <= idx < len(options):
                selected_src = options[idx][1]
                if selected_src == "CUSTOM":
                    custom_path = input("Enter video file path or RTSP stream URL: ").strip()
                    return custom_path if custom_path else default_source
                return selected_src

        return user_choice
    except (EOFError, KeyboardInterrupt):
        return default_source


def build_pipeline(
    config: AppConfig,
    source_override: Optional[str] = None,
    device_override: Optional[str] = None,
    use_synthetic: bool = False,
    initial_target_id: Optional[int] = None,
) -> MultiCameraPipeline:
    """
    Constructs a unified MultiCameraPipeline for a single camera (1-node CameraGraph).
    Eliminates code duplication between single and multi-camera modes.
    """
    if device_override is not None:
        config.inference.device = device_override

    src = source_override if source_override is not None else config.camera.source
    if use_synthetic or str(src).lower() == "synthetic":
        src_type = SourceType.SYNTHETIC
    elif str(src).startswith(("rtsp://", "http://", "https://")):
        src_type = SourceType.RTSP
    elif str(src).isdigit() or isinstance(src, int):
        src_type = SourceType.WEBCAM
    else:
        src_type = SourceType.VIDEO_FILE

    node_cfg = CameraNodeConfig(
        camera_id=config.camera.name or "cam_0",
        name="Primary Camera",
        source=src,
        source_type=src_type,
        enabled=True,
    )
    graph = CameraGraph()
    graph.add_node(node_cfg)

    pipeline = MultiCameraPipeline(
        graph=graph,
        config=config,
    )
    if initial_target_id is not None:
        pipeline.select_target_by_id(node_cfg.camera_id, initial_target_id)
    return pipeline


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


def _render_monitoring_grid(
    frames: Dict[str, np.ndarray],
    all_camera_ids: List[str],
    tile_w: int = 640,
    tile_h: int = 360,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Renders a security-camera monitoring grid from raw camera frames (no AI overlays).
    Returns (canvas, tile_maps) where tile_maps allows mapping click coords to camera IDs.
    """
    n_cams = len(all_camera_ids)
    if n_cams <= 1:
        cols, rows = 1, 1
    elif n_cams == 2:
        cols, rows = 2, 1
    elif n_cams <= 4:
        cols, rows = 2, 2
    elif n_cams <= 6:
        cols, rows = 3, 2
    else:
        cols = 3
        rows = int(np.ceil(n_cams / 3))

    header_h = 40
    grid_w = cols * tile_w
    grid_h = header_h + (rows * tile_h)

    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    canvas[:] = (12, 15, 23)

    tile_maps: List[Dict[str, Any]] = []

    for idx, cid in enumerate(all_camera_ids):
        r = idx // cols
        c = idx % cols
        x_off = c * tile_w
        y_off = header_h + (r * tile_h)

        frame = frames.get(cid)
        orig_w, orig_h = 640, 480

        if frame is not None:
            orig_h, orig_w = frame.shape[:2]
            tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        else:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            tile[:] = (20, 26, 38)
            cv2.putText(tile, f"{cid} [NO SIGNAL]", (tile_w // 2 - 80, tile_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 120, 140), 1, cv2.LINE_AA)

        # Camera label badge
        label = f"{cid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(tile, (4, 4), (12 + tw, 10 + th), (10, 14, 22), -1)
        cv2.rectangle(tile, (4, 4), (12 + tw, 10 + th), (60, 200, 140), 1)
        cv2.putText(tile, label, (8, 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 200, 140), 1, cv2.LINE_AA)

        # Subtle border
        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (40, 55, 70), 1)

        canvas[y_off:y_off + tile_h, x_off:x_off + tile_w] = tile
        tile_maps.append({
            "camera_id": cid, "x1": x_off, "y1": y_off,
            "x2": x_off + tile_w, "y2": y_off + tile_h,
            "scale_x": orig_w / float(tile_w), "scale_y": orig_h / float(tile_h),
        })

    # Header
    hud = f"ARGUS SURVEILLANCE  |  {n_cams} CAMERAS  |  MONITORING  |  Click a camera to focus"
    cv2.putText(canvas, hud, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 180), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, header_h - 2), (grid_w, header_h - 2), (0, 220, 180), 1)

    return canvas, tile_maps


def _render_search_grid(
    camera_results: Dict[str, Tuple[Optional[np.ndarray], Optional[Any], Optional[Any]]],
    active_cam_id: Optional[str],
    search_cam_ids: List[str],
    search_radius: int,
    handoff_cam_id: Optional[str] = None,
    tile_w: int = 640,
    tile_h: int = 360,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Renders the SEARCH_VIEW: active camera + adjacent search cameras.
    Returns (canvas, tile_maps).
    """
    cam_ids = []
    if active_cam_id:
        cam_ids.append(active_cam_id)
    for cid in search_cam_ids:
        if cid != active_cam_id:
            cam_ids.append(cid)

    n_cams = len(cam_ids)
    if n_cams <= 1:
        cols, rows = 1, 1
    elif n_cams == 2:
        cols, rows = 2, 1
    elif n_cams <= 4:
        cols, rows = 2, 2
    else:
        cols = 3
        rows = int(np.ceil(n_cams / 3))

    header_h = 40
    grid_w = cols * tile_w
    grid_h = header_h + (rows * tile_h)

    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    canvas[:] = (12, 15, 23)

    tile_maps: List[Dict[str, Any]] = []

    for idx, cid in enumerate(cam_ids):
        r = idx // cols
        c = idx % cols
        x_off = c * tile_w
        y_off = header_h + (r * tile_h)

        data = camera_results.get(cid)
        frame = data[0] if (data and data[0] is not None) else None
        orig_w, orig_h = 640, 480

        if frame is not None:
            orig_h, orig_w = frame.shape[:2]
            tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        else:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            tile[:] = (20, 26, 38)
            cv2.putText(tile, f"{cid} [NO SIGNAL]", (tile_w // 2 - 80, tile_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 120, 140), 1, cv2.LINE_AA)

        # Status-dependent styling
        is_active = (cid == active_cam_id)
        is_handoff = (cid == handoff_cam_id)

        if is_handoff:
            border_color = (0, 255, 0)  # Green - TARGET FOUND
            border_thick = 3
            tag = f"{cid} [TARGET FOUND]"
        elif is_active:
            border_color = (0, 100, 255)  # Orange-red - LOST
            border_thick = 3
            tag = f"{cid} [ACTIVE - TARGET LOST]"
        else:
            border_color = (0, 165, 255)  # Amber - SEARCHING
            border_thick = 2
            tag = f"{cid} [SEARCHING R{search_radius}]"

        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), border_color, border_thick)

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(tile, (4, 4), (12 + tw, 10 + th), (10, 14, 22), -1)
        cv2.rectangle(tile, (4, 4), (12 + tw, 10 + th), border_color, 1)
        cv2.putText(tile, tag, (8, 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1, cv2.LINE_AA)

        canvas[y_off:y_off + tile_h, x_off:x_off + tile_w] = tile
        tile_maps.append({
            "camera_id": cid, "x1": x_off, "y1": y_off,
            "x2": x_off + tile_w, "y2": y_off + tile_h,
            "scale_x": orig_w / float(tile_w), "scale_y": orig_h / float(tile_h),
        })

    # Header
    if handoff_cam_id:
        hud = f"ARGUS  |  TARGET FOUND ON {handoff_cam_id}  |  Confirming handoff..."
        hud_color = (0, 255, 0)
    else:
        hud = f"ARGUS  |  TARGET LOST  |  Searching adjacent cameras (Radius {search_radius})"
        hud_color = (0, 165, 255)
    cv2.putText(canvas, hud, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (0, header_h - 2), (grid_w, header_h - 2), hud_color, 1)

    return canvas, tile_maps


def run_multi_camera_app(
    config_path: str = "configs/default.yaml",
    graph_path: Optional[str] = None,
    no_gui: bool = True,
    serve_ui: bool = True,
    ui_port: int = 8765,
) -> None:
    """
    Runs the multi-camera surveillance system with the web operations center as primary interface.
    """
    config_file = Path(config_path)
    config = AppConfig.from_yaml(config_file) if config_file.is_file() else AppConfig()

    pipeline = build_multi_camera_pipeline(config, graph_path=graph_path)

    if serve_ui:
        run_ui_server(port=ui_port, graph_file=graph_path or config.multi_camera.graph_file, pipeline=pipeline, block=False)
        logger.info(f"Surveillance Operations Center web UI available at http://127.0.0.1:{ui_port}")
        _launch_browser_when_ready(ui_port)

    show_window = not no_gui and config.visualization.show_window and (cv2 is not None)
    window_name = "Argus Multi-Camera Surveillance"

    # --- UI State Machine ---
    ui_state = "MONITORING"   # MONITORING | CAMERA_FOCUS | TARGET_TRACKING | SEARCH_VIEW | HANDOFF_CONFIRM
    focused_camera_id: Optional[str] = None
    handoff_new_cam_id: Optional[str] = None
    current_tile_maps: List[Dict[str, Any]] = []
    handoff_confirm_delay = config.multi_camera.search.handoff_confirm_delay_s

    def on_mouse(event, x, y, flags, param):
        nonlocal ui_state, focused_camera_id
        if event == cv2.EVENT_LBUTTONDOWN:
            if ui_state == "MONITORING":
                # Click a camera tile → CAMERA_FOCUS
                for tile in current_tile_maps:
                    if tile["x1"] <= x <= tile["x2"] and tile["y1"] <= y <= tile["y2"]:
                        focused_camera_id = tile["camera_id"]
                        ui_state = "CAMERA_FOCUS"
                        logger.info(f"[UI] MONITORING → CAMERA_FOCUS on '{focused_camera_id}'")
                        break

            elif ui_state == "CAMERA_FOCUS":
                # Click on person → select target → TARGET_TRACKING
                if focused_camera_id:
                    selected_id = pipeline.select_target_on_camera(focused_camera_id, float(x), float(y))
                    if selected_id is not None:
                        ui_state = "TARGET_TRACKING"
                        logger.info(f"[UI] CAMERA_FOCUS → TARGET_TRACKING | Target selected on '{focused_camera_id}' Tracker={selected_id}")

            elif ui_state == "TARGET_TRACKING":
                # Re-select target on active camera
                active_id = pipeline.active_camera_id
                if active_id:
                    selected_id = pipeline.select_target_on_camera(active_id, float(x), float(y))
                    if selected_id is not None:
                        logger.info(f"[UI] Re-selected target on '{active_id}' Tracker={selected_id}")

            elif ui_state in ("SEARCH_VIEW", "HANDOFF_CONFIRM"):
                # Click on a search camera tile
                for tile in current_tile_maps:
                    if tile["x1"] <= x <= tile["x2"] and tile["y1"] <= y <= tile["y2"]:
                        cam_id = tile["camera_id"]
                        local_x = (x - tile["x1"]) * tile["scale_x"]
                        local_y = (y - tile["y1"]) * tile["scale_y"]
                        selected_id = pipeline.select_target_on_camera(cam_id, float(local_x), float(local_y))
                        if selected_id is not None:
                            ui_state = "TARGET_TRACKING"
                            logger.info(f"[UI] Manual target selection on '{cam_id}' → TARGET_TRACKING")
                        break

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click: Manual "add to gallery" capture
            if ui_state in ("TARGET_TRACKING", "CAMERA_FOCUS"):
                cam_id = focused_camera_id if ui_state == "CAMERA_FOCUS" else pipeline.active_camera_id
                if cam_id and pipeline.target_manager.is_active():
                    ok = pipeline.add_manual_target_sample(cam_id)
                    if ok:
                        logger.info(
                            f"[UI] Captured manual protected sample on '{cam_id}' "
                            f"(Gallery={pipeline.gallery.size}/{pipeline.gallery.max_size})"
                        )

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

    logger.info(f"Starting Argus Multi-Camera Surveillance. State: MONITORING")

    try:
        while True:
            if not show_window:
                pipeline.step()
                time.sleep(0.01)
                continue

            # ===== MONITORING STATE =====
            if ui_state == "MONITORING":
                frames = pipeline.read_monitoring_frames()
                all_cams = sorted(list(pipeline.graph.all_camera_ids()))
                if not all_cams:
                    all_cams = sorted(list(frames.keys())) if frames else []

                if all_cams:
                    canvas, current_tile_maps = _render_monitoring_grid(frames, all_cams)
                    cv2.imshow(window_name, canvas)
                else:
                    blank = np.zeros((400, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "No cameras configured", (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 120, 140), 1, cv2.LINE_AA)
                    cv2.imshow(window_name, blank)

            # ===== CAMERA_FOCUS STATE =====
            elif ui_state == "CAMERA_FOCUS":
                if focused_camera_id:
                    result = pipeline.process_single_camera_frame(focused_camera_id)
                    if result is not None:
                        ann_frame, track_res, target = result
                        h, w = ann_frame.shape[:2]
                        header_text = f"CAMERA: {focused_camera_id}  |  Left-Click: Select Target  |  [Esc]: Back to grid"
                        cv2.rectangle(ann_frame, (0, 0), (w, 32), (10, 14, 22), -1)
                        cv2.putText(ann_frame, header_text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 180), 1, cv2.LINE_AA)
                        cv2.line(ann_frame, (0, 32), (w, 32), (0, 220, 180), 1)
                        cv2.imshow(window_name, ann_frame)
                    else:
                        blank = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(blank, f"{focused_camera_id} - No frame", (160, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 120, 140), 1, cv2.LINE_AA)
                        cv2.imshow(window_name, blank)

            # ===== TARGET_TRACKING STATE =====
            elif ui_state == "TARGET_TRACKING":
                results = pipeline.step()
                active_id = pipeline.active_camera_id
                active_data = results.get(active_id) if active_id else None

                if active_data and active_data[0] is not None:
                    frame = active_data[0]
                    target = active_data[2]
                    h, w = frame.shape[:2]
                    state_str = target.state.value if target else "UNKNOWN"
                    g_size = pipeline.gallery.size
                    g_max = pipeline.gallery.max_size
                    g_man = pipeline.gallery.manual_count
                    g_auto = pipeline.gallery.auto_count
                    header_text = (
                        f"TRACKING | Camera: {active_id} | Target: [{state_str}] | "
                        f"Gallery: {g_size}/{g_max} (Man: {g_man}, Auto: {g_auto}) | "
                        f"[A/Right-Click]: Add Angle | [C]: Clear | [Q]: Quit"
                    )
                    cv2.rectangle(frame, (0, 0), (w, 32), (10, 14, 22), -1)
                    cv2.putText(frame, header_text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 1, cv2.LINE_AA)
                    cv2.line(frame, (0, 32), (w, 32), (0, 215, 255), 1)
                    cv2.imshow(window_name, frame)

                    # Check for target loss → transition to SEARCH_VIEW only if adjacent search cameras exist
                    if target and target.state in (TargetState.LOST, TargetState.UNCERTAIN):
                        progress = pipeline.get_search_progress()
                        if pipeline.search_manager.is_searching and len(progress.active_cameras) > 0:
                            ui_state = "SEARCH_VIEW"
                            logger.info(f"[UI] TARGET_TRACKING → SEARCH_VIEW (Target lost on '{active_id}')")

                # Check for handoff that happened during step()
                if pipeline.handoff_timestamp > 0 and ui_state == "TARGET_TRACKING":
                    pass

            # ===== SEARCH_VIEW STATE =====
            elif ui_state == "SEARCH_VIEW":
                results = pipeline.step()
                active_id = pipeline.active_camera_id
                progress = pipeline.get_search_progress()
                search_cam_ids = list(progress.active_cameras)

                canvas, current_tile_maps = _render_search_grid(
                    results, active_id, search_cam_ids, progress.search_radius,
                )
                cv2.imshow(window_name, canvas)

                # Check if handoff just happened
                if pipeline.handoff_timestamp > 0:
                    elapsed = time.time() - pipeline.handoff_timestamp
                    if elapsed < handoff_confirm_delay:
                        handoff_new_cam_id = pipeline.active_camera_id
                        ui_state = "HANDOFF_CONFIRM"
                        logger.info(f"[UI] SEARCH_VIEW → HANDOFF_CONFIRM (Target found on '{handoff_new_cam_id}')")

                # Check if search timed out or no search cameras → back to TARGET_TRACKING
                if not pipeline.search_manager.is_searching or len(search_cam_ids) == 0:
                    ui_state = "TARGET_TRACKING"
                    logger.info(f"[UI] SEARCH_VIEW → TARGET_TRACKING (search ended)")

            # ===== HANDOFF_CONFIRM STATE =====
            elif ui_state == "HANDOFF_CONFIRM":
                results = pipeline.step()
                active_id = pipeline.active_camera_id
                progress = pipeline.get_search_progress()
                search_cam_ids = list(progress.active_cameras)

                canvas, current_tile_maps = _render_search_grid(
                    results, active_id, search_cam_ids, progress.search_radius,
                    handoff_cam_id=handoff_new_cam_id,
                )
                cv2.imshow(window_name, canvas)

                # After confirmation delay, transition to TARGET_TRACKING
                elapsed = time.time() - pipeline.handoff_timestamp
                if elapsed >= handoff_confirm_delay:
                    ui_state = "TARGET_TRACKING"
                    handoff_new_cam_id = None
                    logger.info(f"[UI] HANDOFF_CONFIRM → TARGET_TRACKING (confirmed on '{active_id}')")

            # === Handle keyboard ===
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 27:  # Esc
                if ui_state == "CAMERA_FOCUS":
                    ui_state = "MONITORING"
                    focused_camera_id = None
                    logger.info("[UI] CAMERA_FOCUS → MONITORING (Esc)")
                elif ui_state in ("TARGET_TRACKING", "SEARCH_VIEW", "HANDOFF_CONFIRM"):
                    pipeline.clear_target()
                    ui_state = "MONITORING"
                    focused_camera_id = None
                    handoff_new_cam_id = None
                    logger.info("[UI] → MONITORING (Esc, target cleared)")
            elif key == ord("c"):
                pipeline.clear_target()
                ui_state = "MONITORING"
                focused_camera_id = None
                handoff_new_cam_id = None
                logger.info("[UI] Target cleared → MONITORING")
            elif key == ord("a") or key == 32:  # 'a' or Space: Manual Gallery Capture
                if pipeline.target_manager.is_active():
                    ok = pipeline.add_manual_target_sample()
                    if ok:
                        logger.info(
                            f"[UI] Manual sample added to gallery via hotkey "
                            f"(Size={pipeline.gallery.size}/{pipeline.gallery.max_size}, "
                            f"Manual={pipeline.gallery.manual_count}, Auto={pipeline.gallery.auto_count})"
                        )
            elif key == ord("m"):
                pipeline.clear_target()
                ui_state = "MONITORING"
                focused_camera_id = None
                handoff_new_cam_id = None
                logger.info("[UI] → MONITORING (m key)")

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
    _launch_browser_when_ready(port)
    run_ui_server(port=port, graph_file=graph_path, pipeline=None, block=True)


def run_app(
    config_path: str = "configs/default.yaml",
    source: Optional[str] = None,
    device: Optional[str] = None,
    no_gui: bool = False,
    synthetic: bool = False,
    target_id: Optional[int] = None,
) -> None:
    """Loads configuration and runs the live pipeline loop for a single camera."""
    config_file = Path(config_path)
    if config_file.is_file():
        config = AppConfig.from_yaml(config_file)
    else:
        logger.warning(f"Config file '{config_path}' not found. Using defaults.")
        config = AppConfig()

    # Prompt for source if not provided
    if source is None and not synthetic:
        source = _prompt_camera_source(
            graph_path=config.multi_camera.graph_file,
            default_source=config.camera.source,
            allow_interactive=not no_gui,
        )
        if str(source).lower() == "synthetic":
            synthetic = True
            source = None

    pipeline = build_pipeline(
        config=config,
        source_override=source,
        device_override=device,
        use_synthetic=synthetic,
        initial_target_id=target_id,
    )

    window_name = config.visualization.window_name
    show_window = config.visualization.show_window and not no_gui and (cv2 is not None)
    cam_id = pipeline.active_camera_id or "cam_0"

    # Mouse callback to allow clicking directly on targets in GUI window
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_id = pipeline.select_target_on_camera(cam_id, float(x), float(y))
            if selected_id is not None:
                logger.info(f"[USER ACTION] Clicked and locked onto Target ID: {selected_id} (Gallery seeded)")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if pipeline.target_manager.is_active():
                ok = pipeline.add_manual_target_sample(cam_id)
                if ok:
                    logger.info(
                        f"[USER ACTION] Right-click captured manual sample on '{cam_id}' "
                        f"(Gallery: {pipeline.gallery.size}/{pipeline.gallery.max_size})"
                    )

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

    logger.info(
        "Starting Argus pipeline. Controls: Left-Click: Select Target | "
        "Right-Click / 'a': Add Angle to Gallery | 'c': Clear | 'q': Quit"
    )

    try:
        for results in pipeline.stream():
            active_cam = pipeline.active_camera_id or cam_id
            data = results.get(active_cam)
            if data is None or data[0] is None:
                continue
            annotated_frame, track_result, target = data

            if show_window:
                h, w = annotated_frame.shape[:2]
                state_str = target.state.value if target else "UNSELECTED"
                g_size = pipeline.gallery.size
                g_max = pipeline.gallery.max_size
                g_man = pipeline.gallery.manual_count
                g_auto = pipeline.gallery.auto_count

                hud = (
                    f"Camera: {active_cam} | Target: [{state_str}] | "
                    f"Gallery: {g_size}/{g_max} (Man: {g_man}, Auto: {g_auto}) | "
                    f"[A/Right-Click]: Add Angle | [C]: Clear"
                )
                cv2.rectangle(annotated_frame, (0, 0), (w, 30), (10, 14, 22), -1)
                cv2.putText(annotated_frame, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 180), 1, cv2.LINE_AA)
                cv2.line(annotated_frame, (0, 30), (w, 30), (0, 220, 180), 1)

                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or Esc
                    logger.info("Quit key pressed.")
                    break
                elif key == ord("c"):
                    logger.info("[USER ACTION] 'c' key pressed: Deselecting target.")
                    pipeline.clear_target()
                elif key == ord("a") or key == 32:  # 'a' or Space
                    if pipeline.target_manager.is_active():
                        ok = pipeline.add_manual_target_sample(active_cam)
                        if ok:
                            logger.info(
                                f"[USER ACTION] 'a' hotkey captured manual sample on '{active_cam}' "
                                f"(Gallery: {pipeline.gallery.size}/{pipeline.gallery.max_size})"
                            )
            else:
                if track_result and (track_result.count > 0 or (target and target.track_id is not None)):
                    target_info = f", Target ID {target.track_id} [{target.state.value}]" if (target and target.track_id) else ""
                    logger.info(
                        f"Frame {track_result.frame_id}: Active Tracks = {track_result.count}{target_info} | "
                        f"Gallery: {pipeline.gallery.size}/{pipeline.gallery.max_size}"
                    )
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
    finally:
        pipeline.stop()
        if show_window:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Argus Real-Time Surveillance Operations Center")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--graph", type=str, default="configs/camera_graph.json", help="Path to camera topology graph JSON")
    parser.add_argument("--multi-camera", action="store_true", default=True, help="Run in multi-camera web surveillance mode (default)")
    parser.add_argument("--map-ui", action="store_true", help="Launch interactive web camera mapping UI")
    parser.add_argument("--ui-port", type=int, default=8765, help="Port for the surveillance web UI")
    parser.add_argument("--source", type=str, default=None, help="Camera index, video file, or 'synthetic'")
    parser.add_argument("--device", type=str, default=None, help="Inference device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--target-id", type=int, default=None, help="Pre-select specific Track ID as target")
    parser.add_argument("--legacy-desktop", action="store_true", help="[Deprecated / Debug Only] Launch legacy OpenCV desktop window")
    parser.add_argument("--no-gui", action="store_true", default=True, help="Run without legacy graphical display window")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic frame generator")

    args = parser.parse_args()

    if args.map_ui:
        run_map_ui(graph_path=args.graph, port=args.ui_port)
    elif args.legacy_desktop:
        logger.warning("The OpenCV desktop window is deprecated. Running in debug legacy mode.")
        run_app(
            config_path=args.config,
            source=args.source,
            device=args.device,
            no_gui=False,
            synthetic=args.synthetic,
            target_id=args.target_id,
        )
    else:
        # Default: Web-First Multi-Camera Surveillance Operations Center
        run_multi_camera_app(
            config_path=args.config,
            graph_path=args.graph,
            no_gui=True,
            serve_ui=True,
            ui_port=args.ui_port,
        )


if __name__ == "__main__":
    main()
