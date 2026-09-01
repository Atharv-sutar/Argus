"""Lightweight HTTP server and REST API for the interactive Camera Mapping UI."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import mimetypes
import os
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from src.core.multi_camera_types import (
    CameraEdgeConfig,
    CameraNodeConfig,
    EdgeDirection,
    EdgeType,
    SourceType,
)
from src.multi_camera.camera_graph import CameraGraph

logger = logging.getLogger("argus.ui_server")

STATIC_DIR = Path(__file__).parent / "static"

_SHUTDOWN_EVENT = threading.Event()


def is_shutdown_requested() -> bool:
    """Returns True if a system shutdown has been requested."""
    return _SHUTDOWN_EVENT.is_set()


def _probe_single_device(idx: int) -> Optional[Dict[str, Any]]:
    """Probes a single camera index using DirectShow (Windows) or native capture."""
    if cv2 is None or _SHUTDOWN_EVENT.is_set():
        return None
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    t0 = time.time()
    cap = None
    try:
        cap = cv2.VideoCapture(idx, backend)
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)

        if cap and cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            ret, frame = cap.read()
            cap.release()
            cap = None
            elapsed = time.time() - t0
            if ret and frame is not None:
                logger.info(f"[PROBE] Webcam index {idx}: SUCCESS ({w}x{h} @ {fps}fps, took {elapsed:.2f}s)")
                return {
                    "source": idx,
                    "name": f"Webcam {idx}",
                    "source_type": "webcam",
                    "width": w,
                    "height": h,
                    "fps": fps,
                    "status": "available",
                }
            else:
                logger.debug(f"[PROBE] Webcam index {idx}: opened but read() failed (took {elapsed:.2f}s)")
        else:
            elapsed = time.time() - t0
            logger.debug(f"[PROBE] Webcam index {idx}: could not open / device absent (took {elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        logger.debug(f"[PROBE] Webcam index {idx}: probe exception (took {elapsed:.2f}s): {e}")
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    return None


def probe_local_webcams(
    max_indices: int = 4,
    pipeline: Any = None,
    timeout_per_index: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    Safe sequential probe of local webcam devices with timeout per index.
    If a pipeline is active, safely pauses processing and releases camera handles first so that
    DirectShow on Windows can probe devices without access conflict, then restores pipeline cameras.
    """
    if _SHUTDOWN_EVENT.is_set():
        return []

    cameras: List[Dict[str, Any]] = []
    t_start = time.time()
    logger.info(f"[TOPOLOGY/PROBE] Starting hardware camera probe (indices 0..{max_indices - 1})...")

    # 1. If pipeline is running, safely pause processing and release cameras during probe
    was_pipeline_active = False
    if pipeline is not None and hasattr(pipeline, "pause_processing") and not _SHUTDOWN_EVENT.is_set():
        was_pipeline_active = True
        try:
            logger.info("[TOPOLOGY/PROBE] Pausing pipeline processing to release DirectShow camera handles...")
            pipeline.pause_processing()
            time.sleep(0.15)  # brief pause for OS to release DirectShow handles
        except Exception as e:
            logger.warning(f"[TOPOLOGY/PROBE] Error pausing pipeline for probe: {e}")

    if cv2 is not None and not _SHUTDOWN_EVENT.is_set():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for idx in range(max_indices):
                if _SHUTDOWN_EVENT.is_set():
                    break
                future = executor.submit(_probe_single_device, idx)
                try:
                    res = future.result(timeout=timeout_per_index)
                    if res is not None:
                        cameras.append(res)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"[TOPOLOGY/PROBE] Webcam index {idx} probe timed out after {timeout_per_index}s (skipped)")
                except Exception as e:
                    logger.debug(f"[TOPOLOGY/PROBE] Error probing index {idx}: {e}")

        # Sort discovered cameras by source index
        cameras.sort(key=lambda c: str(c["source"]))

    # 2. Restore pipeline cameras if they were active and shutdown not requested
    if was_pipeline_active and pipeline is not None and hasattr(pipeline, "resume_processing") and not _SHUTDOWN_EVENT.is_set():
        try:
            logger.info("[TOPOLOGY/PROBE] Resuming pipeline processing and restoring camera workers...")
            pipeline.resume_processing()
        except Exception as e:
            logger.warning(f"[TOPOLOGY/PROBE] Error resuming pipeline after probe: {e}")

    total_elapsed = time.time() - t_start
    cam_names = [c["name"] for c in cameras]
    logger.info(f"[TOPOLOGY/PROBE] Probe finished in {total_elapsed:.2f}s. Discovered {len(cameras)} camera(s): {cam_names}")
    return cameras


class MappingAPIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for camera mapping UI static files and REST endpoints."""

    graph_file: Path = Path("configs/camera_graph.json")
    runtime_pipeline = None  # Optional MultiCameraPipeline reference

    def log_message(self, format: str, *args: Any) -> None:
        # Route standard HTTP access logs to debug file log
        logger.debug(f"[HTTP ACCESS] {self.address_string()} - {format % args}")

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path) -> None:
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        try:
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path
            query = urllib.parse.parse_qs(parsed_url.query)

            # --- REST Endpoints ---
            if path == "/api/cameras/discover":
                logger.info(f"[TOPOLOGY/DISCOVERY] [GET /api/cameras/discover] Hardware scan initiated from {self.address_string()}")
                webcams = probe_local_webcams(pipeline=self.runtime_pipeline)
                logger.info(f"[TOPOLOGY/DISCOVERY] Returning {len(webcams)} discovered camera(s) to {self.address_string()}")
                self._send_json({"cameras": webcams})
                return

            elif path == "/api/cameras/live":
                if self.runtime_pipeline is not None:
                    cards = self.runtime_pipeline.get_all_camera_cards()
                    active_cam = self.runtime_pipeline.active_camera_id
                    card_summary = [f"{c['camera_id']}:{c['status']}(fps={c.get('fps',0):.1f},frame={c.get('has_frame')})" for c in cards]
                    logger.debug(f"[LIVE MATRIX] [GET /api/cameras/live] Returning {len(cards)} card(s), active='{active_cam}'. Details: {card_summary}")
                    self._send_json({"cameras": cards, "active_camera": active_cam})
                elif self.graph_file.is_file():
                    try:
                        with open(self.graph_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        cams = []
                        for c in data.get("cameras", []):
                            cams.append({
                                "camera_id": c.get("camera_id"),
                                "name": c.get("name"),
                                "source": c.get("source"),
                                "source_type": c.get("source_type", "webcam"),
                                "enabled": c.get("enabled", True),
                                "is_active": False,
                                "is_searching": False,
                                "status": "STANDBY",
                                "fps": 0.0,
                                "floor": c.get("floor"),
                                "zone": c.get("zone"),
                                "has_frame": False,
                            })
                        logger.debug(f"[LIVE MATRIX] [GET /api/cameras/live] Returning {len(cams)} static graph camera(s)")
                        self._send_json({"cameras": cams, "active_camera": None})
                    except Exception as e:
                        logger.warning(f"[LIVE MATRIX] [GET /api/cameras/live] Error loading fallback graph: {e}")
                        self._send_json({"cameras": [], "active_camera": None})
                else:
                    self._send_json({"cameras": [], "active_camera": None})
                return

            elif path.startswith("/api/camera/") and path.endswith("/stream"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[0] == "api" and parts[1] == "camera" and parts[3] == "stream":
                    cam_id = parts[2]
                    logger.info(f"[LIVE MATRIX] [STREAM START] Client {self.address_string()} connected to MJPEG stream for '{cam_id}'")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    try:
                        while not _SHUTDOWN_EVENT.is_set() and (self.runtime_pipeline is None or getattr(self.runtime_pipeline, "is_running", True)):
                            frame_bytes = None
                            if self.runtime_pipeline is not None:
                                frame_bytes = self.runtime_pipeline.get_camera_frame_jpeg(cam_id, quality=75)

                            if frame_bytes is None:
                                if cv2 is not None and np is not None:
                                    blank = np.zeros((360, 640, 3), dtype=np.uint8)
                                    cv2.putText(blank, f"CONNECTING [{cam_id}]...", (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 242, 254), 1, cv2.LINE_AA)
                                    _, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
                                    frame_bytes = buf.tobytes()

                            if frame_bytes is not None:
                                header = (
                                    b"--frame\r\n"
                                    b"Content-Type: image/jpeg\r\n"
                                    b"Content-Length: " + str(len(frame_bytes)).encode("ascii") + b"\r\n\r\n"
                                )
                                self.wfile.write(header + frame_bytes + b"\r\n")
                                self.wfile.flush()
                            time.sleep(0.033)  # ~30 FPS throttle
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError):
                        pass
                    except Exception as e:
                        logger.debug(f"[LIVE MATRIX] Stream client disconnected for camera '{cam_id}': {e}")
                    finally:
                        logger.info(f"[LIVE MATRIX] [STREAM END] Client {self.address_string()} disconnected from MJPEG stream for '{cam_id}'")
                    return

            elif path.startswith("/api/camera/") and (path.endswith("/frame.jpg") or path.endswith("/frame")):
                parts = path.strip("/").split("/")
                if len(parts) >= 3:
                    cam_id = parts[2]
                    frame_bytes = None
                    if self.runtime_pipeline is not None:
                        frame_bytes = self.runtime_pipeline.get_camera_frame_jpeg(cam_id, quality=75)

                    if frame_bytes is None:
                        if cv2 is not None and np is not None:
                            blank = np.zeros((360, 640, 3), dtype=np.uint8)
                            cv2.putText(blank, f"STANDBY [{cam_id}]", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 242, 254), 1, cv2.LINE_AA)
                            _, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
                            frame_bytes = buf.tobytes()

                    if frame_bytes is not None:
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame_bytes)))
                        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND, f"No frame available for camera '{cam_id}'")
                    return

            elif path == "/api/graph":
                logger.info(f"[TOPOLOGY] [GET /api/graph] Reading topology from '{self.graph_file.resolve()}' (exists={self.graph_file.is_file()})")
                if self.graph_file.is_file():
                    try:
                        with open(self.graph_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        cams = data.get("cameras", [])
                        edges = data.get("edges", [])
                        logger.info(f"[TOPOLOGY] Loaded topology: {len(cams)} cameras ({[c.get('camera_id') for c in cams]}), {len(edges)} edges")
                        self._send_json(data)
                    except Exception as e:
                        logger.exception(f"[TOPOLOGY] Error reading graph file '{self.graph_file}': {e}")
                        self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                else:
                    logger.info("[TOPOLOGY] No graph file found. Returning default empty topology.")
                    self._send_json({"version": 1, "cameras": [], "edges": [], "background_map": None})
                return

            elif path == "/api/status":
                if self.runtime_pipeline is not None:
                    progress = self.runtime_pipeline.get_search_progress()
                    statuses = {
                        cid: self.runtime_pipeline.get_camera_status(cid).value
                        for cid in self.runtime_pipeline.graph.all_camera_ids()
                        if self.runtime_pipeline.get_camera_status(cid)
                    }
                    gallery = self.runtime_pipeline.gallery
                    status_data = {
                        "active_camera": self.runtime_pipeline.active_camera_id,
                        "target_state": getattr(self.runtime_pipeline, "target_state", "UNSELECTED"),
                        "target_track_id": self.runtime_pipeline.target_manager.target.track_id if self.runtime_pipeline.target_manager.target else None,
                        "transit_history": getattr(self.runtime_pipeline, "transit_history", []),
                        "search_progress": progress.to_dict(),
                        "camera_statuses": statuses,
                        "candidate_scores": getattr(self.runtime_pipeline, "last_candidate_scores", {}),
                        "gallery_size": gallery.size,
                        "gallery_max": gallery.max_size,
                        "gallery_manual": gallery.manual_count,
                        "gallery_auto": gallery.auto_count,
                    }
                    logger.debug(f"[STATUS] [GET /api/status] active={status_data['active_camera']}, target_state={status_data['target_state']}, gallery={gallery.size}/{gallery.max_size}")
                    self._send_json(status_data)
                else:
                    self._send_json({
                        "active_camera": None,
                        "target_state": "UNSELECTED",
                        "target_track_id": None,
                        "transit_history": [],
                        "search_progress": None,
                        "camera_statuses": {},
                        "candidate_scores": {},
                        "gallery_size": 0,
                        "gallery_max": 25,
                        "gallery_manual": 0,
                        "gallery_auto": 0,
                    })
                return

            elif path == "/api/target/gallery":
                if self.runtime_pipeline is not None:
                    gallery = self.runtime_pipeline.gallery
                    thumbnails = gallery.get_thumbnails(max_count=25)
                    self._send_json({
                        "size": gallery.size,
                        "max_size": gallery.max_size,
                        "manual_count": gallery.manual_count,
                        "auto_count": gallery.auto_count,
                        "thumbnails": thumbnails,
                    })
                else:
                    self._send_json({
                        "size": 0,
                        "max_size": 25,
                        "manual_count": 0,
                        "auto_count": 0,
                        "thumbnails": [],
                    })
                return

            elif path == "/api/preview":
                source_param = query.get("source", ["0"])[0]
                source_type = query.get("type", ["webcam"])[0]
                logger.info(f"[TOPOLOGY/PREVIEW] [GET /api/preview] Snapshot requested for source='{source_param}', type='{source_type}' from {self.address_string()}")

                frame_bytes = self._capture_preview_jpeg(source_param, source_type)
                if frame_bytes is not None:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame_bytes)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                else:
                    logger.warning(f"[TOPOLOGY/PREVIEW] Preview capture failed for source='{source_param}'")
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Could not capture preview from source")
                return

            # --- Static File Serving ---
            if path == "/" or path == "":
                file_path = STATIC_DIR / "index.html"
            else:
                rel_path = path.lstrip("/")
                file_path = STATIC_DIR / rel_path

            self._send_file(file_path)
        except Exception as err:
            logger.exception(f"[HTTP GET ERROR] {err}")
            try:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(err))
            except Exception:
                pass
            return

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception as e:
            logger.warning(f"[HTTP POST] Could not parse JSON body from {self.address_string()}: {e}")
            payload = {}

        if path == "/api/camera/select_active":
            cam_id = payload.get("camera_id")
            logger.info(f"[LIVE MATRIX] [POST /api/camera/select_active] Switching active camera to '{cam_id}' requested from {self.address_string()}")
            if self.runtime_pipeline is not None and cam_id:
                ok = self.runtime_pipeline.set_active_camera(cam_id)
                logger.info(f"[LIVE MATRIX] Set active camera '{cam_id}' -> result={ok} (now active='{self.runtime_pipeline.active_camera_id}')")
                self._send_json({"success": ok, "active_camera": self.runtime_pipeline.active_camera_id})
            else:
                self._send_json({"success": False, "error": "Pipeline not active or invalid camera_id"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/cameras/restart":
            logger.info(f"[LIVE MATRIX] [POST /api/cameras/restart] Hardware camera restart requested from {self.address_string()}")
            if self.runtime_pipeline is not None:
                self.runtime_pipeline.restart_cameras()
                cards = self.runtime_pipeline.get_all_camera_cards()
                logger.info(f"[LIVE MATRIX] All cameras restarted. Active workers: {len(self.runtime_pipeline._workers)}, cards: {len(cards)}")
                self._send_json({
                    "success": True,
                    "message": "All cameras restarted successfully",
                    "cameras": cards,
                    "active_camera": self.runtime_pipeline.active_camera_id,
                })
            else:
                self._send_json({"success": False, "error": "Runtime pipeline not active"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/graph":
            logger.info(f"[TOPOLOGY] [POST /api/graph] Topology save requested from {self.address_string()} ({content_length} bytes)")
            try:
                cam_ids = [c.get("camera_id") for c in payload.get("cameras", [])]
                edges_count = len(payload.get("edges", []))
                logger.info(f"[TOPOLOGY] Received graph with {len(cam_ids)} cameras ({cam_ids}) and {edges_count} edges")
                logger.debug(f"[TOPOLOGY] Full payload JSON: {json.dumps(payload, indent=2)}")

                graph = CameraGraph.from_dict(payload)
                errors = graph.validate()
                if errors:
                    logger.warning(f"[TOPOLOGY] Graph validation failed: {errors}")
                    self._send_json({"success": False, "errors": errors}, status=HTTPStatus.BAD_REQUEST)
                    return

                graph.save(self.graph_file)
                logger.info(f"[TOPOLOGY] Graph saved successfully to '{self.graph_file.resolve()}'")

                # Dynamically sync running pipeline with updated topology graph
                if self.runtime_pipeline is not None:
                    logger.info("[TOPOLOGY] Updating live pipeline with new graph topology...")
                    self.runtime_pipeline.update_graph(graph)
                    logger.info(f"[TOPOLOGY] Live pipeline synced: {len(self.runtime_pipeline._nodes)} nodes configured, active camera='{self.runtime_pipeline.active_camera_id}'")

                self._send_json({"success": True, "message": "Graph saved and live pipeline updated successfully"})
            except Exception as e:
                logger.exception(f"[TOPOLOGY] Exception during graph save: {e}")
                self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        elif path == "/api/graph/validate":
            logger.info(f"[TOPOLOGY] [POST /api/graph/validate] Validating graph from {self.address_string()}...")
            try:
                graph = CameraGraph.from_dict(payload)
                errors = graph.validate()
                logger.info(f"[TOPOLOGY] Validation result: valid={len(errors) == 0}, errors={errors}")
                self._send_json({"valid": len(errors) == 0, "errors": errors})
            except Exception as e:
                logger.warning(f"[TOPOLOGY] Exception during validation: {e}")
                self._send_json({"valid": False, "errors": [str(e)]})
            return

        elif path == "/api/target/select":
            cam_id = payload.get("camera_id")
            track_id = payload.get("track_id")
            x = payload.get("x")
            y = payload.get("y")
            logger.info(f"[TARGET] [POST /api/target/select] Target selection on camera '{cam_id}' (track_id={track_id}, x={x}, y={y})")

            if self.runtime_pipeline is not None and cam_id:
                self.runtime_pipeline.set_active_camera(cam_id)
                selected_id = None
                if track_id is not None:
                    ok = self.runtime_pipeline.select_target_by_id(cam_id, int(track_id))
                    selected_id = int(track_id) if ok else None
                elif x is not None and y is not None:
                    selected_id = self.runtime_pipeline.select_target_on_camera(cam_id, float(x), float(y))

                logger.info(f"[TARGET] Target locked: ID={selected_id}, ActiveCam='{self.runtime_pipeline.active_camera_id}'")
                self._send_json({
                    "success": True,
                    "target_locked": (selected_id is not None),
                    "selected_id": selected_id,
                    "camera_id": cam_id,
                    "active_camera": self.runtime_pipeline.active_camera_id,
                })
            else:
                self._send_json({"success": False, "error": "Runtime pipeline not active or missing camera_id"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/target/add_sample":
            logger.info(f"[TARGET] [POST /api/target/add_sample] Manual sample capture requested from {self.address_string()}")
            if self.runtime_pipeline is not None:
                cam_id = payload.get("camera_id")
                ok = self.runtime_pipeline.add_manual_target_sample(cam_id)
                logger.info(f"[TARGET] Manual sample captured on '{cam_id}' -> success={ok}, gallery size={self.runtime_pipeline.gallery.size}")
                self._send_json({
                    "success": ok,
                    "size": self.runtime_pipeline.gallery.size,
                    "manual_count": self.runtime_pipeline.gallery.manual_count,
                    "auto_count": self.runtime_pipeline.gallery.auto_count,
                })
            else:
                self._send_json({"success": False, "error": "Runtime pipeline not active"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/target/clear":
            logger.info(f"[TARGET] [POST /api/target/clear] Target clear requested from {self.address_string()}")
            if self.runtime_pipeline is not None:
                self.runtime_pipeline.clear_target()
                self._send_json({"success": True, "message": "Target cleared"})
            else:
                self._send_json({"success": False, "error": "Runtime pipeline not active"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/target/gallery/delete":
            entry_id = payload.get("entry_id")
            logger.info(f"[TARGET] [POST /api/target/gallery/delete] Deleting gallery entry '{entry_id}'")
            if self.runtime_pipeline is not None:
                if entry_id:
                    ok = self.runtime_pipeline.gallery.remove_entry(entry_id)
                    self._send_json({
                        "success": ok,
                        "size": self.runtime_pipeline.gallery.size,
                        "manual_count": self.runtime_pipeline.gallery.manual_count,
                        "auto_count": self.runtime_pipeline.gallery.auto_count,
                    })
                else:
                    self._send_json({"success": False, "error": "Missing entry_id"}, status=HTTPStatus.BAD_REQUEST)
            else:
                self._send_json({"success": False, "error": "Runtime pipeline not active"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/system/quit":
            logger.info(f"[SERVER] Safe shutdown requested via /api/system/quit endpoint from {self.address_string()}")
            _SHUTDOWN_EVENT.set()
            try:
                self._send_json({"success": True, "message": "Argus Surveillance shutting down cleanly."})
            except Exception:
                pass

            def _instant_shutdown():
                time.sleep(0.08)
                if self.runtime_pipeline is not None:
                    try:
                        self.runtime_pipeline.stop()
                    except Exception as e:
                        logger.warning(f"Error stopping pipeline: {e}")
                logger.info("[SERVER] All pipeline workers and camera handles released. Process exiting cleanly.")
                time.sleep(0.05)
                if "PYTEST_CURRENT_TEST" not in os.environ:
                    os._exit(0)

            threading.Thread(target=_instant_shutdown, daemon=True, name="shutdown_thread").start()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _capture_preview_jpeg(self, source_param: str, source_type: str) -> Optional[bytes]:
        """Capture a single test frame as JPEG, reusing pipeline frame if available."""
        if source_type == "synthetic" or source_param == "synthetic":
            if cv2 is not None and np is not None:
                img = np.zeros((360, 640, 3), dtype=np.uint8)
                img[:] = (20, 24, 36)
                cv2.putText(img, "SYNTHETIC TEST CAMERA", (180, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 2, cv2.LINE_AA)
                cv2.rectangle(img, (220, 190), (420, 320), (0, 255, 128), 2)
                cv2.putText(img, "Target Simulator Active", (230, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return buf.tobytes()

        # Reuse existing pipeline capture if source is already managed by pipeline
        if self.runtime_pipeline is not None:
            for cid, node in getattr(self.runtime_pipeline, "_nodes", {}).items():
                if str(node.config.source) == str(source_param) or cid == source_param:
                    jpeg = self.runtime_pipeline.get_camera_frame_jpeg(cid)
                    if jpeg:
                        return jpeg

        if cv2 is None:
            return None

        src: Any = source_param
        if source_type == "webcam":
            try:
                src = int(source_param)
            except ValueError:
                src = 0

        try:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(src, backend)
            if not cap.isOpened():
                cap = cv2.VideoCapture(src, cv2.CAP_ANY)
            if not cap or not cap.isOpened():
                return None

            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return None

            h, w = frame.shape[:2]
            if w > 640:
                scale = 640.0 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                return None
            return buf.tobytes()
        except Exception:
            return None


class ArgusHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server with graceful client disconnection handling and daemon threads."""
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc_type, exc_val, _ = sys.exc_info()
        if exc_type in (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError):
            logger.debug(f"Client disconnected gracefully from {client_address}")
            return
        logger.debug(f"HTTP Server exception from {client_address}: {exc_val}")


def run_ui_server(
    port: int = 8765,
    graph_file: str = "configs/camera_graph.json",
    pipeline=None,
    block: bool = True,
) -> ArgusHTTPServer:
    """Starts the Camera Mapping UI server."""
    MappingAPIHandler.graph_file = Path(graph_file)
    MappingAPIHandler.runtime_pipeline = pipeline

    server = ArgusHTTPServer(("127.0.0.1", port), MappingAPIHandler)
    logger.info(f"Camera Mapping UI server running at http://127.0.0.1:{port}")

    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("UI server stopping...")
        finally:
            try:
                server.server_close()
            except Exception:
                pass
    else:
        def _serve_loop():
            try:
                server.serve_forever()
            except Exception:
                pass

        t = threading.Thread(target=_serve_loop, daemon=True, name="ui_server_thread")
        t.start()

    return server
