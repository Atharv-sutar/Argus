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
    CameraStatus,
    EdgeDirection,
    EdgeType,
    SourceType,
)
from src.multi_camera.camera_graph import CameraGraph
from src.playback.annotations import Annotation, AnnotationStore
from src.case.manager import CaseManager
from src.audit.logger import AuditLogger, AuditEventType
from src.case.evidence_exporter import EvidenceExporter
import threading
import os
import uuid
from datetime import datetime, timezone

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
                if sys.platform.startswith("win"):
                    time.sleep(0.2)
            except Exception:
                pass
    return None


def probe_local_webcams(
    start_index: int = 0,
    end_index: int = 4,
    pipeline: Any = None,
    timeout_per_index: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    Safe sequential probe of local webcam devices with timeout per index.
    Skips indices already in use by the live pipeline to prevent flickering.
    """
    if _SHUTDOWN_EVENT.is_set():
        return []

    cameras: List[Dict[str, Any]] = []
    t_start = time.time()
    logger.info(f"[TOPOLOGY/PROBE] Starting hardware camera probe (indices {start_index}..{end_index - 1})...")

    in_use_indices = set()
    if pipeline is not None and getattr(pipeline, "_nodes", None):
        for node in pipeline._nodes.values():
            if node.config.enabled and node.config.source_type.value == "webcam":
                try:
                    in_use_indices.add(int(node.config.source))
                except ValueError:
                    pass

    if cv2 is not None and not _SHUTDOWN_EVENT.is_set():
        for idx in range(start_index, end_index):
            if _SHUTDOWN_EVENT.is_set():
                break
            
            if idx in in_use_indices:
                logger.info(f"[TOPOLOGY/PROBE] Skipping index {idx} (already in use by active pipeline)")
                cameras.append({
                    "source": idx,
                    "name": f"Webcam {idx} (In Use)",
                    "source_type": "webcam",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "status": "in_use",
                })
                continue
                
            try:
                res = _probe_single_device(idx)
                if res is not None:
                    cameras.append(res)
            except Exception as e:
                logger.debug(f"[TOPOLOGY/PROBE] Error probing index {idx}: {e}")

        # Sort discovered cameras by source index
        cameras.sort(key=lambda c: str(c["source"]))

    total_elapsed = time.time() - t_start
    cam_names = [c["name"] for c in cameras]
    logger.info(f"[TOPOLOGY/PROBE] Probe finished in {total_elapsed:.2f}s. Discovered {len(cameras)} camera(s): {cam_names}")
    return cameras


class MappingAPIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for camera mapping UI static files and REST endpoints."""

    graph_file: Path = Path("configs/camera_graph.json")
    runtime_pipeline = None  # Optional MultiCameraPipeline reference
    graph_lock = threading.Lock()
    annotation_store = None
    case_manager = None
    export_status = {}

    def log_message(self, format: str, *args: Any) -> None:
        # Route standard HTTP access logs to debug file log
        logger.debug(f"[HTTP ACCESS] {self.address_string()} - {format % args}")

    def _is_authorized(self) -> bool:
        """Fix P-12: Enforce authorization for administrative endpoints."""
        if self.client_address[0] in ("127.0.0.1", "::1", "localhost"):
            return True
        expected_token = os.environ.get("ARGUS_ADMIN_TOKEN")
        if expected_token:
            auth_header = self.headers.get("Authorization")
            if auth_header == f"Bearer {expected_token}":
                return True
        return False

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
                start_idx = int(query.get("start", ["0"])[0])
                end_idx = int(query.get("end", ["4"])[0])
                logger.info(f"[TOPOLOGY/DISCOVERY] [GET /api/cameras/discover] Hardware scan initiated (indices {start_idx}-{end_idx}) from {self.address_string()}")
                webcams = probe_local_webcams(start_index=start_idx, end_index=end_idx, pipeline=self.runtime_pipeline)
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
                        with self.graph_lock:
                            graph = CameraGraph.load(self.graph_file)
                            data = graph.to_dict()
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
                        last_frame_id = None
                        last_changed_time = time.time()
                        while not _SHUTDOWN_EVENT.is_set() and (self.runtime_pipeline is None or getattr(self.runtime_pipeline, "is_running", True)):
                            frame_seq = None
                            frame_bytes = None
                            if self.runtime_pipeline is not None:
                                frame_seq = getattr(self.runtime_pipeline, "get_camera_frame_seq", lambda c: None)(cam_id)
                            
                            # Skip encoding and streaming if the frame hasn't changed
                            if frame_seq is not None and last_frame_id == frame_seq:
                                time.sleep(0.01)
                                continue

                            if self.runtime_pipeline is not None:
                                frame_bytes = self.runtime_pipeline.get_camera_frame_jpeg(cam_id, quality=75)

                            if frame_bytes is None:
                                if cv2 is not None and np is not None:
                                    blank = np.zeros((360, 640, 3), dtype=np.uint8)
                                    cv2.putText(blank, f"CONNECTING [{cam_id}]...", (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 242, 254), 1, cv2.LINE_AA)
                                    _, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
                                    frame_bytes = buf.tobytes()

                            if frame_bytes is not None:
                                # Staleness detection: using sequence number or length fallback
                                current_id = frame_seq if frame_seq is not None else len(frame_bytes)
                                if current_id != last_frame_id:
                                    last_frame_id = current_id
                                    last_changed_time = time.time()
                                elif time.time() - last_changed_time > 2.0:
                                    # Render STALE overlay
                                    if cv2 is not None and np is not None:
                                        blank = np.zeros((360, 640, 3), dtype=np.uint8)
                                        cv2.putText(blank, f"STALE - RECONNECTING [{cam_id}]", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                                        _, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
                                        frame_bytes = buf.tobytes()

                                header = (
                                    b"--frame\r\n"
                                    b"Content-Type: image/jpeg\r\n"
                                    b"Content-Length: " + str(len(frame_bytes)).encode("ascii") + b"\r\n\r\n"
                                )
                                self.wfile.write(header + frame_bytes + b"\r\n")
                                self.wfile.flush()
                            
                            # Minimal sleep to avoid CPU spinning if frame_seq isn't implemented
                            time.sleep(0.005)

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
                        with self.graph_lock:
                            graph = CameraGraph.load(self.graph_file)
                            data = graph.to_dict()
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

            elif path == "/api/settings":
                if self.runtime_pipeline is not None:
                    self._send_json(self.runtime_pipeline.config.to_dict())
                else:
                    self._send_json({"error": "Pipeline not running"}, status=HTTPStatus.SERVICE_UNAVAILABLE)
                return

            elif path == "/api/status":
                if self.runtime_pipeline is not None:
                    telemetry = self.runtime_pipeline.get_telemetry()
                    import dataclasses
                    status_data = dataclasses.asdict(telemetry)
                    self._send_json(status_data)
                else:
                    self._send_json({
                        "active_camera_id": None,
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
                        "fps": 0.0,
                        "gpu_memory_mb": 0.0,
                        "uptime_s": 0.0
                    })
                return

            elif path == "/api/undo/stack":
                if self.runtime_pipeline is not None and getattr(self.runtime_pipeline, "undo_stack", None):
                    self._send_json(self.runtime_pipeline.undo_stack.get_state())
                else:
                    self._send_json({"can_undo": False, "can_redo": False, "last_action_name": None, "next_redo_name": None})
                return

            elif path.startswith("/api/cases/") and path.endswith("/export/status"):
                case_id = path.split("/")[3]
                status = self.export_status.get(case_id, {"status": "none"})
                self._send_json(status)
                return

            elif path.startswith("/api/cases/") and path.endswith("/export/download"):
                case_id = path.split("/")[3]
                status = self.export_status.get(case_id, {})
                if status.get("status") == "completed" and status.get("zip_path"):
                    zip_path = status["zip_path"]
                    if os.path.exists(zip_path):
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/zip')
                        self.send_header('Content-Disposition', f'attachment; filename="EVIDENCE_{case_id}.zip"')
                        self.send_header('Content-Length', str(os.path.getsize(zip_path)))
                        self.end_headers()
                        with open(zip_path, 'rb') as f:
                            self.wfile.write(f.read())
                        return
                self.send_error(404, "Export not found or not ready")
                return

            elif path == "/api/cases":
                if self.case_manager:
                    cases = [c.to_dict() for c in self.case_manager.list_cases()]
                    self._send_json({"cases": cases, "active_case": self.case_manager.active_case.case_id if self.case_manager.active_case else None})
                else:
                    self._send_json({"cases": []})
                return

            elif path == "/api/audit/log":
                if hasattr(self, 'audit_logger') and self.audit_logger:
                    limit = int(query.get("limit", ["100"])[0])
                    offset = int(query.get("offset", ["0"])[0])
                    logs = self.audit_logger.get_logs(limit=limit, offset=offset)
                    self._send_json({"success": True, "logs": logs})
                else:
                    self._send_json({"success": False, "logs": []})
                return

            elif path == "/api/playback/stats":
                if hasattr(self, 'pipeline') and self.runtime_pipeline and getattr(self.runtime_pipeline, 'playback_controller', None):
                    stats = self.runtime_pipeline.playback_controller.stats
                    self._send_json({"success": True, "stats": stats, "mode": self.runtime_pipeline.playback_controller.mode})
                else:
                    self._send_json({"success": False, "error": "Playback controller not found"}, status=400)
                return

            elif path == "/api/annotations":
                if self.annotation_store:
                    cam_id = query.get("camera_id", [None])[0]
                    if cam_id:
                        annos = self.annotation_store.get_by_camera(cam_id)
                    else:
                        annos = self.annotation_store.get_all()
                    self._send_json([a.to_dict() for a in annos])
                else:
                    self._send_json([])
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

            elif path == "/api/stream/events":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                
                try:
                    while not _SHUTDOWN_EVENT.is_set() and (self.runtime_pipeline is None or getattr(self.runtime_pipeline, "is_running", True)):
                        if self.runtime_pipeline is not None:
                            progress = self.runtime_pipeline.get_search_progress()
                            statuses = {
                                cid: self.runtime_pipeline.get_camera_status(cid).value
                                for cid in self.runtime_pipeline.graph.all_camera_ids()
                                if self.runtime_pipeline.get_camera_status(cid)
                            }
                            gallery = self.runtime_pipeline.gallery
                            thumbnails = gallery.get_thumbnails(max_count=25)
                            
                            
                            pending_handoff = None
                            if getattr(self.runtime_pipeline, "target_state", "UNSELECTED") == "UNCERTAIN":
                                pending_handoff = {
                                    "camera_id": getattr(self.runtime_pipeline, "_pending_handoff_cam", None),
                                    "similarity": getattr(self.runtime_pipeline, "_pending_handoff_sim", 0.0)
                                }
                                
                            import dataclasses
                            telemetry = self.runtime_pipeline.get_telemetry()
                            event_data = dataclasses.asdict(telemetry)
                            
                            payload = json.dumps(event_data)
                            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                            self.wfile.flush()
                        
                        time.sleep(0.1)  # 10Hz telemetry updates
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError):
                    pass
                except Exception as e:
                    logger.debug(f"[SSE] Client disconnected: {e}")
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


    def do_PUT(self) -> None:
        if not self._is_authorized():
            self.send_error(HTTPStatus.FORBIDDEN, "Unauthorized")
            return
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8")) if body else {}

        if path.startswith("/api/cases/"):
            case_id = path.split("/")[-1]
            if self.case_manager:
                self.case_manager.delete_case(case_id)
                self._send_json({"success": True})
            return
            
        elif path.startswith("/api/annotations/"):
            anno_id = path.split("/")[-1]
            if self.annotation_store:
                ok = self.annotation_store.update(anno_id, payload.get("text", ""), payload.get("annotation_type"))
                self._send_json({"success": ok})
            else:
                self._send_json({"success": False}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        if not self._is_authorized():
            self.send_error(HTTPStatus.FORBIDDEN, "Unauthorized")
            return
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        if path.startswith("/api/cases/"):
            case_id = path.split("/")[-1]
            if self.case_manager:
                self.case_manager.delete_case(case_id)
                self._send_json({"success": True})
            return
            
        elif path.startswith("/api/annotations/"):
            anno_id = path.split("/")[-1]
            if self.annotation_store:
                ok = self.annotation_store.remove(anno_id)
                self._send_json({"success": ok})
            else:
                self._send_json({"success": False}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _audit(self, event_type: AuditEventType, details: str):
        if hasattr(self, 'audit_logger') and self.audit_logger:
            self.audit_logger.log(event_type=event_type, detail={"message": details, "user_id": "operator"})

    def do_POST(self) -> None:
        if not self._is_authorized():
            logger.warning(f"[SECURITY] Unauthorized POST request to {self.path} from {self.address_string()}")
            self.send_error(HTTPStatus.FORBIDDEN, "Unauthorized administrative access")
            return

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
                # Reload graph file if present so newly saved or enabled cameras are picked up
                if self.graph_file.is_file():
                    try:
                        with self.graph_lock:
                            fresh_graph = CameraGraph.load(self.graph_file)
                        self.runtime_pipeline.update_graph(fresh_graph)
                    except Exception as e:
                        logger.warning(f"[LIVE MATRIX] Could not reload graph before restart: {e}")
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

                with self.graph_lock:
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

        elif path == "/api/settings":
            if self.runtime_pipeline is not None:
                try:
                    # Update configuration fields based on payload
                    cfg = self.runtime_pipeline.config
                    
                    if "multi_camera" in payload and "search" in payload["multi_camera"]:
                        search_pl = payload["multi_camera"]["search"]
                        if "max_radius" in search_pl:
                            cfg.multi_camera.search.max_radius = int(search_pl["max_radius"])
                        if "total_recovery_timeout" in search_pl:
                            cfg.multi_camera.search.total_recovery_timeout = float(search_pl["total_recovery_timeout"])
                            
                    if "reid" in payload:
                        reid_pl = payload["reid"]
                        if "match_threshold" in reid_pl:
                            cfg.reid.match_threshold = float(reid_pl["match_threshold"])
                            
                    if "tracking" in payload:
                        track_pl = payload["tracking"]
                        if "track_buffer" in track_pl:
                            cfg.tracking.track_buffer = int(track_pl["track_buffer"])
                            
                    # Save to default.yaml
                    cfg.save("configs/default.yaml")
                    if hasattr(self.runtime_pipeline, "apply_config_update"):
                        self.runtime_pipeline.apply_config_update(cfg)
                    self._audit(AuditEventType.CONFIG_CHANGE, "Advanced settings updated")
                    self._send_json({"success": True, "message": "Settings updated"})
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).exception(f"[SETTINGS] Failed to update settings: {e}")
                    self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            else:
                self._send_json({"success": False, "error": "Pipeline not running"}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return

        elif path == "/api/undo":
            if self.runtime_pipeline is not None:
                res = self.runtime_pipeline.undo_last_action()
                if res:
                    self._audit(AuditEventType.UNDO, "Reverted last target action")
                    self._send_json({"success": True, "action": res})
                else:
                    self._send_json({"success": False, "error": "Nothing to undo"}, status=HTTPStatus.BAD_REQUEST)
            else:
                self._send_json({"success": False, "error": "Pipeline not active"}, status=HTTPStatus.BAD_REQUEST)
            return
            
        elif path == "/api/redo":
            if self.runtime_pipeline is not None:
                res = self.runtime_pipeline.redo_last_action()
                if res:
                    self._send_json({"success": True, "action": res})
                else:
                    self._send_json({"success": False, "error": "Nothing to redo"}, status=HTTPStatus.BAD_REQUEST)
            else:
                self._send_json({"success": False, "error": "Pipeline not active"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/cases":
            if self.case_manager:
                try:
                    case_id = payload.get("case_id")
                    if not case_id:
                        self._send_json({"success": False, "error": "case_id required"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    case = self.case_manager.create_case(
                        case_id=case_id,
                        mode=payload.get("mode", "live"),
                        operator=payload.get("operator", ""),
                        notes=payload.get("notes", "")
                    )
                    self._send_json({"success": True, "case": case.to_dict()})
                except Exception as e:
                    self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path.startswith("/api/cases/") and path.endswith("/open"):
            if self.case_manager:
                case_id = path.split("/")[-2]
                try:
                    # Switch cases
                    case = self.case_manager.open_case(case_id)
                    case_dir = self.case_manager.get_active_case_dir()
                    
                    # Update active paths
                    if self.annotation_store:
                        self.annotation_store.set_db_path(str(case_dir / "annotations.db"))
                        
                    if self.runtime_pipeline:
                        if hasattr(self.runtime_pipeline.identity_manager, "set_db_path"):
                            self.runtime_pipeline.identity_manager.set_db_path(str(case_dir / "gallery" / "entries.db"))
                            
                    self._send_json({"success": True, "case": case.to_dict()})
                except Exception as e:
                    self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            return


        elif path == "/api/playback/fast-scan/start":
            try:
                data = payload
                skip_zones = data.get("skip_zones", [])
                if hasattr(self, 'pipeline') and self.runtime_pipeline and getattr(self.runtime_pipeline, 'playback_controller', None):
                    self.runtime_pipeline.playback_controller.start_fast_scan(skip_zones)
                    self._send_json({"success": True})
                else:
                    self._send_json({"success": False, "error": "Playback controller not found"}, status=400)
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=500)
            return

        elif path == "/api/playback/fast-scan/pause":
            if hasattr(self, 'pipeline') and self.runtime_pipeline and getattr(self.runtime_pipeline, 'playback_controller', None):
                self.runtime_pipeline.playback_controller.mode = "playback"
                self.runtime_pipeline.playback_controller.pause()
                self._send_json({"success": True})
            else:
                self._send_json({"success": False, "error": "Playback controller not found"}, status=400)
            return

        elif path == "/api/playback/stats":
            if hasattr(self, 'pipeline') and self.runtime_pipeline and getattr(self.runtime_pipeline, 'playback_controller', None):
                stats = self.runtime_pipeline.playback_controller.stats
                self._send_json({"success": True, "stats": stats, "mode": self.runtime_pipeline.playback_controller.mode})
            else:
                self._send_json({"success": False, "error": "Playback controller not found"}, status=400)
            return

        elif path.startswith("/api/cases/") and path.endswith("/export"):
            case_id = path.split("/")[3]
            if self.case_manager:
                cm = self.case_manager
                try:
                    case = cm.open_case(case_id)  # just to get the metadata
                    self.export_status[case_id] = {"status": "running"}
                    
                    def run_export():
                        try:
                            exporter = EvidenceExporter(cm.cases_dir, "exports")
                            zip_path = exporter.export(case)
                            self.export_status[case_id] = {"status": "completed", "zip_path": zip_path}
                        except Exception as e:
                            self.export_status[case_id] = {"status": "error", "error": str(e)}
                            
                    threading.Thread(target=run_export, daemon=True).start()
                    self._send_json({"success": True, "status": "running"})
                except Exception as e:
                    self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path.startswith("/api/cases/") and path.endswith("/close"):
            if self.case_manager:
                self.case_manager.close_case()
                self._send_json({"success": True})
            return

        elif path == "/api/annotations":
            if self.annotation_store:
                try:
                    anno = Annotation(
                        annotation_id=str(uuid.uuid4()),
                        timestamp_ms=float(payload.get("timestamp_ms", 0.0)),
                        camera_id=payload.get("camera_id", ""),
                        text=payload.get("text", ""),
                        created_at=datetime.now(timezone.utc).isoformat(),
                        bbox=tuple(payload.get("bbox")) if payload.get("bbox") else None,
                        annotation_type=payload.get("annotation_type", "note")
                    )
                    self.annotation_store.add(anno)
                    self._send_json({"success": True, "annotation_id": anno.annotation_id})
                except Exception as e:
                    self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            else:
                self._send_json({"success": False}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        elif path.startswith("/api/annotations/") and self.command == "PUT":
            # Note: Do PUT logic in do_PUT typically, but if routing from POST or handling via PUT
            pass

        elif path == "/api/target/select":
            cam_id = payload.get("camera_id")
            track_id = payload.get("track_id")
            x = payload.get("x")
            y = payload.get("y")
            start_new = payload.get("start_new", False)
            logger.info(f"[TARGET] [POST /api/target/select] Target selection on camera '{cam_id}' (track_id={track_id}, x={x}, y={y}, start_new={start_new})")

            if self.runtime_pipeline is not None and cam_id:
                self.runtime_pipeline.set_active_camera(cam_id)
                selected_id = None
                
                is_active = self.runtime_pipeline.target_manager.is_active()
                is_correction = is_active and not start_new
                
                if track_id is not None:
                    if is_correction:
                        ok = self.runtime_pipeline.correct_target_by_id(cam_id, int(track_id))
                    else:
                        ok = self.runtime_pipeline.select_target_by_id(cam_id, int(track_id))
                    selected_id = int(track_id) if ok else None
                elif x is not None and y is not None:
                    if is_correction:
                        selected_id = self.runtime_pipeline.correct_target_on_camera(cam_id, float(x), float(y))
                    else:
                        selected_id = self.runtime_pipeline.select_target_on_camera(cam_id, float(x), float(y))

                logger.info(f"[TARGET] Target locked: ID={selected_id}, ActiveCam='{self.runtime_pipeline.active_camera_id}', Correction={is_correction}")
                if selected_id is not None:
                    if is_correction:
                        self._audit(AuditEventType.HUMAN_CORRECTION, f"Target corrected to ID={selected_id} on {cam_id}")
                    else:
                        self._audit(AuditEventType.TARGET_SELECTED, f"Target selected: ID={selected_id} on {cam_id}")
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
                if ok:
                    self._audit(AuditEventType.GALLERY_ADD, f"Manual sample captured from {cam_id}")
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
                self._audit(AuditEventType.GALLERY_CLEAR, "Target and gallery cleared")
                self._send_json({"success": True, "message": "Target cleared"})
            else:
                self._send_json({"success": False, "error": "Runtime pipeline not active"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/handoff/confirm":
            if self.runtime_pipeline is not None:
                self.runtime_pipeline.accept_handoff()
                self._audit(AuditEventType.HANDOFF_ACCEPTED, "Operator confirmed uncertain handoff")
                self._send_json({"success": True})
            else:
                self._send_json({"success": False}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/handoff/reject":
            if self.runtime_pipeline is not None:
                self.runtime_pipeline.reject_handoff()
                self._audit(AuditEventType.HANDOFF_REJECTED, "Operator rejected uncertain handoff")
                self._send_json({"success": True})
            else:
                self._send_json({"success": False}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/target/gallery/delete":
            entry_id = payload.get("entry_id")
            logger.info(f"[TARGET] [POST /api/target/gallery/delete] Deleting gallery entry '{entry_id}'")
            if self.runtime_pipeline is not None:
                if entry_id:
                    ok = self.runtime_pipeline.gallery.remove_entry(entry_id)
                    if ok:
                        self._audit(AuditEventType.GALLERY_REMOVE, f"Gallery entry deleted: {entry_id}")
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
        elif path == "/api/system/switch_source":
            if self.runtime_pipeline is None:
                self._send_json({"success": False, "error": "Runtime pipeline not active"}, status=HTTPStatus.BAD_REQUEST)
                return
            
            mode = payload.get("mode", "live")
            cam_id = payload.get("camera_id", "cam_0")
            
            source_val = 0
            source_type = "webcam"
            
            if mode == "video":
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    import sys
                    
                    # Create a hidden root window
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes('-topmost', True)
                    
                    file_path = filedialog.askopenfilename(
                        title="Select Pre-recorded Video File",
                        filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")]
                    )
                    root.destroy()
                    
                    if not file_path:
                        self._send_json({"success": False, "error": "No file selected"})
                        return
                        
                    source_val = file_path
                    source_type = "video_file"
                except Exception as e:
                    logger.exception(f"[SERVER] Failed to open file dialog: {e}")
                    self._send_json({"success": False, "error": f"File dialog failed: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
            
            try:
                with self.graph_lock:
                    if not self.graph_file.is_file():
                        self._send_json({"success": False, "error": "No graph file found"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                        return
                    
                    graph_dict = CameraGraph.load(self.graph_file).to_dict()
                    updated = False
                    for cam in graph_dict.get("cameras", []):
                        if cam.get("camera_id") == cam_id:
                            cam["source"] = source_val
                            cam["source_type"] = source_type
                            updated = True
                            break
                    
                    if not updated:
                        self._send_json({"success": False, "error": f"Camera '{cam_id}' not found in graph"}, status=HTTPStatus.BAD_REQUEST)
                        return
                        
                    new_graph = CameraGraph.from_dict(graph_dict)
                    new_graph.save(self.graph_file)
                    self.runtime_pipeline.update_graph(new_graph)
                    
                    logger.info(f"[SERVER] Switched '{cam_id}' source to {mode} ({source_val})")
                    self._send_json({"success": True, "mode": mode, "source": source_val})
                    
            except Exception as e:
                logger.exception(f"[SERVER] Failed to switch source: {e}")
                self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

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
                if str(node.config.source) == source_param or cid == source_param:
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
    MappingAPIHandler.annotation_store = AnnotationStore()
    MappingAPIHandler.case_manager = CaseManager("cases")
    
    # Initialize global audit logger
    audit_db = Path("cases") / "audit.db"
    audit_db.parent.mkdir(parents=True, exist_ok=True)
    MappingAPIHandler.audit_logger = AuditLogger(audit_db)

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
