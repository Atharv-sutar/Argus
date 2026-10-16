"""Lightweight HTTP server and REST API for the interactive Camera Mapping UI."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import mimetypes
import os
import subprocess
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
from src.camera.directory_loader import auto_load_video_directory, is_video_directory
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


_PROBE_CODE = (
    "import cv2, json, os, sys\n"
    "os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'\n"
    "try:\n"
    "    if hasattr(cv2, 'setLogLevel'):\n"
    "        cv2.setLogLevel(0)\n"
    "    idx = int(sys.argv[1])\n"
    "    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith('win') else [cv2.CAP_ANY]\n"
    "    for backend in backends:\n"
    "        try:\n"
    "            cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)\n"
    "            if cap and cap.isOpened():\n"
    "                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640\n"
    "                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480\n"
    "                raw_fps = cap.get(cv2.CAP_PROP_FPS)\n"
    "                fps = int(raw_fps) if (raw_fps and raw_fps > 0) else 30\n"
    "                ret, frame = cap.read()\n"
    "                cap.release()\n"
    "                if ret and frame is not None and frame.size > 0:\n"
    "                    print(json.dumps({'source': idx, 'width': w, 'height': h, 'fps': fps}))\n"
    "                    sys.exit(0)\n"
    "            if cap:\n"
    "                cap.release()\n"
    "        except Exception:\n"
    "            pass\n"
    "except Exception:\n"
    "    pass\n"
    "sys.exit(1)\n"
)


_CACHED_CAMERA_NAMES: Optional[List[str]] = None
_CACHE_TIMESTAMP: float = 0.0


def _get_windows_camera_friendly_names() -> List[str]:
    """Retrieves hardware friendly names of connected cameras on Windows, ordered by USB physical then virtual."""
    global _CACHED_CAMERA_NAMES, _CACHE_TIMESTAMP
    now = time.time()
    if _CACHED_CAMERA_NAMES is not None and (now - _CACHE_TIMESTAMP) < 15.0:
        return _CACHED_CAMERA_NAMES

    if not sys.platform.startswith("win"):
        return []

    try:
        cmd = [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-PnpDevice -Class Camera | Select-Object FriendlyName, Status, InstanceId | ConvertTo-Json"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=4.0)
        if res.returncode == 0 and res.stdout.strip():
            raw = json.loads(res.stdout)
            if isinstance(raw, dict):
                raw = [raw]
            ok_devs = [d for d in raw if d.get("Status") == "OK" and d.get("FriendlyName")]
            # Sort physical USB devices first (matching DirectShow index 0..N), then virtual/network devices
            ok_devs.sort(key=lambda d: 0 if d.get("InstanceId", "").startswith("USB") else 1)
            names = [d["FriendlyName"].strip() for d in ok_devs]
            _CACHED_CAMERA_NAMES = names
            _CACHE_TIMESTAMP = now
            return names
    except Exception as e:
        logger.debug(f"[PROBE] Could not retrieve Windows camera friendly names: {e}")
    return []


def _probe_single_device(idx: int, timeout: float = 6.0, friendly_names: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Probes a single camera index in an isolated subprocess with strict timeout."""
    if _SHUTDOWN_EVENT.is_set():
        return None
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE, str(idx)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        if proc.returncode == 0 and proc.stdout.strip():
            data = None
            for line in proc.stdout.strip().splitlines():
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            if data is not None:
                w = data.get("width", 640)
                h = data.get("height", 480)
                fps = data.get("fps", 30)
                if friendly_names and idx < len(friendly_names):
                    cam_name = friendly_names[idx]
                else:
                    cam_name = f"Webcam {idx}"
                logger.info(f"[PROBE] Webcam index {idx} ('{cam_name}'): SUCCESS ({w}x{h} @ {fps}fps, took {elapsed:.2f}s)")
                return {
                    "source": idx,
                    "name": cam_name,
                    "source_type": "webcam",
                    "width": w,
                    "height": h,
                    "fps": fps,
                    "status": "available",
                }
            else:
                logger.debug(f"[PROBE] Webcam index {idx}: output was not valid JSON (took {elapsed:.2f}s): {proc.stdout.strip()}")
        else:
            logger.debug(f"[PROBE] Webcam index {idx}: device absent / unreadable (took {elapsed:.2f}s)")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        logger.debug(f"[PROBE] Webcam index {idx}: probe timed out after {elapsed:.2f}s (device busy or invalid)")
    except Exception as e:
        elapsed = time.time() - t0
        logger.debug(f"[PROBE] Webcam index {idx}: error (took {elapsed:.2f}s): {e}")
    return None


def probe_local_webcams(
    start_index: int = 0,
    end_index: int = 4,
    pipeline: Any = None,
    timeout_per_index: float = 6.0,
    max_indices: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Parallel probe of local webcam devices with strict per-device timeout in isolated subprocesses.
    Skips indices already in use by the live pipeline to prevent device lockups while retrieving their active properties.
    """
    if _SHUTDOWN_EVENT.is_set():
        return []

    if max_indices is not None:
        end_index = start_index + max_indices

    cameras: List[Dict[str, Any]] = []
    t_start = time.time()
    logger.info(f"[TOPOLOGY/PROBE] Starting hardware camera probe (indices {start_index}..{end_index - 1})...")

    friendly_names = _get_windows_camera_friendly_names()
    if friendly_names:
        logger.info(f"[TOPOLOGY/PROBE] Windows camera device inventory: {friendly_names}")

    in_use_indices = set()
    if pipeline is not None and getattr(pipeline, "_nodes", None):
        for node in pipeline._nodes.values():
            if getattr(node, "config", None) and getattr(node.config, "enabled", False):
                st = getattr(node.config, "source_type", None)
                st_val = st.value if hasattr(st, "value") else str(st).lower().strip()
                if st_val in ("webcam", "local", "usb"):
                    try:
                        in_use_indices.add(int(node.config.source))
                    except (ValueError, TypeError):
                        pass

    if not _SHUTDOWN_EVENT.is_set():
        indices_to_probe: List[int] = []
        for idx in range(start_index, end_index):
            if _SHUTDOWN_EVENT.is_set():
                break

            if idx in in_use_indices:
                logger.info(f"[TOPOLOGY/PROBE] Index {idx} is active in live pipeline")
                base_name = friendly_names[idx] if (friendly_names and idx < len(friendly_names)) else f"Webcam {idx}"
                w, h = 640, 480
                if pipeline is not None and getattr(pipeline, "_nodes", None):
                    for node in pipeline._nodes.values():
                        try:
                            if int(node.config.source) == idx:
                                if getattr(node.config, "name", None):
                                    base_name = node.config.name
                                if getattr(node.config, "width", None):
                                    w = node.config.width
                                if getattr(node.config, "height", None):
                                    h = node.config.height
                                break
                        except (ValueError, TypeError):
                            pass
                cameras.append({
                    "source": idx,
                    "name": f"{base_name} (Active)",
                    "source_type": "webcam",
                    "width": w,
                    "height": h,
                    "fps": 30,
                    "status": "active",
                })
            else:
                indices_to_probe.append(idx)

        if indices_to_probe:
            # On Windows, keep workers <= 2 to prevent COM/DirectShow hardware driver contention hangs
            worker_limit = 2 if sys.platform.startswith("win") else 4
            max_workers = min(len(indices_to_probe), worker_limit)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_probe_single_device, idx, timeout_per_index, friendly_names): idx
                    for idx in indices_to_probe
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    if _SHUTDOWN_EVENT.is_set():
                        break
                    try:
                        res = future.result()
                        if res is not None:
                            cameras.append(res)
                    except Exception as e:
                        logger.debug(f"[TOPOLOGY/PROBE] Error in probe future: {e}")

        # Sort discovered cameras by integer source index
        cameras.sort(key=lambda c: int(c["source"]) if isinstance(c["source"], int) or str(c["source"]).isdigit() else str(c["source"]))

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
                try:
                    start_idx = max(0, int(query.get("start", ["0"])[0]))
                    end_idx = max(start_idx + 1, int(query.get("end", ["4"])[0]))
                except (ValueError, TypeError):
                    start_idx = 0
                    end_idx = 4
                logger.info(f"[TOPOLOGY/DISCOVERY] [GET /api/cameras/discover] Hardware scan initiated (indices {start_idx}-{end_idx}) from {self.address_string()}")
                webcams = probe_local_webcams(start_index=start_idx, end_index=end_idx, pipeline=self.runtime_pipeline)
                logger.info(f"[TOPOLOGY/DISCOVERY] Returning {len(webcams)} discovered camera(s) to {self.address_string()}")
                self._send_json({"cameras": webcams})
                return

            elif path == "/api/settings/reid":
                if self.runtime_pipeline is not None:
                    current_model = self.runtime_pipeline.config.reid.model_name
                    recommended = self.runtime_pipeline.profile_hardware()
                    self._send_json({
                        "current_model": current_model,
                        "recommended": recommended,
                        "options": [
                            {"id": "osnet_x0_25", "name": "OSNet x0.25 (Fast/Low VRAM)", "recommended": recommended == "osnet_x0_25"},
                            {"id": "osnet_x1_0", "name": "OSNet x1.0 (High Compute/Best Accuracy)", "recommended": recommended == "osnet_x1_0"}
                        ]
                    })
                else:
                    self.send_error(503, "Pipeline not ready")
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
                            is_placeholder = False

                            if self.runtime_pipeline is not None:
                                frame_seq = getattr(self.runtime_pipeline, "get_camera_frame_seq", lambda c: None)(cam_id)
                            is_stale = (time.time() - last_changed_time > 2.0)
                            
                            # Skip encoding and streaming if the frame hasn't changed and is not stale
                            if frame_seq is not None and last_frame_id == frame_seq and not is_stale:
                                time.sleep(0.025)
                                continue

                            if self.runtime_pipeline is not None:
                                frame_bytes = self.runtime_pipeline.get_camera_frame_jpeg(cam_id, quality=75)

                            if frame_bytes is None:
                                is_placeholder = True
                                # If placeholder already sent recently, don't spam the socket
                                if last_frame_id == "placeholder" and not is_stale:
                                    time.sleep(0.5)
                                    continue

                                if cv2 is not None and np is not None:
                                    blank = np.zeros((360, 640, 3), dtype=np.uint8)
                                    blank[:] = (18, 22, 32)
                                    cv2.rectangle(blank, (0, 0), (639, 359), (45, 55, 75), 1)
                                    cv2.putText(blank, f"STANDBY [{cam_id}]", (190, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 242, 254), 1, cv2.LINE_AA)
                                    cv2.putText(blank, "Awaiting signal / Target tracking", (170, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 145, 165), 1, cv2.LINE_AA)
                                    _, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
                                    frame_bytes = buf.tobytes()

                            if frame_bytes is not None:
                                # Staleness detection: using sequence number or length fallback
                                current_id = frame_seq if frame_seq is not None else ("placeholder" if is_placeholder else len(frame_bytes))
                                if current_id != last_frame_id:
                                    last_frame_id = current_id
                                    last_changed_time = time.time()
                                elif frame_seq is not None and (time.time() - last_changed_time > 5.0):
                                    pipeline_running = getattr(self.runtime_pipeline, "is_running", True) if self.runtime_pipeline else True
                                    if cv2 is not None and np is not None and not pipeline_running:
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
                            
                            # Throttle: 30-40 FPS max for active stream, 2 FPS for idle/placeholder
                            time.sleep(0.5 if is_placeholder else 0.02)

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

            elif path == "/api/recording/stats":
                if self.runtime_pipeline is not None and getattr(self.runtime_pipeline, "recording_manager", None):
                    self._send_json(self.runtime_pipeline.recording_manager.get_storage_stats())
                else:
                    self._send_json({
                        "enabled": False,
                        "active_cameras": [],
                        "total_segments": 0,
                        "storage_used_bytes": 0,
                        "storage_used_mb": 0.0,
                        "storage_used_gb": 0.0,
                    })
                return

            elif path == "/api/recording/segments":
                if self.runtime_pipeline is not None and getattr(self.runtime_pipeline, "recording_manager", None):
                    camera_id = query.get("camera_id", [None])[0]
                    start_ms = float(query.get("start_ms", [0.0])[0]) if "start_ms" in query else None
                    end_ms = float(query.get("end_ms", [0.0])[0]) if "end_ms" in query else None
                    limit = int(query.get("limit", [50])[0])
                    segments = self.runtime_pipeline.recording_manager.index.query_segments(
                        camera_id=camera_id,
                        start_time_ms=start_ms,
                        end_time_ms=end_ms,
                        limit=limit,
                    )
                    self._send_json({"segments": segments})
                else:
                    self._send_json({"segments": []})
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
                if self.runtime_pipeline is not None and getattr(self.runtime_pipeline, 'playback_controller', None):
                    pc = self.runtime_pipeline.playback_controller
                    stats = dict(pc.stats)
                    stats["current_time_ms"] = pc.current_time_ms
                    stats["max_duration_ms"] = pc.max_duration_ms
                    stats["is_playing"] = pc.is_playing
                    stats["speed"] = pc.playback_speed
                    self._send_json({"success": True, "stats": stats, "mode": pc.mode})
                else:
                    self._send_json({"success": False, "error": "Playback controller not found"}, status=HTTPStatus.BAD_REQUEST)
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
                            thumbnails = gallery.get_thumbnails(max_count=25) if gallery else []
                            
                            pending_handoff = None
                            if getattr(self.runtime_pipeline, "target_state", "UNSELECTED") == "UNCERTAIN":
                                pending_handoff = {
                                    "camera_id": getattr(self.runtime_pipeline, "_pending_handoff_cam", None),
                                    "similarity": getattr(self.runtime_pipeline, "_pending_handoff_sim", 0.0)
                                }
                                
                            import dataclasses
                            telemetry = self.runtime_pipeline.get_telemetry()
                            event_data = dataclasses.asdict(telemetry)
                            if gallery is not None:
                                event_data["gallery"] = {
                                    "size": gallery.size,
                                    "max_size": gallery.max_size,
                                    "manual_count": gallery.manual_count,
                                    "auto_count": gallery.auto_count,
                                    "thumbnails": thumbnails,
                                }
                            event_data["pending_handoff"] = pending_handoff
                            event_data["topology_version"] = getattr(self.runtime_pipeline.graph, "topology_version", 1)
                            if getattr(self.runtime_pipeline, "playback_controller", None) is not None:
                                pc = self.runtime_pipeline.playback_controller
                                event_data["playback"] = {
                                    "current_time_ms": pc.current_time_ms,
                                    "max_duration_ms": pc.max_duration_ms,
                                    "is_playing": pc.is_playing,
                                    "speed": pc.playback_speed,
                                }
                            
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

            elif path == "/api/forensic/status":
                if self.runtime_pipeline is not None and hasattr(self.runtime_pipeline, 'forensic_recorder'):
                    self._send_json(self.runtime_pipeline.forensic_recorder.status)
                else:
                    self._send_json({"recording": False, "duration_s": 0, "frames": 0, "camera_id": ""})
                return

            elif path == "/api/forensic/download":
                file_path_str = query.get("path", [""])[0]
                if not file_path_str:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Missing 'path' query parameter")
                    return
                download_path = Path(file_path_str)
                if not download_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "Forensic clip file not found")
                    return
                try:
                    file_size = download_path.stat().st_size
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(file_size))
                    self.send_header("Content-Disposition", f'attachment; filename="{download_path.name}"')
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    with open(download_path, "rb") as f:
                        while True:
                            chunk = f.read(1024 * 1024)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                except Exception as e:
                    logger.error(f"[FORENSIC] Download error: {e}")
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

        if path == "/api/settings/reid":
            model_name = payload.get("model_name")
            logger.info(f"[SETTINGS] [POST /api/settings/reid] Changing ReID model to '{model_name}' requested from {self.address_string()}")
            if self.runtime_pipeline is not None and model_name:
                success = self.runtime_pipeline.set_reid_model(model_name=model_name, auto_rollback=True)
                if success:
                    self._send_json({"status": "success", "model": self.runtime_pipeline.config.reid.model_name})
                else:
                    self.send_error(500, "Failed to switch model")
            else:
                self.send_error(400, "Invalid request or pipeline not ready")
            return

        elif path == "/api/camera/select_active":
            cam_id = payload.get("camera_id")
            logger.info(f"[LIVE MATRIX] [POST /api/camera/select_active] Switching active camera to '{cam_id}' requested from {self.address_string()}")
            if self.runtime_pipeline is not None and cam_id:
                ok = self.runtime_pipeline.set_active_camera(cam_id)
                logger.info(f"[LIVE MATRIX] Set active camera '{cam_id}' -> result={ok} (now active='{self.runtime_pipeline.active_camera_id}')")
                self._send_json({"success": ok, "active_camera": self.runtime_pipeline.active_camera_id})
            else:
                self._send_json({"success": False, "error": "Pipeline not active or invalid camera_id"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/forensic/export":
            if self.runtime_pipeline is not None and hasattr(self.runtime_pipeline, 'forensic_recorder'):
                if not self.runtime_pipeline.forensic_recorder.is_recording:
                    self._send_json({"success": False, "error": "No forensic recording in progress"}, status=HTTPStatus.BAD_REQUEST)
                    return
                
                export_status_key = "forensic_export"
                self.export_status[export_status_key] = {"status": "running"}

                def run_forensic_export():
                    try:
                        result_path = self.runtime_pipeline.export_forensic_clip()
                        if result_path:
                            self.export_status[export_status_key] = {"status": "completed", "path": result_path}
                        else:
                            self.export_status[export_status_key] = {"status": "error", "error": "Export returned no path"}
                    except Exception as e:
                        logger.error(f"[FORENSIC] Export failed: {e}")
                        self.export_status[export_status_key] = {"status": "error", "error": str(e)}

                threading.Thread(target=run_forensic_export, daemon=True).start()
                self._send_json({"success": True, "status": "running"})
            else:
                self._send_json({"success": False, "error": "Pipeline not active"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/forensic/export/status":
            export_status_key = "forensic_export"
            status = self.export_status.get(export_status_key, {"status": "idle"})
            self._send_json(status)
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
                    
                    if "multi_camera" in payload:
                        mc_pl = payload["multi_camera"]
                        search_pl = mc_pl.get("search", mc_pl) if isinstance(mc_pl, dict) else {}
                        if "max_radius" in search_pl:
                            cfg.multi_camera.search.max_radius = int(search_pl["max_radius"])
                        elif "max_search_radius" in search_pl:
                            cfg.multi_camera.search.max_radius = int(search_pl["max_search_radius"])
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


        elif path == "/api/playback/play":
            if self.runtime_pipeline is not None and getattr(self.runtime_pipeline, 'playback_controller', None):
                self.runtime_pipeline.playback_controller.play()
                self._send_json({"success": True, "is_playing": True})
            else:
                self._send_json({"success": False, "error": "Playback controller not found"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/playback/pause":
            if self.runtime_pipeline is not None and getattr(self.runtime_pipeline, 'playback_controller', None):
                self.runtime_pipeline.playback_controller.pause()
                self._send_json({"success": True, "is_playing": False})
            else:
                self._send_json({"success": False, "error": "Playback controller not found"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/playback/toggle":
            if self.runtime_pipeline is not None and getattr(self.runtime_pipeline, 'playback_controller', None):
                pc = self.runtime_pipeline.playback_controller
                pc.toggle_play_pause()
                self._send_json({"success": True, "is_playing": pc.is_playing})
            else:
                self._send_json({"success": False, "error": "Playback controller not found"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/playback/speed":
            speed = float(payload.get("speed", 1.0))
            if self.runtime_pipeline is not None and getattr(self.runtime_pipeline, 'playback_controller', None):
                pc = self.runtime_pipeline.playback_controller
                pc.set_speed(speed)
                self._send_json({"success": True, "speed": pc.playback_speed})
            else:
                self._send_json({"success": False, "error": "Playback controller not found"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/playback/seek":
            if self.runtime_pipeline is not None and getattr(self.runtime_pipeline, 'playback_controller', None):
                pc = self.runtime_pipeline.playback_controller
                if "time_ms" in payload:
                    time_ms = float(payload["time_ms"])
                elif "ratio" in payload:
                    time_ms = float(payload["ratio"]) * pc.max_duration_ms
                else:
                    time_ms = 0.0
                pc.seek(time_ms)
                self._send_json({"success": True, "current_time_ms": pc.current_time_ms})
            else:
                self._send_json({"success": False, "error": "Playback controller not found"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/playback/fast-scan/start":
            try:
                data = payload
                skip_zones = data.get("skip_zones", [])
                if self.runtime_pipeline is not None and getattr(self.runtime_pipeline, 'playback_controller', None):
                    self.runtime_pipeline.playback_controller.start_fast_scan(skip_zones)
                    self._send_json({"success": True})
                else:
                    self._send_json({"success": False, "error": "Playback controller not found"}, status=HTTPStatus.BAD_REQUEST)
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        elif path == "/api/playback/fast-scan/pause":
            if self.runtime_pipeline is not None and getattr(self.runtime_pipeline, 'playback_controller', None):
                self.runtime_pipeline.playback_controller.mode = "playback"
                self.runtime_pipeline.playback_controller.pause()
                self._send_json({"success": True})
            else:
                self._send_json({"success": False, "error": "Playback controller not found"}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/playback/stats":
            if self.runtime_pipeline is not None and getattr(self.runtime_pipeline, 'playback_controller', None):
                pc = self.runtime_pipeline.playback_controller
                stats = dict(pc.stats)
                stats["current_time_ms"] = pc.current_time_ms
                stats["max_duration_ms"] = pc.max_duration_ms
                stats["is_playing"] = pc.is_playing
                stats["speed"] = pc.playback_speed
                self._send_json({"success": True, "stats": stats, "mode": pc.mode})
            else:
                self._send_json({"success": False, "error": "Playback controller not found"}, status=HTTPStatus.BAD_REQUEST)
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
                            case_dir = Path(cm.cases_dir) / case_id
                            
                            # 1. Save route.json
                            if hasattr(self, 'runtime_pipeline') and self.runtime_pipeline and self.runtime_pipeline.route_recorder:
                                self.runtime_pipeline.route_recorder.save_json(str(case_dir / "route.json"))
                                
                                # 2. Stitch journey video
                                from src.playback.journey_stitcher import JourneyStitcher
                                video_sources = {}
                                for cid, node in self.runtime_pipeline._nodes.items():
                                    if node.config.source_type in ("video", "video_file"):
                                        video_sources[cid] = node.config.source
                                
                                events = self.runtime_pipeline.route_recorder.get_events()
                                out_path = str(case_dir / f"journey_{case_id}.mp4")
                                stitched = False

                                final_ts = None
                                if getattr(self.runtime_pipeline, 'playback_controller', None):
                                    final_ts = self.runtime_pipeline.playback_controller.current_time_ms

                                if video_sources and events:
                                    stitcher = JourneyStitcher(video_sources, output_path=out_path)
                                    stitched = stitcher.stitch_journey(events, final_time_ms=final_ts)

                                # If stitcher produced no output (e.g. webcams/live feeds), fall back to forensic clip recorder
                                if not stitched and hasattr(self.runtime_pipeline, 'forensic_recorder') and self.runtime_pipeline.forensic_recorder.is_recording:
                                    try:
                                        self.runtime_pipeline.export_forensic_clip(out_path)
                                    except Exception as ex:
                                        logger.warning(f"Forensic clip fallback export failed: {ex}")
                                
                                # 3. Generate Report
                                from src.playback.report import RouteReporter
                                reporter = RouteReporter(output_path=str(case_dir / "report.txt"))
                                reporter.generate_report(events)
                            
                            # 4. Export
                            exporter = EvidenceExporter(cm.cases_dir, "exports")
                            zip_path = exporter.export(case)
                            self.export_status[case_id] = {"status": "completed", "zip_path": zip_path}
                        except Exception as e:
                            logger.error(f"Export failed: {e}")
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
                elif (x is not None and y is not None) or ("norm_x" in payload and "norm_y" in payload):
                    norm_x = payload.get("norm_x")
                    norm_y = payload.get("norm_y")
                    norm_x = float(norm_x) if norm_x is not None else None
                    norm_y = float(norm_y) if norm_y is not None else None
                    px_x = float(x) if x is not None else 0.0
                    px_y = float(y) if y is not None else 0.0
                    if is_correction:
                        selected_id = self.runtime_pipeline.correct_target_on_camera(cam_id, px_x, px_y, norm_x=norm_x, norm_y=norm_y)
                    else:
                        selected_id = self.runtime_pipeline.select_target_on_camera(cam_id, px_x, px_y, norm_x=norm_x, norm_y=norm_y)

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
        elif path == "/api/cameras/load_directory":
            dir_path = payload.get("directory") or payload.get("path")
            transit_s = float(payload.get("transit_time_s", 5.0))
            if not dir_path:
                self._send_json({"success": False, "error": "Missing 'directory' parameter in payload"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                new_graph = auto_load_video_directory(dir_path, default_transit_time_s=transit_s)
                with self.graph_lock:
                    new_graph.save(self.graph_file)
                if self.runtime_pipeline is not None:
                    self.runtime_pipeline.update_graph(new_graph)
                    cams = sorted(list(new_graph.all_camera_ids()))
                    if cams:
                        self.runtime_pipeline.set_active_camera(cams[0])
                all_cams = sorted(list(new_graph.all_camera_ids()))
                self._audit(AuditEventType.CONFIG_CHANGE, f"Loaded video directory: {dir_path} ({len(all_cams)} cameras)")
                logger.info(f"[SERVER] Auto-loaded video directory '{dir_path}' with {len(all_cams)} cameras: {all_cams}")
                self._send_json({
                    "success": True,
                    "message": f"Successfully loaded {len(all_cams)} camera video(s)",
                    "count": len(all_cams),
                    "cameras": all_cams,
                    "active_camera": self.runtime_pipeline.active_camera_id if self.runtime_pipeline else (all_cams[0] if all_cams else None),
                })
            except Exception as e:
                logger.exception(f"[SERVER] Error loading video directory '{dir_path}': {e}")
                self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/cameras/apply_discovered":
            raw_cams = payload.get("cameras")
            if not raw_cams:
                raw_cams = probe_local_webcams(pipeline=self.runtime_pipeline)
            if not raw_cams:
                self._send_json({"success": False, "error": "No cameras discovered to apply"}, status=HTTPStatus.BAD_REQUEST)
                return

            try:
                new_graph = CameraGraph()
                added_cams = []
                for idx, c in enumerate(raw_cams):
                    raw_src = c.get("source", idx)
                    try:
                        src_val = int(raw_src)
                    except (ValueError, TypeError):
                        src_val = raw_src

                    clean_name = str(c.get("name", f"Camera {idx}")).replace(" (Active)", "").replace(" (In Use)", "").replace(" (Offline / In Use)", "").strip()
                    cid = f"cam_{idx}"
                    node_cfg = CameraNodeConfig(
                        camera_id=cid,
                        name=clean_name,
                        source=src_val,
                        source_type=SourceType.WEBCAM if isinstance(src_val, int) else SourceType.RTSP,
                        enabled=True,
                        width=c.get("width", 1280),
                        height=c.get("height", 720),
                        fps=c.get("fps", 30),
                        position_x=220.0 + (idx * 260.0),
                        position_y=160.0,
                    )
                    new_graph.add_node(node_cfg)
                    added_cams.append(cid)

                # Connect all added cameras bidirectionally
                if len(added_cams) >= 2:
                    for i in range(len(added_cams)):
                        for j in range(i + 1, len(added_cams)):
                            new_graph.add_edge(added_cams[i], added_cams[j], bidirectional=True)

                with self.graph_lock:
                    new_graph.save(self.graph_file)

                if self.runtime_pipeline is not None:
                    self.runtime_pipeline.update_graph(new_graph)
                    if added_cams:
                        self.runtime_pipeline.set_active_camera(added_cams[0])

                self._audit(AuditEventType.CONFIG_CHANGE, f"Applied {len(added_cams)} discovered hardware cameras to live topology")
                logger.info(f"[SERVER] Applied {len(added_cams)} discovered cameras to live topology: {added_cams}")
                self._send_json({
                    "success": True,
                    "applied": len(added_cams),
                    "cameras": added_cams,
                    "active_camera": added_cams[0] if added_cams else None
                })
            except Exception as e:
                logger.exception(f"[SERVER] Failed to apply discovered cameras: {e}")
                self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
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
                file_path = payload.get("source")
                if not file_path:
                    self._send_json({"success": False, "error": "No file path provided"})
                    return
                    
                source_val = file_path
                source_type = "video_file" if not str(file_path).startswith(("rtsp://", "http://")) else "rtsp"
            
            try:
                if mode == "video" and Path(file_path).is_dir():
                    new_graph = auto_load_video_directory(file_path)
                    with self.graph_lock:
                        new_graph.save(self.graph_file)
                    self.runtime_pipeline.update_graph(new_graph)
                    all_cams = sorted(list(new_graph.all_camera_ids()))
                    if all_cams:
                        self.runtime_pipeline.set_active_camera(all_cams[0])
                    logger.info(f"[SERVER] Switched system to video directory {file_path} ({len(all_cams)} cameras)")
                    self._send_json({"success": True, "mode": mode, "updated": all_cams, "count": len(all_cams)})
                    return

                if mode == "live":
                    # Check if graph currently has video file cameras that should be replaced with live webcams
                    with self.graph_lock:
                        curr_graph = CameraGraph.load(self.graph_file) if self.graph_file.is_file() else None
                    has_video_nodes = False
                    if curr_graph:
                        for n in curr_graph.nodes.values():
                            if n.source_type == SourceType.VIDEO_FILE or str(n.source).lower().endswith(('.mp4', '.avi', '.mkv')):
                                has_video_nodes = True
                                break
                    if has_video_nodes:
                        # Cleanly purge video files and reinstate live hardware webcams
                        probed = probe_local_webcams(pipeline=self.runtime_pipeline)
                        if probed:
                            new_graph = CameraGraph()
                            for idx, c in enumerate(probed):
                                cid = f"cam_{idx}"
                                clean_name = str(c.get("name", f"Webcam {idx}")).replace(" (Active)", "").replace(" (In Use)", "").strip()
                                new_graph.add_node(CameraNodeConfig(
                                    camera_id=cid,
                                    name=clean_name,
                                    source=int(c.get("source", idx)),
                                    source_type=SourceType.WEBCAM,
                                    enabled=True,
                                    width=c.get("width", 1280),
                                    height=c.get("height", 720),
                                    fps=c.get("fps", 30),
                                    position_x=220.0 + (idx * 260.0),
                                    position_y=160.0,
                                ))
                            if new_graph.node_count() >= 2:
                                ids = list(new_graph.all_camera_ids())
                                for i in range(len(ids)):
                                    for j in range(i + 1, len(ids)):
                                        new_graph.add_edge(ids[i], ids[j], bidirectional=True)
                            with self.graph_lock:
                                new_graph.save(self.graph_file)
                            self.runtime_pipeline.update_graph(new_graph)
                            all_cams = list(new_graph.all_camera_ids())
                            if all_cams:
                                self.runtime_pipeline.set_active_camera(all_cams[0])
                            self._send_json({"success": True, "mode": "live", "updated": all_cams})
                            return

                with self.graph_lock:
                    if not self.graph_file.is_file():
                        self._send_json({"success": False, "error": "No graph file found"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                        return
                    
                    graph_dict = CameraGraph.load(self.graph_file).to_dict()
                    updated_cameras = []
                    
                    for cam in graph_dict.get("cameras", []):
                        if cam.get("camera_id") == cam_id:
                            cam["source"] = source_val
                            cam["source_type"] = source_type
                            updated_cameras.append(cam_id)
                            break
                    
                    if not updated_cameras:
                        self._send_json({"success": False, "error": f"Camera '{cam_id}' not found in graph"}, status=HTTPStatus.BAD_REQUEST)
                        return
                        
                    new_graph = CameraGraph.from_dict(graph_dict)
                    new_graph.save(self.graph_file)
                    self.runtime_pipeline.update_graph(new_graph)
                    
                    logger.info(f"[SERVER] Switched sources for {updated_cameras} to {mode}")
                    self._send_json({"success": True, "mode": mode, "updated": updated_cameras})
                    
            except Exception as e:
                logger.exception(f"[SERVER] Failed to switch source: {e}")
                self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        elif path == "/api/audit/clear":
            if hasattr(self, 'audit_logger') and self.audit_logger:
                self.audit_logger.clear()
                self._send_json({"success": True, "message": "Audit trail cleared."})
            else:
                self._send_json({"success": False, "error": "Audit logger not available"}, status=HTTPStatus.BAD_REQUEST)
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

            # Hard watchdog timer to guarantee process never stays alive in background
            def _watchdog():
                time.sleep(3.0)
                if "PYTEST_CURRENT_TEST" not in os.environ:
                    logger.warning("[SERVER] Watchdog timeout reached during shutdown. Forcing exit.")
                    os._exit(0)

            threading.Thread(target=_instant_shutdown, daemon=True, name="shutdown_thread").start()
            threading.Thread(target=_watchdog, daemon=True, name="shutdown_watchdog").start()
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

        src_str = str(source_param).strip()
        is_url = src_str.startswith(("http://", "https://", "rtsp://"))
        if is_url:
            src = src_str
        elif source_type == "webcam" or src_str.isdigit():
            try:
                src = int(src_str)
            except (ValueError, TypeError):
                src = src_str
        else:
            src = src_str

        # If it's a webcam integer index that is already in use by active pipeline, don't re-open it
        if isinstance(src, int) and self.runtime_pipeline is not None:
            for cid, node in getattr(self.runtime_pipeline, "_nodes", {}).items():
                try:
                    if int(node.config.source) == src:
                        return self.runtime_pipeline.get_camera_frame_jpeg(cid)
                except (ValueError, TypeError):
                    pass

        cap = None
        try:
            if isinstance(src, int) and sys.platform.startswith("win"):
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                if not cap or not cap.isOpened():
                    cap = cv2.VideoCapture(src, cv2.CAP_MSMF)
                if not cap or not cap.isOpened():
                    cap = cv2.VideoCapture(src)
            else:
                cap = cv2.VideoCapture(src)
                # Auto-detect IP Webcam endpoint for HTTP/HTTPS roots
                if (not cap or not cap.isOpened()) and isinstance(src, str) and src.startswith(("http://", "https://")):
                    import urllib.parse
                    try:
                        parsed = urllib.parse.urlparse(src)
                        if parsed.path in ("", "/"):
                            candidate = src.rstrip("/") + "/video"
                            alt_cap = cv2.VideoCapture(candidate)
                            if alt_cap and alt_cap.isOpened():
                                cap = alt_cap
                                src = candidate
                    except Exception:
                        pass
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2500)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2500)

            if not cap or not cap.isOpened():
                return None

            ret, frame = cap.read()
            cap.release()
            cap = None

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
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass


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
            if "PYTEST_CURRENT_TEST" not in os.environ:
                time.sleep(0.1)
                os._exit(0)
    else:
        def _serve_loop():
            try:
                server.serve_forever()
            except Exception:
                pass

        t = threading.Thread(target=_serve_loop, daemon=True, name="ui_server_thread")
        t.start()

    return server
