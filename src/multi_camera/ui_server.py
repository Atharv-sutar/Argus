"""Lightweight HTTP server and REST API for the interactive Camera Mapping UI."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import sys
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import cv2
except ImportError:
    cv2 = None

from src.core.multi_camera_types import (
    CameraEdgeConfig,
    CameraNodeConfig,
    EdgeDirection,
    EdgeType,
    SourceType,
)
from src.multi_camera.camera_graph import CameraGraph

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def probe_local_webcams(max_indices: int = 5) -> List[Dict[str, Any]]:
    """Probes local webcam devices up to max_indices."""
    cameras = []
    if cv2 is None:
        return cameras

    for i in range(max_indices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                cameras.append({
                    "source": i,
                    "name": f"Webcam {i}",
                    "source_type": "webcam",
                    "width": w,
                    "height": h,
                    "fps": fps,
                    "status": "available",
                })
        else:
            cap.release()

    return cameras


class MappingAPIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for camera mapping UI static files and REST endpoints."""

    graph_file: Path = Path("configs/camera_graph.json")
    runtime_pipeline = None  # Optional MultiCameraPipeline reference

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy standard request logging
        logger.debug(f"{self.address_string()} - {format % args}")

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
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # --- REST Endpoints ---
        if path == "/api/cameras/discover":
            webcams = probe_local_webcams()
            self._send_json({"cameras": webcams})
            return

        elif path == "/api/graph":
            if self.graph_file.is_file():
                try:
                    with open(self.graph_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._send_json(data)
                except Exception as e:
                    self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
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
                self._send_json({
                    "active_camera": self.runtime_pipeline.active_camera_id,
                    "search_progress": progress.to_dict(),
                    "camera_statuses": statuses,
                })
            else:
                self._send_json({
                    "active_camera": None,
                    "search_progress": None,
                    "camera_statuses": {},
                })
            return

        elif path == "/api/preview":
            # Return JPEG preview snapshot for a given source
            source_param = query.get("source", ["0"])[0]
            source_type = query.get("type", ["webcam"])[0]

            frame_bytes = self._capture_preview_jpeg(source_param, source_type)
            if frame_bytes is not None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame_bytes)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(frame_bytes)
            else:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Could not capture preview from source")
            return

        # --- Static File Serving ---
        if path == "/" or path == "":
            file_path = STATIC_DIR / "index.html"
        else:
            rel_path = path.lstrip("/")
            file_path = STATIC_DIR / rel_path

        self._send_file(file_path)

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}

        if path == "/api/graph":
            try:
                graph = CameraGraph.from_dict(payload)
                errors = graph.validate()
                if errors:
                    self._send_json({"success": False, "errors": errors}, status=HTTPStatus.BAD_REQUEST)
                    return

                graph.save(self.graph_file)
                self._send_json({"success": True, "message": "Graph saved successfully"})
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        elif path == "/api/graph/validate":
            try:
                graph = CameraGraph.from_dict(payload)
                errors = graph.validate()
                self._send_json({"valid": len(errors) == 0, "errors": errors})
            except Exception as e:
                self._send_json({"valid": False, "errors": [str(e)]})
            return

        elif path == "/api/target/select":
            cam_id = payload.get("camera_id")
            x = float(payload.get("x", 0.0))
            y = float(payload.get("y", 0.0))
            if self.runtime_pipeline is not None and cam_id:
                selected_id = self.runtime_pipeline.select_target_on_camera(cam_id, x, y)
                self._send_json({"success": selected_id is not None, "selected_id": selected_id})
            else:
                self._send_json({"success": False, "error": "Runtime pipeline not active"}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _capture_preview_jpeg(self, source_param: str, source_type: str) -> Optional[bytes]:
        """Capture a single test frame as JPEG."""
        if cv2 is None:
            return None

        src: Any = source_param
        if source_type == "webcam":
            try:
                src = int(source_param)
            except ValueError:
                src = 0

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None

        # Resize preview for fast network transfer
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            return None
        return buf.tobytes()


def run_ui_server(
    port: int = 8765,
    graph_file: str = "configs/camera_graph.json",
    pipeline=None,
    block: bool = True,
) -> ThreadingHTTPServer:
    """Starts the Camera Mapping UI server."""
    MappingAPIHandler.graph_file = Path(graph_file)
    MappingAPIHandler.runtime_pipeline = pipeline

    server = ThreadingHTTPServer(("127.0.0.1", port), MappingAPIHandler)
    logger.info(f"Camera Mapping UI server running at http://127.0.0.1:{port}")

    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("UI server stopping...")
        finally:
            server.server_close()
    else:
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

    return server
