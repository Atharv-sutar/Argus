"""Stitches together clips into a journey video."""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import cv2
import numpy as np

from src.playback.route_recorder import RouteEvent
from src.playback.annotations import AnnotationStore

logger = logging.getLogger(__name__)


def stitch_journey(case_id: str,
                   footage_dir: str,
                   events: List[RouteEvent],
                   output_path: str,
                   fps: float = 30.0,
                   resolution: tuple = (1280, 720),
                   annotation_store: AnnotationStore = None) -> bool:
    """
    Generates a continuous journey video from the route events.
    """
    if not events:
        logger.error("No route events to stitch.")
        return False

    sorted_events = sorted(events, key=lambda e: e.timestamp_ms)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    if not out.isOpened():
        logger.error(f"Failed to open VideoWriter for {output_path}")
        return False

    # Group events by camera segments (enter -> exit/handoff)
    segments = []
    active_cam = None
    start_ms = 0.0
    was_corrected = False

    for e in sorted_events:
        if e.event_type in ("enter", "handoff", "correction"):
            if active_cam is not None and active_cam != e.camera_id:
                segments.append((active_cam, start_ms, e.timestamp_ms, was_corrected))
            active_cam = e.camera_id
            start_ms = e.timestamp_ms
            was_corrected = e.was_human_corrected
        elif e.event_type in ("exit", "lost"):
            if active_cam == e.camera_id:
                segments.append((active_cam, start_ms, e.timestamp_ms, was_corrected))
                active_cam = None

    if active_cam is not None:
        last_ms = sorted_events[-1].timestamp_ms
        if last_ms > start_ms:
            segments.append((active_cam, start_ms, last_ms, was_corrected))

    try:
        last_end_ms = 0.0
        for cam_id, start_ms, end_ms, corrected in segments:
            # Handle gaps (IN TRANSIT)
            if start_ms > last_end_ms + 1000 and last_end_ms > 0:
                _write_transit_frames(out, last_end_ms, start_ms, resolution, fps)
            
            cam_annos = annotation_store.get_by_camera(cam_id) if annotation_store else []
            _write_camera_segment(
                cam_id=cam_id,
                footage_dir=footage_dir,
                start_ms=start_ms,
                end_ms=end_ms,
                corrected=corrected,
                out_writer=out,
                resolution=resolution,
                fps=fps,
                annotations=cam_annos
            )
            last_end_ms = end_ms

        logger.info(f"Journey video stitched successfully: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error during stitching: {e}")
        return False
    finally:
        out.release()


def _write_transit_frames(out: cv2.VideoWriter, start_ms: float, end_ms: float, resolution: tuple, fps: float) -> None:
    gap_s = (end_ms - start_ms) / 1000.0
    num_frames = int(min(3.0, gap_s) * fps)  # Cap transit placeholder to 3 seconds max
    
    frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    text = f"IN TRANSIT ({gap_s:.1f}s)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
    
    tx = (resolution[0] - text_size[0]) // 2
    ty = (resolution[1] + text_size[1]) // 2
    cv2.putText(frame, text, (tx, ty), font, 1.5, (255, 255, 255), 3)

    for _ in range(num_frames):
        out.write(frame)


def _write_camera_segment(cam_id: str, footage_dir: str, start_ms: float, end_ms: float,
                          corrected: bool, out_writer: cv2.VideoWriter, resolution: tuple, target_fps: float,
                          annotations: list = None) -> None:
    
    # Try different extensions
    video_path = ""
    cam_dir = os.path.join(footage_dir, cam_id)
    if os.path.exists(cam_dir):
        for ext in [".mp4", ".avi", ".mkv"]:
            p = os.path.join(cam_dir, f"video{ext}")
            if os.path.exists(p):
                video_path = p
                break
    
    if not video_path:
        logger.warning(f"Could not find video file for {cam_id} in {cam_dir}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        
        while True:
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms > end_ms:
                break
                
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to match output resolution
            frame = cv2.resize(frame, resolution)

            # Overlay metadata
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (resolution[0], 60), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            s = int(pos_ms / 1000)
            m = s // 60
            s = s % 60
            ts_str = f"{m:02d}:{s:02d}"

            tag = "CORRECTED [HUMAN]" if corrected else "AUTO"
            color = (0, 165, 255) if corrected else (0, 255, 0)

            cv2.putText(frame, f"CAM: {cam_id} | TIME: {ts_str} | {tag}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if annotations:
                # Find if any annotation is active (within 3 seconds of pos_ms)
                for a in annotations:
                    if a.timestamp_ms <= pos_ms <= a.timestamp_ms + 3000:
                        cv2.rectangle(frame, (0, resolution[1]-60), (resolution[0], resolution[1]), (0, 0, 0), -1)
                        cv2.putText(frame, f"Note: {a.text}", (20, resolution[1]-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        break

            out_writer.write(frame)
    finally:
        cap.release()
