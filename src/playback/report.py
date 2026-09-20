"""Generates textual reports from recorded routes."""

from __future__ import annotations

import datetime
from typing import List

from src.playback.route_recorder import RouteEvent
from src.playback.annotations import AnnotationStore


def generate_route_report(case_id: str, events: List[RouteEvent], annotation_store: AnnotationStore = None) -> str:
    """
    Generates a human-readable markdown report of the target's route.
    """
    if not events:
        return f"Route Report — Case {case_id}\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nNo route data recorded."

    sorted_events = sorted(events, key=lambda e: e.timestamp_ms)
    
    first_event = sorted_events[0]
    
    lines = []
    lines.append(f"Route Report — Case {case_id}")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    start_time_str = _format_ms(first_event.timestamp_ms)
    lines.append(f"Target: First identified on {first_event.camera_id} at {start_time_str}")
    lines.append("")
    lines.append("Timeline:")

    # Very naive linear pass to group enter->exit segments
    cameras_visited = set()
    corrections = 0
    total_tracked_ms = 0.0
    
    active_cam = None
    start_ms = 0.0
    was_corrected = False

    for e in sorted_events:
        cameras_visited.add(e.camera_id)
        if e.was_human_corrected:
            corrections += 1

        if e.event_type in ("enter", "handoff", "correction"):
            if active_cam is None:
                active_cam = e.camera_id
                start_ms = e.timestamp_ms
                was_corrected = e.was_human_corrected
            elif active_cam != e.camera_id:
                # Transitioned without explicit exit? Close previous.
                duration = e.timestamp_ms - start_ms
                total_tracked_ms += duration
                lines.append(_format_segment(active_cam, start_ms, e.timestamp_ms, was_corrected))
                if annotation_store:
                    annos = annotation_store.get_in_range(start_ms, e.timestamp_ms)
                    for a in annos:
                        if a.camera_id == active_cam:
                            lines.append(f"      📝 {_format_ms(a.timestamp_ms)} \"{a.text}\"")
                
                active_cam = e.camera_id
                start_ms = e.timestamp_ms
                was_corrected = e.was_human_corrected

        elif e.event_type in ("exit", "lost"):
            if active_cam == e.camera_id:
                duration = e.timestamp_ms - start_ms
                total_tracked_ms += duration
                lines.append(_format_segment(active_cam, start_ms, e.timestamp_ms, was_corrected))
                if annotation_store:
                    annos = annotation_store.get_in_range(start_ms, e.timestamp_ms)
                    for a in annos:
                        if a.camera_id == active_cam:
                            lines.append(f"      📝 {_format_ms(a.timestamp_ms)} \"{a.text}\"")
                active_cam = None

    if active_cam is not None:
        last_ms = sorted_events[-1].timestamp_ms
        if last_ms > start_ms:
            duration = last_ms - start_ms
            total_tracked_ms += duration
            lines.append(_format_segment(active_cam, start_ms, last_ms, was_corrected))
            if annotation_store:
                annos = annotation_store.get_in_range(start_ms, last_ms)
                for a in annos:
                    if a.camera_id == active_cam:
                        lines.append(f"      📝 {_format_ms(a.timestamp_ms)} \"{a.text}\"")

    lines.append("")
    lines.append(f"Total tracked time: {_format_duration(total_tracked_ms)}")
    lines.append(f"Cameras visited: {len(cameras_visited)}")
    lines.append(f"Human corrections: {corrections}")
    lines.append(f"Journey video: journey_{case_id}.mp4")

    return "\n".join(lines)


def _format_ms(ms: float) -> str:
    s = int(ms / 1000)
    m = s // 60
    s = s % 60
    return f"{m:02d}:{s:02d}"

def _format_duration(ms: float) -> str:
    s = int(ms / 1000)
    m = s // 60
    s = s % 60
    return f"{m}m {s}s"

def _format_segment(camera: str, start_ms: float, end_ms: float, corrected: bool) -> str:
    tag = "CORRECTED ⚠" if corrected else "AUTO"
    duration = end_ms - start_ms
    return f"  {_format_ms(start_ms)} - {_format_ms(end_ms)}  {camera:15s} ({_format_duration(duration):>7s})  {tag}"
