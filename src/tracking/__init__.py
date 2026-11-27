"""Tracking subsystem with modular tracker factory registry.

UNUSED:
- register_tracker (lines 21-23): Modular tracker extension hook kept for registering custom BaseTracker implementations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.core.interfaces import BaseReID, BaseTracker
from src.tracking.bot_sort import BoTSORTTracker
from src.tracking.byte_tracker import ByteTracker

TRACKER_REGISTRY: Dict[str, Callable[..., BaseTracker]] = {
    "botsort": BoTSORTTracker,
    "bytetrack": ByteTracker,
}


def register_tracker(name: str, builder: Callable[..., BaseTracker]) -> None:
    """Register a BaseTracker implementation under a lowercase identifier."""
    TRACKER_REGISTRY[name.lower().strip()] = builder


def create_tracker(config: Any, reid_extractor: Optional[BaseReID] = None) -> BaseTracker:
    """Instantiate the configured tracker implementation behind the BaseTracker interface."""
    tcfg = getattr(config, "tracking", None)
    tracker_type = str(getattr(tcfg, "tracker_type", "botsort")).lower().strip()
    if tracker_type in TRACKER_REGISTRY:
        key = tracker_type
    elif "bot" in tracker_type:
        key = "botsort"
    else:
        key = "bytetrack"

    builder = TRACKER_REGISTRY[key]
    if key == "botsort":
        return builder(
            track_thresh=getattr(tcfg, "track_thresh", 0.4),
            match_thresh=getattr(tcfg, "match_thresh", 0.8),
            track_buffer=getattr(tcfg, "track_buffer", 30),
            min_box_area=getattr(tcfg, "min_box_area", 100.0),
            gmc_method=getattr(tcfg, "gmc_method", "sparseOptFlow"),
            reid_extractor=reid_extractor,
            reid_weight=getattr(tcfg, "reid_weight", 0.5),
            ema_alpha=getattr(tcfg, "ema_alpha", 0.9),
            fix_association_cost=getattr(tcfg, "fix_association_cost", True),
        )
    if key == "bytetrack":
        return builder(
            track_thresh=getattr(tcfg, "track_thresh", 0.4),
            match_thresh=getattr(tcfg, "match_thresh", 0.8),
            track_buffer=getattr(tcfg, "track_buffer", 30),
            min_box_area=getattr(tcfg, "min_box_area", 100.0),
            reid_extractor=reid_extractor,
            reid_weight=getattr(tcfg, "reid_weight", 0.5),
        )
    return builder(config=config, reid_extractor=reid_extractor)


__all__ = [
    "BoTSORTTracker",
    "ByteTracker",
    "TRACKER_REGISTRY",
    "create_tracker",
    "register_tracker",
]