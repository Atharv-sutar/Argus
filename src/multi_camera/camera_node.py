"""Per-camera runtime state holder separating connection state from AI activity."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.core.multi_camera_types import CameraNodeConfig, CameraStatus

logger = logging.getLogger(__name__)


class CameraNode:
    """
    Represents the runtime state of a single camera in the multi-camera system.

    Separates two concepts:
    - Camera is connected (grabbing frames) — always true when ONLINE
    - Camera is running AI pipeline (detection/tracking/ReID) — only when activated by search manager

    This distinction is the key scalability mechanism: many cameras can be connected
    without running expensive AI processing on all of them simultaneously.
    """

    def __init__(self, config: CameraNodeConfig) -> None:
        self.config = config
        self.status: CameraStatus = CameraStatus.OFFLINE
        self.is_ai_active: bool = False
        self.last_frame: Optional[np.ndarray] = None
        self.fps: float = 0.0
        self._pipeline = None  # Set by MultiCameraPipeline when activated

    @property
    def camera_id(self) -> str:
        return self.config.camera_id

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_online(self) -> bool:
        return self.status not in (CameraStatus.OFFLINE, CameraStatus.DISABLED)

    def activate_ai(self) -> None:
        """Mark this camera as running the full AI pipeline."""
        if self.status == CameraStatus.OFFLINE:
            logger.warning(f"Cannot activate AI on offline camera '{self.camera_id}'")
            return
        self.is_ai_active = True
        if self.status != CameraStatus.ACTIVE_TARGET:
            self.status = CameraStatus.SEARCHING
        logger.info(f"Camera '{self.camera_id}' AI pipeline activated")

    def deactivate_ai(self) -> None:
        """Stop the AI pipeline but keep the camera connected."""
        self.is_ai_active = False
        if self.status == CameraStatus.SEARCHING:
            self.status = CameraStatus.ONLINE
        logger.info(f"Camera '{self.camera_id}' AI pipeline deactivated")

    def mark_online(self) -> None:
        """Mark camera as connected and grabbing frames."""
        if self.status == CameraStatus.DISABLED:
            return
        self.status = CameraStatus.ONLINE

    def mark_offline(self) -> None:
        """Mark camera as disconnected."""
        self.is_ai_active = False
        self.status = CameraStatus.OFFLINE

    def mark_disabled(self) -> None:
        """Mark camera as manually disabled."""
        self.is_ai_active = False
        self.status = CameraStatus.DISABLED

    def mark_active_target(self) -> None:
        """Mark this camera as the one currently tracking the target."""
        self.is_ai_active = True
        self.status = CameraStatus.ACTIVE_TARGET

    def to_status_dict(self) -> dict:
        """Serialize runtime status for the UI."""
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "status": self.status.value,
            "is_ai_active": self.is_ai_active,
            "fps": round(self.fps, 1),
            "enabled": self.config.enabled,
        }
