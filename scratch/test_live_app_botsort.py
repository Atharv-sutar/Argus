"""Test live application initialization with BoT-SORT."""

import logging
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import AppConfig
from src.core.dependencies import ensure_technologies
from src.tracking.bot_sort import BoTSORTTracker
from src.app.main import build_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_live_app")

def test_live_app():
    config = AppConfig.load("configs/default.yaml")
    ensure_technologies(config)

    logger.info("Configured tracker_type: %s", getattr(config.tracking, "tracker_type", None))
    assert getattr(config.tracking, "tracker_type", None) == "botsort"

    # Build pipeline in synthetic mode
    pipeline = build_pipeline(
        config=config,
        use_synthetic=True,
    )

    # Inspect initialized tracker in worker
    cam_id = pipeline.active_camera_id or "camera_0"
    worker = pipeline._get_or_create_worker(cam_id)
    assert worker is not None, "Worker should not be None"
    
    logger.info("Worker tracker class: %s", type(worker.tracker).__name__)
    assert isinstance(worker.tracker, BoTSORTTracker), f"Expected BoTSORTTracker, got {type(worker.tracker)}"

    # Run 10 steps through pipeline
    logger.info("Stepping pipeline for 10 frames...")
    for frame_idx in range(1, 11):
        results = pipeline.step()
        data = results.get(cam_id)
        assert data is not None
        frame, track_res, target = data
        track_count = track_res.count if track_res else 0
        logger.info(
            "Frame %d: frame shape=%s, active tracks=%d, target state=%s",
            frame_idx,
            frame.shape if frame is not None else None,
            track_count,
            target.state.value if target else "NONE",
        )

    pipeline.stop()
    logger.info("LIVE APPLICATION VERIFICATION SUCCESSFUL: BoT-SORT running smoothly without conflicts!")

if __name__ == "__main__":
    test_live_app()
