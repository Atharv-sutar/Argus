"""Unit tests for continuous SegmentWriter."""

import os
import shutil
import tempfile
import time
import cv2
import numpy as np
import pytest

from src.recording.segment_index import SegmentIndex
from src.recording.segment_writer import SegmentWriter


@pytest.fixture
def temp_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath, ignore_errors=True)


def test_segment_writer_flow(temp_dir):
    db_path = os.path.join(temp_dir, "test_segments.db")
    index = SegmentIndex(db_path=db_path)

    # Use small segment duration (0.5s) to test segment rotation
    writer = SegmentWriter(
        camera_id="cam_test",
        output_dir=temp_dir,
        segment_duration_s=0.5,
        fps=10.0,
        resolution=(320, 240),
        segment_index=index,
        queue_size=50,
    )

    writer.start()

    # Feed 15 frames over 1.5 seconds simulated timestamp
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # Draw something so frame is not empty
    cv2.putText(dummy_frame, "TEST", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    base_time_ms = 1000.0
    for i in range(15):
        ts_ms = base_time_ms + (i * 100.0)  # 100ms intervals = 10 fps
        fed = writer.feed_frame(dummy_frame, ts_ms)
        assert fed is True
        time.sleep(0.02)  # Give worker thread a moment to process

    # Stop writer and flush
    writer.stop()

    # Verify at least one segment was written and indexed
    segments = index.query_segments(camera_id="cam_test")
    assert len(segments) >= 1

    total_frames = sum(s["frame_count"] for s in segments)
    assert total_frames > 0

    # Verify video file exists on disk and is non-empty
    for seg in segments:
        file_path = seg["file_path"]
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) > 0

        # Verify video can be opened with OpenCV
        cap = cv2.VideoCapture(file_path)
        assert cap.isOpened()
        ret, frame = cap.read()
        assert ret is True
        assert frame.shape == (240, 320, 3)
        cap.release()
