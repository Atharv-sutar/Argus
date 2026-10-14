"""Unit tests for SQLite SegmentIndex."""

import os
import tempfile
import pytest

from src.recording.segment_index import SegmentIndex


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def test_segment_index_crud(temp_db):
    index = SegmentIndex(db_path=temp_db)

    # Initially empty
    assert index.get_segment_count() == 0
    assert index.get_total_storage_bytes() == 0

    # Insert segments
    index.insert_segment(
        segment_id="cam1_001",
        camera_id="cam1",
        start_time_ms=1000.0,
        end_time_ms=2000.0,
        file_path="/data/cam1_001.mp4",
        file_size_bytes=1024,
        frame_count=30,
        created_at=100.0,
    )

    index.insert_segment(
        segment_id="cam1_002",
        camera_id="cam1",
        start_time_ms=2000.0,
        end_time_ms=3000.0,
        file_path="/data/cam1_002.mp4",
        file_size_bytes=2048,
        frame_count=30,
        created_at=200.0,
    )

    index.insert_segment(
        segment_id="cam2_001",
        camera_id="cam2",
        start_time_ms=1500.0,
        end_time_ms=2500.0,
        file_path="/data/cam2_001.mp4",
        file_size_bytes=4096,
        frame_count=30,
        created_at=150.0,
    )

    assert index.get_segment_count() == 3
    assert index.get_total_storage_bytes() == 1024 + 2048 + 4096

    # Query all
    all_segs = index.query_segments()
    assert len(all_segs) == 3

    # Query by camera
    cam1_segs = index.query_segments(camera_id="cam1")
    assert len(cam1_segs) == 2
    assert all(s["camera_id"] == "cam1" for s in cam1_segs)

    # Query with time range
    range_segs = index.query_segments(camera_id="cam1", start_time_ms=1800.0, end_time_ms=2200.0)
    assert len(range_segs) == 2  # overlaps cam1_001 (end=2000) and cam1_002 (start=2000)

    # Query older than
    older = index.get_segments_older_than(created_at_cutoff=175.0)
    assert len(older) == 2  # cam1_001 (100.0) and cam2_001 (150.0)

    # Query oldest
    oldest = index.get_oldest_segments(limit=2)
    assert len(oldest) == 2
    assert oldest[0]["segment_id"] == "cam1_001"
    assert oldest[1]["segment_id"] == "cam2_001"

    # Delete
    deleted = index.delete_segment("cam1_001")
    assert deleted is True
    assert index.get_segment_count() == 2
    assert index.get_total_storage_bytes() == 2048 + 4096
