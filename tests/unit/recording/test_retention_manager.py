"""Unit tests for RetentionManager."""

import os
import shutil
import tempfile
import time
import pytest

from src.recording.retention_manager import RetentionManager
from src.recording.segment_index import SegmentIndex


@pytest.fixture
def temp_env():
    dirpath = tempfile.mkdtemp()
    db_path = os.path.join(dirpath, "retention_test.db")
    index = SegmentIndex(db_path=db_path)
    yield dirpath, index
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath, ignore_errors=True)


def test_retention_age_cleanup(temp_env):
    dirpath, index = temp_env

    # Create dummy files
    f1 = os.path.join(dirpath, "old_seg.mp4")
    f2 = os.path.join(dirpath, "new_seg.mp4")
    with open(f1, "wb") as f:
        f.write(b"0" * 1024)
    with open(f2, "wb") as f:
        f.write(b"0" * 1024)

    now = time.time()
    old_time = now - (10 * 86400)  # 10 days old

    index.insert_segment(
        segment_id="seg_old",
        camera_id="cam1",
        start_time_ms=1000.0,
        end_time_ms=2000.0,
        file_path=f1,
        file_size_bytes=1024,
        frame_count=10,
        created_at=old_time,
    )

    index.insert_segment(
        segment_id="seg_new",
        camera_id="cam1",
        start_time_ms=3000.0,
        end_time_ms=4000.0,
        file_path=f2,
        file_size_bytes=1024,
        frame_count=10,
        created_at=now,
    )

    retention = RetentionManager(
        segment_index=index,
        output_dir=dirpath,
        max_storage_gb=10.0,
        max_age_days=7.0,  # 7 days max age
        cleanup_interval_s=1000.0,
    )

    deleted = retention.enforce_retention()
    assert deleted >= 1
    assert not os.path.exists(f1)
    assert os.path.exists(f2)
    assert index.get_segment_count() == 1


def test_retention_quota_cleanup(temp_env):
    dirpath, index = temp_env

    # Create 3 files of 1 MB each (1024 * 1024 bytes)
    files = []
    now = time.time()
    for i in range(3):
        fp = os.path.join(dirpath, f"seg_{i}.mp4")
        with open(fp, "wb") as f:
            f.write(b"x" * (1024 * 1024))
        files.append(fp)
        index.insert_segment(
            segment_id=f"seg_{i}",
            camera_id="cam1",
            start_time_ms=float(i * 1000),
            end_time_ms=float((i + 1) * 1000),
            file_path=fp,
            file_size_bytes=1024 * 1024,
            frame_count=10,
            created_at=now + i,
        )

    # Set quota to ~1.5 MB (0.0015 GB)
    retention = RetentionManager(
        segment_index=index,
        output_dir=dirpath,
        max_storage_gb=0.0015,
        max_age_days=30.0,
        cleanup_interval_s=1000.0,
    )

    deleted = retention.enforce_retention()
    assert deleted >= 1
    # Oldest segment should have been deleted to fall under quota
    assert not os.path.exists(files[0])
    assert index.get_segment_count() <= 2
