"""Unit tests for camera capture implementations."""

from src.camera.capture import SyntheticCamera


def test_synthetic_camera_stream():
    camera = SyntheticCamera(width=320, height=240, fps=30, max_frames=10)
    assert camera.is_opened()

    frames_read = 0
    while camera.is_opened():
        success, frame, timestamp_ms = camera.read()
        if not success:
            break
        assert frame is not None
        assert frame.shape == (240, 320, 3)
        assert timestamp_ms > 0
        frames_read += 1

    assert frames_read == 10
    camera.release()
    assert not camera.is_opened()
