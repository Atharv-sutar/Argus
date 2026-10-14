"""Camera acquisition module."""

from src.camera.capture import OpenCVCamera, SyntheticCamera
from src.camera.video_file import VideoFileCamera
from src.camera.directory_loader import auto_load_video_directory, is_video_directory, discover_video_files

__all__ = [
    "OpenCVCamera",
    "SyntheticCamera",
    "VideoFileCamera",
    "auto_load_video_directory",
    "is_video_directory",
    "discover_video_files",
]
