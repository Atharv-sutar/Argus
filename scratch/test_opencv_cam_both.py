import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from src.camera.capture import OpenCVCamera

for idx in [0, 1]:
    cam = OpenCVCamera(source=idx, use_thread=True)
    success, frame, ts = cam.read()
    print(f"Camera index {idx}: success={success}, shape={frame.shape if frame is not None else None}, mean={float(np.mean(frame)) if frame is not None else None:.2f}")
    cam.release()
