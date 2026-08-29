import os
import sys
import cv2
import numpy as np

print("Probing available camera indices and backends...")

for idx in [0, 1, 2]:
    print(f"\n--- Testing index {idx} ---")
    for name, backend in [("Default", cv2.CAP_ANY), ("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]:
        try:
            cap = cv2.VideoCapture(idx, backend)
            opened = cap.isOpened()
            if opened:
                ret, frame = cap.read()
                if ret and frame is not None:
                    mean_val = float(np.mean(frame))
                    shape = frame.shape
                    print(f"  [{name}] SUCCESS! Frame read: shape={shape}, mean_pixel={mean_val:.2f}")
                else:
                    print(f"  [{name}] Opened, but read() returned False / None")
                cap.release()
            else:
                print(f"  [{name}] Failed to open")
        except Exception as e:
            print(f"  [{name}] Exception: {e}")
