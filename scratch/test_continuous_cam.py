import time
import cv2
import numpy as np

print("Testing continuous read on camera 0 and camera 1...")

for idx in [0, 1]:
    print(f"\n--- Testing index {idx} continuous 50 frames ---")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    
    success_count = 0
    t0 = time.time()
    for i in range(50):
        ret, frame = cap.read()
        if ret and frame is not None:
            mean = float(np.mean(frame))
            if mean > 1.0:
                success_count += 1
        time.sleep(0.02)
    t1 = time.time()
    cap.release()
    print(f"Index {idx}: {success_count}/50 valid frames read in {t1 - t0:.2f}s (FPS: {50/(t1-t0):.1f})")
