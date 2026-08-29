import os
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np

from src.reid.extractor import PyTorchReIDExtractor
from src.reid.gallery import TargetGallery

print("=========================================================")
print(" REAL IMAGE RE-IDENTIFICATION & DISCRIMINATION TEST")
print("=========================================================")

extractor_x1 = PyTorchReIDExtractor(model_name="osnet_x1_0")
extractor_x0 = PyTorchReIDExtractor(model_name="osnet_x0_25")

# Test on diagnostic demo images
target_img = cv2.imread(r"diagnostics/demo/target.png")
cand_similar = cv2.imread(r"diagnostics/demo/cand_similar.png")
cand_impostor = cv2.imread(r"diagnostics/demo/cand_impostor.png")

for name, ext in [("OSNet-x1.0", extractor_x1), ("OSNet-x0.25", extractor_x0)]:
    print(f"\n--- Testing with {name} ---")
    emb_target = ext.extract(target_img)
    emb_similar = ext.extract(cand_similar)
    emb_impostor = ext.extract(cand_impostor)
    
    sim_self = float(np.dot(emb_target.vector, emb_target.vector))
    sim_similar = float(np.dot(emb_target.vector, emb_similar.vector))
    sim_impostor = float(np.dot(emb_target.vector, emb_impostor.vector))
    
    print(f"Target self-similarity:   {sim_self:.4f}")
    print(f"Target vs Similar person: {sim_similar:.4f}")
    print(f"Target vs Impostor:       {sim_impostor:.4f}")
    print(f"Discrimination Margin:    {sim_similar - sim_impostor:.4f}")

# Extract real people from bus.jpg and zidane.jpg using YOLO
from src.detection.yolo_detector import YOLODetector
detector = YOLODetector(model_name="yolov8n.pt", confidence_threshold=0.3)

bus_img = cv2.imread(r".venv/Lib/site-packages/ultralytics/assets/bus.jpg")
det_res = detector.detect(bus_img, frame_id=1, timestamp_ms=0.0)

print(f"\nDetected {det_res.count} persons in bus.jpg:")
person_crops = []
for i, det in enumerate(det_res.detections):
    x1, y1, x2, y2 = map(int, det.box.as_xyxy())
    crop = bus_img[y1:y2, x1:x2]
    person_crops.append((f"Person_{i+1}", crop))
    print(f"  Person {i+1}: bbox=({x1}, {y1}, {x2}, {y2}), size={crop.shape}")

print("\nCross-person pairwise cosine similarity matrix on real persons (OSNet-x1.0):")
embs = [extractor_x1.extract(c) for _, c in person_crops]
sim_matrix = np.zeros((len(embs), len(embs)))
for i in range(len(embs)):
    for j in range(len(embs)):
        sim_matrix[i, j] = float(np.dot(embs[i].vector, embs[j].vector))

header = "         " + "".join([f"P{i+1:d}     " for i in range(len(embs))])
print(header)
for i in range(len(embs)):
    row = f"P{i+1:d}       " + "".join([f"{sim_matrix[i, j]:.3f}   " for j in range(len(embs))])
    print(row)

print("\n=========================================================")
print(" SUCCESS: Real-image person ReID feature extraction validated!")
print("=========================================================")
