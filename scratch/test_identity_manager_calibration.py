import glob
import os
import cv2
import numpy as np
import torch
from src.benchmark.dataset import BenchmarkDataset
from src.benchmark.evaluator import ProductionReIDEvaluator
from src.identity.manager import IdentityManager
from src.reid.extractor import PyTorchReIDExtractor
from src.reid.quality import CropQualityEvaluator

def main():
    scratch_dir = r"C:\Users\athar\.gemini\antigravity-ide\brain\c2a87c82-e77c-4b55-8b21-41f2212e7450\scratch"
    dataset = BenchmarkDataset.from_scratch_archive(scratch_dir)
    dev_set, val_set, test_set = dataset.split_by_sequence(train_ratio=0.50, val_ratio=0.25)

    extractor = PyTorchReIDExtractor(model_name="osnet_x0_25", device="cuda")
    quality = CropQualityEvaluator(min_height=35, min_width=16)

    for sim_th, ref_th, upper_th in [
        (0.88, 0.86, 0.82),
        (0.90, 0.88, 0.84),
        (0.92, 0.90, 0.86),
        (0.94, 0.92, 0.88),
    ]:
        id_mgr = IdentityManager(
            reid_extractor=extractor,
            similarity_threshold=sim_th,
            reacquisition_threshold=sim_th + 0.04,
            reference_threshold=ref_th,
            upper_threshold=upper_th,
            min_margin=0.06,
            quality_evaluator=quality,
        )
        evaluator = ProductionReIDEvaluator(identity_manager=id_mgr)
        report = evaluator.evaluate_dataset(test_set if len(test_set.identities) >= 2 else dataset)
        print(f"SimTh: {sim_th:.2f} | RefTh: {ref_th:.2f} | UpperTh: {upper_th:.2f} -> TPR: {report.tpr:6.2f}% | FMR: {report.fmr:6.2f}% | Top1: {report.top1_accuracy:6.2f}% | Reacq: {report.reacquisition_success_rate:6.2f}% | FalseReacq: {report.false_reacquisition_rate:6.2f}%")

if __name__ == "__main__":
    main()
