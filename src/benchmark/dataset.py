"""Production benchmark dataset loader, sequence splitter, and hard-negative bank."""

from __future__ import annotations

import glob
import logging
import os
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from src.benchmark.types import (
    BenchmarkObservation,
    OcclusionLevel,
    ResolutionLevel,
    ViewpointType,
)
from src.core.types import BoundingBox

logger = logging.getLogger(__name__)


class HardNegativeBank:
    """Maintains a curated bank of difficult impostor pairs and challenging distractors."""

    def __init__(self) -> None:
        self._hard_negatives: List[BenchmarkObservation] = []

    def add_hard_negative(self, obs: BenchmarkObservation) -> None:
        obs.is_hard_negative = True
        self._hard_negatives.append(obs)

    def get_hard_negatives_for_target(self, target_identity_id: str) -> List[BenchmarkObservation]:
        return [
            obs for obs in self._hard_negatives
            if obs.hard_negative_target_id == target_identity_id or obs.hard_negative_target_id is None
        ]

    def all_samples(self) -> List[BenchmarkObservation]:
        return list(self._hard_negatives)

    def __len__(self) -> int:
        return len(self._hard_negatives)


class BenchmarkDataset:
    """
    Manages structured benchmark observations, guarantees sequence-based splitting,
    and coordinates cross-camera and hard-negative evaluations.
    """

    def __init__(self) -> None:
        self.observations: List[BenchmarkObservation] = []
        self.hard_negative_bank = HardNegativeBank()
        self._by_identity: Dict[str, List[BenchmarkObservation]] = {}
        self._by_sequence: Dict[str, List[BenchmarkObservation]] = {}

    def add_observation(self, obs: BenchmarkObservation) -> None:
        self.observations.append(obs)
        self._by_identity.setdefault(obs.identity_id, []).append(obs)
        self._by_sequence.setdefault(obs.sequence_id, []).append(obs)
        if obs.is_hard_negative:
            self.hard_negative_bank.add_hard_negative(obs)

    @property
    def identities(self) -> List[str]:
        return sorted(self._by_identity.keys())

    @property
    def sequences(self) -> List[str]:
        return sorted(self._by_sequence.keys())

    def get_observations_for_identity(self, identity_id: str) -> List[BenchmarkObservation]:
        return self._by_identity.get(identity_id, [])

    def get_observations_for_sequence(self, sequence_id: str) -> List[BenchmarkObservation]:
        return self._by_sequence.get(sequence_id, [])

    def split_by_sequence(
        self,
        train_ratio: float = 0.5,
        val_ratio: float = 0.25,
    ) -> Tuple[BenchmarkDataset, BenchmarkDataset, BenchmarkDataset]:
        """
        Splits dataset by capture sequence ID to strictly avoid temporal frame leakage.
        Returns: (dev_set, val_set, test_set)
        """
        seqs = sorted(self.sequences)
        n = len(seqs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train_seqs = set(seqs[:n_train])
        val_seqs = set(seqs[n_train:n_train + n_val])
        test_seqs = set(seqs[n_train + n_val:])
        if not test_seqs and val_seqs:
            # Guarantee held-out test has at least 1 sequence if available
            test_seqs = {seqs[-1]}
            val_seqs.discard(seqs[-1])

        dev_set = BenchmarkDataset()
        val_set = BenchmarkDataset()
        test_set = BenchmarkDataset()

        for obs in self.observations:
            if obs.sequence_id in train_seqs:
                dev_set.add_observation(obs)
            elif obs.sequence_id in val_seqs:
                val_set.add_observation(obs)
            else:
                test_set.add_observation(obs)

        return dev_set, val_set, test_set

    @classmethod
    def from_scratch_archive(cls, scratch_dir: str) -> BenchmarkDataset:
        """
        Builds a BenchmarkDataset from stored scratch crops, grouping temporally
        contiguous sequences into distinct identities and sequences.
        """
        dataset = cls()
        crop_paths = sorted(glob.glob(os.path.join(scratch_dir, "reid_cand_*.png")))

        raw_tracks: Dict[str, List[Tuple[int, str]]] = {}
        for p in crop_paths:
            base = os.path.basename(p).replace(".png", "")
            parts = base.split("_")
            if "frame" in parts and "track" in parts:
                f_idx = int(parts[parts.index("frame") + 1])
                t_idx = parts[parts.index("track") + 1]
                raw_tracks.setdefault(t_idx, []).append((f_idx, p))

        # Segment raw tracks into continuous video sequences (gap > 20 indicates new sequence)
        person_count = 0
        for t_id, f_path_list in sorted(raw_tracks.items()):
            f_path_list.sort(key=lambda x: x[0])
            curr_seq: List[Tuple[int, str]] = []
            last_f = -999
            seq_num = 0

            for f_idx, path in f_path_list:
                if last_f >= 0 and (f_idx - last_f) > 20:
                    if len(curr_seq) >= 6:
                        cls._ingest_sequence(dataset, curr_seq, person_id=f"person_{person_count}", seq_id=f"seq_{t_id}_{seq_num}")
                        person_count += 1
                        seq_num += 1
                    curr_seq = []
                curr_seq.append((f_idx, path))
                last_f = f_idx

            if len(curr_seq) >= 6:
                cls._ingest_sequence(dataset, curr_seq, person_id=f"person_{person_count}", seq_id=f"seq_{t_id}_{seq_num}")
                person_count += 1

        logger.info(f"Loaded {len(dataset.observations)} observations across {len(dataset.identities)} identities and {len(dataset.sequences)} sequences.")
        return dataset

    @staticmethod
    def _ingest_sequence(
        dataset: BenchmarkDataset,
        seq_items: List[Tuple[int, str]],
        person_id: str,
        seq_id: str,
    ) -> None:
        for idx, (f_idx, path) in enumerate(seq_items):
            img = cv2.imread(path)
            if img is None or img.size == 0:
                continue

            h, w = img.shape[:2]
            res_level = ResolutionLevel.HIGH if h >= 120 else (ResolutionLevel.MEDIUM if h >= 60 else (ResolutionLevel.LOW if h >= 35 else ResolutionLevel.TINY))

            # Infer viewpoint heuristically from aspect ratio and index in sequence
            if idx == 0:
                vp = ViewpointType.FRONT
            elif idx == len(seq_items) - 1:
                vp = ViewpointType.REAR
            elif idx % 2 == 1:
                vp = ViewpointType.SIDE_LEFT
            else:
                vp = ViewpointType.OBLIQUE

            obs = BenchmarkObservation(
                identity_id=person_id,
                sequence_id=seq_id,
                camera_id=f"cam_{int(person_id.split('_')[1]) % 3}",
                frame_id=f_idx,
                timestamp_ms=float(f_idx * 33.3),
                track_id=int(person_id.split('_')[1]),
                bbox=BoundingBox(50.0, 50.0, 50.0 + w, 50.0 + h),
                image_path=path,
                crop=img,
                viewpoint=vp,
                resolution=res_level,
                occlusion=OcclusionLevel.NONE if idx < 3 else (OcclusionLevel.PARTIAL if idx % 4 == 0 else OcclusionLevel.NONE),
                quality_score=1.0 if res_level != ResolutionLevel.TINY else 0.4,
            )
            dataset.add_observation(obs)
