"""Unit tests for ReID DecisionLogger harness."""

import json
import os
import tempfile

from src.reid.decision_logger import DecisionLogger
from src.core.config import AppConfig


def test_decision_logger_basic_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test_decision.jsonl")
        logger = DecisionLogger(log_path=log_path, enabled=True)

        entry = {
            "frame_number": 42,
            "camera": "cam_0",
            "timestamp": 1234567.8,
            "state_before": "TRACKING",
            "state_after": "CONFIRMED",
            "locked_track_id": 1,
            "reason": "CONFIRM_INCUMBENT",
            "candidates": [
                {
                    "track_id": 1,
                    "box": [10.0, 20.0, 50.0, 100.0],
                    "blur_score": 85.5,
                    "filtered": False,
                    "filter_reason": None,
                    "effective_sim": 0.82,
                    "manual_sim": 0.80,
                    "auto_sim": 0.82,
                    "top3_gallery_indices": [0, 1],
                }
            ],
            "auto_adds": [1],
            "rollbacks": [],
        }

        logger.log(entry)
        logger.close()

        assert os.path.exists(log_path)
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f.readlines() if l.strip()]

        assert len(lines) == 1
        record = lines[0]
        assert record["frame_number"] == 42
        assert record["camera"] == "cam_0"
        assert record["reason"] == "CONFIRM_INCUMBENT"
        assert record["locked_track_id"] == 1
        assert len(record["candidates"]) == 1
        assert record["candidates"][0]["track_id"] == 1
        assert record["candidates"][0]["blur_score"] == 85.5
        assert record["candidates"][0]["effective_sim"] == 0.82
        assert record["auto_adds"] == [1]


def test_decision_logger_disabled_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test_disabled.jsonl")
        logger = DecisionLogger(log_path=log_path, enabled=False)

        logger.log({"test": 123})
        logger.close()

        assert not os.path.exists(log_path)


def test_decision_log_config_defaults():
    cfg = AppConfig()
    assert hasattr(cfg, "diagnostics")
    assert cfg.diagnostics.decision_log is False
    assert "decision_log.jsonl" in cfg.diagnostics.decision_log_path
