"""Step 2: target.switch_when_better — an incumbent above match_threshold must not keep the
lock (or poison the gallery) when the true target clearly outscores it."""

import pytest

from src.core.config import AppConfig
from src.core.types import TargetState

from tests.unit.pipeline._lock_harness import build, enrolled_track_ids, run_reid_frames, seed_and_lock

TRUE_TARGET, IMPOSTOR = 1, 2


def _cfg(flag: bool) -> AppConfig:
    cfg = AppConfig()
    cfg.target.switch_when_better = flag
    # Fix synthetic test thresholds explicitly so math reflects calibrated 0.75/0.82 fixtures
    cfg.reid.match_threshold = 0.75
    cfg.reid.reacquisition_threshold = 0.78
    cfg.reid.auto_add_threshold = 0.82
    cfg.reid.min_margin = 0.05
    cfg.reid.uncertain_band_min = 0.73
    cfg.reid.uncertain_band_max = 0.78
    return cfg


def test_flag_defaults_on_in_code_and_yaml():
    assert AppConfig().target.switch_when_better is True
    assert AppConfig.from_yaml("configs/default.yaml").target.switch_when_better is True


# (a) lock on impostor 0.79, true target 0.93, 30 ReID frames -> lock must move to the target
def test_a_lock_moves_from_impostor_to_true_target_when_flag_on():
    p, im, tm = build(_cfg(True))
    seed_and_lock(tm, IMPOSTOR)
    hist = run_reid_frames(p, {TRUE_TARGET: 0.93, IMPOSTOR: 0.79}, n=30)
    assert tm.target.track_id == TRUE_TARGET
    assert tm.target.state == TargetState.CONFIRMED
    # keeps the existing 2-frame consensus: margin 0.14 < 0.15 so no immediate switch
    assert hist[0][0] == IMPOSTOR and hist[0][1] == TargetState.UNCERTAIN
    assert hist[1][0] == TRUE_TARGET
    # lock never flips back
    assert all(tid == TRUE_TARGET for tid, _ in hist[1:])


# (b) impostor 0.84 vs target 0.95 -> impostor must never be enrolled
def test_b_impostor_above_auto_add_threshold_is_never_enrolled_when_flag_on():
    p, im, tm = build(_cfg(True))
    seed_and_lock(tm, IMPOSTOR)
    run_reid_frames(p, {TRUE_TARGET: 0.95, IMPOSTOR: 0.84}, n=30)
    assert IMPOSTOR not in enrolled_track_ids(im)
    assert tm.target.track_id == TRUE_TARGET


def test_no_enrollment_when_incumbent_is_not_clear_top1_when_flag_on():
    # Correct lock at 0.86, bystander 0.84: incumbent stays (margin 0.02 < switch margin) but is
    # not top-1 by min_margin (0.05)... and that is the AMBIGUOUS case, so nothing is enrolled.
    p, im, tm = build(_cfg(True))
    seed_and_lock(tm, TRUE_TARGET)
    run_reid_frames(p, {TRUE_TARGET: 0.86, IMPOSTOR: 0.84}, n=10)
    assert enrolled_track_ids(im) == []
    assert tm.target.track_id == TRUE_TARGET


def test_clear_incumbent_still_enrolls_when_flag_on():
    p, im, tm = build(_cfg(True))
    seed_and_lock(tm, TRUE_TARGET)
    run_reid_frames(p, {TRUE_TARGET: 0.95, IMPOSTOR: 0.30}, n=30)
    assert tm.target.track_id == TRUE_TARGET
    assert enrolled_track_ids(im) and set(enrolled_track_ids(im)) == {TRUE_TARGET}


def test_challenger_failing_manual_anchor_does_not_unseat_good_incumbent():
    # Manual-anchor gate needs challenger man_sim >= max(0.55, match - 0.05) = 0.55.
    # Challenger 0.54 beats incumbent 0.45 by 0.09 (>= switch margin) but must not win,
    # and the incumbent (still >= match_threshold) must not be rejected because of it.
    cfg = _cfg(True)
    cfg.reid.match_threshold = 0.40
    p, im, tm = build(cfg)
    seed_and_lock(tm, TRUE_TARGET)
    hist = run_reid_frames(p, {TRUE_TARGET: 0.45, IMPOSTOR: 0.54}, n=10)
    assert all(tid == TRUE_TARGET and st == TargetState.CONFIRMED for tid, st in hist)
    assert enrolled_track_ids(im) == []


# (c) flag off: legacy behaviour unchanged (the scoop persists and the impostor is enrolled)
@pytest.mark.parametrize("sims", [{TRUE_TARGET: 0.93, IMPOSTOR: 0.79}, {TRUE_TARGET: 0.95, IMPOSTOR: 0.84}])
def test_c_flag_off_keeps_legacy_behaviour(sims):
    p, im, tm = build(_cfg(False))
    seed_and_lock(tm, IMPOSTOR)
    hist = run_reid_frames(p, sims, n=30)
    assert all(tid == IMPOSTOR and st == TargetState.CONFIRMED for tid, st in hist)
    if sims[IMPOSTOR] >= 0.82:
        assert IMPOSTOR in enrolled_track_ids(im)
