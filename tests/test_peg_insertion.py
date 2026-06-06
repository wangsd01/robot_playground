# tests/test_peg_insertion.py
import numpy as np
import pytest
from unittest.mock import MagicMock

from peg_insertion import (
    FrankaPegInsertion,
    compute_peg_grasp_z,
    compute_hole_insert_z,
    check_success,
    _PEG_HEIGHT,
    _HOLE_TOP_Z,
    _GRIPPER_HEIGHT_OFFSET,
    _INSERTION_DEPTH,
)


# ── Pure math ─────────────────────────────────────────────────────────────────

def test_compute_peg_grasp_z():
    assert compute_peg_grasp_z(0.04) == pytest.approx(0.04 + _GRIPPER_HEIGHT_OFFSET)


def test_compute_hole_insert_z():
    expected = _HOLE_TOP_Z - _INSERTION_DEPTH + _GRIPPER_HEIGHT_OFFSET + _PEG_HEIGHT / 2
    assert compute_hole_insert_z() == pytest.approx(expected)   # 0.128


def test_check_success_pass():
    # Peg center at z=0.09 → bottom = 0.09 - 0.04 = 0.05 ≤ 0.05 + 0.01; XY on-target
    peg_pos = np.array([-0.1, 0.35, 0.09])
    hole_pos = np.array([-0.1, 0.35, 0.05])
    assert check_success(peg_pos, hole_pos)


def test_check_success_fail_xy_too_far():
    peg_pos = np.array([-0.1 + 0.010, 0.35, 0.09])   # 10 mm off in X → fail
    hole_pos = np.array([-0.1, 0.35, 0.05])
    assert not check_success(peg_pos, hole_pos)


def test_check_success_fail_peg_too_high():
    peg_pos = np.array([-0.1, 0.35, 0.20])   # peg bottom at 0.16 — way above hole
    hole_pos = np.array([-0.1, 0.35, 0.05])
    assert not check_success(peg_pos, hole_pos)


# ── is_done() ─────────────────────────────────────────────────────────────────

def test_is_done_false_initially():
    s = FrankaPegInsertion()
    assert not s.is_done()


def test_is_done_true_when_event_equals_phase_count():
    s = FrankaPegInsertion()
    s._event = len(FrankaPegInsertion.EVENTS_DT)
    assert s.is_done()


def test_events_dt_has_nine_entries():
    assert len(FrankaPegInsertion.EVENTS_DT) == 9
