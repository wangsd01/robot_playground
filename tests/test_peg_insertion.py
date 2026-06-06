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


# ── State machine helper ──────────────────────────────────────────────────────

def _make_scenario() -> FrankaPegInsertion:
    """Return a FrankaPegInsertion with robot and peg mocked out."""
    s = FrankaPegInsertion()
    mock_peg = MagicMock()
    mock_peg.get_world_poses.return_value = (
        MagicMock(numpy=lambda: np.array([[0.4, 0.0, 0.04]])),
        MagicMock(),
    )
    s.robot = MagicMock()
    s.peg = mock_peg
    return s


# ── forward() tests ───────────────────────────────────────────────────────────

def test_forward_returns_true_while_running():
    s = _make_scenario()
    assert s.forward() is True


def test_forward_returns_false_when_done():
    s = _make_scenario()
    s._event = len(FrankaPegInsertion.EVENTS_DT)
    assert s.forward() is False


def test_forward_returns_false_when_final_phase_completes():
    s = _make_scenario()
    s._event = len(FrankaPegInsertion.EVENTS_DT) - 1
    s._step = FrankaPegInsertion.EVENTS_DT[-1] - 1
    s._peg_grasp_z = 0.098
    s._hole_insert_z = 0.128
    assert s.forward() is False
    assert s.is_done()


def test_forward_phase0_advances_after_80_steps():
    s = _make_scenario()
    for _ in range(FrankaPegInsertion.EVENTS_DT[0]):
        s.forward()
    assert s._event == 1
    assert s._step == 0


def test_forward_phase1_caches_heights_on_first_step():
    s = _make_scenario()
    s._event = 1
    s._step = 0
    s.forward()
    assert s._peg_grasp_z == pytest.approx(0.04 + _GRIPPER_HEIGHT_OFFSET)
    assert s._hole_insert_z == pytest.approx(
        _HOLE_TOP_Z - _INSERTION_DEPTH + _GRIPPER_HEIGHT_OFFSET + _PEG_HEIGHT / 2
    )


def test_forward_phase3_calls_close_gripper():
    s = _make_scenario()
    s._event = 3
    s._peg_grasp_z = 0.098
    s._hole_insert_z = 0.128
    s.forward()
    s.robot.close_gripper.assert_called_once()
    s.robot.set_end_effector_pose.assert_not_called()


def test_forward_phase7_calls_open_gripper():
    s = _make_scenario()
    s._event = 7
    s._peg_grasp_z = 0.098
    s._hole_insert_z = 0.128
    s.forward()
    s.robot.open_gripper.assert_called_once()
    s.robot.set_end_effector_pose.assert_not_called()


def test_forward_phase6_targets_hole_insert_z():
    s = _make_scenario()
    s._event = 6
    s._peg_grasp_z = 0.098
    s._hole_insert_z = 0.128
    s.forward()
    call_kwargs = s.robot.set_end_effector_pose.call_args.kwargs
    position = call_kwargs["position"]
    assert position[2] == pytest.approx(0.128)
    assert position[0] == pytest.approx(FrankaPegInsertion.HOLE_POSITION[0])
    assert position[1] == pytest.approx(FrankaPegInsertion.HOLE_POSITION[1])


# ── reset() tests ─────────────────────────────────────────────────────────────

def test_reset_clears_event_and_step():
    s = _make_scenario()
    s._event = 5
    s._step = 30
    s.reset()
    assert s._event == 0
    assert s._step == 0


def test_reset_clears_cached_heights():
    s = _make_scenario()
    s._peg_grasp_z = 0.098
    s._hole_insert_z = 0.128
    s.reset()
    assert s._peg_grasp_z is None
    assert s._hole_insert_z is None


def test_reset_calls_robot_reset_to_default_pose():
    s = _make_scenario()
    s.reset()
    s.robot.reset_to_default_pose.assert_called_once()


def test_reset_teleports_peg_to_initial_position():
    s = _make_scenario()
    s.reset()
    s.peg.set_world_poses.assert_called_once()
    kwargs = s.peg.set_world_poses.call_args.kwargs
    np.testing.assert_array_almost_equal(kwargs["positions"][0], [0.4, 0.0, 0.04])
    np.testing.assert_array_almost_equal(kwargs["orientations"][0], [1.0, 0.0, 0.0, 0.0])
