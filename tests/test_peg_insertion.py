# tests/test_peg_insertion.py
import peg_insertion as peg_insertion_module
import numpy as np
import pytest
from unittest.mock import MagicMock

from peg_insertion import (
    FrankaPegInsertion,
    build_hole_fixture_parts,
    choose_slot_aligned_insertion_orientation,
    compute_insertion_hand_pose,
    get_default_camera_view,
    compute_peg_grasp_z,
    compute_quaternion_angle_error_deg,
    compute_hole_insert_z,
    check_success,
    quaternion_from_z_yaw,
    validate_slot_fit,
    _quaternion_multiply,
    _PEG_SIZE_X,
    _PEG_SIZE_Y,
    _PEG_HEIGHT,
    _HOLE_TOP_Z,
    _INSERTION_DEPTH,
    _HOLE_FIXTURE_HEIGHT,
    _HOLE_FIXTURE_OUTER_SIZE_X,
    _HOLE_FIXTURE_OUTER_SIZE_Y,
    _SLOT_BASE_THICKNESS,
    _SLOT_INNER_SIZE_X,
    _SLOT_INNER_SIZE_Y,
    _SLOT_WALL_THICKNESS_X,
    _SLOT_WALL_THICKNESS_Y,
    _SLOT_YAW,
    _INSERTION_ALIGNMENT_CLEARANCE,
    _INSERTION_ALIGNMENT_STEPS,
    _INSERTION_ALIGN_MAX_XY_STEP,
    _INSERTION_ALIGN_YAW_GAIN,
    _INSERTION_ALIGN_MAX_YAW_STEP,
)

_EXPECTED_HAND_TO_FINGER_PAD_CENTER_OFFSET = 0.10365
_EXPECTED_GRASP_DEPTH_BELOW_PEG_TOP = 0.025
_EXPECTED_GRASP_POINT_ABOVE_CENTER = 0.08 / 2 - _EXPECTED_GRASP_DEPTH_BELOW_PEG_TOP
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
_Z_90_QUAT = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
_Z_NEG_90_QUAT = np.array([np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)])
_Y_90_QUAT = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])


# ── Pure math ─────────────────────────────────────────────────────────────────

def test_compute_peg_grasp_z():
    expected = 0.04 + _EXPECTED_HAND_TO_FINGER_PAD_CENTER_OFFSET + _EXPECTED_GRASP_POINT_ABOVE_CENTER
    assert compute_peg_grasp_z(0.04) == pytest.approx(expected)


def test_compute_hole_insert_z():
    expected = (
        _HOLE_TOP_Z
        - _INSERTION_DEPTH
        + _EXPECTED_HAND_TO_FINGER_PAD_CENTER_OFFSET
        + _PEG_HEIGHT / 2
        + _EXPECTED_GRASP_POINT_ABOVE_CENTER
    )
    assert compute_hole_insert_z() == pytest.approx(expected)


def test_compute_peg_grasp_z_targets_upper_grasp_band():
    peg_center_z = 0.04
    peg_top_z = peg_center_z + _PEG_HEIGHT / 2
    ee_target_z = compute_peg_grasp_z(peg_center_z)
    grip_point_z = ee_target_z - _EXPECTED_HAND_TO_FINGER_PAD_CENTER_OFFSET
    assert grip_point_z == pytest.approx(peg_top_z - _EXPECTED_GRASP_DEPTH_BELOW_PEG_TOP)


def test_compute_hole_insert_z_places_peg_bottom_below_hole_top():
    ee_target_z = compute_hole_insert_z()
    peg_bottom_z = ee_target_z - _EXPECTED_HAND_TO_FINGER_PAD_CENTER_OFFSET - _EXPECTED_GRASP_POINT_ABOVE_CENTER - _PEG_HEIGHT / 2
    assert peg_bottom_z == pytest.approx(_HOLE_TOP_Z - _INSERTION_DEPTH)


def test_get_default_camera_view_moves_camera_closer_to_robot():
    eye, target = get_default_camera_view()

    np.testing.assert_allclose(eye, np.array([1.2, 1.0, 0.9]))
    np.testing.assert_allclose(target, np.array([0.15, 0.1, 0.2]))


def test_compute_insertion_hand_pose_compensates_for_measured_peg_drift():
    current_peg_pos = np.array([-0.092, 0.343, 0.11])
    current_peg_orientation = _Z_90_QUAT
    current_hand_pos = np.array(
        [
            FrankaPegInsertion.HOLE_POSITION[0],
            FrankaPegInsertion.HOLE_POSITION[1],
            FrankaPegInsertion.TRANSPORT_HEIGHT,
        ]
    )
    desired_peg_pos = np.array(
        [
            FrankaPegInsertion.HOLE_POSITION[0],
            FrankaPegInsertion.HOLE_POSITION[1],
            _HOLE_TOP_Z - _INSERTION_DEPTH + _PEG_HEIGHT / 2,
        ]
    )

    hand_pos, hand_orientation = compute_insertion_hand_pose(
        current_peg_pos=current_peg_pos,
        current_peg_orientation=current_peg_orientation,
        current_hand_pos=current_hand_pos,
        current_hand_orientation=_IDENTITY_QUAT,
        desired_peg_pos=desired_peg_pos,
        desired_peg_orientation=_IDENTITY_QUAT,
    )

    np.testing.assert_allclose(hand_pos, np.array([-0.093, 0.358, 0.31]))
    np.testing.assert_allclose(hand_orientation, _Z_NEG_90_QUAT)


def test_compute_quaternion_angle_error_deg():
    assert compute_quaternion_angle_error_deg(_Z_90_QUAT, _IDENTITY_QUAT) == pytest.approx(90.0)


def test_default_slot_yaw_is_stored_in_radians():
    assert _SLOT_YAW == pytest.approx(np.deg2rad(10.0))


def test_default_slot_dimensions_are_increased_for_more_clearance():
    assert _SLOT_INNER_SIZE_X == pytest.approx(0.026)
    assert _SLOT_INNER_SIZE_Y == pytest.approx(0.036)


def test_choose_slot_aligned_insertion_orientation_prefers_slot_yaw_when_dimensions_match():
    orientation = choose_slot_aligned_insertion_orientation(
        slot_yaw=np.pi / 6,
        peg_xy=np.array([0.02, 0.03]),
        slot_xy=np.array([0.021, 0.031]),
    )

    np.testing.assert_allclose(orientation, quaternion_from_z_yaw(np.pi / 6))


def test_choose_slot_aligned_insertion_orientation_rotates_ninety_degrees_when_swap_matches():
    orientation = choose_slot_aligned_insertion_orientation(
        slot_yaw=np.pi / 6,
        peg_xy=np.array([0.02, 0.03]),
        slot_xy=np.array([0.031, 0.021]),
    )

    np.testing.assert_allclose(orientation, quaternion_from_z_yaw(np.pi / 6 + np.pi / 2))


def test_choose_slot_aligned_insertion_orientation_raises_when_no_candidate_fits():
    with pytest.raises(ValueError, match="does not fit"):
        choose_slot_aligned_insertion_orientation(
            slot_yaw=0.0,
            peg_xy=np.array([0.02, 0.03]),
            slot_xy=np.array([0.025, 0.025]),
        )


def test_validate_slot_fit_accepts_direct_fit():
    validate_slot_fit(np.array([0.02, 0.03]), np.array([0.021, 0.031]))


def test_validate_slot_fit_accepts_swapped_fit():
    validate_slot_fit(np.array([0.02, 0.03]), np.array([0.031, 0.021]))


def test_validate_slot_fit_raises_when_neither_candidate_fits():
    with pytest.raises(ValueError, match="does not fit"):
        validate_slot_fit(np.array([0.02, 0.03]), np.array([0.025, 0.025]))


def test_compute_insertion_hand_pose_preserves_rotated_hand_to_peg_offset():
    current_peg_pos = np.array([0.0, 0.0, -1.0])
    current_peg_orientation = _IDENTITY_QUAT
    current_hand_pos = np.array([0.0, 0.0, 0.0])
    desired_peg_pos = np.array([0.0, 0.0, 0.0])

    hand_pos, hand_orientation = compute_insertion_hand_pose(
        current_peg_pos=current_peg_pos,
        current_peg_orientation=current_peg_orientation,
        current_hand_pos=current_hand_pos,
        current_hand_orientation=_IDENTITY_QUAT,
        desired_peg_pos=desired_peg_pos,
        desired_peg_orientation=_Y_90_QUAT,
    )

    np.testing.assert_allclose(hand_pos, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(hand_orientation, _Y_90_QUAT)


def test_check_success_pass():
    # Peg center at z=0.09 → bottom = 0.09 - 0.04 = 0.05 ≤ 0.05 + 0.01; XY on-target
    peg_pos = np.array([-0.1, 0.35, 0.09])
    hole_pos = np.array([-0.1, 0.35, 0.05])
    assert check_success(peg_pos, hole_pos)


def test_check_success_fail_xy_too_far():
    peg_pos = np.array([-0.1 + 0.011, 0.35, 0.09])   # 11 mm off in X → fail
    hole_pos = np.array([-0.1, 0.35, 0.05])
    assert not check_success(peg_pos, hole_pos)


def test_check_success_pass_at_ten_mm_xy_error():
    peg_pos = np.array([-0.1 + 0.010, 0.35, 0.09])   # 10 mm off in X → pass
    hole_pos = np.array([-0.1, 0.35, 0.05])
    assert check_success(peg_pos, hole_pos)


def test_check_success_fail_peg_too_high():
    peg_pos = np.array([-0.1, 0.35, 0.20])   # peg bottom at 0.16 — way above hole
    hole_pos = np.array([-0.1, 0.35, 0.05])
    assert not check_success(peg_pos, hole_pos)


def test_build_hole_fixture_parts_creates_open_top_slot(monkeypatch):
    monkeypatch.setattr(peg_insertion_module, "_SLOT_YAW", 0.0)
    parts = build_hole_fixture_parts(FrankaPegInsertion.HOLE_POSITION, FrankaPegInsertion.HOLE_FIXTURE_PATH)

    assert [part["name"] for part in parts] == ["base", "front_wall", "back_wall", "left_wall", "right_wall"]

    part_by_name = {part["name"]: part for part in parts}
    base = part_by_name["base"]
    front = part_by_name["front_wall"]
    back = part_by_name["back_wall"]
    left = part_by_name["left_wall"]
    right = part_by_name["right_wall"]

    np.testing.assert_array_almost_equal(base["scale"], [_HOLE_FIXTURE_OUTER_SIZE_X, _HOLE_FIXTURE_OUTER_SIZE_Y, _SLOT_BASE_THICKNESS])
    assert base["position"][2] == pytest.approx(_SLOT_BASE_THICKNESS / 2)

    expected_wall_height = _HOLE_FIXTURE_HEIGHT - _SLOT_BASE_THICKNESS
    expected_wall_z = _SLOT_BASE_THICKNESS + expected_wall_height / 2
    expected_x_offset = _SLOT_INNER_SIZE_X / 2 + _SLOT_WALL_THICKNESS_X / 2
    expected_y_offset = _SLOT_INNER_SIZE_Y / 2 + _SLOT_WALL_THICKNESS_Y / 2

    np.testing.assert_array_almost_equal(front["scale"], [_HOLE_FIXTURE_OUTER_SIZE_X, _SLOT_WALL_THICKNESS_Y, expected_wall_height])
    np.testing.assert_array_almost_equal(back["scale"], [_HOLE_FIXTURE_OUTER_SIZE_X, _SLOT_WALL_THICKNESS_Y, expected_wall_height])
    np.testing.assert_array_almost_equal(left["scale"], [_SLOT_WALL_THICKNESS_X, _SLOT_INNER_SIZE_Y, expected_wall_height])
    np.testing.assert_array_almost_equal(right["scale"], [_SLOT_WALL_THICKNESS_X, _SLOT_INNER_SIZE_Y, expected_wall_height])

    assert front["position"][1] == pytest.approx(FrankaPegInsertion.HOLE_POSITION[1] + expected_y_offset)
    assert back["position"][1] == pytest.approx(FrankaPegInsertion.HOLE_POSITION[1] - expected_y_offset)
    assert left["position"][0] == pytest.approx(FrankaPegInsertion.HOLE_POSITION[0] - expected_x_offset)
    assert right["position"][0] == pytest.approx(FrankaPegInsertion.HOLE_POSITION[0] + expected_x_offset)

    for wall in (front, back, left, right):
        assert wall["position"][2] == pytest.approx(expected_wall_z)
        assert wall["path"].startswith(FrankaPegInsertion.HOLE_FIXTURE_PATH + "/")


def test_build_hole_fixture_parts_uses_rectangular_slot_dimensions(monkeypatch):
    monkeypatch.setattr(peg_insertion_module, "_SLOT_YAW", 0.0)
    parts = build_hole_fixture_parts(FrankaPegInsertion.HOLE_POSITION, FrankaPegInsertion.HOLE_FIXTURE_PATH)

    part_by_name = {part["name"]: part for part in parts}
    left = part_by_name["left_wall"]
    front = part_by_name["front_wall"]

    np.testing.assert_array_almost_equal(
        left["scale"],
        [_SLOT_WALL_THICKNESS_X, _SLOT_INNER_SIZE_Y, _HOLE_FIXTURE_HEIGHT - _SLOT_BASE_THICKNESS],
    )
    np.testing.assert_array_almost_equal(
        front["scale"],
        [_HOLE_FIXTURE_OUTER_SIZE_X, _SLOT_WALL_THICKNESS_Y, _HOLE_FIXTURE_HEIGHT - _SLOT_BASE_THICKNESS],
    )


def test_build_hole_fixture_parts_rotates_slot_fixture_with_slot_yaw(monkeypatch):
    monkeypatch.setattr(peg_insertion_module, "_SLOT_YAW", np.pi / 2)
    parts = build_hole_fixture_parts(FrankaPegInsertion.HOLE_POSITION, FrankaPegInsertion.HOLE_FIXTURE_PATH)

    part_by_name = {part["name"]: part for part in parts}
    front = part_by_name["front_wall"]
    left = part_by_name["left_wall"]

    expected_x_offset = _SLOT_INNER_SIZE_X / 2 + _SLOT_WALL_THICKNESS_X / 2
    expected_y_offset = _SLOT_INNER_SIZE_Y / 2 + _SLOT_WALL_THICKNESS_Y / 2

    assert front["position"][0] == pytest.approx(FrankaPegInsertion.HOLE_POSITION[0] - expected_y_offset)
    assert front["position"][1] == pytest.approx(FrankaPegInsertion.HOLE_POSITION[1])
    assert left["position"][0] == pytest.approx(FrankaPegInsertion.HOLE_POSITION[0])
    assert left["position"][1] == pytest.approx(FrankaPegInsertion.HOLE_POSITION[1] - expected_x_offset)
    np.testing.assert_allclose(front["orientation"], quaternion_from_z_yaw(np.pi / 2))
    np.testing.assert_allclose(left["orientation"], quaternion_from_z_yaw(np.pi / 2))


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
        MagicMock(numpy=lambda: np.array([_IDENTITY_QUAT])),
    )
    s.robot = MagicMock()
    s.robot.get_downward_orientation.return_value = _IDENTITY_QUAT
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
    assert s._peg_grasp_z == pytest.approx(0.04 + _EXPECTED_HAND_TO_FINGER_PAD_CENTER_OFFSET + _EXPECTED_GRASP_POINT_ABOVE_CENTER)
    assert s._hole_insert_z == pytest.approx(
        _HOLE_TOP_Z
        - _INSERTION_DEPTH
        + _EXPECTED_HAND_TO_FINGER_PAD_CENTER_OFFSET
        + _PEG_HEIGHT / 2
        + _EXPECTED_GRASP_POINT_ABOVE_CENTER
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


def test_forward_phase6_targets_measured_alignment_pose(monkeypatch):
    s = _make_scenario()
    s._event = 6
    s._step = 0
    s._peg_grasp_z = 0.098
    s._hole_insert_z = 0.128
    monkeypatch.setattr(peg_insertion_module, "_SLOT_YAW", np.pi / 6)
    s.peg.get_world_poses.return_value = (
        MagicMock(numpy=lambda: np.array([[-0.1, 0.35, 0.11]])),
        MagicMock(numpy=lambda: np.array([_IDENTITY_QUAT])),
    )

    s.forward()

    call_kwargs = s.robot.set_end_effector_pose.call_args.kwargs
    desired_alignment_peg_pos = np.array(
        [
            FrankaPegInsertion.HOLE_POSITION[0],
            FrankaPegInsertion.HOLE_POSITION[1],
            _HOLE_TOP_Z + _INSERTION_ALIGNMENT_CLEARANCE + _PEG_HEIGHT / 2,
        ]
    )
    expected_position, _ = compute_insertion_hand_pose(
        current_peg_pos=np.array([-0.1, 0.35, 0.11]),
        current_peg_orientation=_IDENTITY_QUAT,
        current_hand_pos=np.array(
            [
                FrankaPegInsertion.HOLE_POSITION[0],
                FrankaPegInsertion.HOLE_POSITION[1],
                FrankaPegInsertion.TRANSPORT_HEIGHT,
            ]
        ),
        current_hand_orientation=_IDENTITY_QUAT,
        desired_peg_pos=desired_alignment_peg_pos,
        desired_peg_orientation=quaternion_from_z_yaw(np.pi / 6),
    )
    np.testing.assert_allclose(call_kwargs["position"], expected_position)
    np.testing.assert_allclose(call_kwargs["orientation"], quaternion_from_z_yaw(np.pi / 6))


def test_forward_phase6_recomputes_hand_pose_from_measured_peg_pose():
    s = _make_scenario()
    s._event = 6
    s._step = 0
    s._peg_grasp_z = 0.098
    s._hole_insert_z = 0.128
    s.peg.get_world_poses.return_value = (
        MagicMock(numpy=lambda: np.array([[-0.092, 0.343, 0.11]])),
        MagicMock(numpy=lambda: np.array([_Z_90_QUAT])),
    )

    s.forward()

    call_kwargs = s.robot.set_end_effector_pose.call_args.kwargs
    desired_peg_orientation = choose_slot_aligned_insertion_orientation(
        _SLOT_YAW,
        np.array([_PEG_SIZE_X, _PEG_SIZE_Y]),
        np.array([_SLOT_INNER_SIZE_X, _SLOT_INNER_SIZE_Y]),
    )
    expected_position, expected_orientation = compute_insertion_hand_pose(
        current_peg_pos=np.array([-0.092, 0.343, 0.11]),
        current_peg_orientation=_Z_90_QUAT,
        current_hand_pos=np.array(
            [
                FrankaPegInsertion.HOLE_POSITION[0],
                FrankaPegInsertion.HOLE_POSITION[1],
                FrankaPegInsertion.TRANSPORT_HEIGHT,
            ]
        ),
        current_hand_orientation=_IDENTITY_QUAT,
        desired_peg_pos=np.array(
            [
                FrankaPegInsertion.HOLE_POSITION[0],
                FrankaPegInsertion.HOLE_POSITION[1],
                _HOLE_TOP_Z + _INSERTION_ALIGNMENT_CLEARANCE + _PEG_HEIGHT / 2,
            ]
        ),
        desired_peg_orientation=desired_peg_orientation,
    )
    np.testing.assert_allclose(call_kwargs["position"], expected_position)
    np.testing.assert_allclose(call_kwargs["orientation"], expected_orientation)


def test_forward_phase6_logs_peg_orientation_drift(capsys):
    s = _make_scenario()
    s._event = 6
    s._step = 0
    s._peg_grasp_z = 0.098
    s._hole_insert_z = 0.128
    s.peg.get_world_poses.return_value = (
        MagicMock(numpy=lambda: np.array([[-0.092, 0.343, 0.11]])),
        MagicMock(numpy=lambda: np.array([_Z_90_QUAT])),
    )

    s.forward()

    captured = capsys.readouterr()
    expected_orientation_error_deg = compute_quaternion_angle_error_deg(
        _Z_90_QUAT,
        choose_slot_aligned_insertion_orientation(
            _SLOT_YAW,
            np.array([_PEG_SIZE_X, _PEG_SIZE_Y]),
            np.array([_SLOT_INNER_SIZE_X, _SLOT_INNER_SIZE_Y]),
        ),
    )
    assert "[insert]" in captured.out
    assert "step=1/60" in captured.out
    assert f"peg_orientation_error_deg={expected_orientation_error_deg:.2f}" in captured.out


def test_forward_phase6_holds_alignment_height_before_descent():
    s = _make_scenario()
    s._event = 6
    s._step = 0
    s._peg_grasp_z = 0.098
    s._hole_insert_z = 0.128

    peg_measurements = [
        (
            MagicMock(numpy=lambda: np.array([[-0.092, 0.343, 0.11]])),
            MagicMock(numpy=lambda: np.array([_Z_90_QUAT])),
        ),
        (
            MagicMock(numpy=lambda: np.array([[-0.092, 0.343, 0.11]])),
            MagicMock(numpy=lambda: np.array([_Z_90_QUAT])),
        ),
    ]
    s.peg.get_world_poses.side_effect = peg_measurements

    s.forward()
    first_call = s.robot.set_end_effector_pose.call_args.kwargs

    s.forward()
    second_call = s.robot.set_end_effector_pose.call_args.kwargs

    assert second_call["position"][2] == pytest.approx(first_call["position"][2])
    assert np.linalg.norm(second_call["position"][:2] - first_call["position"][:2]) <= (
        _INSERTION_ALIGN_MAX_XY_STEP + 1e-9
    )
    assert compute_quaternion_angle_error_deg(
        first_call["orientation"],
        second_call["orientation"],
    ) <= np.degrees(_INSERTION_ALIGN_MAX_YAW_STEP) + 1e-9


def test_forward_phase6_applies_bounded_xy_and_yaw_corrections_during_alignment():
    s = _make_scenario()
    s._event = 6
    s._step = 0
    s._peg_grasp_z = 0.098
    s._hole_insert_z = 0.128

    peg_measurements = [
        (
            MagicMock(numpy=lambda: np.array([[-0.092, 0.343, 0.11]])),
            MagicMock(numpy=lambda: np.array([_Z_90_QUAT])),
        ),
        (
            MagicMock(numpy=lambda: np.array([[-0.070, 0.320, 0.108]])),
            MagicMock(numpy=lambda: np.array([_IDENTITY_QUAT])),
        ),
    ]
    s.peg.get_world_poses.side_effect = peg_measurements

    s.forward()
    first_call = s.robot.set_end_effector_pose.call_args.kwargs

    s.forward()
    second_call = s.robot.set_end_effector_pose.call_args.kwargs

    xy_delta = np.linalg.norm(second_call["position"][:2] - first_call["position"][:2])
    orientation_delta_deg = compute_quaternion_angle_error_deg(
        first_call["orientation"],
        second_call["orientation"],
    )

    assert xy_delta > 0.0
    assert xy_delta <= _INSERTION_ALIGN_MAX_XY_STEP + 1e-9
    assert orientation_delta_deg > 0.0
    assert orientation_delta_deg <= np.degrees(_INSERTION_ALIGN_MAX_YAW_STEP) + 1e-9
    assert second_call["position"][2] == pytest.approx(first_call["position"][2])


def test_forward_phase6_yaw_correction_preserves_downward_tilt():
    s = _make_scenario()
    s.robot.get_downward_orientation.return_value = _Y_90_QUAT
    s._event = 6
    s._step = 0
    s._peg_grasp_z = 0.098
    s._hole_insert_z = 0.128

    peg_measurements = [
        (
            MagicMock(numpy=lambda: np.array([[-0.1, 0.35, 0.11]])),
            MagicMock(numpy=lambda: np.array([_IDENTITY_QUAT])),
        ),
        (
            MagicMock(numpy=lambda: np.array([[-0.1, 0.35, 0.11]])),
            MagicMock(numpy=lambda: np.array([_IDENTITY_QUAT])),
        ),
    ]
    s.peg.get_world_poses.side_effect = peg_measurements

    s.forward()
    first_call = s.robot.set_end_effector_pose.call_args.kwargs

    s.forward()
    second_call = s.robot.set_end_effector_pose.call_args.kwargs

    desired_peg_orientation = choose_slot_aligned_insertion_orientation(
        _SLOT_YAW,
        np.array([_PEG_SIZE_X, _PEG_SIZE_Y]),
        np.array([_SLOT_INNER_SIZE_X, _SLOT_INNER_SIZE_Y]),
    )
    target_yaw = 2.0 * np.arctan2(desired_peg_orientation[3], desired_peg_orientation[0])
    yaw_correction = np.clip(
        _INSERTION_ALIGN_YAW_GAIN * target_yaw,
        -_INSERTION_ALIGN_MAX_YAW_STEP,
        _INSERTION_ALIGN_MAX_YAW_STEP,
    )
    expected_second_orientation = _quaternion_multiply(
        quaternion_from_z_yaw(yaw_correction),
        first_call["orientation"],
    )

    np.testing.assert_allclose(second_call["orientation"], expected_second_orientation)
    assert abs(second_call["orientation"][2]) > 0.1


def test_forward_phase6_does_not_depend_on_cached_commanded_hand_pose():
    s = _make_scenario()
    s._event = 6
    s._step = 0
    s._peg_grasp_z = 0.098
    s._hole_insert_z = 0.128
    s._commanded_hand_position = np.array([9.0, 9.0, 9.0])
    s._commanded_hand_orientation = _Y_90_QUAT
    s.peg.get_world_poses.return_value = (
        MagicMock(numpy=lambda: np.array([[-0.092, 0.343, 0.11]])),
        MagicMock(numpy=lambda: np.array([_Z_90_QUAT])),
    )

    s.forward()

    call_kwargs = s.robot.set_end_effector_pose.call_args.kwargs
    desired_peg_orientation = choose_slot_aligned_insertion_orientation(
        _SLOT_YAW,
        np.array([_PEG_SIZE_X, _PEG_SIZE_Y]),
        np.array([_SLOT_INNER_SIZE_X, _SLOT_INNER_SIZE_Y]),
    )
    expected_position, expected_orientation = compute_insertion_hand_pose(
        current_peg_pos=np.array([-0.092, 0.343, 0.11]),
        current_peg_orientation=_Z_90_QUAT,
        current_hand_pos=np.array(
            [
                FrankaPegInsertion.HOLE_POSITION[0],
                FrankaPegInsertion.HOLE_POSITION[1],
                FrankaPegInsertion.TRANSPORT_HEIGHT,
            ]
        ),
        current_hand_orientation=_IDENTITY_QUAT,
        desired_peg_pos=np.array(
            [
                FrankaPegInsertion.HOLE_POSITION[0],
                FrankaPegInsertion.HOLE_POSITION[1],
                _HOLE_TOP_Z + _INSERTION_ALIGNMENT_CLEARANCE + _PEG_HEIGHT / 2,
            ]
        ),
        desired_peg_orientation=desired_peg_orientation,
    )
    np.testing.assert_allclose(call_kwargs["position"], expected_position)
    np.testing.assert_allclose(call_kwargs["orientation"], expected_orientation)


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
