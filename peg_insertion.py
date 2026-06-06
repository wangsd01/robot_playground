# peg_insertion.py
from __future__ import annotations
import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
_PEG_HEIGHT = 0.08                  # total peg height (m)
_FINGER_JOINT_Z_OFFSET = 0.0584     # panda_hand frame -> finger link origin (m)
_FINGERTIP_PAD_CENTER_OFFSET = 0.04525  # finger link origin -> center of finger pad contact patch (m)
_GRIPPER_HEIGHT_OFFSET = _FINGER_JOINT_Z_OFFSET + _FINGERTIP_PAD_CENTER_OFFSET
_GRASP_DEPTH_BELOW_PEG_TOP = 0.025  # target grasp band, chosen as a compromise between palm clearance and peg engagement (m)
_INSERTION_DEPTH = 0.02             # how far peg bottom should enter below the hole top (m)
_HOLE_TOP_Z = 0.05                  # z of hole fixture top face (m)
_HOLE_FIXTURE_OUTER_SIZE = 0.12
_HOLE_FIXTURE_HEIGHT = 0.05
_SLOT_INNER_WIDTH = 0.04
_SLOT_BASE_THICKNESS = 0.01
_SLOT_WALL_THICKNESS = (_HOLE_FIXTURE_OUTER_SIZE - _SLOT_INNER_WIDTH) / 2


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    normalized = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(normalized)
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero.")
    normalized = normalized / norm
    if normalized[0] < 0.0:
        normalized = -normalized
    return normalized


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.array([w, -x, -y, -z], dtype=float)


def _quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(lhs, dtype=float)
    w2, x2, y2, z2 = np.asarray(rhs, dtype=float)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = _normalize_quaternion(quaternion)
    vector_quaternion = np.array([0.0, *np.asarray(vector, dtype=float)], dtype=float)
    rotated = _quaternion_multiply(
        _quaternion_multiply(quaternion, vector_quaternion),
        _quaternion_conjugate(quaternion),
    )
    return rotated[1:]


def compute_peg_grasp_z(peg_center_z: float) -> float:
    """EE z-target so the fingers grasp a stable upper-middle band of the peg."""
    peg_top_z = peg_center_z + _PEG_HEIGHT / 2
    grasp_point_z = peg_top_z - _GRASP_DEPTH_BELOW_PEG_TOP
    return grasp_point_z + _GRIPPER_HEIGHT_OFFSET


def compute_hole_insert_z() -> float:
    """EE z-target so the peg bottom finishes slightly below the hole top face."""
    grasp_point_above_center = _PEG_HEIGHT / 2 - _GRASP_DEPTH_BELOW_PEG_TOP
    return _HOLE_TOP_Z - _INSERTION_DEPTH + _GRIPPER_HEIGHT_OFFSET + _PEG_HEIGHT / 2 + grasp_point_above_center


def compute_insertion_hand_pose(
    current_peg_pos: np.ndarray,
    current_peg_orientation: np.ndarray,
    current_hand_pos: np.ndarray,
    current_hand_orientation: np.ndarray,
    desired_peg_pos: np.ndarray,
    desired_peg_orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Retarget the hand pose so a rigidly grasped peg moves onto the desired peg pose."""
    current_peg_pos = np.asarray(current_peg_pos, dtype=float)
    current_hand_pos = np.asarray(current_hand_pos, dtype=float)
    desired_peg_pos = np.asarray(desired_peg_pos, dtype=float)
    current_peg_orientation = _normalize_quaternion(current_peg_orientation)
    current_hand_orientation = _normalize_quaternion(current_hand_orientation)
    desired_peg_orientation = _normalize_quaternion(desired_peg_orientation)

    hand_to_peg_orientation = _quaternion_multiply(
        _quaternion_conjugate(current_hand_orientation),
        current_peg_orientation,
    )
    hand_to_peg_translation = _rotate_vector(
        _quaternion_conjugate(current_hand_orientation),
        current_peg_pos - current_hand_pos,
    )

    hand_orientation = _normalize_quaternion(
        _quaternion_multiply(
            desired_peg_orientation,
            _quaternion_conjugate(hand_to_peg_orientation),
        )
    )
    desired_hand_to_peg_translation = _rotate_vector(hand_orientation, hand_to_peg_translation)
    hand_position = desired_peg_pos - desired_hand_to_peg_translation
    return hand_position, hand_orientation


def compute_quaternion_angle_error_deg(current: np.ndarray, target: np.ndarray) -> float:
    """Return the smallest orientation difference between two quaternions in degrees."""
    current = _normalize_quaternion(current)
    target = _normalize_quaternion(target)
    dot = float(np.clip(abs(np.dot(current, target)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def check_success(peg_pos: np.ndarray, hole_pos: np.ndarray) -> bool:
    """True if peg is within 5 mm XY of hole and peg bottom is at or below hole top + 10 mm."""
    xy_dist = float(np.linalg.norm(peg_pos[:2] - hole_pos[:2]))
    peg_bottom_z = float(peg_pos[2]) - _PEG_HEIGHT / 2
    return xy_dist <= 0.005 and peg_bottom_z <= _HOLE_TOP_Z + 0.010


def build_hole_fixture_parts(hole_top_center: np.ndarray, root_path: str) -> list[dict[str, object]]:
    """Return the base and walls for an open-top slot fixture."""
    hole_x = float(hole_top_center[0])
    hole_y = float(hole_top_center[1])
    fixture_bottom_z = float(hole_top_center[2]) - _HOLE_FIXTURE_HEIGHT
    wall_height = _HOLE_FIXTURE_HEIGHT - _SLOT_BASE_THICKNESS
    base_center_z = fixture_bottom_z + _SLOT_BASE_THICKNESS / 2
    wall_center_z = fixture_bottom_z + _SLOT_BASE_THICKNESS + wall_height / 2
    wall_offset = _SLOT_INNER_WIDTH / 2 + _SLOT_WALL_THICKNESS / 2

    return [
        {
            "name": "base",
            "path": f"{root_path}/base",
            "position": np.array([hole_x, hole_y, base_center_z]),
            "scale": np.array([_HOLE_FIXTURE_OUTER_SIZE, _HOLE_FIXTURE_OUTER_SIZE, _SLOT_BASE_THICKNESS]),
        },
        {
            "name": "front_wall",
            "path": f"{root_path}/front_wall",
            "position": np.array([hole_x, hole_y + wall_offset, wall_center_z]),
            "scale": np.array([_HOLE_FIXTURE_OUTER_SIZE, _SLOT_WALL_THICKNESS, wall_height]),
        },
        {
            "name": "back_wall",
            "path": f"{root_path}/back_wall",
            "position": np.array([hole_x, hole_y - wall_offset, wall_center_z]),
            "scale": np.array([_HOLE_FIXTURE_OUTER_SIZE, _SLOT_WALL_THICKNESS, wall_height]),
        },
        {
            "name": "left_wall",
            "path": f"{root_path}/left_wall",
            "position": np.array([hole_x - wall_offset, hole_y, wall_center_z]),
            "scale": np.array([_SLOT_WALL_THICKNESS, _SLOT_INNER_WIDTH, wall_height]),
        },
        {
            "name": "right_wall",
            "path": f"{root_path}/right_wall",
            "position": np.array([hole_x + wall_offset, hole_y, wall_center_z]),
            "scale": np.array([_SLOT_WALL_THICKNESS, _SLOT_INNER_WIDTH, wall_height]),
        },
    ]


# ── Scenario class ────────────────────────────────────────────────────────────

class FrankaPegInsertion:
    # Scene paths
    ROBOT_PATH        = "/World/robot"
    PEG_PATH          = "/World/Peg"
    HOLE_FIXTURE_PATH = "/World/HoleFixture"

    # Default poses
    PEG_INITIAL_POSITION    = np.array([0.4, 0.0, 0.04])
    PEG_INITIAL_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])   # w-first
    HOLE_POSITION           = np.array([-0.1, 0.35, 0.05])       # hole top-face center

    # Motion constants
    TRANSPORT_HEIGHT = 0.35   # safe travel height for EE (m)
    INSERTION_LOG_INTERVAL = 1

    # Phase step counts  [approach, descend, settle, grasp, lift, transport, insert, release, retract]
    EVENTS_DT = [80, 60, 20, 20, 60, 80, 60, 20, 40]

    def __init__(self) -> None:
        self._event: int = 0
        self._step: int = 0
        self._peg_grasp_z: float | None = None
        self._hole_insert_z: float | None = None
        self._insert_target_position: np.ndarray | None = None
        self._insert_target_orientation: np.ndarray | None = None
        self._insert_start_z: float | None = None
        self._previous_insert_peg_pos: np.ndarray | None = None
        self._previous_insert_orientation_error_deg: float | None = None
        self.robot = None   # Franka — set by setup_scene()
        self.peg = None     # RigidPrim — set by setup_scene()

    def is_done(self) -> bool:
        return self._event >= len(self.EVENTS_DT)

    def _command_hand_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        ik_method: str,
    ) -> None:
        self.robot.set_end_effector_pose(
            position=np.asarray(position, dtype=float),
            orientation=_normalize_quaternion(orientation),
            ik_method=ik_method,
        )

    def _log_insertion_state(self, peg_pos: np.ndarray, peg_orientation: np.ndarray) -> None:
        total_steps = self.EVENTS_DT[6]
        step_index = self._step + 1
        should_log = (
            self._step == 0
            or step_index == total_steps
            or self._step % self.INSERTION_LOG_INTERVAL == 0
        )
        if not should_log:
            return

        orientation_error_deg = compute_quaternion_angle_error_deg(
            peg_orientation,
            self.PEG_INITIAL_ORIENTATION,
        )
        peg_xy_error_mm = float(np.linalg.norm(peg_pos[:2] - self.HOLE_POSITION[:2]) * 1000.0)
        peg_xy_delta_mm = 0.0
        if self._previous_insert_peg_pos is not None:
            peg_xy_delta_mm = float(np.linalg.norm(peg_pos[:2] - self._previous_insert_peg_pos[:2]) * 1000.0)
        orientation_delta_deg = 0.0
        if self._previous_insert_orientation_error_deg is not None:
            orientation_delta_deg = abs(orientation_error_deg - self._previous_insert_orientation_error_deg)

        peg_orientation_str = np.array2string(
            _normalize_quaternion(peg_orientation),
            precision=4,
            suppress_small=True,
        )
        target_orientation_str = np.array2string(
            self.PEG_INITIAL_ORIENTATION,
            precision=4,
            suppress_small=True,
        )
        print(
            "[insert] "
            f"step={step_index}/{total_steps} "
            f"peg_xy=({peg_pos[0]:.4f}, {peg_pos[1]:.4f}) "
            f"peg_xy_error_mm={peg_xy_error_mm:.2f} "
            f"peg_xy_delta_mm={peg_xy_delta_mm:.2f} "
            f"peg_orientation={peg_orientation_str} "
            f"target_orientation={target_orientation_str} "
            f"peg_orientation_error_deg={orientation_error_deg:.2f} "
            f"peg_orientation_delta_deg={orientation_delta_deg:.2f}"
        )
        self._previous_insert_peg_pos = np.asarray(peg_pos, dtype=float).copy()
        self._previous_insert_orientation_error_deg = orientation_error_deg

    def setup_scene(self) -> None:
        import isaacsim.core.experimental.utils.app as app_utils
        app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")

        from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane
        from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
        from isaacsim.robot.experimental.manipulators.examples.franka.franka import Franka

        GroundPlane("/World/ground_plane")
        DomeLight("/World/DomeLight").set_intensities(1000)

        self.robot = Franka(robot_path=self.ROBOT_PATH, create_robot=True)

        peg_cube = Cube(
            paths=self.PEG_PATH,
            positions=self.PEG_INITIAL_POSITION,
            orientations=self.PEG_INITIAL_ORIENTATION,
            sizes=1.0,
            scales=np.array([0.03, 0.03, 0.08]),
            colors="red",
        )
        GeomPrim(paths=peg_cube.paths, apply_collision_apis=True)
        self.peg = RigidPrim(paths=peg_cube.paths)

        for part in build_hole_fixture_parts(self.HOLE_POSITION, self.HOLE_FIXTURE_PATH):
            hole_part = Cube(
                paths=part["path"],
                positions=part["position"],
                orientations=np.array([1.0, 0.0, 0.0, 0.0]),
                sizes=1.0,
                scales=part["scale"],
                colors="gray",
            )
            GeomPrim(paths=hole_part.paths, apply_collision_apis=True)

    def forward(self, ik_method: str = "damped-least-squares") -> bool:
        if self.is_done():
            return False

        peg_positions, peg_orientations = self.peg.get_world_poses()
        peg_pos = peg_positions.numpy()[0]
        peg_orientation = peg_orientations.numpy()[0]
        goal_orientation = self.robot.get_downward_orientation()

        if self._event == 1 and self._step == 0:
            self._peg_grasp_z = compute_peg_grasp_z(float(peg_pos[2]))
            self._hole_insert_z = compute_hole_insert_z()

        peg_x, peg_y = float(peg_pos[0]), float(peg_pos[1])
        hole_x, hole_y = float(self.HOLE_POSITION[0]), float(self.HOLE_POSITION[1])

        if self._event == 0:
            self._command_hand_pose(
                position=np.array([peg_x, peg_y, self.TRANSPORT_HEIGHT]),
                orientation=goal_orientation,
                ik_method=ik_method,
            )
        elif self._event == 1:
            self._command_hand_pose(
                position=np.array([peg_x, peg_y, self._peg_grasp_z]),
                orientation=goal_orientation,
                ik_method=ik_method,
            )
        elif self._event == 2:
            self._command_hand_pose(
                position=np.array([peg_x, peg_y, self._peg_grasp_z]),
                orientation=goal_orientation,
                ik_method=ik_method,
            )
        elif self._event == 3:
            self.robot.close_gripper()
        elif self._event == 4:
            self._command_hand_pose(
                position=np.array([peg_x, peg_y, self.TRANSPORT_HEIGHT]),
                orientation=goal_orientation,
                ik_method=ik_method,
            )
        elif self._event == 5:
            self._command_hand_pose(
                position=np.array([hole_x, hole_y, self.TRANSPORT_HEIGHT]),
                orientation=goal_orientation,
                ik_method=ik_method,
            )
        elif self._event == 6:
            if self._previous_insert_peg_pos is None:
                self._previous_insert_peg_pos = np.asarray(peg_pos, dtype=float).copy()
            if self._previous_insert_orientation_error_deg is None:
                self._previous_insert_orientation_error_deg = compute_quaternion_angle_error_deg(
                    peg_orientation,
                    self.PEG_INITIAL_ORIENTATION,
                )
            desired_peg_pos = np.array(
                [hole_x, hole_y, _HOLE_TOP_Z - _INSERTION_DEPTH + _PEG_HEIGHT / 2],
                dtype=float,
            )
            current_hand_pos = np.array([hole_x, hole_y, self.TRANSPORT_HEIGHT], dtype=float)
            current_hand_orientation = goal_orientation
            if self._step == 0:
                (
                    self._insert_target_position,
                    self._insert_target_orientation,
                ) = compute_insertion_hand_pose(
                    current_peg_pos=peg_pos,
                    current_peg_orientation=peg_orientation,
                    current_hand_pos=current_hand_pos,
                    current_hand_orientation=current_hand_orientation,
                    desired_peg_pos=desired_peg_pos,
                    desired_peg_orientation=self.PEG_INITIAL_ORIENTATION,
                )
                self._insert_start_z = float(current_hand_pos[2])

            z_progress = (self._step + 1) / self.EVENTS_DT[6]
            command_position = self._insert_target_position.copy()
            command_position[2] = (
                self._insert_start_z
                + z_progress * (float(self._insert_target_position[2]) - self._insert_start_z)
            )
            self._log_insertion_state(peg_pos, peg_orientation)
            self._command_hand_pose(
                position=command_position,
                orientation=self._insert_target_orientation,
                ik_method=ik_method,
            )
        elif self._event == 7:
            self.robot.open_gripper()
        elif self._event == 8:
            self._command_hand_pose(
                position=np.array([hole_x, hole_y, self.TRANSPORT_HEIGHT]),
                orientation=goal_orientation,
                ik_method=ik_method,
            )

        self._step += 1
        if self._step >= self.EVENTS_DT[self._event]:
            self._event += 1
            self._step = 0

        return not self.is_done()

    def reset(self) -> None:
        self._event = 0
        self._step = 0
        self._peg_grasp_z = None
        self._hole_insert_z = None
        self._insert_target_position = None
        self._insert_target_orientation = None
        self._insert_start_z = None
        self._previous_insert_peg_pos = None
        self._previous_insert_orientation_error_deg = None
        self.robot.reset_to_default_pose()
        self.peg.set_world_poses(
            positions=self.PEG_INITIAL_POSITION[np.newaxis],
            orientations=self.PEG_INITIAL_ORIENTATION[np.newaxis],
        )


def main() -> None:
    from isaacsim.simulation_app import SimulationApp
    simulation_app = SimulationApp({"headless": False})

    import omni.timeline
    from isaacsim.core.simulation_manager import SimulationManager

    scenario = FrankaPegInsertion()
    scenario.setup_scene()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    for _ in range(10):
        SimulationManager.step(steps=1)
        simulation_app.update()

    scenario.reset()

    while simulation_app.is_running() and scenario.forward():
        SimulationManager.step(steps=1)
        simulation_app.update()

    peg_pos = scenario.peg.get_world_poses()[0].numpy()[0]
    success = check_success(peg_pos, FrankaPegInsertion.HOLE_POSITION)
    status = "SUCCEEDED" if success else "FAILED"
    print(f"\nInsertion {status}")
    print(f"  Peg position:   {peg_pos}")
    print(f"  Hole center:    {FrankaPegInsertion.HOLE_POSITION}")
    xy_dist = float(np.linalg.norm(peg_pos[:2] - FrankaPegInsertion.HOLE_POSITION[:2]))
    print(f"  XY error:       {xy_dist * 1000:.1f} mm  (threshold 5 mm)")

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
