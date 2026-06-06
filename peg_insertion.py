# peg_insertion.py
from __future__ import annotations
import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
_PEG_HEIGHT = 0.08            # total peg height (m)
_GRIPPER_HEIGHT_OFFSET = 0.058  # panda_hand frame → finger grip point (m)
_HOLE_TOP_Z = 0.05            # z of hole fixture top face (m)
_INSERTION_DEPTH = 0.02       # how far peg bottom enters below hole top (m)


def compute_peg_grasp_z(peg_center_z: float) -> float:
    """EE z-target so fingers align with peg center."""
    return peg_center_z + _GRIPPER_HEIGHT_OFFSET


def compute_hole_insert_z() -> float:
    """EE z-target so peg bottom is INSERTION_DEPTH below hole top face."""
    return _HOLE_TOP_Z - _INSERTION_DEPTH + _GRIPPER_HEIGHT_OFFSET + _PEG_HEIGHT / 2


def check_success(peg_pos: np.ndarray, hole_pos: np.ndarray) -> bool:
    """True if peg is within 5 mm XY of hole and peg bottom is at or below hole top + 10 mm."""
    xy_dist = float(np.linalg.norm(peg_pos[:2] - hole_pos[:2]))
    peg_bottom_z = float(peg_pos[2]) - _PEG_HEIGHT / 2
    return xy_dist <= 0.005 and peg_bottom_z <= _HOLE_TOP_Z + 0.010


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

    # Phase step counts  [approach, descend, settle, grasp, lift, transport, insert, release, retract]
    EVENTS_DT = [80, 60, 20, 20, 60, 80, 60, 20, 40]

    def __init__(self) -> None:
        self._event: int = 0
        self._step: int = 0
        self._peg_grasp_z: float | None = None
        self._hole_insert_z: float | None = None
        self.robot = None   # Franka — set by setup_scene()
        self.peg = None     # RigidPrim — set by setup_scene()

    def is_done(self) -> bool:
        return self._event >= len(self.EVENTS_DT)

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

        Cube(
            paths=self.HOLE_FIXTURE_PATH,
            positions=np.array([-0.1, 0.35, 0.025]),
            orientations=np.array([1.0, 0.0, 0.0, 0.0]),
            sizes=1.0,
            scales=np.array([0.12, 0.12, 0.05]),
            colors="gray",
        )

    def forward(self, ik_method: str = "damped-least-squares") -> bool:
        if self.is_done():
            return False

        peg_pos = self.peg.get_world_poses()[0].numpy()[0]
        goal_orientation = self.robot.get_downward_orientation()

        if self._event == 1 and self._step == 0:
            self._peg_grasp_z = compute_peg_grasp_z(float(peg_pos[2]))
            self._hole_insert_z = compute_hole_insert_z()

        peg_x, peg_y = float(peg_pos[0]), float(peg_pos[1])
        hole_x, hole_y = float(self.HOLE_POSITION[0]), float(self.HOLE_POSITION[1])

        if self._event == 0:
            self.robot.set_end_effector_pose(
                position=np.array([peg_x, peg_y, self.TRANSPORT_HEIGHT]),
                orientation=goal_orientation,
                ik_method=ik_method,
            )
        elif self._event == 1:
            self.robot.set_end_effector_pose(
                position=np.array([peg_x, peg_y, self._peg_grasp_z]),
                orientation=goal_orientation,
                ik_method=ik_method,
            )
        elif self._event == 2:
            self.robot.set_end_effector_pose(
                position=np.array([peg_x, peg_y, self._peg_grasp_z]),
                orientation=goal_orientation,
                ik_method=ik_method,
            )
        elif self._event == 3:
            self.robot.close_gripper()
        elif self._event == 4:
            self.robot.set_end_effector_pose(
                position=np.array([peg_x, peg_y, self.TRANSPORT_HEIGHT]),
                orientation=goal_orientation,
                ik_method=ik_method,
            )
        elif self._event == 5:
            self.robot.set_end_effector_pose(
                position=np.array([hole_x, hole_y, self.TRANSPORT_HEIGHT]),
                orientation=goal_orientation,
                ik_method=ik_method,
            )
        elif self._event == 6:
            self.robot.set_end_effector_pose(
                position=np.array([hole_x, hole_y, self._hole_insert_z]),
                orientation=goal_orientation,
                ik_method=ik_method,
            )
        elif self._event == 7:
            self.robot.open_gripper()
        elif self._event == 8:
            self.robot.set_end_effector_pose(
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
        raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
