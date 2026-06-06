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
        raise NotImplementedError

    def forward(self, ik_method: str = "damped-least-squares") -> bool:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
