"""Robot description shared by the planner, the simulator and the real robot.

Leg order is FL, FR, RL, RR everywhere: per-leg tensors, footstep commands, terrain
patches and joint blocks (hip, thigh, calf within a leg).
"""

from __future__ import annotations

from dataclasses import dataclass

LEG_NAMES: tuple[str, ...] = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class RobotSpec:
    name: str
    hip_offsets: tuple[tuple[float, float, float], ...]
    """Hip (abduction joint) positions in the base frame, per leg (m)."""
    abad_length: float
    thigh_length: float
    calf_length: float
    nominal_height: float
    """Base height above the stance feet that the controller tracks (m)."""
    reach_band: tuple[float, float]
    """(lowest, highest) foothold height relative to the hip, in the hip's gravity-aligned
    yaw frame (m). Negative is below the hip."""
    swing_duration_range: tuple[float, float] = (0.1, 0.3)
    """(min, max) swing duration the planner may command (s)."""
    leg_names: tuple[str, ...] = LEG_NAMES

    @property
    def num_legs(self) -> int:
        return len(self.leg_names)

    @property
    def num_joints(self) -> int:
        return 3 * self.num_legs


GO1 = RobotSpec(
    name="go1",
    # matches Quadruped(RobotType.GO1) in gaitnet_mpc
    hip_offsets=(
        (0.1881, 0.04675, 0.0),
        (0.1881, -0.04675, 0.0),
        (-0.1881, 0.04675, 0.0),
        (-0.1881, -0.04675, 0.0),
    ),
    abad_length=0.08,
    thigh_length=0.213,
    calf_length=0.213,
    nominal_height=0.26,
    # leg fully extended is ~0.43 m; leave margin at both ends of the workspace
    reach_band=(-0.38, -0.12),
)

ROBOTS: dict[str, RobotSpec] = {GO1.name: GO1}
