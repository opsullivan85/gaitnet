"""What the planner observes. The simulator and the real robot both fill these.

Frames:
- base frame: attached to the trunk, x forward, y left, z up.
- yaw frame: origin at the base (or at a hip, where noted), rotated with the base's
  yaw only, so z is true vertical height.

All tensors are batched over robots (first dimension N) and legs are FL, FR, RL, RR.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch

from gaitnet_core.grid import FootholdGrid


@dataclass
class RobotState:
    foot_pos: torch.Tensor
    """(N, L, 3) foot positions relative to the base origin, in the yaw frame (m)."""
    foot_vel: torch.Tensor
    """(N, L, 3) foot velocities relative to the base, in the base frame (m/s). This is
    what leg kinematics measure (J(q) qd): a planted foot reads -(v_base + w x r_foot)."""
    base_lin_vel: torch.Tensor
    """(N, 3) base linear velocity in the base frame (m/s)."""
    base_ang_vel: torch.Tensor
    """(N, 3) base angular velocity in the base frame (rad/s)."""
    projected_gravity: torch.Tensor
    """(N, 3) unit gravity direction in the base frame ((0, 0, -1) when level)."""
    contact: torch.Tensor
    """(N, L) measured foot contact, bool."""
    gait_timing: torch.Tensor
    """(N, L, 3) the low-level controller's *scheduled* timing, not a measurement:
    swing phase in [0, 1] (0 in stance), remaining swing time (s, 0 in stance), time since
    scheduled touchdown (s, 0 in swing). A foot that lands early still reads as swinging
    until its scheduled touchdown, and the step eligibility rules rely on that."""
    command: torch.Tensor
    """(N, 3) velocity command the controller is tracking (vx, vy, yaw rate), base frame.
    When a feedback observer nudges the command, this is the nudged value."""
    base_command: torch.Tensor
    """(N, 3) the operator's velocity command before any nudge, base frame. Observers
    compute their nudges from it; the policy's features use `command`."""

    @property
    def num_robots(self) -> int:
        return self.command.shape[0]

    @property
    def num_legs(self) -> int:
        return self.contact.shape[1]

    @property
    def device(self) -> torch.device:
        return self.command.device

    def to(self, device: torch.device | str) -> "RobotState":
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def __getitem__(self, index) -> "RobotState":
        """Select a subset of robots."""
        return replace(self, **{f.name: getattr(self, f.name)[index] for f in fields(self)})


@dataclass
class TerrainPatch:
    heights: torch.Tensor
    """(N, L, *grid.patch_size) terrain height at each cell centre relative to that leg's
    hip, in the hip's yaw frame (m). -inf where the height is unknown (no return / void).
    The patch is the candidate grid plus `grid.border` cells of context on every side."""
    grid: FootholdGrid

    def to(self, device: torch.device | str) -> "TerrainPatch":
        return replace(self, heights=self.heights.to(device))

    def __getitem__(self, index) -> "TerrainPatch":
        return replace(self, heights=self.heights[index])


@dataclass
class Observation:
    """Everything the planner needs for one planning tick."""

    state: RobotState
    terrain: TerrainPatch

    @property
    def num_robots(self) -> int:
        return self.state.num_robots

    def to(self, device: torch.device | str) -> "Observation":
        return Observation(self.state.to(device), self.terrain.to(device))

    def __getitem__(self, index) -> "Observation":
        return Observation(self.state[index], self.terrain[index])
