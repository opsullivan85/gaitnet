"""The boundaries between the planner, the robot and its low-level controller.

`RobotInterface` is what the planner runtime talks to. It is implemented by the
simulator (Isaac Lab plus our `LowLevelController`) and by the real robot (a ROS adapter
in front of the robot's own controller). The planner never sees torques.

`LowLevelController` turns footstep commands into joint torques. The simulator runs
one; the real robot may run ours or its own.

Everything is batched over robots (first dimension N); the real robot is N = 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from gaitnet_core.robot_spec import RobotSpec
from gaitnet_core.state import Observation


@dataclass
class FootstepCommand:
    """At most one new footstep per robot."""

    active: torch.Tensor
    """(N,) bool, False where the robot takes no new step this tick."""
    leg: torch.Tensor
    """(N,) long leg index (FL, FR, RL, RR), meaningless where inactive."""
    target: torch.Tensor
    """(N, 3) foothold in the leg's hip yaw frame (m), z is the terrain height there."""
    duration: torch.Tensor
    """(N,) swing duration (s)."""

    @classmethod
    def none(cls, num_robots: int, device: torch.device | str | None = None) -> "FootstepCommand":
        return cls(
            active=torch.zeros(num_robots, dtype=torch.bool, device=device),
            leg=torch.zeros(num_robots, dtype=torch.long, device=device),
            target=torch.zeros(num_robots, 3, device=device),
            duration=torch.zeros(num_robots, device=device),
        )


@dataclass
class Nudge:
    """Feedback from the planner to the controller, outside the learned policy.

    Only the velocity command for now; add fields as controllers learn to use them.
    """

    command_delta: torch.Tensor
    """(N, 3) added to the (vx, vy, yaw rate) command the controller tracks."""

    @classmethod
    def zeros(cls, num_robots: int, device: torch.device | str | None = None) -> "Nudge":
        return cls(command_delta=torch.zeros(num_robots, 3, device=device))

    def __add__(self, other: "Nudge") -> "Nudge":
        return Nudge(command_delta=self.command_delta + other.command_delta)


class RobotInterface(Protocol):
    """A robot (or a batch of simulated robots) as the planner runtime sees it."""

    spec: RobotSpec

    @property
    def num_robots(self) -> int: ...

    def observe(self) -> Observation:
        """The latest state and terrain."""
        ...

    def command(self, footsteps: FootstepCommand | Sequence[FootstepCommand], nudge: Nudge | None = None) -> None:
        """Start the given footsteps (several per tick allowed) and apply a nudge."""
        ...

    def reset(self, robot_ids: torch.Tensor | None = None) -> None: ...


class LowLevelController(Protocol):
    """Batched footstep-following controller.

    Joint tensors are (N, 12), leg-major FL, FR, RL, RR, and hip, thigh, calf within a
    leg. The base pose is (N, 7), world position then orientation as an xyzw quaternion.
    The base velocity is (N, 6), world linear then angular velocity.
    """

    @property
    def num_robots(self) -> int: ...

    def reset(self, robot_ids: torch.Tensor) -> None: ...

    def command_footsteps(self, footsteps: FootstepCommand) -> None: ...

    def compute_torques(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        base_pose: torch.Tensor,
        base_vel: torch.Tensor,
        velocity_command: torch.Tensor,
    ) -> torch.Tensor:
        """(N, 12) joint torques for one control step."""
        ...

    def gait_timing(self) -> torch.Tensor:
        """(N, L, 3) scheduled gait timing, see `gaitnet_core.state.RobotState.gait_timing`."""
        ...

    def close(self) -> None:
        """Release any worker processes."""
        ...
