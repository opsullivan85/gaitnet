"""The simulator's action vector.

The RL library stores and replays actions, so the vector carries both what the policy
chose (the candidate index and duration, needed to recompute log-probabilities) and what
the environment should do (a concrete footstep and a nudge). The environment reads only
the latter, so it never needs the candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaitnet_core.interfaces import FootstepCommand, Nudge

FIELDS: tuple[str, ...] = (
    "choice_index",
    "duration",
    "leg",
    "x",
    "y",
    "z",
    "nudge_vx",
    "nudge_vy",
    "nudge_wz",
)
DIM = len(FIELDS)
NO_STEP_LEG = -1


@dataclass
class EnvAction:
    choice_index: torch.Tensor
    """(N,) flat candidate index the policy chose, the no-op index for no step."""
    duration: torch.Tensor
    """(N,) swing duration (s)."""
    leg: torch.Tensor
    """(N,) leg to step, -1 for no step."""
    target: torch.Tensor
    """(N, 3) foothold in the leg's hip yaw frame (m)."""
    nudge: torch.Tensor
    """(N, 3) velocity command delta."""

    def encode(self) -> torch.Tensor:
        """(N, DIM) float tensor."""
        return torch.cat(
            [
                self.choice_index.float().unsqueeze(-1),
                self.duration.float().unsqueeze(-1),
                self.leg.float().unsqueeze(-1),
                self.target.float(),
                self.nudge.float(),
            ],
            dim=-1,
        )

    @classmethod
    def decode(cls, action: torch.Tensor) -> "EnvAction":
        return cls(
            choice_index=action[:, 0].round().long(),
            duration=action[:, 1],
            leg=action[:, 2].round().long(),
            target=action[:, 3:6],
            nudge=action[:, 6:9],
        )

    def footstep_command(self) -> FootstepCommand:
        active = self.leg != NO_STEP_LEG
        return FootstepCommand(
            active=active,
            leg=self.leg.clamp(min=0),
            target=self.target,
            duration=self.duration,
        )

    def nudge_command(self) -> Nudge:
        return Nudge(command_delta=self.nudge)
