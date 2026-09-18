"""A RobotInterface that replays recorded observations, for testing the deploy path
without a simulator or a robot."""

from __future__ import annotations

from typing import Sequence

import torch

from gaitnet_core.interfaces import FootstepCommand, Nudge
from gaitnet_core.robot_spec import GO1, RobotSpec
from gaitnet_core.state import Observation


class ReplayRobot:
    def __init__(self, observations: Sequence[Observation], spec: RobotSpec = GO1, loop: bool = False):
        self.spec = spec
        self.observations = list(observations)
        self.loop = loop
        self.index = 0
        self.commands: list[tuple[list[FootstepCommand], Nudge | None]] = []
        """Every (footsteps, nudge) sent, in order."""

    @property
    def num_robots(self) -> int:
        return self.observations[0].num_robots

    def observe(self) -> Observation:
        if self.index >= len(self.observations):
            if not self.loop:
                raise StopIteration("replay exhausted")
            self.index = 0
        observation = self.observations[self.index]
        self.index += 1
        return observation

    def command(self, footsteps, nudge: Nudge | None = None) -> None:
        if isinstance(footsteps, FootstepCommand):
            footsteps = [footsteps]
        self.commands.append((list(footsteps), nudge))

    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        self.index = 0
