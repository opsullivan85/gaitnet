"""A constant velocity command, for evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class FixedVelocityCommand(CommandTerm):
    """The same (vx, vy, yaw rate) for every robot, until `set_command` changes it."""

    cfg: "FixedVelocityCommandCfg"

    def __init__(self, cfg: "FixedVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self._command = torch.zeros(env.num_envs, 3, device=env.device)
        self.set_command(cfg.command)

    def set_command(self, command: tuple[float, float, float]) -> None:
        """Overwrite every robot's command; how an evaluation sweeps velocities in one scene."""
        self._command[:] = torch.as_tensor(command, device=self._command.device, dtype=self._command.dtype)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@configclass
class FixedVelocityCommandCfg(CommandTermCfg):
    class_type: type = FixedVelocityCommand
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    command: tuple[float, float, float] = MISSING
    """(vx, vy, yaw rate), base frame."""
