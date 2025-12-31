from dataclasses import MISSING
from typing import Sequence
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
import torch

class FixedVelocityCommand(CommandTerm):
    r"""Command generator that generates a fixed velocity command in SE(3).
    """
    def __init__(self, cfg: "FixedVelocityCommandCfg", env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)  # type: ignore
        self._command = torch.tensor(cfg.command, device=env.device).unsqueeze(0).expand(env.num_envs, -1)  # type: ignore
    
    @property
    def command(self) -> torch.Tensor:
        """Get the current command.

        Returns:
            The current command, shape (num_envs, 3).
        """
        return self._command
    
    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

@configclass
class FixedVelocityCommandCfg(CommandTermCfg):
    """Configuration for the fixed velocity command generator."""

    class_type: type = FixedVelocityCommand
    resampling_time_range: tuple[float, float] = (0,0)
    command: tuple[float, float, float] = MISSING  # type: ignore
    """The fixed command to be used, in the form of (vx, vy, wz)."""
