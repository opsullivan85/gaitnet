"""The simulated robots behind `gaitnet_core.interfaces.RobotInterface`, so evaluation runs
the same `PlannerRuntime` as deployment.

Observations come from the env's footstep action term. A command is one env step: the
first footstep and the nudge go through the action vector, like a policy's; any further
footsteps go straight to the low-level controller first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from gaitnet_core.action_layout import NO_STEP_LEG, EnvAction
from gaitnet_core.interfaces import FootstepCommand, Nudge
from gaitnet_core.state import Observation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn

    from gaitnet_sim.env.actions import FootstepControlAction


class IsaacRobot:
    def __init__(self, env: "ManagerBasedRLEnv", action_name: str = "footstep"):
        self.env = env
        self.term: "FootstepControlAction" = env.action_manager.get_term(action_name)  # type: ignore[assignment]
        self.spec = self.term.spec
        self.last_step: "VecEnvStepReturn | None" = None
        """What the last `command` returned from `env.step`: observations, rewards,
        terminated, truncated, extras. Robots that terminated have already been reset."""

    @property
    def num_robots(self) -> int:
        return self.env.num_envs

    def observe(self) -> Observation:
        return self.term.observation()

    def command(self, footsteps: FootstepCommand | Sequence[FootstepCommand], nudge: Nudge | None = None) -> None:
        if isinstance(footsteps, FootstepCommand):
            footsteps = [footsteps]
        n, device = self.num_robots, self.env.device
        first = footsteps[0] if footsteps else FootstepCommand.none(n, device)
        for extra in footsteps[1:]:
            self.term.controller.command_footsteps(extra)
        action = EnvAction(
            # the choice index only matters for training's log-probabilities
            choice_index=torch.zeros(n, dtype=torch.long, device=device),
            duration=first.duration,
            leg=torch.where(first.active, first.leg, torch.full_like(first.leg, NO_STEP_LEG)),
            target=first.target,
            nudge=nudge.command_delta if nudge is not None else torch.zeros(n, 3, device=device),
        )
        self.last_step = self.env.step(action.encode())

    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        self.env.reset(env_ids=robot_ids)
