"""The one action term: execute the planner's footstep and nudge through a low-level controller.

The action vector is `gaitnet_core.action_layout`: the policy's choice (candidate index,
duration), the concrete footstep it resolves to, and a velocity command nudge. The term
reads only the footstep and nudge, so it never needs the candidate set.

The term owns the controller. Observation and reward terms reach the controller, the
nudged command and the robot's state through it (`footstep_action(env)`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from gaitnet_core import action_layout
from gaitnet_core.action_layout import EnvAction
from gaitnet_core.interfaces import FootstepCommand, LowLevelController
from gaitnet_core.state import Observation, RobotState, TerrainPatch
from gaitnet_sim import robot as go1
from gaitnet_sim.controllers import PooledMpcControllerCfg
from gaitnet_sim.env.scene import SCANNER_NAMES
from gaitnet_sim.robot_io import RobotIO

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class FootstepControlAction(ActionTerm):
    cfg: "FootstepControlActionCfg"
    _asset: Articulation
    _env: "ManagerBasedRLEnv"

    def __init__(self, cfg: "FootstepControlActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        contract = env.cfg.gaitnet
        self.spec = contract.robot_spec()
        self.grid = contract.foothold_grid()
        self.io = RobotIO(
            env.scene,
            self.spec,
            self.grid,
            robot_name=cfg.asset_name,
            joint_names=cfg.joint_names,
            foot_names=cfg.foot_names,
            contact_sensor_name=cfg.contact_sensor_name,
            scanner_names=cfg.scanner_names,
            contact_threshold=cfg.contact_threshold,
        )
        self.controller: LowLevelController = cfg.controller.class_type(
            cfg.controller, num_robots=self.num_envs, dt=env.physics_dt, device=self.device
        )
        self._raw_actions = torch.zeros(self.num_envs, action_layout.DIM, device=self.device)
        self._nudge = torch.zeros(self.num_envs, 3, device=self.device)
        self._footsteps = FootstepCommand.none(self.num_envs, device=self.device)

    def __del__(self):
        controller = getattr(self, "controller", None)
        if controller is not None:
            controller.close()
        super().__del__()

    @property
    def action_dim(self) -> int:
        return action_layout.DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def footsteps(self) -> FootstepCommand:
        """The footsteps started on the latest env step."""
        return self._footsteps

    @property
    def nudge(self) -> torch.Tensor:
        """(N, 3) the velocity command delta applied this env step."""
        return self._nudge

    def base_command(self) -> torch.Tensor:
        """(N, 3) the command manager's velocity command, before the nudge."""
        return self._env.command_manager.get_command(self.cfg.command_name)

    def effective_command(self) -> torch.Tensor:
        """(N, 3) the velocity command the controller tracks: the base command plus the nudge."""
        return self.base_command() + self._nudge

    def robot_state(self) -> RobotState:
        return self.io.robot_state(self.controller.gait_timing(), self.effective_command())

    def terrain(self) -> TerrainPatch:
        return self.io.terrain()

    def observation(self) -> Observation:
        return Observation(self.robot_state(), self.terrain())

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        action = EnvAction.decode(actions)
        self._footsteps = action.footstep_command()
        if self.cfg.apply_nudge:
            self._nudge = action.nudge.clone()
        self.controller.command_footsteps(self._footsteps)

    def apply_actions(self):
        joint_pos, joint_vel = self.io.joint_state()
        torques = self.controller.compute_torques(
            joint_pos, joint_vel, self.io.base_pose(), self.io.base_vel(), self.effective_command()
        )
        self._asset.actuators.target_command.set_effort_index(value=torques, joint_ids=self.io.joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._raw_actions[ids] = 0.0
        self._nudge[ids] = 0.0
        self._footsteps.active[ids] = False
        self.controller.reset(ids)


@configclass
class FootstepControlActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FootstepControlAction
    asset_name: str = "robot"

    controller: PooledMpcControllerCfg = PooledMpcControllerCfg()
    """Any controller cfg whose `class_type` implements `LowLevelController`."""
    command_name: str = "base_velocity"
    """The command term holding the velocity command before the nudge."""
    apply_nudge: bool = True
    """Add the action's nudge to the command. The nudge is zero unless a feedback observer
    produced one."""

    joint_names: tuple[str, ...] = go1.JOINT_NAMES
    foot_names: tuple[str, ...] = go1.FOOT_NAMES
    contact_sensor_name: str = "contact_forces"
    scanner_names: tuple[str, ...] = SCANNER_NAMES
    """One foothold scanner per leg, in leg order."""
    contact_threshold: float = 1.0
    """Normal force above which a foot counts as in contact (N)."""
