"""The one action term: execute the planner's footstep and nudge through a low-level controller.

The action vector is `gaitnet_core.action_layout`: the policy's choice (candidate index,
duration), the concrete footstep it resolves to, and a velocity command nudge. The term
reads only the footstep and nudge, so it never needs the candidate set.

The term owns the controller. Observation and reward terms reach the controller, the
nudged command and the robot's state through it (`footstep_action(env)`). It also keeps the
per-episode gait statistics of the training log (`gaitnet_sim.env.metrics`). Its cfg is in
`gaitnet_sim.env.actions_cfg`, which names this class by string so task cfgs import
without the simulator.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm

from gaitnet_core import action_layout
from gaitnet_core.action_layout import EnvAction
from gaitnet_core.interfaces import FootstepCommand, LowLevelController
from gaitnet_core.state import Observation, RobotState, TerrainPatch
from gaitnet_sim.env.metrics import GaitMetrics
from gaitnet_sim.env.noise import corrupt
from gaitnet_sim.robot_io import RobotIO

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from gaitnet_sim.env.actions_cfg import FootstepControlActionCfg


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
        self._planner_observation: Observation | None = None
        self._metrics = GaitMetrics(self.io, self.spec, env.step_dt)

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
        return self.io.robot_state(self.controller.gait_timing(), self.effective_command(), self.base_command())

    def terrain(self) -> TerrainPatch:
        return self.io.terrain()

    def observation(self) -> Observation:
        """The truth: what terminations, rewards and privileged observations read."""
        return Observation(self.robot_state(), self.terrain())

    def planner_observation(self) -> Observation:
        """What the planner sees: the truth with `cfg.observation_noise` added, if any. One draw
        per env step, so every observation group sees the same corrupted world."""
        if self._planner_observation is None:
            observation = self.observation()
            if self.cfg.observation_noise is not None:
                observation = corrupt(observation, self.cfg.observation_noise)
            self._planner_observation = observation
        return self._planner_observation

    def process_actions(self, actions: torch.Tensor):
        # the robots are about to move
        self._planner_observation = None
        # copied into the buffers made at init: tensors made here under torch.inference_mode
        # (RSL-RL, evaluation) would refuse the in-place updates of a later reset outside it
        self._raw_actions[:] = actions
        action = EnvAction.decode(actions)
        footsteps = action.footstep_command()
        self._footsteps.active[:] = footsteps.active
        self._footsteps.leg[:] = footsteps.leg
        self._footsteps.target[:] = footsteps.target
        # the sampled duration is the network's mean plus unbounded noise
        low, high = self.spec.swing_duration_range
        self._footsteps.duration[:] = torch.where(footsteps.active, footsteps.duration.clamp(low, high), 0.0)
        if self.cfg.apply_nudge:
            self._nudge[:] = action.nudge
        self._metrics.record(self._footsteps, self._nudge)
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
        # the action manager drops what its terms' reset returns, so the metrics go in directly
        self._env.extras.setdefault("log", {}).update(self._metrics.pop(ids))
        self._raw_actions[ids] = 0.0
        self._nudge[ids] = 0.0
        self._footsteps.active[ids] = False
        self.controller.reset(ids)
        self._planner_observation = None
