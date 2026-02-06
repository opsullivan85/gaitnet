"""Action for running low level control with an MPC"""

from dataclasses import MISSING
from typing import Sequence
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
import torch

from gaitnet import sim2real
from gaitnet.util import VectorPool
from gaitnet.simulation.util import controls_to_joint_efforts
from gaitnet import get_logger

logger = get_logger()


class MPCActionTerm(ActionTerm):
    """MPC Action Term, gets commands from action input"""

    def __init__(self, cfg: "MPCActionCfg", env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        # for type hinting
        self.cfg: "MPCActionCfg"
        self.env_cfg = env.cfg  # type: ignore
        self._asset: Articulation  # type: ignore

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)

        self._raw_actions = torch.zeros(
            (self.num_envs, self.action_dim), device=self.device
        )
        self._processed_actions = self._raw_actions

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return 3  # 3D velocity command: vx, vy, wz

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions to process.
        """
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions  # no processing for now

    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """
        processed_actions_cpu = self.processed_actions.cpu().numpy()
        robot_controllers: VectorPool[sim2real.Sim2RealInterface] = self.env_cfg.robot_controllers  # type: ignore
        torques = controls_to_joint_efforts(
            scene=self._env.scene,
            controllers=robot_controllers,
            controls=processed_actions_cpu,
            asset_name=self.cfg.asset_name,
        )
        self._asset.set_joint_effort_target(torques, self._joint_ids)
    
    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the action term.

        Args:
            env_ids: The environment IDs to reset.
        """
        # convert env_ids to numpy arrray if not none
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.cpu().numpy()  # type: ignore

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        robot_controllers: VectorPool[sim2real.Sim2RealInterface] = self.env_cfg.robot_controllers  # type: ignore
        robot_controllers.call(
            function=sim2real.Sim2RealInterface.reset,
            mask=env_ids,  # type: ignore
        )


@configclass
class MPCActionCfg(ActionTermCfg):
    """Configuration for the MPC Action Term"""

    class_type: type[ActionTerm] = MPCActionTerm

    joint_names: Sequence[str] = [".*"]
    """Regex patterns for the joint names to apply the action over.
    This should match what the controller expects.
    """

###############################################################################
###############################################################################

class MPCControlActionTerm(MPCActionTerm):
    """MPC Control Action Term, gets commands from control input"""

    def __init__(self, cfg: "MPCControlActionCfg", env: ManagerBasedRLEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        super().__init__(cfg, env)  # type: ignore
        # for type hinting
        self.cfg: "MPCControlActionCfg"  # type: ignore
        self._env: ManagerBasedRLEnv  # type: ignore

        # override action dimensions
        self._raw_actions = torch.zeros(
            (self.num_envs, self.control_dim), device=self.device
        )
        self._processed_actions = self._raw_actions

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return 0
    
    @property
    def control_dim(self) -> int:
        """Dimension of the control input."""
        return 3  # 3D velocity command: vx, vy, wz

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions to process.
        """
        # we expect actions to be empty, get real actions from command

        self._raw_actions[:] = self._env.command_manager.get_command(self.cfg.command_name)
        self._processed_actions = self._raw_actions  # no processing for now


@configclass
class MPCControlActionCfg(ActionTermCfg):
    """Configuration for the MPC Action Term"""

    class_type: type[ActionTerm] = MPCControlActionTerm

    command_name: str = MISSING  # type: ignore
    """Name of the command to use as input."""

    joint_names: Sequence[str] = [".*"]
    """Regex patterns for the joint names to apply the action over.
    This should match what the controller expects.
    """