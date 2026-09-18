"""Action for running footstep controller"""

from typing import Sequence, TextIO
from gaitnet.constants import NO_STEP
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
import torch

from gaitnet.sim2real.siminterface import Sim2RealInterface, SimInterface
from gaitnet_mpc.pool import VectorPool
import numpy as np
import gaitnet.constants as const
from gaitnet import get_logger, PROJECT_ROOT

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaitnet.gaitnet.components.gaitnet_observation_manager import (
        GaitNetObservationManager,
    )

logger = get_logger()

_log_files: dict[str, TextIO] = {}


def _log_file(name: str) -> TextIO:
    """Open (once) a csv under data/<name>/ for the experiment logging flags in `const.experiments`."""
    if name not in _log_files:
        folder = PROJECT_ROOT / "data" / name
        folder.mkdir(parents=True, exist_ok=True)
        _log_files[name] = open(folder / f"{name}_log.csv", "a")
    return _log_files[name]


class FSCActionTerm(ActionTerm):
    """Footstep Controller (FSC) Action Term"""

    def __init__(self, cfg: "FSCActionCfg", env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        # for type hinting
        self.cfg: "FSCActionCfg"
        self.env_cfg = env.cfg
        self._asset: Articulation  # type: ignore

        self._raw_actions = torch.zeros(
            (self.num_envs, self.action_dim), device=self.device
        )
        self._processed_actions = self._raw_actions

    def _get_option_manager(self) -> "GaitNetObservationManager":
        """Get the footstep option manager.

        note that it isn't initilized until after the action term is initialized

        Returns:
            The footstep option manager.
        """
        footstep_option_manager: "GaitNetObservationManager" = self._env.observation_manager  # type: ignore
        return footstep_option_manager

    @property
    def action_dim(self) -> int:
        """Dimension of the action term.

        Returns 2: flat candidate index (the no-op is num_legs * candidates_per_leg) and duration.
        """
        return 2  # Action index + duration

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @staticmethod
    def footstep_kwargs(processed_actions: np.ndarray) -> dict[str, np.ndarray]:
        """Generate the kwargs for the footstep initiation call.

        Args:
            processed_actions: The processed actions (on cpu).

        Returns:
            The kwargs for the footstep initiation call.
        """
        # legs are in FL, FR, RL, RR order, which is also the Sim2RealInterface order
        legs = processed_actions[:, 0].astype(np.int32)
        return {
            "leg": legs,
            "location_hip": processed_actions[:, 1:3],
            "duration": processed_actions[:, 3],
        }

    def action_indices_to_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert actions (index + duration) to footstep actions.

        Args:
            actions: The actions from the policy. (num_envs, 2) where:
                     - Column 0: flat candidate index, see gaitnet_core.candidates
                     - Column 1: duration value

        Returns:
            Footstep actions.
            (num_envs, 4) where each action is (leg, x, y, duration), leg NO_STEP for the no-op
        """
        # Extract action indices and durations
        action_indices = actions[:, 0].round().long()  # (num_envs,)
        durations = actions[:, 1]  # (num_envs,)

        # the candidates the policy chose from, see GaitNetObservationManager
        candidates = self._get_option_manager().candidates
        is_step, leg, xyz = candidates.gather(action_indices)
        leg = torch.where(is_step, leg, torch.full_like(leg, NO_STEP))

        # Durations are intentionally not clipped to the valid range: clipping made
        # overlong swings free, so the policy pushed the mean duration to the cap.
        selected_actions = torch.cat(
            [leg.float().unsqueeze(-1), xyz[:, :2], durations.unsqueeze(-1)], dim=-1
        )

        return selected_actions  # (num_envs, 4) - (leg, x, y, duration)

    @staticmethod
    def log_actions(processed_actions: np.ndarray):
        """Log the processed actions to files if logging is enabled.

        Args:
            processed_actions: The processed actions (on cpu).
        """
        if const.experiments.contact_schedule_logging:
            action = processed_actions[0]
            contact_schedule_log = _log_file("contact_schedule")
            contact_schedule_log.write(
                f"{action[0]},{action[1]},{action[2]},{action[3]}\n"
            )
            contact_schedule_log.flush()

        if const.experiments.swing_duration_logging:
            # filter by valid steps
            valid_swing_mask = processed_actions[:, 0] != NO_STEP
            valid_swing_durations = processed_actions[valid_swing_mask][:, 3]
            valid_swing_legs = processed_actions[valid_swing_mask][:, 0]

            swing_duration_log = _log_file("swing_duration")
            swing_duration_log.writelines(
                [f"{d},{l}\n" for l, d in zip(valid_swing_legs, valid_swing_durations)]
            )
            swing_duration_log.flush()

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions from the policy (num_envs, 2) where:
                     - Column 0: flat candidate index
                     - Column 1: duration value
        """
        # Store raw actions
        self._raw_actions = actions

        # Convert actions (index + duration) to footstep actions (leg, x, y, duration)
        self._processed_actions = self.action_indices_to_actions(actions)
        processed_actions_cpu = self.processed_actions.cpu().numpy()

        # perform logging if enabled
        FSCActionTerm.log_actions(processed_actions_cpu)

        # ablate swing duration if specified
        if const.experiments.ablate_swing_duration:
            processed_actions_cpu[:, 3] = const.experiments.constant_swing_duration

        # mask out invalid steps
        mask = processed_actions_cpu[:, 0] != NO_STEP
        footstep_parameters = self.footstep_kwargs(processed_actions_cpu)

        # initiate the footsteps
        robot_controllers: VectorPool[Sim2RealInterface] = self.env_cfg.robot_controllers  # type: ignore
        robot_controllers.call(
            function=Sim2RealInterface.initiate_footstep,
            mask=mask,
            **footstep_parameters,
        )

    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """
        # we don't do anything here
        pass

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the action term.

        Args:
            env_ids: The environment IDs to reset.
        """
        super().reset(env_ids)


@configclass
class FSCActionCfg(ActionTermCfg):
    """Configuration for the Footstep Controller (FSC) Action Term"""

    class_type: type[ActionTerm] = FSCActionTerm
