"""Observation manager that swaps the footstep scanner values for sampled foothold candidates.

Replaced in the new sim layer (restructure plan P3) by a plain observation term.
"""

import numpy as np
import torch
from isaaclab.managers import (
    ObservationManager,
)

import gaitnet.constants as const
import gaitnet.gaitnet.env_cfg.observations_utils as obs_utils
from gaitnet.gaitnet.actions.mpc_action import ManagerBasedEnv
from gaitnet import get_logger
from gaitnet_core.candidates import Candidates
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.samplers import CandidateSampler

logger = get_logger()




class GaitNetObservationManager(ObservationManager):
    """Assumes the 4 footstep scanner values are at the end of the observation.

    Replaces the footstep scanner values with the robot's foothold candidates, packed as
    (N, L * K * 5), see `gaitnet_core.candidates.Candidates.pack`.
    """

    def __init__(
        self,
        cfg: object,
        env: ManagerBasedEnv,
        sampler: CandidateSampler,
        candidates_per_leg: int,
    ):
        self.logged_update_history_warning = False
        self.sampler = sampler
        self.candidates_per_leg = candidates_per_leg
        self.grid = FootholdGrid(
            resolution=const.footstep_scanner.grid_resolution,
            size=tuple(int(n) for n in const.footstep_scanner.grid_size),
            border=0,
        )
        super().__init__(cfg, env)
        self.candidates: Candidates
        """The latest candidates, which the footstep action term resolves the chosen index against."""
        self.most_recent_terrain_obs: torch.Tensor
        self._check_robot_state_layout()
        self._overwrite_obs_dim()

    def _check_robot_state_layout(self) -> None:
        """Check the configured policy terms match the layout the rest of the code indexes into."""
        names = list(self._group_obs_term_names["policy"])
        dims = [int(np.prod(dim)) for dim in self._group_obs_term_dim["policy"]]
        starts = np.cumsum([0] + dims[:-1]).tolist()
        actual = [(name, (start, start + dim)) for name, start, dim in zip(names, starts, dims)]

        expected = list(obs_utils.robot_state_layout.items())
        if actual[: len(expected)] != expected:
            raise ValueError(
                "Policy observation terms don't match observations_utils.robot_state_layout."
                f"\n\texpected: {expected}\n\tactual:   {actual[: len(expected)]}"
            )
        robot_state_dim = expected[-1][1][1]
        if robot_state_dim != const.gait_net.robot_state_dim:
            raise ValueError(
                f"robot_state_layout is {robot_state_dim} dims, but "
                f"const.gait_net.robot_state_dim is {const.gait_net.robot_state_dim}."
            )
        remaining = actual[len(expected) :]
        if (
            [name for name, _ in remaining] != obs_utils.footstep_scanner_terms
            or sum(end - start for _, (start, end) in remaining) != const.footstep_scanner.total_robot_features
        ):
            raise ValueError(
                f"Only {obs_utils.footstep_scanner_terms}, in that order, may follow the robot state, got {remaining}"
            )

    def _overwrite_obs_dim(self) -> None:
        """Overwrite the observation dimensions to account for the candidates."""
        policy_obs_dim = self._group_obs_dim["policy"][0]
        # if this is a list throw an error
        if isinstance(policy_obs_dim, tuple):
            raise NotImplementedError(
                "FootstepObservationManager does not support list observation dimensions."
            )
        obs_dim: int = policy_obs_dim - const.footstep_scanner.total_robot_features
        obs_dim += const.robot.num_legs * self.candidates_per_leg * Candidates.PACKED_DIM
        self._group_obs_dim["policy"] = (obs_dim,)

    def _modify_obs(self, obs: torch.Tensor) -> torch.Tensor:
        valid = obs_utils.legacy_valid_footholds(obs)
        self.candidates = self.sampler.sample(valid, self.grid)

        # replace the footstep scanner values at the end of the observation with the candidates
        self.most_recent_terrain_obs = obs[:, -const.footstep_scanner.total_robot_features :]
        obs = obs[:, : -const.footstep_scanner.total_robot_features]
        return torch.cat([obs, self.candidates.pack().flatten(start_dim=1)], dim=1)

    def compute_group(
        self, group_name: str, update_history: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        obs = super().compute_group(group_name, update_history)
        # only modify the policy observation group
        if group_name != "policy":
            return obs

        if isinstance(obs, dict):
            raise NotImplementedError(
                "FootstepObservationManager does not support dict observations."
            )

        # the history will not be in the same shape as the observations
        if update_history:
            if not self.logged_update_history_warning:
                logger.warning(
                    "FootstepObservationManager history might not work as expected."
                )
                self.logged_update_history_warning = True

        obs = self._modify_obs(obs)
        return obs
