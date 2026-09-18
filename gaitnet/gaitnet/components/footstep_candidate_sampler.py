from __future__ import annotations

import torch

import gaitnet.constants as const
from gaitnet.constants import NO_STEP
from gaitnet.gaitnet.env_cfg.observations_utils import get_terrain_mask, scheduled_contact
from gaitnet.simulation.cfg.footstep_scanner_constants import idx_to_xy


class FootstepCandidateSampler:
    def __init__(self, options_per_leg: int):
        self.options_per_leg = options_per_leg

    @staticmethod
    def valid_footholds(obs: torch.Tensor) -> torch.Tensor:
        """Which footstep scanner cells each leg may step to.

        Args:
            obs: (num_envs, obs_dim) observation tensor containing the robot state,
                with the footstep scanner values at the end

        Returns:
            valid: (num_envs, 4, H, W) True where the leg may step to the cell
        """
        # remove options with invalid terrain
        valid = get_terrain_mask(const.gait_net.valid_height_range, obs)  # (num_envs, 4, H, W)

        # use the controller's schedule rather than measured contact, since a leg
        # the controller still has in swing can't be given a new footstep
        contact_states = scheduled_contact(obs)  # (num_envs, 4)
        valid = valid & contact_states.unsqueeze(-1).unsqueeze(-1)

        # require a minimum number of legs to be in contact
        min_legs_in_contact = 2
        num_legs_in_contact = contact_states.sum(dim=1)  # (num_envs,)
        contact_limit = num_legs_in_contact <= min_legs_in_contact  # (num_envs,)
        valid = valid & ~contact_limit.view(-1, 1, 1, 1)

        return valid

    @staticmethod
    def _random_options_per_leg(
        valid: torch.Tensor, options_per_leg: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample up to `options_per_leg` distinct valid cells per leg, uniformly at random.

        Args:
            valid: (num_envs, 4, H, W) valid foothold mask
            options_per_leg: number of options to sample per leg

        Returns:
            options: (num_envs, 4 * options_per_leg, 3) as (leg, x, y), leg-major
            option_valid: (num_envs, 4 * options_per_leg) False where a leg had fewer
                valid cells than `options_per_leg`
        """
        num_envs, num_legs, height, width = valid.shape
        # uniform without replacement: the smallest k of iid uniform keys over the valid cells
        keys = torch.rand(valid.shape, device=valid.device).masked_fill(~valid, float("inf"))
        keys, flat_indices = torch.topk(
            keys.view(num_envs, num_legs, height * width), options_per_leg, largest=False
        )  # (num_envs, 4, k)
        option_valid = ~torch.isinf(keys)

        cell_indices = torch.stack(torch.unravel_index(flat_indices, (height, width)), dim=-1)
        xy = idx_to_xy(cell_indices)  # (num_envs, 4, k, 2)
        legs = torch.arange(num_legs, device=valid.device, dtype=xy.dtype)
        legs = legs.view(1, num_legs, 1, 1).expand(num_envs, -1, options_per_leg, -1)
        options = torch.cat([legs, xy], dim=-1)  # (num_envs, 4, k, 3)

        return options.reshape(num_envs, -1, 3), option_valid.reshape(num_envs, -1)

    def get_footstep_options(self, obs: torch.Tensor) -> torch.Tensor:
        """Generate footstep options based on the current environment state.

        Returns:
            torch.Tensor: Footstep options of shape (num_envs, options_per_leg * 4 + 1, 3),
                each (leg_index, x_offset, y_offset), leg-major with a trailing NO_STEP
                option. Options that could not be filled carry the NO_STEP encoding.
        """
        valid = self.valid_footholds(obs)
        options, option_valid = self._random_options_per_leg(valid, self.options_per_leg)

        # unfilled options get the no-op encoding, which the policy masks out
        options[:, :, 0] = torch.where(option_valid, options[:, :, 0], float(NO_STEP))
        options[:, :, 1:] = torch.where(option_valid.unsqueeze(-1), options[:, :, 1:], 0.0)

        no_op = torch.zeros((obs.shape[0], 1, 3), device=obs.device, dtype=options.dtype)
        no_op[:, 0, 0] = NO_STEP
        return torch.cat([options, no_op], dim=1)
