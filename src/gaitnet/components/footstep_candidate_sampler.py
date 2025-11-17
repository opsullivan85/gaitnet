import src.constants as const
from src.contactnet.contactnet import CostMapGenerator
from src.contactnet.debug import view_footstep_cost_map
from src.gaitnet.actions.footstep_action import NO_STEP
from src.gaitnet.actions.mpc_action import ManagerBasedEnv
from src.gaitnet.env_cfg.observations import contact_state_indices, get_terrain_mask
from src.simulation.cfg.footstep_scanner_constants import idx_to_xy
from src.util.math import seeded_uniform_noise

import torch
import torch.nn as nn

_debug_footstep_cost_map = False
_debug_footstep_cost_map_all = False

class FootstepCandidateSampler:
    def __init__(self, env: ManagerBasedEnv, options_per_leg: int, noise: bool = True):
        self.env = env
        self.cost_map_generator = CostMapGenerator(device=env.device)
        self.options_per_leg = options_per_leg
        self.noise = noise

    def _filter_cost_map(
        self, cost_map: torch.Tensor, obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Filter the cost map to remove invalid footstep options based on terrain and robot state.

        Args:
            cost_map: (num_envs, 4, H, W) raw cost maps for each leg
            obs: (num_envs, obs_dim) observation tensor containing robot state

        Returns:
            filtered_cost_map: (num_envs, 4, H, W) cost maps with invalid options set to inf
        """
        # remove options with invalid terrain
        terrain_mask = get_terrain_mask(
            const.gait_net.valid_height_range, obs
        )  # (num_envs, 4, H, W)
        masked_cost_maps = torch.where(
            terrain_mask, cost_map, torch.tensor(float("inf"), device=cost_map.device)
        )

        contact_states = obs[:, contact_state_indices].bool()  # (num_envs, 4)
        swing_states = ~contact_states  # (num_envs, 4)
        masked_cost_maps = torch.where(
            swing_states.unsqueeze(-1).unsqueeze(-1),
            torch.tensor(float("inf"), device=cost_map.device),
            masked_cost_maps,
        )

        # require minimum number leg to be in contact
        # by setting all costs to inf if not enough legs in contact
        min_legs_in_contact = 2
        num_legs_in_contact = contact_states.sum(dim=1)  # (num_envs,)
        contact_limit = num_legs_in_contact <= min_legs_in_contact  # (num_envs,)
        masked_cost_maps = torch.where(
            contact_limit.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            torch.tensor(float("inf"), device=cost_map.device),
            masked_cost_maps,
        )

        return masked_cost_maps

    @staticmethod
    def _overall_best_options(
        cost_map: torch.Tensor, options_per_leg: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the overall best footstep options across all legs.

        Args:
            cost_map: (num_envs, 4, H, W) filtered cost maps for each leg

        Returns:
            best_options: (num_envs, num_options, 3) best options as (leg_idx, dx, dy)
            best_values: (num_envs, num_options) corresponding costs
        """
        # get the best options
        flat_cost_map = cost_map.flatten()
        flat_cost_map = flat_cost_map.reshape(
            cost_map.shape[0], -1
        )  # (num_envs, 4*H*W)
        # technically there is no need to sort here, but
        # it will guarantee determinism
        topk_values, topk_flat_indices = torch.topk(
            flat_cost_map, options_per_leg, largest=False, sorted=True
        )  # (num_envs, num_options)

        # unravel indices to (leg, idx, idy)
        topk_indices = torch.unravel_index(topk_flat_indices, cost_map.shape[1:])
        # to (num_envs, num_options, 3)
        topk_indices = torch.stack(topk_indices, dim=-1)

        # convert to (leg, x, y) to (leg, dx, dy)
        topk_pos = idx_to_xy(topk_indices)  # (num_envs, num_options, 3)

        return topk_pos, topk_values

    @staticmethod
    def _best_options_per_leg(
        cost_map: torch.Tensor, options_per_leg: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the best footstep options for each leg.

        includes a NO_STEP option at the end.

        Args:
            cost_map: (num_envs, 4, H, W) filtered cost maps for each leg

        Returns:
            best_options: (num_envs, num_options, 3) best options as (leg_idx, dx, dy)
            best_values: (num_envs, num_options) corresponding costs
        """

        best_options = []
        best_values = []
        for leg in range(cost_map.shape[1]):  # iterate over legs
            leg_cost_map = cost_map[:, leg, :, :].unsqueeze(1)  # (num_envs, 1, H, W)
            topk_pos, topk_values = FootstepCandidateSampler._overall_best_options(
                leg_cost_map, options_per_leg
            )
            # manually set leg index
            topk_pos[:, :, 0] = leg
            best_options.append(topk_pos)
            best_values.append(topk_values)

        # add in the NO_STEP option
        num_envs = cost_map.shape[0]
        no_step_option = torch.zeros(
            (num_envs, 1, 3),
            dtype=torch.float32,
            device=cost_map.device,
        )
        no_step_option[:, 0, 0] = NO_STEP
        no_step_value = torch.zeros(
            (num_envs, 1),
            dtype=torch.float32,
            device=cost_map.device,
        )
        best_options.append(no_step_option)
        best_values.append(no_step_value)

        best_options = torch.cat(best_options, dim=1)  # (num_envs, num_options, 3)
        best_values = torch.cat(best_values, dim=1)  # (num_envs, num_options)

        return best_options, best_values

    def _apply_options_to_cost_map(
        self, cost_map: torch.Tensor, options: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the selected footstep options to the cost map.

        Args:
            cost_map: (num_envs, 4, H, W) filtered cost maps for each leg
            options: (num_envs, num_options, 3) footstep options as (leg_idx, dx, dy)

        Returns:
            torch.Tensor: Updated cost map with applied footstep options.
        """
        zero = idx_to_xy(torch.tensor([0, 0], device=options.device))
        one = idx_to_xy(torch.tensor([1, 1], device=options.device))
        slope = (one[0] - zero[0], one[1] - zero[1])
        intercept = (zero[0], zero[1])
        # Apply each footstep option to the cost map
        for leg_idx, dx, dy in options[0, :-1]:  # exclude the NO_STEP option
            idx = torch.round((dx - intercept[0]) / slope[0]).long()
            idy = torch.round((dy - intercept[1]) / slope[1]).long()
            cost_map[0, int(leg_idx), idx, idy] = -0.3  # Set cost for selected options

        return cost_map

    def get_footstep_options(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Generate footstep options based on the current environment state.

        Returns:
            torch.Tensor: Footstep options of shape (num_envs, options_per_leg*4, 4)
                          Each option is represented as (leg_index, x_offset, y_offset, cost)
                          appends a NO_STEP option at the end of the list.
        """
        contactnet_obs = obs[:, :18]
        with torch.inference_mode():
            cost_maps = self.cost_map_generator.predict(
                contactnet_obs
            )  # (num_envs, 4, H, W)
        # switch from (FL, FR, RL, RR) to (FR, FL, RR, RL)
        cost_maps = cost_maps[:, [1, 0, 3, 2], :, :]
        if _debug_footstep_cost_map_all:
            view_footstep_cost_map(
                cost_map=cost_maps[0][[1, 0, 3, 2]].cpu().numpy(),
                title="Default Footstep Cost Map",
                save_figure=True,
                show_ticks=False,
            )

        # interpolate cost map
        cost_maps = nn.functional.interpolate(
            cost_maps,  # (num_envs, 4, h, w) 4 is being used as the channel dimension here
            size=const.footstep_scanner.grid_size.tolist(),  # (H, W)
            mode="bilinear",
            align_corners=True,
        ).squeeze(
            1
        )  # (num_envs, 4, H, W)
        if _debug_footstep_cost_map_all:
            view_footstep_cost_map(
                cost_map=cost_maps[0][[1, 0, 3, 2]].cpu().numpy(),
                title="Scaled Footstep Cost Map",
                save_figure=True,
                show_ticks=False,
            )

        if self.noise:
            noise = seeded_uniform_noise(cost_maps, cost_maps.shape[1:])
            noise = (
                noise * 2 * const.gait_net.upscale_costmap_noise
                - const.gait_net.upscale_costmap_noise
            )
            cost_maps += noise
            if _debug_footstep_cost_map_all:
                view_footstep_cost_map(
                    cost_map=cost_maps[0][[1, 0, 3, 2]].cpu().numpy(),
                    title="Noisy Footstep Cost Map",
                    save_figure=True,
                    show_ticks=False,
                )

        cost_maps = self._filter_cost_map(cost_maps, obs)  # (num_envs, 4, H, W)
        if _debug_footstep_cost_map_all:
            view_footstep_cost_map(
                cost_maps[0][[1, 0, 3, 2]].cpu().numpy(),
                title="Masked Footstep Cost Map",
                save_figure=True,
                # show_ticks=False,
                tick_skip=4,
                tick_rotations=(90, 0),
            )

        # best_options, best_values = self._overall_best_options(masked_cost_maps, self.num_options)
        best_options, best_values = self._best_options_per_leg(
            cost_maps, self.options_per_leg
        )
        if _debug_footstep_cost_map:
            applied_map = self._apply_options_to_cost_map(
                cost_maps, best_options
            )
            view_footstep_cost_map(
                applied_map[0][[1, 0, 3, 2]].cpu().numpy(),
                title="Applied Footstep Cost Map",
                save_figure=True,
                # show_ticks=False,
                tick_skip=4,
                tick_rotations=(90, 0),
            )

        # replace any options with inf cost with NO_STEP option and 0 cost
        inf_mask = torch.isinf(best_values)  # (num_envs, num_options)
        if inf_mask.any():
            # Use unsqueeze to broadcast the mask to match best_options shape
            inf_mask_expanded = inf_mask.unsqueeze(-1)  # (num_envs, num_options, 1)
            best_options[:, :, 0] = torch.where(
                inf_mask,
                torch.tensor(
                    NO_STEP, device=best_options.device, dtype=best_options.dtype
                ),
                best_options[:, :, 0],
            )
            best_options[:, :, 1:3] = torch.where(
                inf_mask_expanded.expand(-1, -1, 2),
                torch.tensor(0.0, device=best_options.device, dtype=best_options.dtype),
                best_options[:, :, 1:3],
            )
            best_values[inf_mask] = 0.0

        result = torch.cat([best_options, best_values.unsqueeze(-1)], dim=-1)
        # (num_envs, num_options, 4) where each option is (leg, dx, dy, cost)

        # zero out cost for ablation study
        if const.experiments.ablate_footstep_cost:
            result[:, :, 3] = 0.0

        return result