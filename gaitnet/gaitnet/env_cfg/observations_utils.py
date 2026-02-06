from __future__ import annotations
import numpy as np
import gaitnet.constants as const


import torch


def get_terrain_mask(
    valid_height_range: tuple[float, float], obs: torch.Tensor
) -> torch.Tensor:
    """Get a mask for the terrain observations.

    0 indicates invalid terrain (too high or too low)
    1 indicates valid terrain
    """
    terrain_terms = (
        const.footstep_scanner.grid_size[0] * const.footstep_scanner.grid_size[1] * const.robot.num_legs
    )
    terrain_obs = obs[:, -terrain_terms:]
    # reshape to (N, 4, H, W)
    terrain_obs = terrain_obs.reshape(
        terrain_obs.shape[0],
        const.robot.num_legs,
        const.footstep_scanner.grid_size[0],
        const.footstep_scanner.grid_size[1],
    )
    # mask out values outside of allowed height range
    min_height, max_height = valid_height_range
    terrain_mask = (terrain_obs > min_height) & (terrain_obs < max_height)
    return terrain_mask


contact_state_indices = np.asarray([18, 19, 20, 21])