"""Termination terms isaaclab's stock mdp doesn't provide."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def bodies_below_height(
    env: "ManagerBasedRLEnv", minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Any of `asset_cfg.body_ids` below `minimum_height` in world z.

    World z is only meaningful on terrain whose surface is at z = 0 (holes); terrain with
    height needs a terrain-relative check.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    heights = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]
    return torch.any(heights < minimum_height, dim=1)


def out_of_terrain(
    env: "ManagerBasedRLEnv", distance_buffer: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The robot came within `distance_buffer` of the generated terrain's outer edge."""
    generator = env.scene.cfg.terrain.terrain_generator
    size_x = generator.num_rows * generator.size[0] + 2 * generator.border_width
    size_y = generator.num_cols * generator.size[1] + 2 * generator.border_width
    position = env.scene[asset_cfg.name].data.root_link_pos_w.torch[:, :2]
    return (torch.abs(position[:, 0]) > 0.5 * size_x - distance_buffer) | (
        torch.abs(position[:, 1]) > 0.5 * size_y - distance_buffer
    )


def out_of_sub_terrain(
    env: "ManagerBasedRLEnv", distance_buffer: float = 0.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The robot left the sub-terrain it spawned on.

    For the evaluation grid, where each row is a different difficulty: `out_of_terrain`
    would only fire once the robot had walked onto a neighbouring row.
    """
    size_x, size_y = env.scene.cfg.terrain.terrain_generator.size
    # a sub-terrain's origin is its centre
    offset = env.scene[asset_cfg.name].data.root_link_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    return (torch.abs(offset[:, 0]) > 0.5 * size_x - distance_buffer) | (
        torch.abs(offset[:, 1]) > 0.5 * size_y - distance_buffer
    )
