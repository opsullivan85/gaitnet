"""Termination terms for the evaluation sweep."""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def out_of_sub_terrain(
    env: ManagerBasedRLEnv,
    distance_buffer: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot leaves the sub-terrain it was spawned on.

    ``isaaclab_tasks``' :func:`terrain_out_of_bounds` measures against the extent of the
    whole terrain grid. That was equivalent while the grid was a single row, but the
    evaluation terrain now puts one difficulty per row, so a robot would have to walk
    onto a neighbouring row's terrain -- at a different difficulty -- before that check
    fired. Measuring per sub-terrain keeps each sample on the difficulty it was assigned.

    Args:
        env: The environment.
        distance_buffer: How far from the sub-terrain edge to terminate, in metres.
        asset_cfg: The asset to track.

    Returns:
        (num_envs,) bool, True where the robot left its sub-terrain.
    """
    size_x, size_y = env.scene.cfg.terrain.terrain_generator.size  # type: ignore
    asset: RigidObject = env.scene[asset_cfg.name]

    # the sub-terrain origin is its own centre, so this is the offset from that centre
    offset = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    x_out_of_bounds = torch.abs(offset[:, 0]) > 0.5 * size_x - distance_buffer
    y_out_of_bounds = torch.abs(offset[:, 1]) > 0.5 * size_y - distance_buffer
    return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
