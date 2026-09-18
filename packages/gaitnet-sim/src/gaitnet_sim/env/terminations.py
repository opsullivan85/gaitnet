"""Termination terms isaaclab's stock mdp doesn't provide."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from gaitnet_sim.env.observations import footstep_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster


def base_below_terrain_clearance(
    env: "ManagerBasedRLEnv", minimum_height: float, sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_scanner")
) -> torch.Tensor:
    """The base came within `minimum_height` of the highest surface under the trunk.

    `sensor_cfg` is a ray caster on the base, yaw aligned, whose pattern covers the trunk's
    footprint. On flat ground this is the base's height above the ground.
    """
    scanner: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits = scanner.data.ray_hits_w.torch[..., 2]
    hits = torch.where(torch.isfinite(hits), hits, torch.full_like(hits, float("-inf")))
    clearance = scanner.data.pos_w.torch[:, 2] - hits.amax(dim=1)
    return clearance < minimum_height


def feet_below_walkable_terrain(
    env: "ManagerBasedRLEnv", margin: float = 0.05, action_name: str = "footstep"
) -> torch.Tensor:
    """A foot is more than `margin` below the lowest walkable surface around its hip: it has
    gone into a hole or off the edge of a pillar.

    Walkable cells are those of the leg's terrain patch within the robot's reach band;
    voids fall below it. A leg with no walkable cell in view is judged against the bottom
    of the reach band. On flat holed ground this is the foot's depth below the ground.
    """
    term = footstep_action(env, action_name)
    heights = term.terrain().heights
    lowest, highest = term.spec.reach_band
    walkable = (heights >= lowest) & (heights <= highest)
    lowest_walkable = torch.where(walkable, heights, torch.full_like(heights, float("inf"))).amin(dim=(-2, -1))
    reference = torch.where(torch.isinf(lowest_walkable), torch.full_like(lowest_walkable, lowest), lowest_walkable)
    return torch.any(term.io.foot_heights() < reference - margin, dim=1)


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
