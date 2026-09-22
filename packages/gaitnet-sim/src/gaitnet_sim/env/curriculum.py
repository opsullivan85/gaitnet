"""Terrain curriculum, and carrying its state across a resumed training run (the
checkpoint side lives in `gaitnet_sim.scripts.train`)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter


def terrain_levels_survival(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    p_up_given_success: float = 0.1,
    p_down_given_failure: float = 0.5,
    p_random: float = 0.02,
) -> torch.Tensor:
    """Move robots that survived their episode up a level, and robots that fell down one,
    each with some probability.

    The level settles where P(fall) = p_up / (p_up + p_down), and the probabilities set how
    fast it moves and how much it spreads. `p_random` moves a robot up and down alike,
    which leaves it in place; it keeps some robots on levels the rule wouldn't pick.

    Returns:
        The mean difficulty over all robots.
    """
    terrain: TerrainImporter = env.scene.terrain
    termination = env.termination_manager
    n = len(env_ids)
    noise = torch.rand(n, device=env.device) < p_random
    move_up = (termination.time_outs[env_ids] & (torch.rand(n, device=env.device) < p_up_given_success)) | noise
    move_down = (termination.terminated[env_ids] & (torch.rand(n, device=env.device) < p_down_given_failure)) | noise
    terrain.update_env_origins(env_ids, move_up, move_down)

    low, high = terrain.cfg.terrain_generator.difficulty_range
    level = torch.mean(terrain.terrain_levels.float()) / terrain.max_terrain_level
    return low + level * (high - low)


def _row_layout(generator: "TerrainGeneratorCfg") -> str:
    """What a terrain row means: everything in the generator cfg that sets a row's
    difficulty or what its sub-terrains look like, as a canonical JSON string.

    The number of columns, the seed and the border are left out: they don't change what
    standing on row i means.
    """
    layout = {
        "class_type": generator.class_type,
        "curriculum": generator.curriculum,
        "num_rows": generator.num_rows,
        "difficulty_range": generator.difficulty_range,
        "size": generator.size,
        "horizontal_scale": generator.horizontal_scale,
        "vertical_scale": generator.vertical_scale,
        "slope_threshold": generator.slope_threshold,
        "sub_terrains": {name: cfg.to_dict() for name, cfg in generator.sub_terrains.items()},
    }
    return json.dumps(layout, sort_keys=True, default=str)


def terrain_levels_state(env: "ManagerBasedRLEnv") -> dict | None:
    """How many robots stand on each terrain row, and what the rows are, for a checkpoint.

    Returns:
        `{"counts": [robots on row 0, row 1, ...], "layout": _row_layout(...)}`, or None
        when the env has no terrain curriculum.
    """
    terrain: TerrainImporter = env.scene.terrain
    generator = terrain.cfg.terrain_generator
    if terrain.terrain_origins is None or generator is None or not generator.curriculum:
        return None
    counts = torch.bincount(terrain.terrain_levels, minlength=generator.num_rows)
    return {"counts": counts.tolist(), "layout": _row_layout(generator)}


def restore_terrain_levels(env: "ManagerBasedRLEnv", state: dict | None) -> str | None:
    """Put robots back on the terrain rows a checkpoint recorded, sampling each robot's row
    from the saved counts (the number of robots may differ from the saved run's), and
    leave the stock random placement when the rows no longer mean the same thing.

    The robots only move at their next reset; the caller resets the env.

    Returns:
        None when restored, otherwise why not.
    """
    terrain: TerrainImporter = env.scene.terrain
    current = terrain_levels_state(env)
    if current is None:
        return "the env has no terrain curriculum"
    if state is None:
        return "the checkpoint has no terrain levels"
    if state["layout"] != current["layout"]:
        return "the terrain rows differ from the checkpoint's (num_rows, difficulties or sub-terrains)"
    counts = torch.tensor(state["counts"], dtype=torch.float, device=env.device)
    levels = torch.multinomial(counts, env.num_envs, replacement=True)
    terrain.terrain_levels[:] = levels.to(terrain.terrain_levels.dtype)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    return None
