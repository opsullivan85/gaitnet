"""The pillar height field and the terrain-relative terminations, without the simulator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("isaaclab")

from gaitnet_core.grid import FootholdGrid  # noqa: E402
from gaitnet_core.robot_spec import GO1  # noqa: E402
from gaitnet_core.state import TerrainPatch  # noqa: E402
from gaitnet_sim.env.terminations import base_below_terrain_clearance, feet_below_walkable_terrain  # noqa: E402
from gaitnet_sim.terrain_generation import pillar_terrain  # noqa: E402
from gaitnet_sim.terrains import HfPillarsTerrainCfg  # noqa: E402

GRID = FootholdGrid(resolution=0.015, size=(9, 9), border=2)


def pillars(difficulty: float, seed: int = 0) -> tuple[np.ndarray, HfPillarsTerrainCfg]:
    cfg = HfPillarsTerrainCfg(size=(3.0, 3.0), horizontal_scale=0.025, vertical_scale=0.005)
    np.random.seed(seed)
    # the undecorated height field, before Isaac Lab's mesh conversion
    return pillar_terrain.__wrapped__(difficulty, cfg), cfg


def test_pillars_at_difficulty_zero_are_flat():
    heights, _ = pillars(0.0)
    assert heights.shape == (120, 120)
    assert (heights == 0).all()


def test_pillars_have_gaps_and_bounded_heights():
    heights, cfg = pillars(1.0)
    void = round(cfg.hole_depth / cfg.vertical_scale)
    tops = heights[heights != void]
    limit = cfg.max_height_offset / cfg.vertical_scale
    assert (heights == void).mean() > 0.2
    assert tops.min() >= -limit and tops.max() <= limit
    assert len(np.unique(tops)) > 10
    # the spawn platform
    platform = round(cfg.platform_size / cfg.horizontal_scale)
    start = (heights.shape[0] - platform) // 2
    assert (heights[start : start + platform, start : start + platform] == 0).all()


def fake_env_with_term(heights: torch.Tensor, foot_heights: torch.Tensor):
    term = SimpleNamespace(
        spec=GO1,
        terrain=lambda: TerrainPatch(heights=heights, grid=GRID),
        io=SimpleNamespace(foot_heights=lambda: foot_heights),
    )
    return SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))


def patch(value: float) -> torch.Tensor:
    return torch.full((1, 4, *GRID.patch_size), value)


def test_feet_below_walkable_terrain_on_flat_ground():
    heights = patch(-0.27)
    on_ground = torch.tensor([[-0.25, -0.25, -0.25, -0.25]])
    assert not feet_below_walkable_terrain(fake_env_with_term(heights, on_ground)).item()
    in_hole = torch.tensor([[-0.25, -0.33, -0.25, -0.25]])
    assert feet_below_walkable_terrain(fake_env_with_term(heights, in_hole)).item()


def test_feet_below_walkable_terrain_on_pillars():
    # a low pillar, a high pillar and a void in each leg's patch
    heights = patch(-0.77)
    heights[..., :4, :] = -0.31
    heights[..., -4:, :] = -0.21
    on_low_pillar = torch.full((1, 4), -0.29)
    assert not feet_below_walkable_terrain(fake_env_with_term(heights, on_low_pillar)).item()
    in_gap = torch.tensor([[-0.29, -0.29, -0.40, -0.29]])
    assert feet_below_walkable_terrain(fake_env_with_term(heights, in_gap)).item()


def test_feet_below_walkable_terrain_over_a_void():
    heights = patch(float("-inf"))
    assert not feet_below_walkable_terrain(fake_env_with_term(heights, torch.full((1, 4), -0.3))).item()
    # judged against the bottom of the reach band
    assert feet_below_walkable_terrain(fake_env_with_term(heights, torch.full((1, 4), -0.45))).item()


def test_base_clearance_uses_the_highest_surface_under_the_trunk():
    hits = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, float("inf")]]])
    hits = torch.cat([hits, hits.clone()])
    hits[1, 1, 2] = 0.05  # the second robot's highest surface is lower
    scanner = SimpleNamespace(
        data=SimpleNamespace(ray_hits_w=SimpleNamespace(torch=hits), pos_w=SimpleNamespace(torch=torch.tensor([[0, 0, 0.3]] * 2)))
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"base_scanner": scanner}))
    assert base_below_terrain_clearance(env, minimum_height=0.15).tolist() == [True, False]
