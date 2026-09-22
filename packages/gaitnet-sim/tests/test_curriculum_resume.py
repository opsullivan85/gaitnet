"""Saving and restoring the terrain curriculum's levels across a resumed run, without the simulator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab")

from gaitnet_sim.env.curriculum import restore_terrain_levels, terrain_levels_state  # noqa: E402
from gaitnet_sim.terrains import holes_terrain_cfg, pillars_terrain_cfg  # noqa: E402


def fake_env(num_envs: int, terrain_cfg=None):
    cfg = terrain_cfg or holes_terrain_cfg()
    generator = cfg.terrain_generator
    rows, cols = generator.num_rows, generator.num_cols
    terrain = SimpleNamespace(
        cfg=cfg,
        terrain_origins=torch.arange(rows * cols * 3, dtype=torch.float).reshape(rows, cols, 3),
        terrain_levels=torch.zeros(num_envs, dtype=torch.long),
        terrain_types=torch.arange(num_envs) % cols,
        env_origins=torch.zeros(num_envs, 3),
    )
    return SimpleNamespace(scene=SimpleNamespace(terrain=terrain), num_envs=num_envs, device="cpu")


def test_levels_survive_a_change_in_env_count():
    saved = fake_env(64)
    saved.scene.terrain.terrain_levels[:] = 7
    saved.scene.terrain.terrain_levels[:16] = 9
    state = terrain_levels_state(saved)

    resumed = fake_env(512)
    assert restore_terrain_levels(resumed, state) is None
    terrain = resumed.scene.terrain
    assert set(terrain.terrain_levels.unique().tolist()) == {7, 9}
    assert torch.equal(terrain.env_origins, terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])


def test_changed_rows_fall_back_to_random_placement():
    state = terrain_levels_state(fake_env(64))
    for change in (
        lambda g: setattr(g, "num_rows", 10),
        lambda g: setattr(g, "difficulty_range", (0.0, 0.8)),
        lambda g: setattr(next(iter(g.sub_terrains.values())), "hole_depth", 0.5),
    ):
        cfg = holes_terrain_cfg()
        change(cfg.terrain_generator)
        resumed = fake_env(64, cfg)
        before = resumed.scene.terrain.terrain_levels.clone()
        assert restore_terrain_levels(resumed, state) is not None
        assert torch.equal(resumed.scene.terrain.terrain_levels, before)
    assert restore_terrain_levels(fake_env(64, pillars_terrain_cfg()), state) is not None


def test_more_columns_keep_the_rows():
    cfg = holes_terrain_cfg()
    cfg.terrain_generator.num_cols = 20
    assert restore_terrain_levels(fake_env(64, cfg), terrain_levels_state(fake_env(64))) is None


def test_old_checkpoints_and_no_curriculum():
    assert restore_terrain_levels(fake_env(64), None) is not None
    cfg = holes_terrain_cfg()
    cfg.terrain_generator.curriculum = False
    assert terrain_levels_state(fake_env(64, cfg)) is None
