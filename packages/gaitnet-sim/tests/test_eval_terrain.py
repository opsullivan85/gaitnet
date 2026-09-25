"""The evaluation terrain without the simulator: the spawn platform at the start, and mesh
compaction leaving the surface exactly as Isaac Lab's height-field conversion made it."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("isaaclab")

from gaitnet_sim.terrain_generation import compact_height_field_mesh, hole_terrain, pillar_terrain  # noqa: E402
from gaitnet_sim.terrains import HfHolesTerrainCfg, HfPillarsTerrainCfg  # noqa: E402

SCALES = {"horizontal_scale": 0.025, "vertical_scale": 0.005, "slope_threshold": 0.0}
TERRAINS = [(hole_terrain, HfHolesTerrainCfg), (pillar_terrain, HfPillarsTerrainCfg)]


@pytest.mark.parametrize("terrain, cfg_type", TERRAINS)
def test_platform_at_start(terrain, cfg_type):
    np.random.seed(0)
    cfg = cfg_type(size=(4.0, 1.0), platform_at_start=True, **SCALES)
    heights = terrain.__wrapped__(1.0, cfg)
    # less a hole cell (5 samples): the holes' upsampling doesn't put their edges exactly on it
    platform = round(cfg.platform_size / cfg.horizontal_scale) - 5
    assert (heights[:platform, 5:-5] == 0).all()
    assert (heights[platform:] != 0).any()


def surface(mesh, xy: np.ndarray) -> np.ndarray:
    """Height of the highest surface over each point, NaN where there is none."""
    origins = np.column_stack([xy, np.full(len(xy), 10.0)])
    directions = np.tile([0.0, 0.0, -1.0], (len(xy), 1))
    hits, rays, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
    heights = np.full(len(xy), np.nan)
    heights[rays] = hits[:, 2]
    return heights


@pytest.mark.parametrize("difficulty", [0.0, 0.05, 0.5])
@pytest.mark.parametrize("terrain, cfg_type", TERRAINS)
def test_compaction_keeps_the_surface(terrain, cfg_type, difficulty):
    np.random.seed(1)
    cfg = cfg_type(size=(4.0, 1.0), platform_at_start=True, **SCALES)
    (mesh,), _ = terrain(difficulty, cfg)
    compact = compact_height_field_mesh(mesh, cfg.horizontal_scale)
    assert len(compact.faces) < len(mesh.faces) / 1.5

    lower, upper = mesh.bounds[:, :2]
    xy = np.random.default_rng(0).uniform(lower, upper, size=(20_000, 2))
    np.testing.assert_allclose(surface(compact, xy), surface(mesh, xy), atol=1e-6)
