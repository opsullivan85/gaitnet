"""Terrain implementations, kept apart from `gaitnet_sim.terrains` (the cfgs): they import
Isaac Lab's terrain generator and importer, which load USD, and task cfgs have to be
importable before the simulator starts. The cfgs name these by "module:name" strings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage
import torch

from isaaclab.terrains import TerrainGenerator, TerrainImporter
from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from gaitnet_sim.terrains import EvalTerrainGeneratorCfg, HfHolesTerrainCfg, HfPillarsTerrainCfg


@height_field_to_mesh
def hole_terrain(difficulty: float, cfg: "HfHolesTerrainCfg") -> np.ndarray:
    """Flat ground with a `difficulty` fraction of square holes and a solid central platform.

    Returns:
        (width, length) heights in units of `cfg.vertical_scale`.
    """
    # holes are drawn on a grid SCALE times coarser than the height field: the height field
    # to mesh conversion smooths single-sample steps into slopes, so each hole spans
    # SCALE x SCALE samples to keep its walls vertical
    scale = 5
    output_size = (int(cfg.size[0] / cfg.horizontal_scale), int(cfg.size[1] / cfg.horizontal_scale))
    shape = (output_size[0] // scale, output_size[1] // scale)
    void = cfg.hole_depth / cfg.vertical_scale
    terrain = np.zeros(shape, dtype=np.float32)

    # drawn from numpy's global state, not a per-difficulty seed, so equal-difficulty cells
    # (e.g. the columns of an evaluation row) still differ
    cells = np.random.permutation(np.argwhere(np.ones_like(terrain)))
    holes = cells[: int(difficulty * len(cells))]
    terrain[tuple(holes.T)] = void

    platform = int(cfg.platform_size / cfg.horizontal_scale / scale)
    start = (int(shape[0] / 2 - platform / 2), int(shape[1] / 2 - platform / 2))
    terrain[start[0] : start[0] + platform, start[1] : start[1] + platform] = 0

    terrain = scipy.ndimage.zoom(terrain, scale, order=0)
    pad = ((0, output_size[0] - terrain.shape[0]), (0, output_size[1] - terrain.shape[1]))
    return np.pad(terrain, pad, mode="constant", constant_values=void)


@height_field_to_mesh
def pillar_terrain(difficulty: float, cfg: "HfPillarsTerrainCfg") -> np.ndarray:
    """A grid of square pillars at random heights over a void, with a flat central platform.

    Difficulty scales the gaps between pillars (to `cfg.max_gap`) and the spread of their
    heights (to +-`cfg.max_height_offset`), so difficulty 0 is flat ground.

    Returns:
        (width, length) heights in units of `cfg.vertical_scale`.
    """
    pixels = (int(cfg.size[0] / cfg.horizontal_scale), int(cfg.size[1] / cfg.horizontal_scale))
    width = max(1, round(cfg.pillar_width / cfg.horizontal_scale))
    pitch = width + round(difficulty * cfg.max_gap / cfg.horizontal_scale)
    max_offset = difficulty * cfg.max_height_offset / cfg.vertical_scale
    terrain = np.full(pixels, cfg.hole_depth / cfg.vertical_scale)

    # a random phase so pillar edges fall differently relative to the spawn on each
    # sub-terrain; drawn from numpy's global state like the holes, see hole_terrain
    phase = np.random.randint(0, pitch, size=2)
    for x0 in range(phase[0] - pitch, pixels[0], pitch):
        for y0 in range(phase[1] - pitch, pixels[1], pitch):
            height = np.random.uniform(-max_offset, max_offset)
            terrain[max(x0, 0) : max(x0 + width, 0), max(y0, 0) : max(y0 + width, 0)] = height

    platform = int(cfg.platform_size / cfg.horizontal_scale)
    start = ((pixels[0] - platform) // 2, (pixels[1] - platform) // 2)
    terrain[start[0] : start[0] + platform, start[1] : start[1] + platform] = 0
    return np.rint(terrain).astype(np.int16)


class EvalTerrainGenerator(TerrainGenerator):
    """Generates row i at exactly `cfg.difficulties[i]`.

    Neither stock mode does: `curriculum=True` jitters each row's difficulty and
    `curriculum=False` samples it uniformly from `difficulty_range`.
    """

    cfg: "EvalTerrainGeneratorCfg"

    def __init__(self, cfg: "EvalTerrainGeneratorCfg", device: str = "cpu"):
        if len(cfg.difficulties) != cfg.num_rows:
            raise ValueError(f"expected one difficulty per row, got {len(cfg.difficulties)} for {cfg.num_rows} rows")
        if len(cfg.sub_terrains) != 1:
            raise ValueError(f"the difficulty is the only axis varied, expected one sub-terrain, got {list(cfg.sub_terrains)}")
        if cfg.curriculum:
            raise ValueError("curriculum generation would override the per-row difficulties")
        if cfg.use_cache:
            raise ValueError("use_cache would make every column of a row the same cached sub-terrain")
        super().__init__(cfg, device)

    def _generate_random_terrains(self):
        sub_terrain_cfg = next(iter(self.cfg.sub_terrains.values()))
        for row, difficulty in enumerate(self.cfg.difficulties):
            for col in range(self.cfg.num_cols):
                mesh, origin = self._get_terrain_mesh(float(difficulty), sub_terrain_cfg)
                self._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)


class EvalTerrainImporter(TerrainImporter):
    """Pins environment i to row i // num_cols, column i % num_cols, so the row an
    environment runs on (its difficulty) is fixed and known."""

    def configure_env_origins(self, origins=None):
        super().configure_env_origins(origins)
        if self.terrain_origins is None:
            return
        num_rows, num_cols = self.terrain_origins.shape[:2]
        if self.cfg.num_envs != num_rows * num_cols:
            raise ValueError(f"expected one environment per sub-terrain, got {self.cfg.num_envs} for {num_rows}x{num_cols}")
        env_ids = torch.arange(self.cfg.num_envs, device=self.device)
        self.terrain_levels = torch.div(env_ids, num_cols, rounding_mode="floor")
        self.terrain_types = torch.remainder(env_ids, num_cols)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
