"""A terrain that holds the whole difficulty sweep at once.

The sweep used to run one difficulty per process: every configuration booted
Omniverse, cooked a fresh collision mesh and forked a controller pool, then threw
all of it away. Nothing about the difficulty axis needs a new scene though -- it
only needs different ground under the robot -- so this module lays the difficulties
out along the rows of a single terrain grid and pins each environment to one
(row, column) cell. One scene then covers every difficulty simultaneously.

Row ``i`` is generated at exactly ``difficulties[i]``. Neither stock generation mode
can do that: ``curriculum=True`` jitters the difficulty to ``(row + U(0,1)) / num_rows``
and ``curriculum=False`` samples it uniformly from ``difficulty_range``.

Environment ``i`` lands on row ``i // envs_per_difficulty``, column
``i % envs_per_difficulty`` -- see :func:`envs_for_difficulty`.
"""

from __future__ import annotations

import torch
from isaaclab.terrains import (
    TerrainGenerator,
    TerrainGeneratorCfg,
    TerrainImporter,
    TerrainImporterCfg,
)
from isaaclab.utils import configclass


class EvalTerrainGenerator(TerrainGenerator):
    """Terrain generator that takes an explicit difficulty per row."""

    cfg: "EvalTerrainGeneratorCfg"

    def __init__(self, cfg: "EvalTerrainGeneratorCfg", device: str = "cpu"):
        if len(cfg.difficulties) != cfg.num_rows:
            raise ValueError(
                f"Expected one difficulty per row, got {len(cfg.difficulties)}"
                f" difficulties for {cfg.num_rows} rows."
            )
        if len(cfg.sub_terrains) != 1:
            raise ValueError(
                "The difficulty is the only axis varied across the grid, so exactly one"
                f" sub-terrain is expected. Got {list(cfg.sub_terrains)}."
            )
        if cfg.curriculum:
            raise ValueError(
                "curriculum routes generation through the base class' difficulty ramp,"
                " which overrides the explicit per-row difficulties."
            )
        if cfg.use_cache:
            raise ValueError(
                "use_cache would collapse this terrain: every column of a row is built"
                " from an identical sub-terrain config, so they all hash to the same"
                " cache entry and the columns stop being independent samples."
            )
        super().__init__(cfg, device)

    def _generate_random_terrains(self):
        """Generate row ``i`` at ``cfg.difficulties[i]``.

        ``curriculum=False`` routes generation here. The base class samples a fresh
        difficulty per cell from ``difficulty_range``; we use the explicit schedule
        instead. Columns within a row still differ, because the hole layout is drawn
        from the global numpy random state rather than from the difficulty.
        """
        sub_terrain_cfg = next(iter(self.cfg.sub_terrains.values()))
        for row, difficulty in enumerate(self.cfg.difficulties):
            for col in range(self.cfg.num_cols):
                mesh, origin = self._get_terrain_mesh(float(difficulty), sub_terrain_cfg)
                self._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)


@configclass
class EvalTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Configuration for :class:`EvalTerrainGenerator`."""

    class_type: type = EvalTerrainGenerator

    curriculum: bool = False

    difficulties: tuple[float, ...] = ()
    """The difficulty of each row, one entry per ``num_rows``."""


class EvalTerrainImporter(TerrainImporter):
    """Terrain importer that pins each environment to a fixed sub-terrain.

    The base class assigns environments to rows at random and lets the terrain-level
    curriculum move them around. Here the row *is* the difficulty being measured, so
    the assignment has to be fixed and known to the caller.
    """

    def configure_env_origins(self, origins=None):
        super().configure_env_origins(origins)
        if self.terrain_origins is None:
            return

        num_rows, num_cols = self.terrain_origins.shape[:2]
        if self.cfg.num_envs != num_rows * num_cols:
            raise ValueError(
                f"Expected one environment per sub-terrain: {self.cfg.num_envs} environments"
                f" for a {num_rows}x{num_cols} grid."
            )

        env_ids = torch.arange(self.cfg.num_envs, device=self.device)
        self.terrain_levels = torch.div(env_ids, num_cols, rounding_mode="floor").to(torch.long)
        self.terrain_types = torch.remainder(env_ids, num_cols).to(torch.long)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]


def envs_for_difficulty(difficulty_index: int, envs_per_difficulty: int) -> slice:
    """The environments :class:`EvalTerrainImporter` placed on row ``difficulty_index``."""
    start = difficulty_index * envs_per_difficulty
    return slice(start, start + envs_per_difficulty)


def make_eval_terrain(
    terrain: TerrainImporterCfg,
    difficulties: tuple[float, ...],
    envs_per_difficulty: int,
    sub_terrain_size: tuple[float, float],
) -> None:
    """Rewrite ``terrain`` in place as a one-difficulty-per-row evaluation grid.

    The sub-terrain type, physics material and scales are kept as configured; only the
    grid layout and the generation strategy change.

    Args:
        terrain: The terrain importer config to rewrite.
        difficulties: One difficulty per row, in the order the rows are laid out.
        envs_per_difficulty: Columns per row, i.e. samples per difficulty.
        sub_terrain_size: The (x, y) extent of a single sub-terrain, in metres.
    """
    generator = terrain.terrain_generator
    if generator is None:
        raise ValueError("Expected a generated terrain, got terrain_generator=None.")

    terrain.class_type = EvalTerrainImporter
    terrain.terrain_generator = EvalTerrainGeneratorCfg(
        size=sub_terrain_size,
        horizontal_scale=generator.horizontal_scale,
        vertical_scale=generator.vertical_scale,
        slope_threshold=generator.slope_threshold,
        border_width=generator.border_width,
        # copied because TerrainGenerator writes the grid's scales into each sub-terrain
        sub_terrains={name: cfg.copy() for name, cfg in generator.sub_terrains.items()},
        num_rows=len(difficulties),
        num_cols=envs_per_difficulty,
        difficulties=tuple(difficulties),
    )
