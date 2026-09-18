"""Terrains: randomly holed ground for training, and an evaluation grid that holds a whole
difficulty sweep in one scene."""

from __future__ import annotations

import numpy as np
import scipy.ndimage
import torch

from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg


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


@configclass
class HfHolesTerrainCfg(HfTerrainBaseCfg):
    function = hole_terrain

    hole_depth: float = -0.5
    """Depth of the holes (m), negative."""
    platform_size: float = 1.0
    """Side of the hole-free square at the centre, where robots spawn (m)."""


TERRAIN_MATERIAL = PhysxRigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
)


def holes_terrain_cfg() -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(4.0, 4.0),
            horizontal_scale=0.025,
            slope_threshold=0.0,
            sub_terrains={"holes": HfHolesTerrainCfg()},
            curriculum=True,
            num_rows=12,
            num_cols=12,
            difficulty_range=(0.0, 0.5),
        ),
        physics_material=TERRAIN_MATERIAL,
    )


##
# Evaluation grid
##


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


@configclass
class EvalTerrainGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = EvalTerrainGenerator
    curriculum: bool = False
    difficulties: tuple[float, ...] = ()
    """One difficulty per row."""


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


def envs_for_difficulty(difficulty_index: int, envs_per_difficulty: int) -> slice:
    """The environments `EvalTerrainImporter` put on row `difficulty_index`."""
    start = difficulty_index * envs_per_difficulty
    return slice(start, start + envs_per_difficulty)


def make_eval_terrain(
    terrain: TerrainImporterCfg,
    difficulties: tuple[float, ...],
    envs_per_difficulty: int,
    sub_terrain_size: tuple[float, float],
) -> None:
    """Rewrite `terrain` in place as a one-difficulty-per-row evaluation grid, keeping its
    sub-terrain type, material and scales."""
    generator = terrain.terrain_generator
    if generator is None:
        raise ValueError("expected a generated terrain")
    terrain.class_type = EvalTerrainImporter
    terrain.terrain_generator = EvalTerrainGeneratorCfg(
        size=sub_terrain_size,
        horizontal_scale=generator.horizontal_scale,
        vertical_scale=generator.vertical_scale,
        slope_threshold=generator.slope_threshold,
        border_width=generator.border_width,
        # copied: the generator writes the grid's scales into each sub-terrain cfg
        sub_terrains={name: cfg.copy() for name, cfg in generator.sub_terrains.items()},
        num_rows=len(difficulties),
        num_cols=envs_per_difficulty,
        difficulties=tuple(difficulties),
    )
