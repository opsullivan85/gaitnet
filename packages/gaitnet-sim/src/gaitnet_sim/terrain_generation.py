"""Terrain implementations, kept apart from `gaitnet_sim.terrains` (the cfgs): they import
Isaac Lab's terrain generator and importer, which load USD, and task cfgs have to be
importable before the simulator starts. The cfgs name these by "module:name" strings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage
import torch
import trimesh

from isaaclab.terrains import TerrainGenerator, TerrainImporter
from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from gaitnet_sim.terrains import EvalTerrainGeneratorCfg, HfHolesTerrainCfg, HfPillarsTerrainCfg

logger = logging.getLogger(__name__)

# the largest collision mesh known to work; beyond it robots have fallen through the ground
TRIANGLE_BUDGET = 6.4e6


@height_field_to_mesh
def hole_terrain(difficulty: float, cfg: "HfHolesTerrainCfg") -> np.ndarray:
    """Flat ground with a `difficulty` fraction of square holes and a solid spawn platform.

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
    start = (0 if cfg.platform_at_start else int(shape[0] / 2 - platform / 2), int(shape[1] / 2 - platform / 2))
    terrain[start[0] : start[0] + platform, start[1] : start[1] + platform] = 0

    terrain = scipy.ndimage.zoom(terrain, scale, order=0)
    pad = ((0, output_size[0] - terrain.shape[0]), (0, output_size[1] - terrain.shape[1]))
    return np.pad(terrain, pad, mode="constant", constant_values=void)


@height_field_to_mesh
def pillar_terrain(difficulty: float, cfg: "HfPillarsTerrainCfg") -> np.ndarray:
    """A grid of square pillars at random heights over a void, with a flat spawn platform.

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
    start = (0 if cfg.platform_at_start else (pixels[0] - platform) // 2, (pixels[1] - platform) // 2)
    terrain[start[0] : start[0] + platform, start[1] : start[1] + platform] = 0
    return np.rint(terrain).astype(np.int16)


def compact_height_field_mesh(mesh: trimesh.Trimesh, horizontal_scale: float) -> trimesh.Trimesh:
    """The surface of `mesh`, one of Isaac Lab's height-field meshes, in fewer triangles.

    Isaac Lab spends two triangles on every cell of the height field, flat or not. A cell
    whose two triangles are still its own two halves (the slope correction moved none of its
    corners) and lie flat at one height is merged with its neighbours at that height into
    rectangles. Every other triangle is kept as it is, so the surface is unchanged, quirks of
    the slope correction included, and the rectangles' corners are the grid's own vertices.
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    xy = vertices[:, :2] - vertices[:, :2].min(axis=0)
    lattice = np.rint(xy / horizontal_scale).astype(np.int64)
    on_lattice = (np.abs(xy - lattice * horizontal_scale) < 1e-3 * horizontal_scale).all(axis=1)

    corners = lattice[faces]  # (F, 3, 2)
    z = vertices[faces, 2]  # (F, 3)
    low = corners.min(axis=1)
    offset = corners - low[:, None]
    half = (
        on_lattice[faces].all(axis=1)
        & (corners.max(axis=1) - low == 1).all(axis=1)
        & (z == z[:, :1]).all(axis=1)
        & (np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])[:, 2] != 0)
    )
    # the two halves Isaac Lab splits a cell into, told apart by their corner offsets:
    # (0,0) (1,1) (0,1) and (0,0) (1,0) (1,1)
    sum_x, sum_y = offset[..., 0].sum(axis=1), offset[..., 1].sum(axis=1)
    upper = half & (sum_x == 1) & (sum_y == 2)
    lower = half & (sum_x == 2) & (sum_y == 1)

    levels, level_of = np.unique(z[:, 0], return_inverse=True)
    n_i, n_j = lattice.max(axis=0) + 2
    key = (level_of * n_i + low[:, 0]) * n_j + low[:, 1]
    cells = np.intersect1d(key[upper], key[lower])
    merged = (upper | lower) & np.isin(key, cells)
    if not merged.any():
        return mesh

    # runs of cells along i at one (level, j), then equal runs in consecutive j
    level, rest = np.divmod(cells, n_i * n_j)
    i, j = np.divmod(rest, n_j)
    order = np.lexsort((i, j, level))
    level, i, j = level[order], i[order], j[order]
    starts = np.flatnonzero(np.r_[True, (level[1:] != level[:-1]) | (j[1:] != j[:-1]) | (i[1:] != i[:-1] + 1)])
    ends = np.r_[starts[1:], len(i)]
    level, j, i0, i1 = level[starts], j[starts], i[starts], i[ends - 1] + 1
    order = np.lexsort((j, i1, i0, level))
    level, j, i0, i1 = level[order], j[order], i0[order], i1[order]
    same = (level[1:] == level[:-1]) & (i0[1:] == i0[:-1]) & (i1[1:] == i1[:-1]) & (j[1:] == j[:-1] + 1)
    starts = np.flatnonzero(np.r_[True, ~same])
    ends = np.r_[starts[1:], len(j)]
    level, i0, i1, j0, j1 = level[starts], i0[starts], i1[starts], j[starts], j[ends - 1] + 1

    # each rectangle corner is a corner of one of its cells: look it up by (level, i, j)
    used = np.unique(faces[merged])
    used_key = (np.searchsorted(levels, vertices[used, 2]) * n_i + lattice[used, 0]) * n_j + lattice[used, 1]
    order = np.argsort(used_key)
    used_key, used = used_key[order], used[order]

    def vertex(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return used[np.searchsorted(used_key, (level * n_i + i) * n_j + j)]

    a, b, c, d = vertex(i0, j0), vertex(i1, j0), vertex(i0, j1), vertex(i1, j1)
    # wound like Isaac Lab's halves, normals up
    rectangles = np.concatenate([np.stack([a, d, c], axis=1), np.stack([a, b, d], axis=1)])
    compact = trimesh.Trimesh(vertices, np.concatenate([faces[~merged], rectangles]), process=False)
    compact.remove_unreferenced_vertices()
    return compact


class EvalTerrainGenerator(TerrainGenerator):
    """Generates row i at exactly `cfg.difficulties[i]`, each sub-terrain compacted by
    `compact_height_field_mesh`.

    Neither stock mode does the first: `curriculum=True` jitters each row's difficulty and
    `curriculum=False` samples it uniformly from `difficulty_range`. The second is what makes
    a sweep's worth of long sub-terrains fit under `TRIANGLE_BUDGET`.
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
        triangles = len(self.terrain_mesh.faces)
        logger.info(f"terrain: {triangles / 1e6:.2f}M collision triangles")
        if triangles > TRIANGLE_BUDGET:
            logger.warning(
                f"{triangles / 1e6:.2f}M collision triangles is over the {TRIANGLE_BUDGET / 1e6:.1f}M known to"
                " work. If robots end on foot_below_ground at once, they are falling through the terrain:"
                " use shorter or fewer sub-terrains."
            )

    def _get_terrain_mesh(self, difficulty: float, cfg):
        mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
        # the spawn platform is at height 0; the stock origin is the highest point within a
        # metre of the centre, which on pillars is a pillar top
        origin[2] = 0.0
        return compact_height_field_mesh(mesh, self.cfg.horizontal_scale), origin

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
