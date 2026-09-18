"""Terrain cfgs: randomly holed flat ground and pillars of varying height for training, and
an evaluation grid that holds a whole difficulty sweep in one scene. The implementations are in
`gaitnet_sim.terrain_generation`, referenced by name so these import without the simulator."""

from __future__ import annotations

from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.utils import configclass
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg


@configclass
class HfHolesTerrainCfg(HfTerrainBaseCfg):
    function: str = "gaitnet_sim.terrain_generation:hole_terrain"

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


@configclass
class HfPillarsTerrainCfg(HfTerrainBaseCfg):
    """Square pillars at random heights; see `gaitnet_sim.terrain_generation.pillar_terrain`."""

    function: str = "gaitnet_sim.terrain_generation:pillar_terrain"

    pillar_width: float = 0.2
    """Side of each pillar's top (m)."""
    max_gap: float = 0.15
    """Gap between neighbouring pillars at difficulty 1 (m); none at difficulty 0."""
    max_height_offset: float = 0.1
    """Pillar heights are uniform in +-(difficulty * this) (m)."""
    hole_depth: float = -0.5
    """Height of the void between pillars (m), negative."""
    platform_size: float = 1.0
    """Side of the flat square at the centre, where robots spawn (m)."""


def _generated_terrain_cfg(name: str, sub_terrain: HfTerrainBaseCfg) -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(4.0, 4.0),
            horizontal_scale=0.025,
            # every height step becomes a vertical wall
            slope_threshold=0.0,
            sub_terrains={name: sub_terrain},
            curriculum=True,
            num_rows=12,
            num_cols=12,
            difficulty_range=(0.0, 0.5),
        ),
        physics_material=TERRAIN_MATERIAL,
    )


def holes_terrain_cfg() -> TerrainImporterCfg:
    return _generated_terrain_cfg("holes", HfHolesTerrainCfg())


def pillars_terrain_cfg() -> TerrainImporterCfg:
    return _generated_terrain_cfg("pillars", HfPillarsTerrainCfg())


##
# Evaluation grid
##


@configclass
class EvalTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Row i at exactly `difficulties[i]`, see `gaitnet_sim.terrain_generation.EvalTerrainGenerator`."""

    class_type: str = "gaitnet_sim.terrain_generation:EvalTerrainGenerator"
    curriculum: bool = False
    difficulties: tuple[float, ...] = ()
    """One difficulty per row."""


def envs_for_difficulty(difficulty_index: int, envs_per_difficulty: int) -> slice:
    """The environments `EvalTerrainImporter` puts on row `difficulty_index`."""
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
    from gaitnet_sim.terrain_generation import EvalTerrainImporter

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
