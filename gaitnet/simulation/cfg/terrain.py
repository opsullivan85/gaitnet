from typing import Callable

import isaaclab.sim as sim_utils  # type: ignore
import numpy as np
import scipy.ndimage  # type: ignore
from isaaclab.terrains import TerrainGeneratorCfg  # type: ignore
from isaaclab.terrains import TerrainImporterCfg  # type: ignore
from isaaclab.terrains.height_field import hf_terrains_cfg  # type: ignore
from isaaclab.terrains.height_field.utils import height_field_to_mesh  # type: ignore
from isaaclab.utils import configclass  # type: ignore


@height_field_to_mesh
def hole_terrain(difficulty: float, cfg: "HfVoidTerrainCfg") -> np.ndarray:
    """

    Note:
        cfg.vertical_scale is not used

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
            0 is flat ground, 1 is all holes
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # this scale variable fixes the "smoothing" done by the rest of the isaac heightfield pipeline
    # the heightfield pipeline behaves unexpectdley when heights have high, alternating slopes.
    # we are essentially going from this [ 0 1 0 1 1 0 ]
    # to [ 0 0 0 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0] (while actually maintaining the final output size)
    SCALE = 5
    output_size = (
        int(cfg.size[0] / cfg.horizontal_scale),
        int(cfg.size[1] / cfg.horizontal_scale),
    )
    shape_px = (output_size[0] // SCALE, output_size[1] // SCALE)
    terrain = np.zeros(shape_px, dtype=np.float32)

    # add voids
    terrain_indices = np.argwhere(np.ones_like(terrain))  # all (i, j) pairs
    shuffled_indices = np.random.permutation(terrain_indices)
    void_indices = shuffled_indices[: int(difficulty * len(shuffled_indices))]
    terrain[tuple(void_indices.T)] = cfg.void_depth / cfg.vertical_scale

    # add solid platform in center
    platform_size_px = int(cfg.platform_size / cfg.horizontal_scale / SCALE)
    platform_start = (
        int(shape_px[0] / 2 - platform_size_px / 2),
        int(shape_px[1] / 2 - platform_size_px / 2),
    )
    platform_end = (
        int(shape_px[0] / 2 + platform_size_px / 2),
        int(shape_px[1] / 2 + platform_size_px / 2),
    )
    terrain[
        platform_start[0] : platform_end[0], platform_start[1] : platform_end[1]
    ] = 0

    scaled_terrain = scipy.ndimage.zoom(terrain, SCALE, order=0)
    # pad with voids upto the output_size
    padded_terrain = np.pad(
        scaled_terrain,
        (
            (0, output_size[0] - scaled_terrain.shape[0]),
            (0, output_size[1] - scaled_terrain.shape[1]),
        ),
        mode="constant",
        constant_values=cfg.void_depth,
    )
    return padded_terrain


@configclass
class HfVoidTerrainCfg(hf_terrains_cfg.HfTerrainBaseCfg):
    """Configuration for a random uniform height field terrain."""

    function = hole_terrain

    void_depth: float = -10.0
    """The depth of the voids (negative obstacles). Defaults to -10.0."""
    platform_size: float = 1.0
    """The width of the platform at the center of the terrain. Defaults to 1.0."""


VoidTerrainImporterCfg: Callable[[], TerrainImporterCfg] = lambda: TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    # https://isaac-sim.github.io/IsaacLab/v2.1.0/source/api/lab/isaaclab.terrains.html#isaaclab.terrains.TerrainGeneratorCfg
    terrain_generator=TerrainGeneratorCfg(
        size=(4, 4),
        horizontal_scale=0.025,
        slope_threshold=0,
        sub_terrains={
            "holes": HfVoidTerrainCfg(
                void_depth=-0.5,
                platform_size=1.0,
            ),
        },
        curriculum=True,
        num_rows=12,
        num_cols=12,
        difficulty_range=(0.0, 0.5),
    ),
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
)
