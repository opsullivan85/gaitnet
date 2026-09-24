"""The scene: terrain, the torque-controlled Go1, a foothold scanner on each hip, and foot
contact sensing."""

from __future__ import annotations

from collections.abc import Callable

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from gaitnet_core.grid import FootholdGrid
from gaitnet_core.robot_spec import LEG_NAMES
from gaitnet_sim.env.contract import GaitNetCfg
from gaitnet_sim.robot import BASE_NAME, GO1_TORQUE_CFG, HIP_NAMES
from gaitnet_sim.terrains import holes_terrain_cfg

SCANNER_NAMES: tuple[str, ...] = tuple(f"{leg}_scanner" for leg in LEG_NAMES)


def _shifted_grid_pattern(cfg: ShiftedGridPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ray_starts, ray_directions = patterns.grid_pattern(cfg, device)
    ray_starts[:, :2] += torch.tensor(cfg.shift, device=device)
    return ray_starts, ray_directions


@configclass
class ShiftedGridPatternCfg(patterns.GridPatternCfg):
    """Isaac Lab's grid pattern, centred at `shift` in the sensor's frame rather than on it.
    Only the ray starts move, so the sensor's frame (`data.pos_w`) stays where it is mounted."""

    func: Callable = _shifted_grid_pattern
    shift: tuple[float, float] = (0.0, 0.0)
    """(x, y) of the pattern's centre in the sensor's frame (m)."""


def foothold_scanner_cfg(hip_name: str, grid: FootholdGrid, leg: int) -> RayCasterCfg:
    """A downward grid of rays over leg `leg`'s foothold grid, covering `grid.patch_size`
    cells around the grid's centre.

    Attached to the hip link and yaw-aligned: the ray starts turn with the base's heading
    but not its roll or pitch, so the patch lies in the hip's gravity-aligned yaw frame.
    The 20 m offset only lifts the ray starts, and the pattern's shift only moves them to
    the grid's centre; the sensor's frame (`data.pos_w`) stays at the hip, which is what
    terrain heights are measured from.
    """
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{hip_name}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=ShiftedGridPatternCfg(
            resolution=grid.resolution,
            size=((grid.patch_size[0] - 1) * grid.resolution, (grid.patch_size[1] - 1) * grid.resolution),
            # x outer, y inner: rays reshape to (size_x, size_y), the core grid's layout
            ordering="yx",
            shift=tuple(grid.leg_centers()[leg].tolist()),
        ),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


_GRID = GaitNetCfg().foothold_grid()


@configclass
class GaitNetSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = holes_terrain_cfg()

    robot: ArticulationCfg = GO1_TORQUE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # named SCANNER_NAMES, in leg order
    FL_scanner: RayCasterCfg = foothold_scanner_cfg(HIP_NAMES[0], _GRID, 0)
    FR_scanner: RayCasterCfg = foothold_scanner_cfg(HIP_NAMES[1], _GRID, 1)
    RL_scanner: RayCasterCfg = foothold_scanner_cfg(HIP_NAMES[2], _GRID, 2)
    RR_scanner: RayCasterCfg = foothold_scanner_cfg(HIP_NAMES[3], _GRID, 3)

    contact_forces: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot")
    # the ground under the trunk, for terrain-relative terminations; yaw aligned like the
    # foothold scanners, so its frame (`data.pos_w`) is the base origin
    base_scanner: RayCasterCfg = RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_NAME}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.3, 0.15)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
