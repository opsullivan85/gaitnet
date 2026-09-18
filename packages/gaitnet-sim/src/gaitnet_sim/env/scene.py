"""The scene: terrain, the torque-controlled Go1, a foothold scanner on each hip, and foot
contact sensing."""

from __future__ import annotations

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


def foothold_scanner_cfg(hip_name: str, grid: FootholdGrid) -> RayCasterCfg:
    """A downward grid of rays centred on a hip, covering `grid.patch_size` cells.

    Attached to the hip link and yaw-aligned: the ray starts turn with the base's heading
    but not its roll or pitch, so the patch lies in the hip's gravity-aligned yaw frame.
    The 20 m offset only lifts the ray starts; the sensor's frame (`data.pos_w`) stays at
    the hip, which is what terrain heights are measured from.
    """
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{hip_name}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=grid.resolution,
            size=((grid.patch_size[0] - 1) * grid.resolution, (grid.patch_size[1] - 1) * grid.resolution),
            # x outer, y inner: rays reshape to (size_x, size_y), the core grid's layout
            ordering="yx",
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
    FL_scanner: RayCasterCfg = foothold_scanner_cfg(HIP_NAMES[0], _GRID)
    FR_scanner: RayCasterCfg = foothold_scanner_cfg(HIP_NAMES[1], _GRID)
    RL_scanner: RayCasterCfg = foothold_scanner_cfg(HIP_NAMES[2], _GRID)
    RR_scanner: RayCasterCfg = foothold_scanner_cfg(HIP_NAMES[3], _GRID)

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
