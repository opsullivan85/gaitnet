import isaaclab.sim as sim_utils  # type: ignore
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # type: ignore
from isaaclab.scene import InteractiveSceneCfg  # type: ignore
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, FrameTransformerCfg  # type: ignore
from isaaclab.utils import configclass  # type: ignore

from gaitnet.simulation.cfg.footstep_scanner import (
    FL_FootstepScannerCfg,
    FR_FootstepScannerCfg,
    RL_FootstepScannerCfg,
    RR_FootstepScannerCfg,
)
from gaitnet.simulation.cfg.robot import ROBOT_CFG
from gaitnet.simulation.cfg.terrain import VoidTerrainImporterCfg

import numpy as np


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    terrain = VoidTerrainImporterCfg()

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    FR_foot_scanner: RayCasterCfg = FR_FootstepScannerCfg()
    FL_foot_scanner: RayCasterCfg = FL_FootstepScannerCfg()
    RL_foot_scanner: RayCasterCfg = RL_FootstepScannerCfg()
    RR_foot_scanner: RayCasterCfg = RR_FootstepScannerCfg()

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.0,
        debug_vis=False,
        track_air_time=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    foot_transforms = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/FL_foot"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/FR_foot"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/RL_foot"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/RR_foot"),
        ],
        debug_vis=False,
    )
    """FL_foot FR_foot RL_foot RR_foot"""