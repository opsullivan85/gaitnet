from typing import Callable

from isaaclab.sensors import RayCasterCfg, patterns
import gaitnet.constants as const

_hip_names = ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]

# Configure offset positions for the raycasters relative to each hip
# note that all axes are in the same orientation as the robot body frame
_stable_footstep_offsets = {
    _hip_names[0]: (0.0, 0.0),
    _hip_names[1]: (0.0, -0.0),
    _hip_names[2]: (-0.0, 0.0),
    _hip_names[3]: (-0.0, -0.0),
}
_height_scanner_offsets = {
    hip_name: RayCasterCfg.OffsetCfg(pos=(*_stable_footstep_offsets[hip_name], 20.0))
    for hip_name in _hip_names
}


# Configure other raycaster settings
_height_scanner_settings = {
    hip_name: {
        "prim_path": f"{{ENV_REGEX_NS}}/Robot/{hip_name}",
        "offset": _height_scanner_offsets[hip_name],
        "update_period": 0.00,  # every sim step
        "ray_alignment": "yaw",
        "pattern_cfg": patterns.GridPatternCfg(
            resolution=const.footstep_scanner.grid_resolution,
            size=(
                (const.footstep_scanner.sensor_grid_size[0] - 1) * const.footstep_scanner.grid_resolution,
                (const.footstep_scanner.sensor_grid_size[1] - 1) * const.footstep_scanner.grid_resolution,
            ),
            # importantly, this is the ordering that 
            # contact-net expects (was used in training data generation)
            ordering="yx",
        ),
        "debug_vis": True,
        "mesh_prim_paths": ["/World/ground"],
    }
    for hip_name in _hip_names
}

FR_FootstepScannerCfg: Callable[[], RayCasterCfg] = lambda: RayCasterCfg(
    **_height_scanner_settings["FR_hip"]
)
FL_FootstepScannerCfg: Callable[[], RayCasterCfg] = lambda: RayCasterCfg(
    **_height_scanner_settings["FL_hip"]
)
RL_FootstepScannerCfg: Callable[[], RayCasterCfg] = lambda: RayCasterCfg(
    **_height_scanner_settings["RL_hip"]
)
RR_FootstepScannerCfg: Callable[[], RayCasterCfg] = lambda: RayCasterCfg(
    **_height_scanner_settings["RR_hip"]
)
