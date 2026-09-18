"""Low-level controllers the footstep action term can run.

A controller implements `gaitnet_core.interfaces.LowLevelController` and has a configclass
whose `class_type` builds it as `class_type(cfg, num_robots=..., dt=..., device=...)`.
Swap controllers by assigning a different cfg to the action term's `controller` field.
"""

from gaitnet_sim.controllers.pooled_mpc import PooledMpcController, PooledMpcControllerCfg

__all__ = ["PooledMpcController", "PooledMpcControllerCfg"]
