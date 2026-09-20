"""Low-level controllers the footstep action term can run.

A controller implements `gaitnet_core.interfaces.LowLevelController` and has a configclass
whose `class_type` builds it as `class_type(cfg, num_robots=..., dt=..., device=...)`.
Swap controllers by assigning a different cfg to the action term's `controller` field.

Both of the ones here run the same convex MPC for the same robot. `PooledMpcController`
runs `gaitnet_mpc` one robot per CPU worker and is the reference; `BatchedMpc` runs
`gaitnet_core.control` for the whole batch on the GPU, which is what makes large env
counts affordable. See `gaitnet_core/control/README.md`.
"""

from gaitnet_sim.controllers.batched_mpc import BatchedMpc, BatchedMpcControllerCfg
from gaitnet_sim.controllers.pooled_mpc import PooledMpcController, PooledMpcControllerCfg

__all__ = [
    "BatchedMpc",
    "BatchedMpcControllerCfg",
    "PooledMpcController",
    "PooledMpcControllerCfg",
]
