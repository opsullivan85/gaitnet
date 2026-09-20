"""The low-level controller, batched over robots and run on the GPU.

Footsteps in, joint torques out, for N robots at once: a convex single-rigid-body MPC
for the stance forces, a Bezier arc for the swinging foot, and a Cartesian PD through
the leg Jacobian. `BatchedMpcController` implements
`gaitnet_core.interfaces.LowLevelController`, so it goes wherever the CPU process pool
in `gaitnet_sim.controllers.PooledMpcController` goes.

It is a copy of `gaitnet-mpc`, not an improvement on it; `README.md` in this directory
says what it copies, what it deliberately does differently and why.
"""

from gaitnet_core.control.admm import AdmmSettings, QpSolution
from gaitnet_core.control.controller import BatchedMpcConfig, BatchedMpcController
from gaitnet_core.control.gait import GaitSchedule
from gaitnet_core.control.model import GO1_SRBD, MODELS, SrbdModel
from gaitnet_core.control.mpc import BatchedConvexMpc, MpcSettings

__all__ = [
    "AdmmSettings",
    "BatchedConvexMpc",
    "BatchedMpcConfig",
    "BatchedMpcController",
    "GO1_SRBD",
    "GaitSchedule",
    "MODELS",
    "MpcSettings",
    "QpSolution",
    "SrbdModel",
]
