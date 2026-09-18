"""CPU convex MPC with specified-footstep swing control, and the process pool that runs
one controller per simulated robot.

Vendored from rl-mpc-locomotion (MIT, Yulun Zhuang), via the opsullivan85 fork. Kept
as the baseline low-level controller until a GPU controller replaces it.
"""

from gaitnet_mpc.controller import MpcFootstepController
from gaitnet_mpc.mpc.common.Quadruped import RobotType
from gaitnet_mpc.pool import SharedMemoryVectorPool, VectorPool

__all__ = ["MpcFootstepController", "RobotType", "SharedMemoryVectorPool", "VectorPool"]
