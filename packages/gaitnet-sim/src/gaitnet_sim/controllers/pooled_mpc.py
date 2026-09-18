"""`LowLevelController` over the CPU MPC: one `gaitnet_mpc` controller per robot, in a
process pool."""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.utils import configclass

from gaitnet_core.interfaces import FootstepCommand
from gaitnet_mpc.controller import MpcFootstepController
from gaitnet_mpc.pool import SharedMemoryVectorPool, VectorPool


class PooledMpcController:
    """Implements `gaitnet_core.interfaces.LowLevelController`.

    Every call crosses to the CPU workers, so tensors are moved to the host and results
    back to `device`.
    """

    def __init__(self, cfg: "PooledMpcControllerCfg", num_robots: int, dt: float, device: str | torch.device):
        """
        Args:
            dt: control period (s), the physics step: torques are computed every step
        """
        self.cfg = cfg
        self.device = torch.device(device)
        self._num_robots = num_robots
        pool_class = SharedMemoryVectorPool if cfg.shared_memory else VectorPool
        self._pool: VectorPool[MpcFootstepController] = pool_class(
            instances=num_robots,
            cls=MpcFootstepController,
            num_workers=cfg.num_workers,
            dt=dt,
            iterations_between_mpc=cfg.iterations_between_mpc,
        )
        self._gait_timing: torch.Tensor | None = None

    @property
    def num_robots(self) -> int:
        return self._num_robots

    def _mask(self, robot_ids: torch.Tensor | None) -> np.ndarray:
        mask = np.zeros(self._num_robots, dtype=bool)
        mask[slice(None) if robot_ids is None else robot_ids.cpu().numpy()] = True
        return mask

    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        self._pool.call(MpcFootstepController.reset, mask=self._mask(robot_ids))
        self._gait_timing = None

    def command_footsteps(self, footsteps: FootstepCommand) -> None:
        active = footsteps.active.cpu().numpy()
        if not active.any():
            return
        # footholds are points on the terrain surface; the MPC places the foot's centre
        location = footsteps.target.cpu().numpy().copy()
        location[:, 2] += self.cfg.foot_radius
        self._pool.call(
            MpcFootstepController.initiate_footstep,
            mask=active,
            leg=footsteps.leg.cpu().numpy().astype(np.int32),
            location_hip=location,
            duration=footsteps.duration.cpu().numpy(),
        )
        self._gait_timing = None

    def compute_torques(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        base_pose: torch.Tensor,
        base_vel: torch.Tensor,
        velocity_command: torch.Tensor,
    ) -> torch.Tensor:
        # one host transfer instead of five, since each .cpu() is its own sync
        host = torch.cat([joint_pos, joint_vel, base_pose, base_vel, velocity_command], dim=-1).cpu().numpy()
        n = host.shape[0]
        pos, vel = host[:, :12], host[:, 12:24]
        torques = self._pool.call(
            MpcFootstepController.get_torques,
            mask=None,
            joint_states=np.stack([pos, vel], axis=-1).reshape(n, 4, 3, 2),
            body_state=host[:, 24:37],
            command=host[:, 37:40],
        )
        self._gait_timing = None
        return torch.from_numpy(torques.reshape(n, 12).astype(np.float32)).to(self.device)

    def gait_timing(self) -> torch.Tensor:
        """(N, L, 3), fetched once per change of controller state."""
        if self._gait_timing is None:
            timing = self._pool.call(MpcFootstepController.get_gait_timing, mask=None)
            self._gait_timing = torch.from_numpy(timing).to(self.device)
        return self._gait_timing

    def estimated_rpy(self) -> torch.Tensor:
        """(N, 3) base roll, pitch, yaw as the controllers' state estimators last saw them."""
        return torch.from_numpy(self._pool.call(MpcFootstepController.get_estimated_rpy, mask=None)).to(self.device)

    def close(self) -> None:
        self._pool.close()


@configclass
class PooledMpcControllerCfg:
    class_type: type = PooledMpcController

    iterations_between_mpc: int = 5
    """Physics steps per MPC solve; the leg PD and swing control run every step."""
    foot_radius: float = 0.02
    """Radius of the Go1's spherical foot (m). The MPC's foot point is the sphere's centre,
    so a foothold on the surface is raised by this much."""
    num_workers: int | None = None
    """Worker processes, default one per CPU core."""
    shared_memory: bool = False
    """Pass arrays through shared memory instead of pipes. Off by default: transport is
    under 2% of call time, the MPC solves dominate."""
