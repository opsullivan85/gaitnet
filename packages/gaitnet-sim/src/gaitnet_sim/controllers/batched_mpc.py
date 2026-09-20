"""`LowLevelController` over the GPU MPC: every robot solved together, nothing on the CPU.

The controller itself lives in `gaitnet_core.control`, which knows nothing about Isaac
Lab; all this adds is the configclass the env cfg holds and the command-line overrides
reach, and the translation from its flat fields into the nested settings the controller
takes. It is a drop-in alternative to `PooledMpcController`, solving the same convex MPC
for the same robot - see `gaitnet_core/control/README.md` for what it matches and where
it deliberately differs.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from gaitnet_core.control import AdmmSettings, BatchedMpcConfig, BatchedMpcController, MpcSettings
from gaitnet_core.control.model import MODELS


class BatchedMpc(BatchedMpcController):
    """`BatchedMpcController` built from an Isaac Lab cfg."""

    def __init__(
        self,
        cfg: "BatchedMpcControllerCfg",
        num_robots: int,
        dt: float,
        device: str | torch.device,
    ) -> None:
        super().__init__(
            BatchedMpcConfig(
                model=MODELS[cfg.robot],
                mpc=MpcSettings(
                    horizon=cfg.horizon,
                    control_steps_per_update=cfg.iterations_between_mpc,
                    admm=AdmmSettings(
                        iterations=cfg.solver_iterations,
                        rho_update_interval=cfg.solver_rho_update_interval,
                    ),
                ),
                foot_radius=cfg.foot_radius,
                dtype=torch.float64 if cfg.double_precision else torch.float32,
            ),
            num_robots=num_robots,
            dt=dt,
            device=device,
        )


@configclass
class BatchedMpcControllerCfg:
    class_type: type = BatchedMpc

    robot: str = "go1"
    """Which `gaitnet_core.control.model.MODELS` entry to take the dynamics and gains
    from. Must be the robot the scene spawns."""
    iterations_between_mpc: int = 5
    """Physics steps per MPC solve; the leg PD and swing control run every step."""
    horizon: int = 10
    """MPC prediction steps. Peak GPU memory grows with its square."""
    foot_radius: float = 0.02
    """Radius of the Go1's spherical foot (m). The MPC's foot point is the sphere's centre,
    so a foothold on the surface is raised by this much."""
    solver_iterations: int = 80
    """ADMM iterations per solve, the main throughput-against-accuracy dial. The default
    tracks the CPU controller's torques to well under a percent; see
    `gaitnet_core/control/README.md` for what lowering it costs."""
    solver_rho_update_interval: int = 20
    """Iterations between step-size updates inside the solver. Each one re-inverts, so
    this trades directly against `solver_iterations`; they were tuned together."""
    double_precision: bool = False
    """Run the controller and its QP in float64. Roughly doubles the solve cost and is
    only useful for checking the float32 path."""
