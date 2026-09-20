"""The convex MPC itself: condense, solve, hand back this step's contact forces.

Thin on purpose. `srbd.build` owns the model and `admm.solve` owns the numerics; what
lives here is the batch's warm start, which is most of why a fixed iteration budget is
enough. The problem moves slowly between updates - the body has travelled a couple of
centimetres and one foot may have changed state - so starting from the previous
solution puts the solver near the answer before it takes a step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from gaitnet_core.control import srbd
from gaitnet_core.control.admm import AdmmSettings, solve
from gaitnet_core.control.model import SrbdModel


@dataclass(frozen=True)
class MpcSettings:
    """Horizon, update rate and solver budget."""

    horizon: int = 10
    """Prediction steps. Peak memory in `srbd.build` grows with its square."""
    control_steps_per_update: int = 5
    """Control steps between MPC solves. The prediction step is this times the control
    period, so with the defaults the MPC runs at 50 Hz and looks 0.2 s ahead."""
    admm: AdmmSettings = field(default_factory=AdmmSettings)
    """Solver tuning."""


class BatchedConvexMpc:
    """One convex MPC per robot, all solved together.

    Holds the warm start across calls, so `reset` matters: a robot that has been
    teleported to a new episode must not start from the forces it wanted in the old one.
    """

    def __init__(
        self,
        model: SrbdModel,
        settings: MpcSettings,
        num_robots: int,
        control_dt: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.settings = settings
        self.control_dt = control_dt
        self.device = torch.device(device)
        self.dtype = dtype

        legs = model.num_legs
        variables = settings.horizon * legs * 3
        blocks = settings.horizon * legs
        self._primal = torch.zeros(num_robots, variables, device=device, dtype=dtype)
        self._dual = torch.zeros(num_robots, blocks, srbd.CONE_ROWS, device=device, dtype=dtype)
        self._weights = torch.tensor(model.weights, device=device, dtype=dtype).expand(
            num_robots, model.state_dim
        )
        self.primal_residual = torch.zeros(num_robots, device=device, dtype=dtype)
        """(N,) worst constraint violation of the last solve (N)."""
        self.dual_residual = torch.zeros(num_robots, device=device, dtype=dtype)
        """(N,) worst stationarity violation of the last solve (N)."""

    @property
    def timestep(self) -> float:
        """Seconds per prediction step (s)."""
        return self.control_dt * self.settings.control_steps_per_update

    def solve(
        self,
        rpy: torch.Tensor,
        position: torch.Tensor,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        foot_positions: torch.Tensor,
        contact: torch.Tensor,
        desired_velocity: torch.Tensor,
        desired_yaw_rate: torch.Tensor,
    ) -> torch.Tensor:
        """(N, L, 3) force each foot should apply this control step (N).

        Arguments are as in `srbd.build`. The sign is the one the leg controller
        expects: the solver's variables are the forces the ground applies to the body,
        and what comes back is their negation, the force the leg pushes with, which is
        what the leg Jacobian turns into joint torques.
        """
        problem = srbd.build(
            self.model,
            horizon=self.settings.horizon,
            timestep=self.timestep,
            rpy=rpy,
            position=position,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            foot_positions=foot_positions,
            contact=contact,
            desired_velocity=desired_velocity,
            desired_yaw_rate=desired_yaw_rate,
            weights=self._weights,
        )
        solution = solve(
            problem.hessian,
            problem.gradient,
            problem.cone,
            problem.lower,
            problem.upper,
            self._primal,
            self._dual,
            self.settings.admm,
        )
        self._primal = solution.primal
        self._dual = solution.dual
        self.primal_residual = solution.primal_residual
        self.dual_residual = solution.dual_residual

        # only the first horizon step is applied; the rest of the plan is discarded and
        # re-solved from the new state next time
        forces = solution.primal.view(-1, self.settings.horizon, self.model.num_legs, 3)
        return -forces[:, 0]

    def reset(self, robot_ids: torch.Tensor) -> None:
        """Drop the warm start for the named robots."""
        self._primal[robot_ids] = 0.0
        self._dual[robot_ids] = 0.0
