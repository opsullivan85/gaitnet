"""Turning the robot's state into the dense QP the MPC solves, batched over robots.

The model is the usual convex single-rigid-body one [1]: the body is a rigid mass with a
fixed inertia, each stance foot applies a force inside a friction cone at a fixed offset
from the body, and the orientation dynamics are linearised about level. The state the
MPC predicts is

    [roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz, -g]

in a gravity-aligned frame carried with the body, with gravity as a constant thirteenth
state so the dynamics stay linear. The decision variables are every foot's force at
every horizon step, ordered step-major then leg then xyz:

    [step 0: FL xyz, FR xyz, RL xyz, RR xyz, step 1: ...]

Eliminating the states leaves a dense QP in the forces alone, which is what `build`
returns.

This reproduces `ConvexMpc::ComputeContactForces` in
`packages/gaitnet-mpc/cpp/mpc_osqp.cc` - Google's implementation of [1], vendored via
rl-mpc-locomotion, see `packages/gaitnet-mpc/THIRD_PARTY.md` - term for term, including
two things that look like mistakes and are marked where they appear. They are kept
because the point of this module is to be the same controller as the CPU one, and every
policy bundle and evaluation baseline in the repo was produced against its behaviour.
Correcting them is a separate, measurable change, not a side effect of moving to the
GPU.

[1] Di Carlo, Wensing, Katz, Bledt and Kim, "Dynamic Locomotion in the MIT Cheetah 3
    Through Convex Model-Predictive Control", IROS 2018.
    https://ieeexplore.ieee.org/document/8594448
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaitnet_core.control.model import GRAVITY, SrbdModel
from gaitnet_core.control.rotations import rpy_to_rotation_xyz, rpy_to_rotation_zyx, skew

CONE_ROWS = 5
"""Constraint rows per foot per horizon step: four sides of the friction pyramid, then
the normal force's own bounds."""


@dataclass
class SrbdQp:
    """One condensed QP per robot, ready for `gaitnet_core.control.admm.solve`."""

    hessian: torch.Tensor
    """(N, n, n) P, with n = horizon * legs * 3 force components."""
    gradient: torch.Tensor
    """(N, n) q."""
    cone: torch.Tensor
    """(N, B, 5, 3) the friction pyramid acting on each of the B = horizon * legs force
    triples. Every block holds the same matrix; it is expanded, not copied."""
    lower: torch.Tensor
    """(N, B, 5) constraint lower bounds (N). Zero for a foot scheduled off the ground."""
    upper: torch.Tensor
    """(N, B, 5) constraint upper bounds (N). Zero for a foot scheduled off the ground,
    which pins that foot's three forces to zero exactly."""

    @property
    def num_variables(self) -> int:
        return self.hessian.shape[-1]


def build(
    model: SrbdModel,
    *,
    horizon: int,
    timestep: float,
    rpy: torch.Tensor,
    position: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    foot_positions: torch.Tensor,
    contact: torch.Tensor,
    desired_velocity: torch.Tensor,
    desired_yaw_rate: torch.Tensor,
    weights: torch.Tensor,
    ground_normal: torch.Tensor | None = None,
) -> SrbdQp:
    """Condense the MPC into a QP in the contact forces.

    Every input is in the estimator's gravity-aligned body frame, the frame the CPU
    controller calls "global": its origin rides with the body, so the x and y of
    `position` are always zero and only its z, the height above the stance feet, carries
    information.

    Args:
        model: dynamics, weights and force bounds.
        horizon: prediction steps.
        timestep: seconds per prediction step, the control period times the number of
            control steps between MPC updates.
        rpy: (N, 3) base roll, pitch, yaw relative to the yaw frame (rad), so yaw is ~0.
        position: (N, 3) base position (m); x and y are zero, z is the height estimate.
        linear_velocity: (N, 3) base linear velocity in the base frame (m/s).
        angular_velocity: (N, 3) base angular velocity in the base frame (rad/s).
        foot_positions: (N, L, 3) foot positions relative to the base, base frame (m).
        contact: (N, H, L) scheduled contact over the horizon, 1 in stance.
        desired_velocity: (N, 2) commanded x and y velocity, base frame (m/s).
        desired_yaw_rate: (N,) commanded yaw rate (rad/s).
        weights: (N, 13) state-tracking weights, per robot so an experiment can vary them.
        ground_normal: (N, 3) surface normal in the yaw frame, default straight up.

    Returns:
        The QP. Building it dominates the solve for short iteration budgets, and its
        peak memory grows with the square of the horizon: the intermediate that carries
        every (step, step) pair of input-to-state blocks is the largest tensor here.
    """
    num_robots = rpy.shape[0]
    legs = model.num_legs
    states = model.state_dim
    inputs = 3 * legs
    device, dtype = rpy.device, rpy.dtype

    state = torch.cat(
        [
            rpy,
            position,
            angular_velocity,
            linear_velocity,
            torch.full((num_robots, 1), -GRAVITY, device=device, dtype=dtype),
        ],
        dim=-1,
    )
    desired = _desired_trajectory(
        model, horizon, timestep, rpy, position, desired_velocity, desired_yaw_rate
    )

    dynamics = _state_matrix(rpy, ground_normal, states=states)
    actuation = _input_matrix(model, rpy, foot_positions)
    discrete_dynamics, discrete_actuation = _discretise(dynamics, actuation, timestep)

    # anb[:, i] = A^i B, the effect of a force i steps ago on the state now
    powers = [discrete_actuation]
    for _ in range(1, horizon):
        powers.append(discrete_dynamics @ powers[-1])
    anb = torch.stack(powers, dim=1)

    free_response = torch.zeros(num_robots, horizon, states, states, device=device, dtype=dtype)
    free_response[:, 0] = discrete_dynamics
    running = discrete_dynamics
    for step in range(1, horizon - 1):
        running = discrete_dynamics @ running
        free_response[:, step] = running
    # Upstream's loop stops one step early (`for (i = 1; i < horizon - 1; ++i)` in
    # CalculateQpMats), leaving the last block row zero, so the final horizon step
    # predicts no free response at all. Reproduced deliberately; see the module note.

    forced_response = _forced_response(anb, horizon)

    weighted = forced_response * weights.repeat(1, horizon).unsqueeze(-1)
    # B' W B is symmetric, but float32 leaves the product asymmetric by ~1e-4, which is
    # ten times the ridge below. `admm._invert` factorises the lower triangle alone, so
    # that asymmetry becomes a different, indefinite P, and ADMM diverges on it. Summing
    # the product with its transpose is the same value, exactly symmetric, for free.
    product = forced_response.transpose(-1, -2) @ weighted
    hessian = product + product.transpose(-1, -2)
    hessian.diagonal(dim1=-2, dim2=-1).add_(model.force_regularization)

    predicted = torch.einsum(
        "nij,nj->ni", free_response.reshape(num_robots, horizon * states, states), state
    )
    error = predicted - desired.reshape(num_robots, horizon * states)
    gradient = 2.0 * torch.einsum(
        "nij,ni->nj", forced_response, weights.repeat(1, horizon) * error
    )

    cone, lower, upper = _constraints(model, contact, horizon, legs)
    assert hessian.shape[-1] == horizon * inputs
    return SrbdQp(hessian=hessian, gradient=gradient, cone=cone, lower=lower, upper=upper)


def _desired_trajectory(
    model: SrbdModel,
    horizon: int,
    timestep: float,
    rpy: torch.Tensor,
    position: torch.Tensor,
    desired_velocity: torch.Tensor,
    desired_yaw_rate: torch.Tensor,
) -> torch.Tensor:
    """(N, H, 13) reference the MPC tracks.

    Level and at the nominal height, holding the commanded velocity, with the position
    and yaw references integrated forward from where the robot is now. The vertical
    velocity reference is zero rather than the integral of anything, which is what keeps
    the body height steady.
    """
    num_robots = rpy.shape[0]
    device, dtype = rpy.device, rpy.dtype
    elapsed = torch.arange(1, horizon + 1, device=device, dtype=dtype) * timestep

    desired = torch.zeros(num_robots, horizon, model.state_dim, device=device, dtype=dtype)
    desired[..., 2] = rpy[:, 2:3] + elapsed * desired_yaw_rate.unsqueeze(-1)
    desired[..., 3] = position[:, 0:1] + elapsed * desired_velocity[:, 0:1]
    desired[..., 4] = position[:, 1:2] + elapsed * desired_velocity[:, 1:2]
    desired[..., 5] = model.spec.nominal_height
    desired[..., 8] = desired_yaw_rate.unsqueeze(-1)
    desired[..., 9] = desired_velocity[:, 0:1]
    desired[..., 10] = desired_velocity[:, 1:2]
    desired[..., 12] = -GRAVITY
    return desired


def _state_matrix(
    rpy: torch.Tensor, ground_normal: torch.Tensor | None, states: int
) -> torch.Tensor:
    """(N, 13, 13) continuous-time A.

    Angular velocity drives the roll, pitch and yaw rates through a transform that is
    singular at a vertical pitch; linear velocity drives position; and gravity enters
    the linear acceleration along the ground normal.
    """
    num_robots = rpy.shape[0]
    device, dtype = rpy.device, rpy.dtype
    _, pitch, yaw = rpy.unbind(-1)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    cos_pitch, tan_pitch = torch.cos(pitch), torch.tan(pitch)

    matrix = torch.zeros(num_robots, states, states, device=device, dtype=dtype)
    matrix[:, 0, 6] = cos_yaw / cos_pitch
    matrix[:, 0, 7] = sin_yaw / cos_pitch
    matrix[:, 1, 6] = -sin_yaw
    matrix[:, 1, 7] = cos_yaw
    matrix[:, 2, 6] = cos_yaw * tan_pitch
    matrix[:, 2, 7] = sin_yaw * tan_pitch
    matrix[:, 2, 8] = 1.0
    matrix[:, 3, 9] = 1.0
    matrix[:, 4, 10] = 1.0
    matrix[:, 5, 11] = 1.0
    if ground_normal is None:
        matrix[:, 11, 12] = 1.0
    else:
        matrix[:, 9:12, 12] = ground_normal
    return matrix


def _input_matrix(
    model: SrbdModel, rpy: torch.Tensor, foot_positions: torch.Tensor
) -> torch.Tensor:
    """(N, 13, 3L) continuous-time B: how each foot's force moves the body.

    A force at a foot accelerates the centre of mass by f/m and torques the body by
    r x f through the inertia, both in the gravity-aligned frame.
    """
    num_robots, legs = foot_positions.shape[:2]
    device, dtype = rpy.device, rpy.dtype

    inverse_inertia = torch.diag(
        torch.tensor([1.0 / value for value in model.inertia], device=device, dtype=dtype)
    )
    # the inertia and the foot positions are carried into the yaw frame by rotations
    # built from the same angles in opposite orders; see `rotations.rpy_to_rotation_xyz`
    inertia_rotation = rpy_to_rotation_zyx(rpy)
    inverse_inertia_world = inertia_rotation @ inverse_inertia @ inertia_rotation.transpose(-1, -2)
    feet = torch.einsum("nij,nlj->nli", rpy_to_rotation_xyz(rpy), foot_positions)

    matrix = torch.zeros(num_robots, model.state_dim, 3 * legs, device=device, dtype=dtype)
    torque = inverse_inertia_world.unsqueeze(1) @ skew(feet)
    matrix[:, 6:9, :] = torque.permute(0, 2, 1, 3).reshape(num_robots, 3, 3 * legs)
    identity = torch.eye(3, device=device, dtype=dtype).view(1, 3, 1, 3)
    matrix[:, 9:12, :] = identity.expand(num_robots, 3, legs, 3).reshape(
        num_robots, 3, 3 * legs
    ) / model.mass
    return matrix


def _discretise(
    dynamics: torch.Tensor, actuation: torch.Tensor, timestep: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-order-hold discretisation of (A, B) by one matrix exponential.

    exp([[A dt, B dt], [0, 0]]) has the discrete A in its top-left block and the
    discrete B in its top-right one.
    """
    states = dynamics.shape[-1]
    inputs = actuation.shape[-1]
    augmented = torch.zeros(
        dynamics.shape[0],
        states + inputs,
        states + inputs,
        device=dynamics.device,
        dtype=dynamics.dtype,
    )
    augmented[:, :states, :states] = dynamics * timestep
    augmented[:, :states, states:] = actuation * timestep
    exponential = torch.linalg.matrix_exp(augmented)
    return exponential[:, :states, :states], exponential[:, :states, states:]


def _forced_response(anb: torch.Tensor, horizon: int) -> torch.Tensor:
    """(N, H * 13, H * 3L) block-lower-triangular map from forces to predicted states.

    Block (i, j) is `anb[i - j]`, so a force applied at step j reaches step i through
    i - j applications of the dynamics. The physically correct power is i - j - 1, since
    a force applied at step j only acts over the interval that ends at step j + 1;
    upstream uses i - j (`power = i - j` in CalculateQpMats) and builds its cost matrix
    to match, so the two are at least consistent with each other. Reproduced
    deliberately; see the module note.
    """
    num_robots, _, states, inputs = anb.shape
    steps = torch.arange(horizon, device=anb.device)
    shift = steps.unsqueeze(-1) - steps.unsqueeze(0)
    causal = (shift >= 0).to(anb.dtype)
    blocks = anb[:, shift.clamp_min(0)] * causal.view(1, horizon, horizon, 1, 1)
    return blocks.permute(0, 1, 3, 2, 4).reshape(num_robots, horizon * states, horizon * inputs)


def _constraints(
    model: SrbdModel, contact: torch.Tensor, horizon: int, legs: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The friction pyramid and the normal-force bounds, per (step, leg) force triple.

    A foot scheduled off the ground gets zero for both bounds, which pins all three of
    its force components: the pyramid rows then force the tangential components to zero
    as well. The CPU solver drops those variables from the problem instead, which is the
    same optimisation problem solved a different way.
    """
    num_robots = contact.shape[0]
    device, dtype = contact.device, contact.dtype
    friction = torch.full((num_robots,), model.friction, device=device, dtype=dtype)
    one = torch.ones_like(friction)
    zero = torch.zeros_like(friction)
    rows = (
        (-one, zero, friction),
        (one, zero, friction),
        (zero, -one, friction),
        (zero, one, friction),
        (zero, zero, one),
    )
    cone = torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)
    blocks = horizon * legs
    cone = cone.unsqueeze(1).expand(num_robots, blocks, CONE_ROWS, 3)

    normal_min = model.weight * model.force_scale[0] * contact
    normal_max = model.weight * model.force_scale[1] * contact
    tangential_max = (model.friction + 1.0) * normal_max

    lower = torch.zeros(num_robots, horizon, legs, CONE_ROWS, device=device, dtype=dtype)
    lower[..., 4] = normal_min
    upper = tangential_max.unsqueeze(-1).expand(num_robots, horizon, legs, CONE_ROWS).clone()
    upper[..., 4] = normal_max
    return cone, lower.reshape(num_robots, blocks, CONE_ROWS), upper.reshape(
        num_robots, blocks, CONE_ROWS
    )
