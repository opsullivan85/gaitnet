"""The batched controller's pieces, checked against their own definitions.

Parity with the CPU controller it replaces is `test_control_parity.py`; this file checks
the things that have to hold whether or not that one can run.
"""

import math

import pytest
import torch

from gaitnet_core.control import srbd
from gaitnet_core.control.admm import AdmmSettings, solve
from gaitnet_core.control.controller import BatchedMpcConfig, BatchedMpcController
from gaitnet_core.control.gait import GaitSchedule
from gaitnet_core.control.kinematics import foot_position_and_jacobian, leg_offsets
from gaitnet_core.control.model import GO1_SRBD
from gaitnet_core.control.mpc import MpcSettings
from gaitnet_core.control.rotations import (
    quat_to_rotation,
    quat_to_rpy,
    rotation_to_rpy,
    rpy_to_rotation_zyx,
    skew,
)
from gaitnet_core.control.swing import swing_trajectory
from gaitnet_core.interfaces import FootstepCommand

DT = 0.004


def crouch(height: float) -> tuple[float, float, float]:
    """Joint angles putting the foot `height` straight below its hip.

    The thigh and calf are the same length, so a knee folded by twice the thigh's pitch
    keeps the foot under the hip and the two links form an isosceles triangle.
    """
    knee = math.acos(height / (2 * GO1_SRBD.spec.thigh_length))
    return (0.0, knee, -2 * knee)


STANCE = crouch(GO1_SRBD.spec.nominal_height)
"""All four feet under the robot, with the body at exactly the height the MPC tracks."""


def make_controller(n: int = 4, dtype: torch.dtype = torch.float64, **mpc) -> BatchedMpcController:
    settings = MpcSettings(**mpc) if mpc else MpcSettings()
    return BatchedMpcController(BatchedMpcConfig(mpc=settings, dtype=dtype), n, DT, "cpu")


def standing(n: int) -> tuple[torch.Tensor, ...]:
    """Inputs for a robot standing level and still."""
    joint_pos = torch.tensor(STANCE, dtype=torch.float64).repeat(n, 4)
    pose = torch.zeros(n, 7, dtype=torch.float64)
    pose[:, 2], pose[:, 6] = 0.3, 1.0
    return (
        joint_pos,
        torch.zeros(n, 12, dtype=torch.float64),
        pose,
        torch.zeros(n, 6, dtype=torch.float64),
        torch.zeros(n, 3, dtype=torch.float64),
    )


def random_quaternion(n: int, generator: torch.Generator) -> torch.Tensor:
    quat = torch.randn(n, 4, generator=generator, dtype=torch.float64)
    return quat / quat.norm(dim=-1, keepdim=True)


# --- rotations -------------------------------------------------------------------


def test_quaternion_rotation_is_a_frame_rotation():
    """`quat_to_rotation` maps world components into the base, so it is the transpose of
    the rotation that turns a base vector into a world one, and it is orthonormal."""
    generator = torch.Generator().manual_seed(0)
    quat = random_quaternion(16, generator)
    rotation = quat_to_rotation(quat)
    identity = torch.eye(3, dtype=torch.float64).expand(16, 3, 3)
    assert torch.allclose(rotation @ rotation.transpose(-1, -2), identity, atol=1e-12)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(16, dtype=torch.float64))


def test_rpy_agrees_between_the_quaternion_and_the_matrix():
    """The two ways the controller reads roll, pitch and yaw - off a quaternion for the
    world orientation, off a matrix for the tilt out of the yaw frame - are one map."""
    generator = torch.Generator().manual_seed(1)
    quat = random_quaternion(64, generator)
    assert torch.allclose(quat_to_rpy(quat), rotation_to_rpy(quat_to_rotation(quat)), atol=1e-9)


def test_rpy_to_rotation_inverts_rotation_to_rpy():
    """Angles taken off a rotation rebuild it, which pins the composition order: the
    active z-y-x rotation is the transpose of the frame rotation the angles came from."""
    generator = torch.Generator().manual_seed(2)
    # stay well away from a vertical pitch, where the decomposition is singular
    rpy = torch.rand(32, 3, generator=generator, dtype=torch.float64) - 0.5
    rebuilt = rotation_to_rpy(rpy_to_rotation_zyx(rpy).transpose(-1, -2))
    assert torch.allclose(rebuilt, rpy, atol=1e-9)


def test_skew_is_the_cross_product():
    generator = torch.Generator().manual_seed(3)
    left = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    right = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    product = torch.einsum("nij,nj->ni", skew(left), right)
    assert torch.allclose(product, torch.cross(left, right, dim=-1), atol=1e-12)


# --- kinematics ------------------------------------------------------------------


def test_jacobian_is_the_derivative_of_the_foot_position():
    """Central differences of the closed-form foot position must reproduce the
    closed-form Jacobian, which is the only thing tying the two expressions together."""
    offsets = leg_offsets(GO1_SRBD, "cpu", torch.float64)
    generator = torch.Generator().manual_seed(4)
    joint_pos = torch.randn(6, 4, 3, generator=generator, dtype=torch.float64) * 0.5
    joint_pos[..., 1] += 0.8
    joint_pos[..., 2] -= 1.6
    _, jacobian = foot_position_and_jacobian(joint_pos, offsets)

    step = 1e-6
    for joint in range(3):
        delta = torch.zeros_like(joint_pos)
        delta[..., joint] = step
        ahead, _ = foot_position_and_jacobian(joint_pos + delta, offsets)
        behind, _ = foot_position_and_jacobian(joint_pos - delta, offsets)
        assert torch.allclose((ahead - behind) / (2 * step), jacobian[..., joint], atol=1e-7)


def test_feet_are_under_the_robot_and_mirrored():
    """The nominal crouch puts every foot below its hip, and the left and right legs are
    mirror images, which is what fixes the sign convention on the abduction offset."""
    offsets = leg_offsets(GO1_SRBD, "cpu", torch.float64)
    position, _ = foot_position_and_jacobian(
        torch.tensor(STANCE, dtype=torch.float64).expand(1, 4, 3), offsets
    )
    assert (position[..., 2] < -0.2).all()
    assert torch.allclose(position[0, 0, 1], -position[0, 1, 1])
    assert torch.allclose(position[0, 2, 1], -position[0, 3, 1])


# --- gait schedule ---------------------------------------------------------------


def test_a_leg_swings_for_exactly_its_commanded_duration():
    """Phase runs from 0 up towards 1 over the swing and returns to 0 at touchdown, and
    the leg reads as in contact then and not before."""
    gait = GaitSchedule(1, 4, DT, "cpu", torch.float64)
    gait.begin_step()
    gait.initiate(
        torch.ones(1, dtype=torch.bool), torch.tensor([2]), torch.tensor([0.2], dtype=torch.float64)
    )
    phases = []
    for _ in range(60):
        gait.begin_step()
        phases.append(float(gait.swing_phase()[0, 2]))
        assert bool(gait.in_contact()[0, 2]) == (float(gait.time) >= 0.2)
        gait.end_step()
    swinging = [p for p in phases if p > 0]
    # 0.2 s at a 0.004 s control period is fifty steps, less the step the swing is
    # commanded on, where no time has passed yet, and the step it touches down on
    assert len(swinging) == 49
    assert swinging == sorted(swinging)
    assert phases[50:] == [0.0] * 10
    assert (gait.in_contact()[0, [0, 1, 3]]).all()


def test_timing_columns_match_the_planner_contract():
    """Swing phase, remaining swing time and time since touchdown, with each zero in the
    phase it does not describe, as `RobotState.gait_timing` promises."""
    gait = GaitSchedule(1, 4, DT, "cpu", torch.float64)
    gait.begin_step()
    gait.initiate(
        torch.ones(1, dtype=torch.bool), torch.tensor([0]), torch.tensor([0.2], dtype=torch.float64)
    )
    for _ in range(25):
        gait.begin_step()
        gait.end_step()
    gait.begin_step()
    phase, remaining, stance = gait.timing()[0, 0]
    assert phase == pytest.approx(0.5, abs=1e-9)
    assert remaining == pytest.approx(0.1, abs=1e-9)
    assert stance == 0.0
    for _ in range(50):
        gait.begin_step()
        gait.end_step()
    gait.begin_step()
    phase, remaining, stance = gait.timing()[0, 0]
    assert (phase, remaining) == (0.0, 0.0)
    assert stance == pytest.approx(0.1, abs=1e-9)


def test_the_contact_table_keeps_stance_legs_down():
    """Nothing has decided when a stance foot next lifts, so the horizon assumes it does
    not; a swinging foot comes back down at its scheduled touchdown and stays."""
    gait = GaitSchedule(1, 4, DT, "cpu", torch.float64)
    gait.begin_step()
    gait.initiate(
        torch.ones(1, dtype=torch.bool),
        torch.tensor([1]),
        torch.tensor([0.12], dtype=torch.float64),
    )
    gait.begin_step()
    table = gait.contact_table(horizon=10, timestep=0.02)
    assert table.shape == (1, 10, 4)
    assert (table[0, :, [0, 2, 3]] == 1).all()
    assert (table[0, :6, 1] == 0).all()  # 0.12 s is six 0.02 s horizon steps
    assert (table[0, 6:, 1] == 1).all()


def test_reset_only_touches_the_named_robots():
    gait = GaitSchedule(3, 4, DT, "cpu", torch.float64)
    gait.begin_step()
    gait.initiate(
        torch.ones(3, dtype=torch.bool),
        torch.tensor([0, 1, 2]),
        torch.full((3,), 0.2, dtype=torch.float64),
    )
    for _ in range(10):
        gait.begin_step()
        gait.end_step()
    gait.reset(torch.tensor([1]))
    gait.begin_step()
    assert gait.in_contact()[1].all()
    assert not gait.in_contact()[0, 0]
    assert not gait.in_contact()[2, 2]


# --- swing trajectory ------------------------------------------------------------


def test_the_swing_arc_starts_stops_and_clears_the_higher_end():
    """The Bezier is flat at both ends, so the foot lifts off and touches down with no
    velocity, and its apex clears whichever end is higher."""
    start = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64)
    end = torch.tensor([[[0.2, 0.0, 0.08]]], dtype=torch.float64)
    duration = torch.full((1, 1), 0.2, dtype=torch.float64)
    clearance = 0.09

    begin, begin_rate = swing_trajectory(
        start, end, clearance, torch.zeros(1, 1, dtype=torch.float64), duration
    )
    finish, finish_rate = swing_trajectory(
        start, end, clearance, torch.ones(1, 1, dtype=torch.float64), duration
    )
    assert torch.allclose(begin, start, atol=1e-12)
    assert torch.allclose(finish, end, atol=1e-12)
    assert torch.allclose(begin_rate, torch.zeros(1, 1, 3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(finish_rate, torch.zeros(1, 1, 3, dtype=torch.float64), atol=1e-12)

    heights = [
        float(
            swing_trajectory(
                start, end, clearance, torch.full((1, 1), p, dtype=torch.float64), duration
            )[0][0, 0, 2]
        )
        for p in torch.linspace(0, 1, 51)
    ]
    assert max(heights) == pytest.approx(float(end[0, 0, 2]) + clearance, abs=1e-9)


# --- the QP solver ---------------------------------------------------------------


def optimality(problem: srbd.SrbdQp, result) -> tuple[float, float, float]:
    """How far the solution is from satisfying the KKT conditions of a convex QP.

    Returns the worst primal infeasibility, the worst stationarity violation, and the
    worst complementary-slackness violation. All three near zero is a certificate that
    this is the minimiser, with no reference solver needed.
    """
    x = result.primal
    y = result.dual
    constrained = torch.einsum("nbmk,nbk->nbm", problem.cone, x.view(x.shape[0], -1, 3))
    primal = torch.maximum(problem.lower - constrained, constrained - problem.upper).clamp_min(0)
    stationarity = (
        torch.einsum("nij,nj->ni", problem.hessian, x)
        + problem.gradient
        + torch.einsum("nbmk,nbm->nbk", problem.cone, y).reshape(x.shape)
    )
    # a positive dual pushes against the upper bound, a negative one against the lower,
    # and a constraint strictly inside its bounds carries no dual at all
    slack = torch.minimum(
        (constrained - problem.lower).abs(), (problem.upper - constrained).abs()
    )
    complementarity = y.abs() * slack
    return (
        float(primal.max()),
        float(stationarity.abs().max()),
        float(complementarity.max()),
    )


def make_problem(contact: list[float], dtype: torch.dtype = torch.float64) -> srbd.SrbdQp:
    feet = torch.tensor(
        [
            [0.186, 0.130, -0.296],
            [0.191, -0.129, -0.296],
            [-0.191, 0.129, -0.296],
            [-0.185, -0.129, -0.296],
        ],
        dtype=dtype,
    ).view(1, 4, 3)
    return srbd.build(
        GO1_SRBD,
        horizon=10,
        timestep=0.02,
        rpy=torch.tensor([[0.02, 0.04, 0.0]], dtype=dtype),
        position=torch.tensor([[0.0, 0.0, 0.306]], dtype=dtype),
        linear_velocity=torch.tensor([[0.3, 0.02, 0.01]], dtype=dtype),
        angular_velocity=torch.tensor([[0.01, 0.02, 0.05]], dtype=dtype),
        foot_positions=feet,
        contact=torch.tensor(contact, dtype=dtype).repeat(10).view(1, 10, 4),
        desired_velocity=torch.tensor([[0.4, 0.0]], dtype=dtype),
        desired_yaw_rate=torch.tensor([0.1], dtype=dtype),
        weights=torch.tensor(GO1_SRBD.weights, dtype=dtype).view(1, 13),
    )


@pytest.mark.parametrize(
    "contact", [[1, 1, 1, 1], [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1]]
)
def test_the_solver_reaches_the_minimiser(contact):
    """Whatever the contact pattern, a long solve satisfies the KKT conditions. The
    tolerances are in newtons, against contact forces of order a hundred."""
    problem = make_problem(contact)
    result = solve(
        problem.hessian,
        problem.gradient,
        problem.cone,
        problem.lower,
        problem.upper,
        torch.zeros(1, problem.num_variables, dtype=torch.float64),
        torch.zeros_like(problem.lower),
        AdmmSettings(iterations=400),
    )
    primal, stationarity, complementarity = optimality(problem, result)
    assert primal < 1e-4
    assert stationarity < 1e-4
    assert complementarity < 1e-3


def test_the_default_budget_is_close_to_the_minimiser():
    """The shipped iteration count has to be good enough on its own, since a rollout
    never gets to look at the residual and try again."""
    problem = make_problem([1, 1, 0, 1])
    settled = solve(
        problem.hessian,
        problem.gradient,
        problem.cone,
        problem.lower,
        problem.upper,
        torch.zeros(1, problem.num_variables, dtype=torch.float64),
        torch.zeros_like(problem.lower),
        AdmmSettings(iterations=400),
    )
    default = solve(
        problem.hessian,
        problem.gradient,
        problem.cone,
        problem.lower,
        problem.upper,
        torch.zeros(1, problem.num_variables, dtype=torch.float64),
        torch.zeros_like(problem.lower),
        AdmmSettings(),
    )
    first_step = slice(0, 12)
    assert (settled.primal[0, first_step] - default.primal[0, first_step]).abs().max() < 0.2


def test_a_foot_off_the_ground_carries_no_force():
    """A swing leg's bounds close to zero, which has to pin all three of its components,
    not just the normal one - the pyramid rows do the other two."""
    problem = make_problem([1, 0, 1, 0])
    result = solve(
        problem.hessian,
        problem.gradient,
        problem.cone,
        problem.lower,
        problem.upper,
        torch.zeros(1, problem.num_variables, dtype=torch.float64),
        torch.zeros_like(problem.lower),
        AdmmSettings(iterations=200),
    )
    forces = result.primal.view(1, 10, 4, 3)
    assert forces[0, :, [1, 3]].abs().max() < 1e-6
    assert forces[0, 0, [0, 2], 2].min() > 1.0


def test_forces_stay_inside_the_friction_cone():
    """The pyramid is what stops the MPC asking for a force the ground cannot give."""
    problem = make_problem([1, 1, 1, 1])
    result = solve(
        problem.hessian,
        problem.gradient,
        problem.cone,
        problem.lower,
        problem.upper,
        torch.zeros(1, problem.num_variables, dtype=torch.float64),
        torch.zeros_like(problem.lower),
        AdmmSettings(iterations=200),
    )
    forces = result.primal.view(1, 10, 4, 3)
    tangential = forces[..., :2].abs().amax(dim=-1)
    assert (tangential <= GO1_SRBD.friction * forces[..., 2] + 1e-4).all()


def test_the_solver_is_independent_across_the_batch():
    """Robots share a factorisation call but not a problem; solving two together must
    give what solving each alone gives."""
    first = make_problem([1, 1, 1, 1])
    second = make_problem([0, 1, 1, 0])
    together = srbd.SrbdQp(
        hessian=torch.cat([first.hessian, second.hessian]),
        gradient=torch.cat([first.gradient, second.gradient]),
        cone=torch.cat([first.cone, second.cone]),
        lower=torch.cat([first.lower, second.lower]),
        upper=torch.cat([first.upper, second.upper]),
    )
    settings = AdmmSettings(iterations=200)

    def run(problem, rows):
        return solve(
            problem.hessian,
            problem.gradient,
            problem.cone,
            problem.lower,
            problem.upper,
            torch.zeros(rows, problem.num_variables, dtype=torch.float64),
            torch.zeros_like(problem.lower),
            settings,
        ).primal

    batched = run(together, 2)
    assert torch.allclose(batched[0], run(first, 1)[0], atol=1e-9)
    assert torch.allclose(batched[1], run(second, 1)[0], atol=1e-9)


# --- the controller --------------------------------------------------------------


def test_a_standing_robot_pushes_down_evenly():
    """Four feet down, level, at the height the MPC tracks: each leg carries the same
    share of the load and pushes down on the ground.

    The total comes out about a fifth above the robot's weight. That is the CPU
    controller's own steady state, to six figures - see `test_control_parity` - and not
    a porting error: it follows from the two upstream quirks reproduced in `srbd.py`,
    which leave the last horizon step mispredicted. The number is pinned here so that
    correcting either quirk shows up as a failing test rather than as a body height
    that quietly sits somewhere new.
    """
    controller = make_controller(1)
    for _ in range(MpcSettings().control_steps_per_update):
        controller.compute_torques(*standing(1))
    forces = controller._feedforward
    assert (forces[..., 2] < 0).all()  # the leg pushes down on the ground
    per_leg = -forces[0, :, 2]
    assert torch.allclose(per_leg, per_leg[0].expand(4), rtol=1e-6)
    assert float(per_leg.sum()) / GO1_SRBD.weight == pytest.approx(1.222, rel=1e-3)


def test_a_robot_standing_too_tall_unloads_its_legs():
    """The height is read off the feet, not the base pose, and it is tracked hard
    (weight 50). Crouching further than the nominal height has to push harder and
    standing taller has to push less, or the body height would drift."""
    nominal, tall, low = (
        make_controller(1) for _ in (GO1_SRBD.spec.nominal_height, 0.30, 0.22)
    )
    totals = []
    for controller, height in ((nominal, GO1_SRBD.spec.nominal_height), (tall, 0.30), (low, 0.22)):
        joint_pos = torch.tensor(crouch(height), dtype=torch.float64).repeat(1, 4)
        inputs = list(standing(1))
        inputs[0] = joint_pos
        for _ in range(MpcSettings().control_steps_per_update):
            controller.compute_torques(*inputs)
        totals.append(float(-controller._feedforward[..., 2].sum()))
    at_nominal, standing_tall, crouched = totals
    assert standing_tall < at_nominal < crouched


def test_the_batch_does_not_mix_robots():
    """Eight robots in one batch must each get what they would have got alone, including
    after one of them is reset mid-episode."""
    single = make_controller(1)
    batch = make_controller(8)
    step = FootstepCommand.none(8)
    step.active[3] = True
    step.leg[3] = 1
    step.target[3] = torch.tensor([0.1, -0.1, -0.26])
    step.duration[3] = 0.2

    for index in range(12):
        if index == 2:
            batch.command_footsteps(step)
        alone = single.compute_torques(*standing(1))
        torques = batch.compute_torques(*standing(8))
        assert torch.allclose(torques[0], torques[5], atol=1e-12)
        assert torch.allclose(torques[0], alone[0], atol=1e-9)
    assert not torch.allclose(torques[3], torques[0])

    batch.reset(torch.tensor([3]))
    # a reset drops the robot's feed-forward forces, so it stands on its leg PD alone
    # until the next scheduled MPC update refills them
    for _ in range(MpcSettings().control_steps_per_update):
        torques = batch.compute_torques(*standing(8))
    # not exactly equal: the reset also dropped this robot's warm start, and at a fixed
    # iteration budget where the solver starts from changes where it stops
    assert torch.allclose(torques[3], torques[0], rtol=1e-3, atol=1e-6)


def test_reset_clears_the_schedule_and_the_warm_start():
    controller = make_controller(2)
    step = FootstepCommand.none(2)
    step.active[:] = True
    step.leg[:] = 0
    step.target[:] = torch.tensor([0.1, 0.1, -0.26])
    step.duration[:] = 0.2
    controller.command_footsteps(step)
    for _ in range(6):
        controller.compute_torques(*standing(2))
    assert not controller.gait_timing()[0, 0, 0] == 0.0

    controller.reset(torch.tensor([0]))
    assert float(controller.gait_timing()[0].abs().max()) == 0.0
    assert float(controller._mpc._primal[0].abs().max()) == 0.0
    assert float(controller.gait_timing()[1, 0, 0]) > 0.0


def test_a_commanded_foothold_is_raised_by_the_foot_radius():
    """Footstep targets name the terrain surface; the controller drives the centre of a
    spherical foot, which sits one radius above it."""
    controller = make_controller(1)
    step = FootstepCommand.none(1)
    step.active[:] = True
    step.leg[:] = 0
    step.target[:] = torch.tensor([0.12, 0.09, -0.3])
    step.duration[:] = 0.2
    controller.command_footsteps(step)
    assert float(controller._footstep_offsets[0, 0, 2]) == pytest.approx(
        -0.3 + controller.cfg.foot_radius
    )


def test_torques_are_finite_through_a_full_swing():
    """A swing that runs to touchdown must not produce a NaN anywhere: the trajectory
    divides by the swing duration and the state matrix divides by the pitch's cosine."""
    controller = make_controller(2, dtype=torch.float32)
    step = FootstepCommand.none(2)
    step.active[:] = True
    step.leg[:] = torch.tensor([0, 3])
    step.target[:] = torch.tensor([0.12, 0.09, -0.26])
    step.duration[:] = 0.15
    controller.command_footsteps(step)
    for _ in range(80):
        torques = controller.compute_torques(*[a.float() for a in standing(2)])
        assert torch.isfinite(torques).all()
    assert torch.isfinite(controller._feedforward).all()


# --- pinned footholds -------------------------------------------------------------

VELOCITY = torch.tensor([0.3, -0.1, 0.02], dtype=torch.float64)
TARGET = (0.12, 0.09, -0.3)


def quaternion(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """(4,) xyzw for z-y'-x" intrinsic angles."""
    (cr, sr), (cp, sp), (cy, sy) = [(math.cos(a / 2), math.sin(a / 2)) for a in (roll, pitch, yaw)]
    return torch.tensor(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=torch.float64,
    )


def travelling(step: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """A body moving at a constant world velocity, as the simulator integrates it, while it
    turns, pitches and rolls. Returns the pose, the twist and the yaw."""
    yaw = 0.4 + 0.01 * step
    pose = torch.zeros(1, 7, dtype=torch.float64)
    pose[0, :3] = torch.tensor([1.0, 2.0, 0.3], dtype=torch.float64) + VELOCITY * DT * step
    pose[0, 3:] = quaternion(0.03 * math.sin(step / 7), 0.05 - 0.001 * step, yaw)
    twist = torch.zeros(1, 6, dtype=torch.float64)
    twist[0, :3] = VELOCITY
    return pose, twist, yaw


def run_footstep(controller: BatchedMpcController, steps: int) -> list[torch.Tensor]:
    """Command leg 0 to `TARGET` after one step of `travelling`, then keep going. Returns,
    per step from the one after the command, where the swing is aimed in the world, and
    the pose."""
    joint_pos, joint_vel, _, _, command = standing(1)
    controller.compute_torques(joint_pos, joint_vel, *travelling(0)[:2], command)
    step = FootstepCommand.none(1)
    step.active[:] = True
    step.leg[:] = 0
    step.target[:] = torch.tensor(TARGET)
    step.duration[:] = 0.2
    controller.command_footsteps(step)
    aimed = []
    for k in range(1, steps + 1):
        pose, twist, _ = travelling(k)
        controller.compute_torques(joint_pos, joint_vel, pose, twist, command)
        world_from_base = quat_to_rotation(pose[:, 3:7])[0].T
        from_base = controller._swing_end[0, 0] - controller._position()[0]
        aimed.append(pose[0, :3] + world_from_base @ from_base)
    return aimed


def test_a_pinned_foothold_stays_put_in_the_world():
    """The target is fixed where the planner measured it, from the hip and heading of the
    step after the command, and the swing keeps aiming there however the body moves."""
    controller = make_controller(1)
    aimed = run_footstep(controller, steps=40)

    pose, _, yaw = travelling(1)
    hip = pose[0, :3] + quat_to_rotation(pose[:, 3:7])[0].T @ controller._hip_offsets[0]
    heading = torch.tensor(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    target = torch.tensor(TARGET, dtype=torch.float64) + torch.tensor([0.0, 0.0, controller.cfg.foot_radius])
    expected = hip + heading @ target
    for point in aimed:
        assert torch.allclose(point, expected, atol=1e-9)


def test_a_new_command_or_a_reset_releases_the_pin():
    controller = make_controller(1)
    run_footstep(controller, steps=2)
    assert bool(controller._pinned[0, 0])
    again = FootstepCommand.none(1)
    again.active[:] = True
    again.leg[:] = 0
    again.target[:] = torch.tensor(TARGET)
    again.duration[:] = 0.2
    controller.command_footsteps(again)
    # re-pinned on the next step, from that step's state
    assert not bool(controller._pinned[0, 0]) and bool(controller._pin_pending[0, 0])
    assert not controller._pinned[0, 1:].any() and not controller._pin_pending[0, 1:].any()
    controller.reset(torch.tensor([0]))
    assert not controller._pinned.any() and not controller._pin_pending.any()
    assert float(controller._odometry.abs().max()) == 0.0
