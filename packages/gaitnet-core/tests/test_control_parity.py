"""The batched controller against the CPU one it stands in for.

`gaitnet_core.control` exists to be the same controller as `gaitnet_mpc`, run for a
whole batch at once. Everything else about it - the horizon, the gains, the frames, even
two upstream results that look like mistakes - is chosen to keep that true, because the
policy bundles and evaluation baselines in this repo were all produced against the CPU
controller. These tests are what makes "the same" a claim and not a hope.

They need `gaitnet-mpc`, which has a compiled extension, so they skip where it is not
installed. CI covers core without it; run them wherever the extension builds (the sim
image always has it).
"""

import math

import numpy as np
import pytest
import torch

from gaitnet_core.control import srbd
from gaitnet_core.control.admm import AdmmSettings, solve
from gaitnet_core.control.controller import BatchedMpcConfig, BatchedMpcController
from gaitnet_core.control.model import GO1_SRBD
from gaitnet_core.control.mpc import MpcSettings
from gaitnet_core.interfaces import FootstepCommand

cpu_controller = pytest.importorskip(
    "gaitnet_mpc.controller", reason="gaitnet-mpc and its compiled extension are not installed"
)
cpp = pytest.importorskip("gaitnet_mpc._mpc_osqp")

DT = 0.004
UPDATE = 5
HORIZON = 10
STEPS = 40
FOOT_RADIUS = 0.02
DOUBLE = torch.float64


def scripted_state(step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A body that rolls, pitches and yaws while moving, and legs that keep moving too.

    Nothing here is a trajectory the robot would actually follow; the point is to sweep
    both controllers through states where the frames, the rate limit on the command and
    the pitch-dependent terms of the dynamics all matter, rather than through a
    symmetric stance where a sign error would cancel.
    """
    time = step * DT
    roll, pitch, yaw = 0.05 * math.sin(3 * time), 0.04 * math.cos(2 * time), 0.02 * time
    half = [(math.cos(angle / 2), math.sin(angle / 2)) for angle in (roll, pitch, yaw)]
    (cr, sr), (cp, sp), (cy, sy) = half
    pose = np.array(
        [
            0.1 * time,
            0.0,
            0.3,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )
    twist = np.array([0.3, 0.02, 0.0, 0.01, 0.02, 0.05])
    knee = math.acos(GO1_SRBD.spec.nominal_height / (2 * GO1_SRBD.spec.thigh_length))
    joints = np.arange(12).reshape(4, 3)
    joint_pos = np.tile([0.0, knee, -2 * knee], (4, 1)) + 0.01 * np.sin(step + joints)
    joint_vel = 0.02 * np.cos(step + joints)
    return joint_pos, joint_vel, pose, twist


FOOTSTEPS = {3: 0, 11: 3, 17: 1, 28: 0}
"""Control step to the leg that starts swinging on it."""


def foothold(leg: int) -> np.ndarray:
    """A reachable target in the leg's hip frame, on the terrain surface."""
    return np.array([0.12, 0.09 * (1 if leg in (0, 2) else -1), -0.28])


def run_both(
    iterations: int, dtype: torch.dtype
) -> tuple[list[np.ndarray], list[np.ndarray], object, BatchedMpcController]:
    """Drive both controllers through the same states, returning their torques."""
    cpu = cpu_controller.MpcFootstepController(dt=DT, iterations_between_mpc=UPDATE)
    config = BatchedMpcConfig(
        mpc=MpcSettings(
            horizon=HORIZON,
            control_steps_per_update=UPDATE,
            admm=AdmmSettings(iterations=iterations),
        ),
        foot_radius=FOOT_RADIUS,
        dtype=dtype,
    )
    batched = BatchedMpcController(config, 1, DT, "cpu")

    cpu_torques, batched_torques = [], []
    for step in range(STEPS):
        joint_pos, joint_vel, pose, twist = scripted_state(step)
        if step in FOOTSTEPS:
            leg = FOOTSTEPS[step]
            target = foothold(leg)
            # the CPU controller is handed the foot centre, the batched one the terrain
            # surface, which it raises by the foot radius itself
            cpu.initiate_footstep(leg=leg, location_hip=target, duration=0.2)
            command = FootstepCommand.none(1)
            command.active[:] = True
            command.leg[:] = leg
            command.target[:] = torch.tensor(target - np.array([0.0, 0.0, FOOT_RADIUS]))
            command.duration[:] = 0.2
            batched.command_footsteps(command)

        cpu_torques.append(
            cpu.get_torques(
                joint_states=np.stack([joint_pos, joint_vel], axis=-1),
                body_state=np.concatenate([pose, twist]),
                command=np.array([0.4, 0.0, 0.1]),
            ).reshape(-1)
        )
        batched_torques.append(
            batched.compute_torques(
                torch.tensor(joint_pos.reshape(1, 12), dtype=dtype),
                torch.tensor(joint_vel.reshape(1, 12), dtype=dtype),
                torch.tensor(pose, dtype=dtype).view(1, 7),
                torch.tensor(twist, dtype=dtype).view(1, 6),
                torch.tensor([0.4, 0.0, 0.1], dtype=dtype).view(1, 3),
            )
            .numpy()
            .reshape(-1)
        )
    return cpu_torques, batched_torques, cpu, batched


def test_torques_match_the_cpu_controller():
    """With the solver run to convergence, the two controllers agree to a hundredth of
    a newton metre over forty steps covering three swings.

    What is left is the CPU path's precision, not the port: `orientation_tools` computes
    in float16, so every rotation matrix over there is good to about three decimals.
    """
    cpu_torques, batched_torques, _, _ = run_both(iterations=400, dtype=DOUBLE)
    worst = max(np.abs(a - b).max() for a, b in zip(cpu_torques, batched_torques))
    peak = max(np.abs(a).max() for a in cpu_torques)
    assert peak > 5.0, "the scripted states should be loading the legs"
    assert worst < 0.02


def test_the_shipped_settings_stay_close():
    """The defaults trade solver iterations for throughput. This pins what that costs,
    so a change to the budget or the step-size schedule cannot quietly widen it."""
    cpu_torques, batched_torques, _, _ = run_both(
        iterations=AdmmSettings().iterations, dtype=torch.float32
    )
    worst = max(np.abs(a - b).max() for a, b in zip(cpu_torques, batched_torques))
    peak = max(np.abs(a).max() for a in cpu_torques)
    median = float(np.median([np.abs(a - b).max() for a, b in zip(cpu_torques, batched_torques)]))
    # the worst step is a transient right after a foot changes state, where the warm
    # start is least useful; the typical step is an order of magnitude closer
    assert worst / peak < 0.01
    assert median < 0.05


def test_the_gait_schedule_matches():
    """The planner reads the schedule, not the torques, so the timing the two
    controllers report has to agree exactly - it is an observation, and a bundle trained
    against one must see the same numbers from the other."""
    _, _, cpu, batched = run_both(iterations=400, dtype=DOUBLE)
    assert np.allclose(cpu.get_gait_timing(), batched.gait_timing()[0].numpy(), atol=1e-9)
    assert np.allclose(cpu.get_estimated_rpy(), batched.estimated_rpy()[0].numpy(), atol=1e-3)


@pytest.mark.parametrize(
    "contact", [[1, 1, 1, 1], [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1]]
)
def test_the_condensed_qp_matches_the_c_plus_plus_one(contact):
    """The same state into both formulations has to give the same contact forces.

    This is the sharper of the two comparisons: it takes the controller plumbing out and
    leaves only `srbd.build` against `ConvexMpc::ComputeContactForces`, so a wrong sign,
    a transposed rotation or a misread constraint bound shows up here as newtons rather
    than being absorbed by the leg PD.
    """
    rpy = np.array([0.0226, 0.0381, 0.0])
    position = np.array([0.0, 0.0, 0.3062])
    velocity = np.array([0.2999, 0.0193, 0.0110])
    omega = np.array([0.0082, 0.0211, 0.0499])
    feet = np.array(
        [
            [0.1861, 0.1296, -0.2958],
            [0.1905, -0.1295, -0.2961],
            [-0.1910, 0.1293, -0.2963],
            [-0.1849, -0.1290, -0.2958],
        ]
    )
    table = np.tile(np.asarray(contact, dtype=float), HORIZON)
    desired = np.array([0.4, 0.0])
    yaw_rate = 0.1

    solver = cpp.ConvexMpc(
        GO1_SRBD.mass,
        list(np.diag(GO1_SRBD.inertia).ravel()),
        GO1_SRBD.num_legs,
        HORIZON,
        DT * UPDATE,
        GO1_SRBD.force_regularization,
        cpp.QPOASES,
    )
    reference = np.array(
        solver.compute_contact_forces(
            list(GO1_SRBD.weights),
            list(position),
            list(velocity),
            list(rpy),
            [0.0, 0.0, 1.0],
            list(omega),
            list(table),
            list(feet.ravel()),
            [GO1_SRBD.friction] * GO1_SRBD.num_legs,
            [0.0, 0.0, GO1_SRBD.spec.nominal_height],
            [desired[0], desired[1], 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, yaw_rate],
        )
    )[:12]

    problem = srbd.build(
        GO1_SRBD,
        horizon=HORIZON,
        timestep=DT * UPDATE,
        rpy=torch.tensor(rpy, dtype=DOUBLE).view(1, 3),
        position=torch.tensor(position, dtype=DOUBLE).view(1, 3),
        linear_velocity=torch.tensor(velocity, dtype=DOUBLE).view(1, 3),
        angular_velocity=torch.tensor(omega, dtype=DOUBLE).view(1, 3),
        foot_positions=torch.tensor(feet, dtype=DOUBLE).view(1, 4, 3),
        contact=torch.tensor(table, dtype=DOUBLE).view(1, HORIZON, 4),
        desired_velocity=torch.tensor(desired, dtype=DOUBLE).view(1, 2),
        desired_yaw_rate=torch.tensor([yaw_rate], dtype=DOUBLE),
        weights=torch.tensor(GO1_SRBD.weights, dtype=DOUBLE).view(1, 13),
    )
    result = solve(
        problem.hessian,
        problem.gradient,
        problem.cone,
        problem.lower,
        problem.upper,
        torch.zeros(1, problem.num_variables, dtype=DOUBLE),
        torch.zeros_like(problem.lower),
        AdmmSettings(iterations=400),
    )
    # the solver's variables are the forces on the body; the controller applies their
    # negation, which is what the C++ returns
    mine = -result.primal[0, :12].numpy()
    assert np.abs(reference).max() > 10.0, "the reference should be carrying the robot"
    assert np.abs(mine - reference).max() < 0.05
