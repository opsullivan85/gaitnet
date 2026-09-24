"""The rigid-body model and the gains the batched controller runs on.

One frozen dataclass holding everything `gaitnet_mpc` spreads over `Quadruped`,
`Parameters` and the `ConvexMPCLocomotion` constructor, so the numbers a GPU rollout
uses are in one readable place and can be swapped per experiment. `GO1_SRBD` carries the
values the CPU controller uses today; changing any of them changes how the robot walks,
so treat it as part of the sim/real contract rather than as tuning knobs.

Kinematics (hip offsets, link lengths, nominal height) come from the shared
`RobotSpec`, which is already the single source of truth for them.
"""

from __future__ import annotations

from dataclasses import dataclass

from gaitnet_core.robot_spec import GO1, RobotSpec

GRAVITY = 9.8
"""Gravitational acceleration the MPC's state vector carries (m/s^2).

9.8, not 9.81: `kGravity` in `mpc_osqp.cc`. It enters the prediction as a constant
thirteenth state, so it has to match whatever the model was tuned against.
"""


@dataclass(frozen=True)
class SrbdModel:
    """Single-rigid-body dynamics, MPC tuning and leg gains for one robot.

    All of it is per robot type, none of it is per robot instance, so a batch of N
    robots shares one of these.
    """

    spec: RobotSpec
    """Kinematics and leg order."""
    mass: float
    """Trunk mass used by the single-rigid-body model (kg)."""
    inertia: tuple[float, float, float]
    """Diagonal of the trunk inertia in the base frame (kg m^2)."""
    friction: float
    """Friction coefficient the force cone is built from."""
    weights: tuple[float, ...]
    """(13,) state-tracking weights: roll, pitch, yaw, x, y, z, then the angular and
    linear velocities, then the gravity state (always weighted 0)."""
    force_regularization: float
    """Weight on the squared contact forces, `Parameters.cmpc_alpha`."""
    force_scale: tuple[float, float]
    """(min, max) normal force per foot, as a multiple of the robot's total weight."""
    max_linear_accel: float
    """Rate limit on the tracked x/y velocity command (m/s^2)."""
    swing_apex_fraction: float
    """Swing clearance above the higher end of the step, as a fraction of the nominal
    base height (m/m)."""
    swing_kp: tuple[float, float, float]
    """Diagonal Cartesian position gain on a swinging foot, in the leg frame (N/m)."""
    swing_kd: tuple[float, float, float]
    """Diagonal Cartesian damping on a swinging foot (N s/m)."""
    stance_kp: tuple[float, float, float]
    """Diagonal Cartesian position gain on a stance foot; zero, the MPC holds the body."""
    stance_kd: tuple[float, float, float]
    """Diagonal Cartesian damping on a stance foot (N s/m)."""
    stance_joint_kd: float
    """Joint-space damping on a stance leg (N m s/rad). Swing legs get none."""
    nominal_footstep_xy: tuple[tuple[float, float], ...]
    """Per leg, the hip-frame (x, y) a foot is placed at until the planner commands a
    footstep (m). Wider and longer than the hip so the untouched stance is stable."""

    @property
    def num_legs(self) -> int:
        return self.spec.num_legs

    @property
    def num_joints(self) -> int:
        return self.spec.num_joints

    @property
    def state_dim(self) -> int:
        """Rigid-body states the MPC predicts: 6 pose, 6 twist, and gravity."""
        return 13

    @property
    def weight(self) -> float:
        """Total weight of the robot (N), what the normal-force bounds scale."""
        return self.mass * GRAVITY


GO1_SRBD = SrbdModel(
    spec=GO1,
    # mass and inertia are measured from the Go1 USD articulation used in sim, summed
    # over all links and taken from the trunk body respectively; the inertia is doubled
    # to stand in for the leg links this single-body model does not carry. Matches
    # Quadruped(RobotType.GO1) in gaitnet_mpc.
    mass=13.100449,
    inertia=(0.0181444194 * 2, 0.0679929703 * 2, 0.0774220750 * 2),
    friction=0.8,
    weights=(1.0, 1.5, 0.0, 0.0, 0.0, 50.0, 0.4, 0.4, 0.1, 1.0, 1.0, 0.1, 0.0),
    force_regularization=1e-5,
    force_scale=(0.1, 10.0),
    max_linear_accel=3.0,
    swing_apex_fraction=1.0 / 3.0,
    swing_kp=(700.0, 700.0, 150.0),
    swing_kd=(7.0, 7.0, 7.0),
    stance_kp=(0.0, 0.0, 0.0),
    stance_kd=(14.0, 14.0, 14.0),
    stance_joint_kd=0.6,
    nominal_footstep_xy=((0.1, 0.1), (0.1, -0.1), (-0.1, 0.1), (-0.1, -0.1)),
)

MODELS: dict[str, SrbdModel] = {GO1_SRBD.spec.name: GO1_SRBD}
