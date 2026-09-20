"""The batched low-level controller: footsteps in, joint torques out, nothing on the CPU.

Implements `gaitnet_core.interfaces.LowLevelController`, so it drops in wherever
`gaitnet_sim.controllers.PooledMpcController` goes. That one runs one
`gaitnet_mpc.MpcFootstepController` per robot in a process pool, which is exact but
costs a CPU core's worth of convex solve per robot and a host round trip every physics
step; this one runs the same controller for every robot at once as batched tensor
operations on whatever device the robot's state is already on.

Each control step it does what the CPU controller does, in the same order:

1. leg kinematics, to find where each foot is and how fast it is moving;
2. state estimation, which here means orientation plus a body height read off the
   stance feet - there is no position estimate, and nothing needs one;
3. footstep placement, turning the planner's hip-frame targets into touchdown points;
4. the convex MPC, on a schedule of one solve every `control_steps_per_update` steps;
5. a Cartesian PD per foot, feed-forward from the MPC in stance and a Bezier swing arc
   in the air, through the leg Jacobian into joint torques.

The frames are the CPU controller's and are worth stating, because their names there are
not always their contents. The estimator never sees a world position, so its "global"
frame is really an instantaneous frame with its origin under the body: x and y of the
estimated position are always zero and only z, the height above the stance feet, means
anything. Footstep targets arrive in each leg's gravity-aligned hip frame and are used
against the hip at *touchdown*, which is what the body-travel compensation below is for.

Two things differ from the CPU controller on purpose, and neither changes what the robot
does:

- The MPC's update *phase* is shared by the batch rather than being counted per robot,
  so one solve covers every robot. Resetting one robot no longer shifts which steps its
  MPC refreshes on. The gait clock stays per robot, because that one is observable.
- The QP is solved by a fixed budget of batched ADMM iterations rather than by an
  active-set method to convergence. See `admm.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from gaitnet_core.control import kinematics, rotations
from gaitnet_core.control.gait import GaitSchedule
from gaitnet_core.control.model import GO1_SRBD, SrbdModel
from gaitnet_core.control.mpc import BatchedConvexMpc, MpcSettings
from gaitnet_core.control.swing import swing_trajectory
from gaitnet_core.interfaces import FootstepCommand


@dataclass
class BatchedMpcConfig:
    """What to build a `BatchedMpcController` with."""

    model: SrbdModel = GO1_SRBD
    """Dynamics, gains and force bounds."""
    mpc: MpcSettings = field(default_factory=MpcSettings)
    """Horizon, update rate and QP solver budget."""
    foot_radius: float = 0.02
    """Radius of the robot's spherical foot (m). Footstep targets name a point on the
    terrain surface and the controller places the sphere's centre, so targets are raised
    by this much."""
    dtype: torch.dtype = torch.float32
    """Precision for the controller and the QP. float64 roughly doubles the solve cost
    and is mostly useful for checking the float32 path against the CPU controller."""


class BatchedMpcController:
    """Implements `gaitnet_core.interfaces.LowLevelController` for N robots at once."""

    def __init__(
        self,
        cfg: BatchedMpcConfig,
        num_robots: int,
        dt: float,
        device: str | torch.device,
    ) -> None:
        """
        Args:
            cfg: model, MPC settings and foot geometry.
            num_robots: N.
            dt: control period (s), the physics step: torques are computed every step.
            device: where the robot's state already lives.
        """
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = cfg.dtype
        self.dt = dt
        self._num_robots = num_robots

        model = cfg.model
        self.model = model
        legs = model.num_legs

        self._hip_offsets = torch.tensor(
            model.spec.hip_offsets, device=device, dtype=self.dtype
        )
        self._leg_offsets = kinematics.leg_offsets(model, device, self.dtype)
        self._nominal_offsets = self._nominal_footstep_offsets()

        self._gait = GaitSchedule(num_robots, legs, dt, device, self.dtype)
        self._mpc = BatchedConvexMpc(model, cfg.mpc, num_robots, dt, device, self.dtype)
        self._update_counter = 0

        def per_foot() -> torch.Tensor:
            return torch.zeros(num_robots, legs, 3, device=device, dtype=self.dtype)

        self._footstep_offsets = self._nominal_offsets.expand(num_robots, legs, 3).clone()
        self._feedforward = per_foot()
        self._swing_start = per_foot()
        self._swing_end = per_foot()
        self._foot_target = per_foot()
        self._foot_target_velocity = per_foot()
        self._first_swing = torch.ones(num_robots, legs, device=device, dtype=torch.bool)
        self._contact_phase = torch.zeros(num_robots, legs, device=device, dtype=self.dtype)
        self._desired_velocity = torch.zeros(num_robots, 2, device=device, dtype=self.dtype)
        self._height = torch.full(
            (num_robots,), model.spec.nominal_height, device=device, dtype=self.dtype
        )
        self._rpy = torch.zeros(num_robots, 3, device=device, dtype=self.dtype)
        self._before_first_step = torch.ones(num_robots, device=device, dtype=torch.bool)

        def gain(value: tuple[float, float, float]) -> torch.Tensor:
            return torch.tensor(value, device=device, dtype=self.dtype)

        self._swing_kp, self._stance_kp = gain(model.swing_kp), gain(model.stance_kp)
        self._swing_kd, self._stance_kd = gain(model.swing_kd), gain(model.stance_kd)

    @property
    def num_robots(self) -> int:
        return self._num_robots

    def _nominal_footstep_offsets(self) -> torch.Tensor:
        """(1, L, 3) where a foot is put before the planner has said anything.

        Wider and longer than the hips, and a nominal body height below them, so a robot
        that is never commanded to step still stands.
        """
        xy = torch.tensor(
            self.model.nominal_footstep_xy, device=self.device, dtype=self.dtype
        )
        z = torch.full_like(xy[:, :1], -self.model.spec.nominal_height)
        return torch.cat([xy, z], dim=-1).unsqueeze(0)

    def command_footsteps(self, footsteps: FootstepCommand) -> None:
        """Start at most one swing per robot."""
        target = footsteps.target.to(self.dtype).clone()
        # the target names a point on the terrain surface; the controller places the
        # foot sphere's centre, one radius above it
        target[:, 2] += self.cfg.foot_radius

        index = footsteps.leg.view(-1, 1, 1).expand(-1, 1, 3)
        held = self._footstep_offsets.gather(1, index)
        chosen = torch.where(footsteps.active.view(-1, 1, 1), target.unsqueeze(1), held)
        self._footstep_offsets.scatter_(1, index, chosen)
        self._gait.initiate(footsteps.active, footsteps.leg, footsteps.duration)

    def compute_torques(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        base_pose: torch.Tensor,
        base_vel: torch.Tensor,
        velocity_command: torch.Tensor,
    ) -> torch.Tensor:
        """(N, 12) joint torques for one control step.

        `base_pose`'s world position is ignored, as it is by the CPU controller: nothing
        downstream of here knows where the robot is, only how it is oriented and how far
        it is above its own feet.
        """
        model = self.model
        legs = model.num_legs
        joint_pos = joint_pos.to(self.dtype).view(-1, legs, 3)
        joint_vel = joint_vel.to(self.dtype).view(-1, legs, 3)

        self._gait.begin_step()
        foot_position, jacobian = kinematics.foot_position_and_jacobian(
            joint_pos, self._leg_offsets
        )
        foot_velocity = torch.einsum("nlij,nlj->nli", jacobian, joint_vel)

        tilt, rpy, rpy_base, linear_velocity, angular_velocity = self._estimate(
            base_pose.to(self.dtype), base_vel.to(self.dtype)
        )
        self._rpy = rpy

        # the command is rate limited before the MPC sees it, so a step change in the
        # operator's command cannot ask the body for an impossible acceleration
        limit = model.max_linear_accel * self.dt
        command = velocity_command.to(self.dtype)
        self._desired_velocity = self._desired_velocity + torch.clamp(
            command[:, :2] - self._desired_velocity, -limit, limit
        )
        desired_yaw_rate = command[:, 2]

        foot_in_base = self._hip_offsets + foot_position
        # where the feet are is read against *last* step's height estimate, and the
        # estimate is only then refreshed, as upstream does
        foot_in_world = foot_in_base + self._position().unsqueeze(1)

        # a robot that has not taken a step yet has no arc behind it, so anchor one at
        # the feet; the far end is set by `_touchdown_targets` every step anyway
        first_step = self._before_first_step.view(-1, 1, 1)
        self._swing_start = torch.where(first_step, foot_in_world, self._swing_start)
        self._before_first_step = torch.zeros_like(self._before_first_step)

        self._update_height(tilt, foot_in_base)
        position = self._position()
        self._swing_end = self._touchdown_targets(linear_velocity, position)

        contact = self._gait.in_contact()
        swing_phase = self._gait.swing_phase()

        self._update_counter += 1
        if self._update_counter % self.cfg.mpc.control_steps_per_update == 0:
            self._feedforward = self._mpc.solve(
                rpy=rpy_base,
                position=position,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                foot_positions=foot_in_base,
                contact=self._gait.contact_table(self.cfg.mpc.horizon, self._mpc.timestep),
                desired_velocity=self._desired_velocity,
                desired_yaw_rate=desired_yaw_rate,
            )

        swinging = swing_phase > 0.0
        self._track_swing(swinging, swing_phase, foot_in_world)
        torques = self._leg_torques(
            swinging, position, foot_position, foot_velocity, jacobian, joint_vel, linear_velocity
        )
        self._contact_phase = contact.to(self.dtype)
        self._gait.end_step()
        return torques.reshape(-1, model.num_joints)

    def _estimate(
        self, base_pose: torch.Tensor, base_vel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Orientation and base-frame twist, as `StateEstimator.update` computes them.

        Returns the base's tilt out of its own yaw frame, its roll, pitch and yaw in the
        world, the same angles relative to the yaw frame (so the third is ~0, and that
        is what the MPC tracks), and the linear and angular velocity in the base frame.
        """
        quat = base_pose[:, 3:7]
        base_rotation = rotations.quat_to_rotation(quat)
        linear_velocity = torch.einsum("nij,nj->ni", base_rotation, base_vel[:, :3])
        angular_velocity = torch.einsum("nij,nj->ni", base_rotation, base_vel[:, 3:6])
        rpy = rotations.quat_to_rpy(quat)
        tilt = base_rotation @ rotations.rotation_z(rpy[:, 2])
        return tilt, rpy, rotations.rotation_to_rpy(tilt), linear_velocity, angular_velocity

    def _position(self) -> torch.Tensor:
        """(N, 3) the estimated base position: zero in x and y, height above the feet in z."""
        zeros = torch.zeros_like(self._height)
        return torch.stack([zeros, zeros, self._height], dim=-1)

    def _update_height(self, tilt: torch.Tensor, foot_in_base: torch.Tensor) -> None:
        """Average how far the stance feet hang below the body, and call that the height.

        Upstream (`StateEstimator._update_com_position_ground_frame`) applies the *yaw
        frame to base* rotation to vectors that are already in the base frame, which is
        its transpose; the difference is second order in the body's tilt. Reproduced
        deliberately, see the note in `srbd.py`. The contact mask is last step's, since
        that is the one the estimator was handed at the end of the previous call.
        """
        in_ground_frame = torch.einsum("nij,nlj->nli", tilt, foot_in_base)
        heights = -in_ground_frame[..., 2]
        weight = self._contact_phase
        total = weight.sum(dim=-1)
        average = (heights * weight).sum(dim=-1) / total.clamp_min(1e-6)
        self._height = torch.where(total > 0, average, self._height)

    def _touchdown_targets(
        self, linear_velocity: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        """(N, L, 3) where each foot is aiming, in the estimator's frame.

        The planner picks a spot on the ground while the foot is still at the far end of
        its swing, but the offset is executed against the hip at *touchdown*, and the
        body carries the hip forward by `velocity * swing duration` in between. Taking
        that travel back off the target cancels it, so the foot lands on the patch of
        ground that was chosen rather than several scan cells past it. This is the
        opposite sign to the Raibert-style lead a gaited controller would apply, which
        pushes the foot ahead of the body for balance rather than onto one spot.
        """
        travel = linear_velocity[:, None, :2] * self._gait.swing_duration.unsqueeze(-1)
        # a velocity spike must not throw the target out of the leg's workspace
        limit = self.model.max_body_travel_compensation
        travel = travel * (limit / travel.norm(dim=-1, keepdim=True).clamp_min(limit))

        offsets = self._footstep_offsets.clone()
        offsets[..., :2] -= travel
        return self._hip_offsets + offsets + position.unsqueeze(1)

    def _track_swing(
        self, swinging: torch.Tensor, swing_phase: torch.Tensor, foot_in_world: torch.Tensor
    ) -> None:
        """Advance each swinging foot along its arc; hold the last point in stance.

        A leg that has just left the ground starts its arc from wherever the foot
        actually is, not from where the last arc ended.
        """
        starting = swinging & self._first_swing
        self._swing_start = torch.where(starting.unsqueeze(-1), foot_in_world, self._swing_start)
        self._first_swing = ~swinging

        clearance = self.model.spec.nominal_height * self.model.swing_apex_fraction
        position, velocity = swing_trajectory(
            self._swing_start,
            self._swing_end,
            clearance,
            swing_phase,
            self._gait.swing_duration,
        )
        airborne = swinging.unsqueeze(-1)
        self._foot_target = torch.where(airborne, position, self._foot_target)
        self._foot_target_velocity = torch.where(
            airborne, velocity, self._foot_target_velocity
        )

    def _leg_torques(
        self,
        swinging: torch.Tensor,
        position: torch.Tensor,
        foot_position: torch.Tensor,
        foot_velocity: torch.Tensor,
        jacobian: torch.Tensor,
        joint_vel: torch.Tensor,
        linear_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """(N, L, 3) joint torques from a Cartesian PD per foot plus the MPC's forces.

        A swinging leg is position controlled onto its arc and gets no feed-forward; a
        stance leg is only damped and carries the force the MPC asked for. That force is
        applied in the leg frame without being rotated out of the MPC's gravity-aligned
        one, as upstream does.
        """
        airborne = swinging.unsqueeze(-1)
        kp = torch.where(airborne, self._swing_kp, self._stance_kp)
        kd = torch.where(airborne, self._swing_kd, self._stance_kd)
        feedforward = torch.where(airborne, torch.zeros_like(self._feedforward), self._feedforward)
        joint_damping = torch.where(
            swinging,
            torch.zeros_like(self._contact_phase),
            torch.full_like(self._contact_phase, self.model.stance_joint_kd),
        )

        desired_position = self._foot_target - position.unsqueeze(1) - self._hip_offsets
        desired_velocity = self._foot_target_velocity - linear_velocity.unsqueeze(1)
        foot_force = (
            feedforward
            + kp * (desired_position - foot_position)
            + kd * (desired_velocity - foot_velocity)
        )
        torque = torch.einsum("nlij,nli->nlj", jacobian, foot_force)
        return torque - joint_damping.unsqueeze(-1) * joint_vel

    def gait_timing(self) -> torch.Tensor:
        """(N, L, 3), see `gaitnet_core.state.RobotState.gait_timing`."""
        return self._gait.timing()

    def estimated_rpy(self) -> torch.Tensor:
        """(N, 3) base roll, pitch, yaw (rad) as the estimator last saw it, for checking
        the body state convention."""
        return self._rpy

    def reset(self, robot_ids: torch.Tensor | None = None) -> None:
        """Put the named robots back to standing, with nothing commanded and no history."""
        if robot_ids is None:
            robot_ids = torch.arange(self._num_robots, device=self.device)
        self._gait.reset(robot_ids)
        self._mpc.reset(robot_ids)
        self._footstep_offsets[robot_ids] = self._nominal_offsets
        self._feedforward[robot_ids] = 0.0
        self._swing_start[robot_ids] = 0.0
        self._swing_end[robot_ids] = 0.0
        self._foot_target[robot_ids] = 0.0
        self._foot_target_velocity[robot_ids] = 0.0
        self._first_swing[robot_ids] = True
        self._contact_phase[robot_ids] = 0.0
        self._desired_velocity[robot_ids] = 0.0
        self._height[robot_ids] = self.model.spec.nominal_height
        self._rpy[robot_ids] = 0.0
        self._before_first_step[robot_ids] = True

    def close(self) -> None:
        """Nothing to release: there are no worker processes."""
