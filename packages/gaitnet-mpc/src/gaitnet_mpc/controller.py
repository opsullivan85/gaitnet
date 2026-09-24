"""Per-robot footstep-driven MPC controller, the unit a VectorPool worker holds."""

from __future__ import annotations

import logging

import numpy as np

from gaitnet_mpc.mpc.common.Quadruped import RobotType
from gaitnet_mpc.mpc.robot_runner.RobotRunnerMin import RobotRunnerMin

logger = logging.getLogger(__name__)


class MpcFootstepController:
    """Convex MPC stance control plus specified-footstep swing control for one robot.

    Leg order is FL, FR, RL, RR throughout; joint order within a leg is hip, thigh, calf.

    Each footstep target is fixed in the world when it is commanded and re-aimed from the
    hip every step, by rewriting the vendored controller's hip-relative target before it
    runs. The vendored controller would instead execute the target against the hip at
    touchdown less a predicted body travel, which misses by the body's turning, tilt and
    change of speed during the swing; that prediction is switched off, since there is no
    travel left to predict. The world is tracked by integrating the base's world velocity,
    as the simulator integrates it, so no position estimate is needed.
    """

    def __init__(
        self,
        dt: float,
        iterations_between_mpc: int = 1,
        robot_type: RobotType = RobotType.GO1,
        debug_logging: bool = False,
    ) -> None:
        self._dt = dt
        self._iterations_between_mpc = iterations_between_mpc
        self._robot_type = robot_type
        self._debug_logging = debug_logging
        self.robot_runner: RobotRunnerMin
        self.reset()

    def get_torques(
        self,
        joint_states: np.ndarray,
        body_state: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        """Compute the joint torques for the current state and velocity command.

        Args:
            joint_states: (4, 3, 2) leg, joint (hip, thigh, calf), (position, velocity)
            body_state: (13,) position [0:3], orientation xyzw [3:7], linear velocity
                [7:10], angular velocity [10:13], all in the world frame
            command: (3,) x velocity, y velocity, yaw rate

        Returns:
            (4, 3) joint torques, leg by joint
        """
        self._aim_pinned_footholds(body_state)
        torques = self.robot_runner.run(
            dof_states=self._convert_joint_states(joint_states),
            body_states=body_state,
            commands=command,
        )

        if self._debug_logging:
            gait = self.robot_runner.cMPC.gait
            with np.printoptions(precision=5, suppress=True):
                logger.info(f"Contact states: {gait.getContactPhase().flatten()}")
                logger.info(f"Swing phase: {gait.getSwingPhase().flatten()}")
                mpc_table = np.asarray(gait.getMpcTable()).reshape(
                    (self.robot_runner.cMPC.horizon_length, -1)
                )
                logger.info(f"MPC table:\n{mpc_table}")

        return self._convert_torques(torques)

    def reset(self) -> None:
        """Reset the controller to its initial state."""
        # TODO: find out why robot_runner.reset() causes issues
        self.robot_runner = RobotRunnerMin()
        self.robot_runner.init(
            self._robot_type,
            dt=self._dt,
            iterations_between_mpc=self._iterations_between_mpc,
        )
        self.robot_runner.cMPC.compensate_body_travel = False
        quadruped = self.robot_runner._quadruped
        self._hips = np.stack([quadruped.getHipLocation(leg).flatten() for leg in range(4)]).astype(np.float64)
        """(4, 3) each hip from the base, base frame (m)."""
        self._odometry = np.zeros(3)
        """(3,) the base's integrated world velocity (m), the frame pinned footholds live in."""
        self._targets_hip = np.zeros((4, 3))
        """(4, 3) the last target commanded per leg, the foot centre in its hip yaw frame (m)."""
        self._foothold_world = np.zeros((4, 3))
        """(4, 3) each pinned foothold (the foot centre) in the odometry frame (m)."""
        self._pin_pending = np.zeros(4, dtype=bool)
        self._pinned = np.zeros(4, dtype=bool)

    def _aim_pinned_footholds(self, body_state: np.ndarray) -> None:
        """Advance the odometry, pin footsteps commanded since the last step (their target
        was measured in this state), and point every pinned leg's hip-relative target, in
        the base frame, at its foothold."""
        self._odometry = self._odometry + np.asarray(body_state[7:10], dtype=np.float64) * self._dt
        world_R_base = _quat_to_matrix(np.asarray(body_state[3:7], dtype=np.float64))
        if self._pin_pending.any():
            yaw = np.arctan2(world_R_base[1, 0], world_R_base[0, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            heading = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            legs = self._pin_pending
            self._foothold_world[legs] = (
                self._odometry + self._hips[legs] @ world_R_base.T + self._targets_hip[legs] @ heading.T
            )
            self._pinned |= legs
            self._pin_pending[:] = False
        if self._pinned.any():
            legs = self._pinned
            aimed = (self._foothold_world[legs] - self._odometry) @ world_R_base - self._hips[legs]
            self.robot_runner.cMPC.footstep_locations_hip[legs] = aimed

    @staticmethod
    def _convert_joint_states(joint_states_interface: np.ndarray) -> np.ndarray:
        """(4, 3, 2) leg-major joint states to the controller's (12, 2) rows,
        FL hip, FL thigh, FL calf, FR hip, ..."""
        return joint_states_interface.reshape((12, 2))

    @staticmethod
    def _convert_torques(torques_control: np.ndarray) -> np.ndarray:
        """The controller's (12,) torques, FL hip, FL thigh, ..., to (4, 3) leg by joint."""
        return torques_control.reshape((4, 3))

    def initiate_footstep(self, leg: int, location_hip: np.ndarray, duration: float) -> None:
        """Start a swing of `leg` to `location_hip`, (x, y, z) relative to that leg's hip in
        its gravity-aligned yaw frame (z negative below the hip), over `duration` s."""
        self.robot_runner.cMPC.initiate_footstep(leg, location_hip, duration)
        self._targets_hip[leg] = np.asarray(location_hip, dtype=np.float64).reshape(-1)
        self._pin_pending[leg] = True
        self._pinned[leg] = False

    def get_contact_state(self) -> np.ndarray:
        """(4,) bool, whether the schedule has each leg in stance."""
        return self.robot_runner.cMPC.gait.getContactPhase().flatten().astype(bool)

    def get_swing_phase(self) -> np.ndarray:
        """(4,) float32 swing phase in [0, 1], 0 in stance."""
        return self.robot_runner.cMPC.gait.getSwingPhase().flatten()

    def get_swing_durations(self) -> np.ndarray:
        """(4, 1) float32 duration of each leg's most recent swing, kept after touchdown."""
        return self.robot_runner.cMPC.gait.swing_durations

    def get_estimated_rpy(self) -> np.ndarray:
        """(3,) roll, pitch, yaw (rad) of the base as the controller's state estimator sees
        it after the last `get_torques` call. For checking the body state convention."""
        return self.robot_runner._stateEstimator.getResult().rpy.flatten().astype(np.float32)

    def get_gait_timing(self) -> np.ndarray:
        """The scheduled gait timing of each leg, (4, 3) float32.

        This is the controller's plan, not a measurement: a foot that strikes the
        ground early is still reported as swinging until its scheduled touchdown.

        Columns: swing phase in [0, 1] (0 in stance), remaining swing time (s, 0 in
        stance), time since scheduled touchdown (s, 0 in swing).
        """
        gait = self.robot_runner.cMPC.gait
        # use the gait's own contact definition so the swing/stance split here
        # always agrees with the contact schedule the MPC is running
        in_contact = gait.getContactPhase().flatten().astype(bool)
        start = gait.swing_start_times.flatten()
        duration = gait.swing_durations.flatten()
        touchdown = start + duration

        elapsed_swing = gait.time - start
        swing_phase = np.divide(
            elapsed_swing, duration, out=np.zeros_like(duration), where=duration > 0
        )
        swing_phase = np.where(in_contact, 0.0, np.clip(swing_phase, 0.0, 1.0))
        swing_remaining = np.where(in_contact, 0.0, np.maximum(touchdown - gait.time, 0.0))
        stance_time = np.where(in_contact, np.maximum(gait.time - touchdown, 0.0), 0.0)

        return np.stack([swing_phase, swing_remaining, stance_time], axis=1).astype(
            np.float32
        )


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """(3, 3) rotation taking base vectors into the world, for an xyzw quaternion. In
    float64: the vendored `orientation_tools` computes in float16."""
    x, y, z, w = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
