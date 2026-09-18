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
        """Start a swing of `leg` to `location_hip`, (x, y) in that leg's hip frame, over `duration` s."""
        self.robot_runner.cMPC.initiate_footstep(leg, location_hip, duration)

    def get_contact_state(self) -> np.ndarray:
        """(4,) bool, whether the schedule has each leg in stance."""
        return self.robot_runner.cMPC.gait.getContactPhase().flatten().astype(bool)

    def get_swing_phase(self) -> np.ndarray:
        """(4,) float32 swing phase in [0, 1], 0 in stance."""
        return self.robot_runner.cMPC.gait.getSwingPhase().flatten()

    def get_swing_durations(self) -> np.ndarray:
        """(4, 1) float32 duration of each leg's most recent swing, kept after touchdown."""
        return self.robot_runner.cMPC.gait.swing_durations

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
