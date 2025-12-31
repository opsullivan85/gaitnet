
import numpy as np
from nptyping import Float32, NDArray, Shape, Bool

try:
    from gaitnet.control import RobotRunnerMin, RobotType
except ImportError as e:
    raise ImportError(
        "Failed to import from src.control. Make sure the control module is correctly installed."
    ) from e
from gaitnet.sim2real.abstractinterface import Sim2RealInterface
import logging
from gaitnet import get_logger
logger = get_logger()


class SimInterface(Sim2RealInterface):
    def __init__(
        self, dt: float, iterations_between_mpc: int = 1, debug_logging: bool = False
    ) -> None:
        self.logger: None | logging.Logger = None
        if debug_logging:
            self.logger = logger

        self._dt = dt
        self._iterations_between_mpc = iterations_between_mpc
        # initialize the robot runner
        self.robot_runner: RobotRunnerMin
        self.reset()

    # override
    def get_torques(
        self,
        joint_states: NDArray[Shape["4, 3, 2"], Float32],
        body_state: NDArray[Shape["13"], Float32],
        command: NDArray[Shape["3"], Float32],
    ) -> NDArray[Shape["4, 3"], Float32]:
        joint_states_converted = SimInterface._convert_joint_states(
            joint_states_interface=joint_states
        )
        torques = self.robot_runner.run(
            dof_states=joint_states_converted, body_states=body_state, commands=command
        )
        torques_converted = SimInterface._convert_torques(torques_control=torques)

        if self.logger is not None:
            # make each array print on the next line, with a tab indent
            formatter = "\n\t\t"
            with np.printoptions(precision=5, suppress=True):
                logger.info(
                    f"Contact states: {self.robot_runner.cMPC.gait.getContactPhase().flatten()}"
                )
                logger.info(
                    f"Swing phase: {self.robot_runner.cMPC.gait.getSwingPhase().flatten()}"
                )
                mpc_table = self.robot_runner.cMPC.gait.getMpcTable()
                mpc_table = np.asarray(mpc_table).reshape(
                    (self.robot_runner.cMPC.horizon_length, -1)
                )
                logger.info(f"MPC table:\n{mpc_table}")
                logger.info("")

        return torques_converted

    # override
    def reset(self) -> None:
        """Resets the robot"""
        # TODO: find out why robot_runner.reset() causes issues
        self.robot_runner = RobotRunnerMin()
        self.robot_runner.init(
            RobotType.GO1,
            dt=self._dt,
            iterations_between_mpc=self._iterations_between_mpc,
        )
        # self.robot_runner.reset()

    @staticmethod
    def _convert_joint_states(
        joint_states_interface: NDArray[Shape["4, 3, 2"], Float32],
    ) -> NDArray[Shape["12, 2"], Float32]:
        """Convert joint states from interface order to control order.

        Args:
            joint_states_interface (np.ndarray): (4, 3, 2) joint states in interface order
                index 0: leg index (0-3)
                index 1: joint index (0-2) (hip, upper leg, lower leg)
                index 2: state (0: position, 1: velocity)

        Returns:
            joint_states_control (np.ndarray): (12, 2) joint states in control order
                [
                    [FL_hip_pos, FL_hip_vel]
                    [FL_knee_pos, FL_knee_vel]
                    [FL_ankle_pos, FL_ankle_vel]
                    [FR_hip_pos, FR_hip_vel]
                    ...
                ]
        """
        return joint_states_interface.reshape((12, 2))

    @staticmethod
    def _convert_torques(
        torques_control: NDArray[Shape["12"], Float32],
    ) -> NDArray[Shape["4, 3"], Float32]:
        """Convert torques from control order to interface order.

        Args:
            torques_control (np.ndarray): (12,) torques in control order
                [
                    FL_hip_torque
                    FL_knee_torque
                    FL_ankle_torque
                    FR_hip_torque
                    ...
                ]

        Returns:
            torques_interface (np.ndarray): (4, 3) torques in interface order
                index 0: leg index (0-3)
                index 1: joint index (0-2) (hip, upper leg, lower leg)
        """
        return torques_control.reshape((4, 3))

    # override
    def initiate_footstep(
        self,
        leg: int,
        location_hip: NDArray[Shape["2"], Float32],
        duration: float,
    ):
        self.robot_runner.cMPC.initiate_footstep(leg, location_hip, duration)

    # override
    def get_contact_state(self) -> NDArray[Shape["4"], Bool]:
        return self.robot_runner.cMPC.gait.getContactPhase().flatten().astype(bool)

    # override
    def get_swing_phase(self) -> NDArray[Shape["4"], Float32]:
        return self.robot_runner.cMPC.gait.getSwingPhase().flatten()
    
    # override
    def get_swing_durations(self) -> NDArray[Shape["4, 1"], Float32]:
        return self.robot_runner.cMPC.gait.swing_durations
