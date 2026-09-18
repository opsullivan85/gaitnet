import numpy as np
from gaitnet_mpc.mpc.common.DesiredStateCommand import DesiredStateCommand
from gaitnet_mpc.mpc.FSM_states.ControlFSMData import ControlFSMData
from gaitnet_mpc.mpc.common.Quadruped import Quadruped, RobotType
from gaitnet_mpc.mpc.common.LegController import LegController
from gaitnet_mpc.mpc.common.StateEstimator import StateEstimator
from gaitnet_mpc.mpc.convex_MPC.SpecifiedFootstepLocomotion import SpecifiedFootstepLocomotion



class RobotRunnerMin:
    def __init__(self):
        pass

    def init(self, robotType: RobotType, dt: float = 0.01, iterations_between_mpc: int = 1):
        """
        Initializes the robot model, state estimator, leg controller,
        robot data, and any control logic specific data.
        """
        self.robotType = robotType

        # TODO tune these parameters
        self.cMPC = SpecifiedFootstepLocomotion(dt, iterations_between_mpc)

        # init quadruped
        self._quadruped = Quadruped(self.robotType)

        # init leg controller
        self._legController = LegController(self._quadruped)

        # init state estimator
        self._stateEstimator = StateEstimator(self._quadruped)

        # init desired state command
        self._desiredStateCommand = DesiredStateCommand()

        # init controller data
        self.data = ControlFSMData()
        self.data._quadruped = self._quadruped
        self.data._stateEstimator = self._stateEstimator
        self.data._legController = self._legController
        self.data._desiredStateCommand = self._desiredStateCommand

        # init convex MPC controller
        self.cMPC.initialize(self.data)

    def reset(self):
        self.cMPC.initialize(self.data)
        self._desiredStateCommand.reset()
        self._stateEstimator.reset()

    def run(
        self,
        dof_states: np.ndarray,
        body_states: np.ndarray,
        commands: np.ndarray,
    ) -> np.ndarray:
        """Runs the overall robot control system by calling each of the major components
        to run each of their respective steps.

        Args:
            dof_states (np.ndarray): (12,2) float32 array of dof states:
                pos = [0:12,0]
                vel = [0:12,1]
                TODO: verify this
                where each leg's 3 dofs are in row order: hip, upper leg, lower leg
            body_states (np.ndarray): (13,) float32 array of body states:
                position = [0:3]
                orientation xyzw = [3:7]
                velocity xyz = [7:10]
                omega xyz = [10:13]
            commands (np.ndarray): (3,) float32 array of user commanded
                x velocity = [0]
                y velocity = [1]
                omega velocity = [2]

        Returns:
            np.ndarray: (12,) float32 array of leg torques
                TODO: verify this
                where each leg's 3 dofs are in row order: hip, upper leg, lower leg
        """
        # Update desired commands
        self._desiredStateCommand.updateCommand(commands)

        # Update the joint states
        self._legController.updateData(dof_states)
        self._legController.zeroCommand()
        # self._legController.setEnable(True)

        # Update robot states
        self._stateEstimator.update(body_states)

        # Run the Control FSM code
        self.cMPC.run(self.data)

        # Sets the leg controller commands for the robot
        legTorques = self._legController.updateCommand()

        return legTorques  # numpy (12,) float32
