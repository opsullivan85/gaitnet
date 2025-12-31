from abc import ABC, abstractmethod

import numpy as np
from nptyping import Float32, NDArray, Shape, Bool

from gaitnet import get_logger
logger = get_logger()


class Sim2RealInterface(ABC):
    @abstractmethod
    def __init__(self, dt: float) -> None:
        pass

    @abstractmethod
    def get_torques(
        self,
        joint_states: NDArray[Shape["4, 3, 2"], Float32],
        body_state: NDArray[Shape["13"], Float32],
        command: NDArray[Shape["3"], Float32],
    ) -> NDArray[Shape["4, 3"], Float32]:
        """Compute and return the joint torques based on the current state and command.

        Args:
            joint_states (np.ndarray): (4,3,2) array of joint states:
                index 0: leg index (0-3)
                index 1: joint index (0-2) (hip, upper leg, lower leg)
                index 2: state (0: position, 1: velocity)
            body_state (np.ndarray): (13,) array of body state:
                position = [0:3]
                orientation (xyzw quaternion) = [3:7]
                velocity = [7:10]
                angular velocity = [10:13]
            command (np.ndarray): (3,) array of command:
                x velocity = [0]
                y velocity = [1]
                yaw rate = [2]

        Returns:
            np.ndarray: (4,3) array of joint torques
                index 0: leg index (0-3)
                index 1: joint index (0-2) (hip, upper leg, lower leg)
        """
        pass

    @abstractmethod
    def initiate_footstep(
        self,
        leg: int,
        location_hip: NDArray[Shape["2"], Float32],
        duration: float,
    ):
        """initiates a footstep for the specified leg at the specified location

        TODO: add height in here

        Args:
            leg (int): Index of the leg (0-3)
                in FR, FL, RR, RL order
            location_hip (NDArray[Shape["2"], Float32]): Desired foot position in the respective hip frame (x, y)
                This position is relative to the hip of the specified leg.
                z will be projected down to zero.
            duration (float): Duration of the footstep
        """
        pass

    @abstractmethod
    def get_contact_state(self) -> NDArray[Shape["4"], Bool]:
        """Get the contact states of the feet.

        Returns:
            np.ndarray: (4,) boolean array indicating contact state of each foot
                index 0: leg index (0-3) in FR, FL, RR, RL order
                True if in contact, False otherwise
        """
        pass

    @abstractmethod
    def get_swing_phase(self) -> NDArray[Shape["4"], Float32]:
        """Get the swing phase of each leg.

        Returns:
            np.ndarray: (4,) float32 array indicating swing phase of each leg
                index 0: leg index (0-3) in FR, FL, RR, RL order
                value in [0, 1], where 0 is the start of swing and 1 is the end of swing
        """
        pass
    
    @abstractmethod
    def get_swing_durations(self) -> NDArray[Shape["4, 1"], Float32]:
        """Get the swing durations of each leg.

        Returns:
            np.ndarray: (4, 1) float32 array indicating swing duration of each leg
                index 0: leg index (0-3) in FR, FL, RR, RL order
                value in seconds
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the internal state of the controller."""
        pass
