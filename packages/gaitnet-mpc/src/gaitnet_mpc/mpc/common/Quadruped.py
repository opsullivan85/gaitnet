from enum import Enum, auto
import numpy as np
from gaitnet_mpc.mpc.utils import DTYPE

# Data structure containing parameters for quadruped robot
class RobotType(Enum):
    ALIENGO = auto()
    # MINI_CHEETAH = auto()
    A1 = auto()
    GO1 = auto()

class Quadruped:

    def __init__(self, robotype:RobotType):

        if robotype.value == RobotType.ALIENGO.value:
            self._abadLinkLength = 0.083
            self._hipLinkLength = 0.25
            self._kneeLinkLength = 0.25
            self._kneeLinkY_offset = 0.0
            self._abadLocation = np.array([0.2399, 0.051, 0], dtype=DTYPE).reshape((3,1))
            self._bodyName = "trunk"
            self._bodyMass = 9.041 * 2
            self._bodyInertia = np.array([0.033260231, 0, 0, 
                                      0, 0.16117211, 0, 
                                      0, 0, 0.17460442])
            self._bodyHeight = 0.35
            self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
            # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
            self._mpc_weights = np.array([1.0, 1.5, 0.0,
                                 0.0, 0.0, 50,
                                 0.0, 0.0, 0.1,
                                 1.0, 1.0, 0.1,
                                 0.0], dtype=DTYPE)
            # self._mpc_weights = [0.25, 0.25, 10, 2, 2, 50, 0, 0, 0.3, 0.2, 0.2, 0.1, 0]
            # self._mpc_weights = [1., 1., 0, 0, 0, 10, 0., 0., .1, .1, .1, .0, 0]

        elif robotype.value == RobotType.GO1.value:
            self._abadLinkLength = 0.08
            self._hipLinkLength = 0.213
            self._kneeLinkLength = 0.213
            self._kneeLinkY_offset = 0.0
            self._abadLocation = np.array([0.1881, 0.04675, 0], dtype=DTYPE).reshape((3,1))
            self._bodyName = "trunk"
            # Measured from the actual Go1 USD articulation used in sim via
            # scene["robot"].root_physx_view.get_masses()/get_inertias() (summed
            # over all links / trunk-body inertia respectively). Total mass is the
            # real whole-robot mass (trunk + 4 legs), not a trunk-only fudge factor.
            # Trunk inertia already includes the knee actuators, which are
            # physically mounted on the trunk and connected to the calf via a rod;
            # it is scaled by a modest 2x (not the previous unfounded 5x) to
            # approximate the remaining un-lumped leg-link inertia in this
            # single-rigid-body model.
            self._bodyMass = 13.100449
            self._bodyInertia = np.array([0.0181444194, 0, 0,
                                      0, 0.0679929703, 0,
                                      0, 0, 0.0774220750]) * 2
            self._bodyHeight = 0.26
            # TODO: sync this up with the sim friction
            self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.8
            # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
            self._mpc_weights = np.array([1.0, 1.5, 0.0,
                                 0.0, 0.0, 50,
                                 0.4, 0.4, 0.1,
                                 1.0, 1.0, 0.1,
                                 0.0], dtype=DTYPE)

        elif robotype.value == RobotType.A1.value:
            self._abadLinkLength = 0.08505
            self._hipLinkLength = 0.2
            self._kneeLinkLength = 0.2
            self._kneeLinkY_offset = 0.0
            self._abadLocation = np.array([0.183, 0.047, 0], dtype=DTYPE).reshape((3,1))
            self._bodyName = "trunk"
            self._bodyMass = 8.5 * 3
            self._bodyInertia = np.array([0.017, 0, 0, 
                                      0, 0.057, 0, 
                                      0, 0, 0.064]) * 10
            self._bodyHeight = 0.26
            self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
            # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
            # self._mpc_weights = [1., 1., 0, 0, 0, 20, 0., 0., .1, .1, .1, .0, 0]
            self._mpc_weights = np.array([0.25, 0.25, 10, 2, 2, 50, 0, 0, 0.3, 0.5, 0.5, 0.1, 0], dtype=DTYPE)

        # elif robotype is RobotType.MINI_CHEETAH:
        #     self._abadLinkLength = 0.062
        #     self._hipLinkLength = 0.209
        #     self._kneeLinkLength = 0.195
        #     self._kneeLinkY_offset = 0.004
        #     self._abadLocation = np.array([0.19, 0.049, 0], dtype=DTYPE).reshape((3,1))
        #     self._bodyName = "body"
        #     self._bodyMass = 3.3 * 3
        #     self._bodyInertia = np.array([0.011253, 0, 0, 
        #                               0, 0.036203, 0, 
        #                               0, 0, 0.042673]) * 10
        #     self._bodyHeight = 0.29
        #     self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
        #     # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
        #     self._mpc_weights = np.array([0.25, 0.25, 10, 2, 2, 50, 0, 0, 0.3, 0.2, 0.2, 0.1, 0], dtype=DTYPE)
        
        else:
            raise Exception("Invalid RobotType")
            
        self._robotType = robotype

    def getHipLocation(self, leg:int):
        """
        Get location of the hip for the given leg in robot frame
        """
        assert leg >= 0 and leg < 4
        pHip = np.array([
            self._abadLocation[0] if (leg == 0 or leg == 1) else -self._abadLocation[0],
            self._abadLocation[1] if (leg == 0 or leg == 2) else -self._abadLocation[1],
            self._abadLocation[2]
            ], dtype=DTYPE).reshape((3,1))

        return pHip

    