from gaitnet_mpc.mpc.common.StateEstimator import StateEstimate
from gaitnet_mpc.mpc.convex_MPC.Gait import CalculatedGait
from gaitnet_mpc.mpc.FSM_states.ControlFSMData import ControlFSMData
from gaitnet_mpc.mpc.convex_MPC.Gait import GaitABC
from gaitnet_mpc.mpc.math_utils.orientation_tools import rpy_to_rot
import numpy as np
from gaitnet_mpc.mpc.convex_MPC.ConvexMPCLocomotion import ConvexMPCLocomotion
from gaitnet_mpc.mpc.utils import DTYPE


class SpecifiedFootstepLocomotion(ConvexMPCLocomotion):
    def __init__(self, dt: float, iterations_between_mpc: int):
        super().__init__(dt, iterations_between_mpc)
        # self.footstep_locations_hip = np.zeros((4, 2), dtype=DTYPE)
        # offset nominal stance so feet are out to the sides and further
        # out from the robot front and back
        # Leg ordering: [FL, FR, RL, RR], see Quadruped.getHipLocation
        self.footstep_locations_hip = np.asarray(
            [
                [0.1, 0.1],  # Front Left
                [0.1, -0.1],  # Front Right
                [-0.1, 0.1],  # Rear Left
                [-0.1, -0.1],  # Rear Right
            ]
        )

        """Four feet, desired x, y positions in respective hip frames
        Leg ordering: [FL, FR, RL, RR] (Front Left, Front Right, Rear Left, Rear Right)
        """

        self.gait = CalculatedGait(dt, iterations_between_mpc, self.horizon_length)

        self.compensate_body_travel = True
        """Whether to compensate specified footsteps for the body travel over the swing.

        See `_update_footstep_placement`. Turn off to command the offsets exactly as
        they were specified."""
        self.max_body_travel_compensation = 0.1
        """Largest distance (m) a footstep may be shifted by that compensation.

        Keeps a velocity spike from throwing the target out of the leg's workspace."""

    def _get_gait(self, gait_number: int) -> GaitABC:
        return self.gait

    # override
    def initialize(self, *args, **kwargs) -> None:
        super().initialize(*args, **kwargs)
        # override default gait with our specified footstep gait
        self.gait = CalculatedGait(
            self.dt, self.iterations_between_mpc, self.horizon_length
        )

    def _update_footstep_placement(
        self,
        i: int,
        gait: GaitABC,
        data: ControlFSMData,
        state_estimator_result: StateEstimate,
        desired_velocity_robot_frame: np.ndarray,
    ):
        """Calculate the footstep placement for swing trajectory.

        Args:
            i: Index of the leg (0-3)
            gait: Current gait object
            data: Control FSM data
            state_estimator_result: State estimator result
            desired_velocity_robot_frame: Desired velocity in robot frame (3x1 array)
        """
        # Set swing height
        self.foot_swing_trajectories[i].setHeight(self.body_height / 3)

        # The specified footstep names a spot on the ground that was picked out while
        # the foot was still on the other end of the swing, but this offset is executed
        # relative to the hip at *touchdown*: the state estimator is never handed a world
        # position (see StateEstimator.update), so the "global" frame everything here is
        # built in is really the instantaneous body frame. The body carries the hip
        # `v * swing_time` forward while the foot is in the air, so commanding the offset
        # as-is lands the foot that far past the spot that was chosen - 3 to 6 cm at our
        # speeds, which is several scan cells of whatever picked the footstep. Subtracting
        # the travel cancels it, since the target is evaluated against the hip at
        # touchdown: hip(touchdown) + offset - v*swing_time == hip(liftoff) + offset.
        #
        # Note this is the opposite sign to the Raibert style lead in the base class,
        # which pushes the foot *ahead* of the body to keep it balanced, rather than onto
        # one particular patch of ground.
        body_travel = np.zeros((3, 1), dtype=DTYPE)
        if self.compensate_body_travel:
            body_travel[:2] = (
                state_estimator_result.vBody[:2] * self.swing_times[i].item()
            )
            travel_distance = np.linalg.norm(body_travel)
            if travel_distance > self.max_body_travel_compensation:
                body_travel *= self.max_body_travel_compensation / travel_distance

        # Get the specified footstep location in the respective hip frame
        footstep_hip_frame = np.array(
            [
                self.footstep_locations_hip[i, 0] - body_travel[0, 0],
                self.footstep_locations_hip[i, 1] - body_travel[1, 0],
                # this should (roughly) put the foot in contact with the ground
                # assuming the body frame has this height in the world frame
                # there will be some sin error if the body is not horizontal
                # but that should be minimal
                -state_estimator_result.position[2, 0],
            ],
            dtype=DTYPE,
        ).reshape((3, 1))

        # Transform from hip frame to global frame
        # 1. Get hip position in robot body frame
        hip_position_body_frame = data._quadruped.getHipLocation(i)

        # 2. Transform footstep from hip frame to body frame
        # For most quadruped robots, the hip frame is aligned with the body frame
        # (same orientation, just translated), so we can directly add the position.
        # If there were any hip joint rotations to consider, they would be applied here.
        #
        # I'm not sure why, but this actually ends up in the world frame? or I have all
        # the frames mis-labeled. Either way this works. I'm just making up the frame
        # names as I go since the base code base wasn't very clear.
        foot_position_world_frame = hip_position_body_frame + footstep_hip_frame

        # 4. Add robot's global position to get final global position
        foot_position_global = (
            state_estimator_result.position + foot_position_world_frame
        )
        foot_position_global[2] = 0.0  # Project z down to zero

        self.foot_swing_trajectories[i].setFinalPosition(foot_position_global)

    def initiate_footstep(
        self,
        leg: int,
        location_hip: np.ndarray,
        duration: float,
    ):
        """initiates a footstep for the specified leg at the specified location

        Args:
            leg (int): Index of the leg (0-3)
            location_hip (np.ndarray): Desired foot position in the respective hip frame (x, y)
                This position is relative to the hip of the specified leg.
                z will be projected down to zero.
            duration (float): Duration of the footstep
        """
        # Store the position in the respective hip frame - x, y from input, z projected to zero
        self.footstep_locations_hip[leg] = location_hip
        self.swing_times[leg] = duration
        self.gait.initiate_footstep(leg, duration)
