# import math
import sys
import time

from gaitnet_mpc.mpc.common.StateEstimator import StateEstimate
import numpy as np
from gaitnet_mpc.mpc.common.FootSwingTrajectory import FootSwingTrajectory
from gaitnet_mpc.mpc.convex_MPC.Gait import GaitABC, OffsetDurationGait
from gaitnet_mpc.mpc.FSM_states.ControlFSMData import ControlFSMData
from gaitnet_mpc.mpc.Logger import Logger
from gaitnet_mpc.mpc.math_utils.orientation_tools import (
    CoordinateAxis,
    coordinateRotation,
)
from gaitnet_mpc.mpc.Parameters import Parameters
from gaitnet_mpc.mpc.utils import CASTING, DTYPE, NUM_LEGS, getSideSign

import gaitnet_mpc._mpc_osqp as mpc


class ConvexMPCLocomotion:
    def __init__(self, dt: float, iterations_between_mpc: int):
        self.iterations_between_mpc = int(iterations_between_mpc)
        self.horizon_length = 10  # a fixed number for all mpc gait
        self.dt = dt

        self.trotting = OffsetDurationGait(
            10 * 3,
            np.array([0 * 3, 5 * 3, 5 * 3, 0 * 3], dtype=DTYPE),
            np.array([5 * 3, 5 * 3, 5 * 3, 5 * 3], dtype=DTYPE),
            "Trotting",
        )

        self.bounding = OffsetDurationGait(
            10 * 3,
            np.array([5 * 3, 5 * 3, 0 * 3, 0 * 3], dtype=DTYPE),
            np.array([4 * 3, 4 * 3, 4 * 3, 4 * 3], dtype=DTYPE),
            "Bounding",
        )

        self.pronking = OffsetDurationGait(
            10 * 3,
            np.array([0 * 3, 0 * 3, 0 * 3, 0 * 3], dtype=DTYPE),
            np.array([4 * 3, 4 * 3, 4 * 3, 4 * 3], dtype=DTYPE),
            "Pronking",
        )

        self.pacing = OffsetDurationGait(
            10 * 3,
            np.array([5 * 3, 0 * 3, 5 * 3, 0 * 3], dtype=DTYPE),
            np.array([5 * 3, 5 * 3, 5 * 3, 5 * 3], dtype=DTYPE),
            "Pacing",
        )

        self.galloping = OffsetDurationGait(
            10 * 3,
            np.array([0 * 3, 2 * 3, 7 * 3, 9 * 3], dtype=DTYPE),
            np.array([4 * 3, 4 * 3, 4 * 3, 4 * 3], dtype=DTYPE),
            "Galloping",
        )

        self.walking = OffsetDurationGait(
            10 * 3,
            np.array([0 * 3, 3 * 3, 5 * 3, 8 * 3], dtype=DTYPE),
            np.array([5 * 3, 5 * 3, 5 * 3, 5 * 3], dtype=DTYPE),
            "Walking",
        )

        self.trot_running = OffsetDurationGait(
            10 * 3,
            np.array([0 * 3, 5 * 3, 5 * 3, 0 * 3], dtype=DTYPE),
            np.array([4 * 3, 4 * 3, 4 * 3, 4 * 3], dtype=DTYPE),
            "Trot Running",
        )

        self.mpc_dt = self.dt * self.iterations_between_mpc
        self.default_iterations_between_mpc = self.iterations_between_mpc
        # print(
        #     "[Convex MPC] dt: %.3f iterations: %d, dtMPC: %.3f"
        #     % (self.dt, self.iterations_between_mpc, self.mpc_dt)
        # )

        self.first_swing_flags = [True, True, True, True]
        self.is_first_run = True
        self.iteration_counter = 0
        self.foot_positions_global = np.zeros((4, 3, 1), dtype=DTYPE)
        self.feedforward_forces = np.zeros((4, 3, 1), dtype=DTYPE)

        self.foot_positions_body_frame = np.zeros((4, 3, 1), dtype=DTYPE)

        self.current_gait = 0
        self.desired_x_velocity = 0.0
        self.desired_y_velocity = 0.0
        self.desired_yaw_rate = 0.0

        self.desired_roll = 0.0
        self.desired_pitch = 0.0

        self.foot_swing_trajectories = [FootSwingTrajectory() for _ in range(4)]
        self.swing_times = np.zeros((4, 1), dtype=DTYPE)
        self.swing_time_remaining = [0.0 for _ in range(4)]

        self.kp_swing = np.array(
            [700, 0, 0, 0, 700, 0, 0, 0, 150], dtype=DTYPE
        ).reshape((3, 3))
        self.kd_swing = np.array([7, 0, 0, 0, 7, 0, 0, 0, 7], dtype=DTYPE).reshape(
            (3, 3)
        )
        self.kp_stance = np.zeros_like(self.kp_swing)
        self.kd_stance = self.kd_swing * 2

        self.logger = Logger("logs/")

    def initialize(self, data: ControlFSMData):
        if Parameters.cmpc_alpha > 1e-4:
            print(
                "Alpha was set too high ("
                + str(Parameters.cmpc_alpha)
                + ") adjust to 1e-5\n"
            )
            Parameters.cmpc_alpha = 1e-5

        if Parameters.cmpc_enable_log:
            # flush last log
            if not self.logger.is_empty():
                self.logger.flush_logging()
            # start new logs
            self.logger.start_logging()

        self.iteration_counter = 0
        self._cpp_mpc = mpc.ConvexMpc(
            data._quadruped._bodyMass,
            list(data._quadruped._bodyInertia),
            NUM_LEGS,
            self.horizon_length,
            self.mpc_dt,
            Parameters.cmpc_alpha,
            mpc.QPOASES,
        )

        self.desired_x_velocity = 0.0
        self.desired_y_velocity = 0.0
        self.desired_yaw_rate = 0.0
        self.first_swing_flags = [True for _ in range(4)]
        self.is_first_run = True

    def recompute_timing(self, iterations_per_mpc: int):
        self.iterations_between_mpc = iterations_per_mpc
        self.mpc_dt = self.dt * iterations_per_mpc

    def setup_command(self, data: ControlFSMData):

        self.body_height = data._quadruped._bodyHeight

        max_delta_v = Parameters.cmpc_max_linear_accel * self.dt
        self.desired_x_velocity += np.clip(
            data._desiredStateCommand.x_vel_cmd - self.desired_x_velocity,
            -max_delta_v, max_delta_v,
        )
        self.desired_y_velocity += np.clip(
            data._desiredStateCommand.y_vel_cmd - self.desired_y_velocity,
            -max_delta_v, max_delta_v,
        )

        self.desired_yaw_rate = data._desiredStateCommand.yaw_turn_rate

    def solve_dense_mpc(self, mpc_table: list, data: ControlFSMData):
        state_estimator_result = data._stateEstimator.getResult()

        # *MPC Weights
        if data._desiredStateCommand.mpc_weights is None:
            mpc_weight = data._quadruped._mpc_weights
        else:
            mpc_weight = data._desiredStateCommand.mpc_weights

        timer = time.time()

        # *Normal Vector of ground
        if Parameters.flat_ground:
            gravity_projection_vec = np.array([0, 0, 1], dtype=DTYPE)
        else:
            gravity_projection_vec = state_estimator_result.ground_normal_yaw

        # *Google's way of states
        com_roll_pitch_yaw = state_estimator_result.rpyBody.flatten()
        # com_roll_pitch_yaw = np.array([state_estimator_result.rpyBody[0], state_estimator_result.rpyBody[1], 0], dtype=DTYPE)
        com_position = state_estimator_result.position.flatten()
        com_angular_velocity = state_estimator_result.omegaBody.flatten()
        com_velocity = state_estimator_result.vBody.flatten()

        desired_com_position = np.array([0.0, 0.0, self.body_height], dtype=DTYPE)
        desired_com_velocity = np.array(
            [self.desired_x_velocity, self.desired_y_velocity, 0], dtype=DTYPE
        )
        desired_com_roll_pitch_yaw = np.zeros(
            3, dtype=DTYPE
        )  # walk parallel to the ground
        desired_com_angular_velocity = np.array(
            [0, 0, self.desired_yaw_rate], dtype=DTYPE
        )

        if Parameters.cmpc_print_states:
            print("------------------------------------------")
            print(
                "COM RPY: {: .4f}, {: .4f}, {: .4f}".format(
                    *np.rad2deg(com_roll_pitch_yaw)
                )
            )
            print("COM Pos: {: .4f}, {: .4f}, {: .4f}".format(*com_position))
            print("COM Ang: {: .4f}, {: .4f}, {: .4f}".format(*com_angular_velocity))
            print("COM Vel: {: .4f}, {: .4f}, {: .4f}".format(*com_velocity))
            # print("------------------------------------------")
            # print("DES RPY: {: .4f}, {: .4f}, {: .4f}".format(*np.rad2deg(desired_com_roll_pitch_yaw)))
            # print("DES Pos: {: .4f}, {: .4f}, {: .4f}".format(*desired_com_position))
            # print("DES Ang: {: .4f}, {: .4f}, {: .4f}".format(*desired_com_angular_velocity))
            # print("DES Vel: {: .4f}, {: .4f}, {: .4f}".format(*desired_com_velocity))
            print("------------------------------------------")
            print("GND Vec: {: .4f}, {: .4f}, {: .4f}".format(*gravity_projection_vec))

        predicted_contact_forces = self._cpp_mpc.compute_contact_forces(
            mpc_weight,  # mpc weights list(12,)
            com_position,  # com_position (set x y to 0.0)
            com_velocity,  # com_velocity
            com_roll_pitch_yaw,  # com_roll_pitch_yaw (set yaw to 0.0)
            gravity_projection_vec,  # Normal Vector of ground
            com_angular_velocity,  # com_angular_velocity
            np.asarray(mpc_table, dtype=DTYPE),  # Foot contact states
            np.array(
                self.foot_positions_body_frame.flatten(), dtype=DTYPE
            ),  # foot_positions_base_frame
            data._quadruped._friction_coeffs,  # foot_friction_coeffs
            desired_com_position,  # desired_com_position
            desired_com_velocity,  # desired_com_velocity
            desired_com_roll_pitch_yaw,  # desired_com_roll_pitch_yaw
            desired_com_angular_velocity,  # desired_com_angular_velocity
        )
        for leg in range(4):
            self.feedforward_forces[leg] = np.array(
                predicted_contact_forces[leg * 3 : (leg + 1) * 3], dtype=DTYPE
            ).reshape((3, 1))

        if Parameters.cmpc_print_update_time:
            print("MPC Update Time %.3f s\n" % (time.time() - timer))

        if Parameters.cmpc_enable_log:
            mpc_state_loss = (
                (com_roll_pitch_yaw - desired_com_roll_pitch_yaw).dot(mpc_weight[0:3])
                + (com_position - desired_com_position).dot(mpc_weight[3:6])
                + (com_angular_velocity - desired_com_velocity).dot(mpc_weight[6:9])
                + (com_velocity - desired_com_velocity).dot(mpc_weight[9:12])
            )

            mpc_torque_loss = Parameters.cmpc_alpha * np.sum(
                predicted_contact_forces[:12]
            )

            log_data_frame = dict(
                COM_RPY=com_roll_pitch_yaw,  # COM_RPY
                COM_POS=com_position,  # COM_POS
                COM_ANG=com_angular_velocity,  # COM_ANG
                COM_VEL=com_velocity,  # COM_VEL
                DES_RPY=desired_com_roll_pitch_yaw,  # DES_RPY
                DES_POS=desired_com_position,  # DES_POS
                DES_ANG=desired_com_angular_velocity,  # DES_ANG
                DES_VEL=desired_com_velocity,  # DES_VEL
                MPC_GRF=predicted_contact_forces[:12],  # MPC_GRF
                MPC_LOS=mpc_state_loss + mpc_torque_loss,  # MPC_LOS
                MPC_WEI=mpc_weight,  # MPC_WEI
                TIM_STA=self.iteration_counter,  # TIM_STA
            )
            self.logger.update_logging(log_data_frame)

    def update_mpc_if_needed(self, mpc_table: list, data: ControlFSMData):
        # self.solve_dense_mpc(mpc_table, data)
        if (self.iteration_counter % self.iterations_between_mpc) == 0:
            self.solve_dense_mpc(mpc_table, data)

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
        if self.first_swing_flags[i]:
            self.swing_time_remaining[i] = self.swing_times[i].item()
        else:
            self.swing_time_remaining[i] -= self.dt

        # Set swing height
        self.foot_swing_trajectories[i].setHeight(self.body_height / 3)

        # Calculate hip offset and foot position in robot frame
        hip_offset = np.array(
            [0, getSideSign(i) * data._quadruped._abadLinkLength, 0], dtype=DTYPE
        ).reshape((3, 1))
        foot_position_robot_frame = data._quadruped.getHipLocation(i) + hip_offset

        # Apply yaw correction for stance time
        stance_time = gait.getCurrentStanceTime(self.mpc_dt, i)
        foot_position_yaw_corrected = (
            coordinateRotation(
                CoordinateAxis.Z, -self.desired_yaw_rate * stance_time / 2
            )
            @ foot_position_robot_frame
        )

        # Calculate basic foot position in global frame
        foot_position_global = state_estimator_result.position + (
            foot_position_yaw_corrected
            + desired_velocity_robot_frame * self.swing_time_remaining[i]
        )

        # Calculate relative position offsets for better tracking
        max_relative_position = 0.3
        foot_x_offset_relative = (
            state_estimator_result.vBody[0]
            * (0.5 + Parameters.cmpc_bonus_swing)
            * stance_time
            + 0.03 * (state_estimator_result.vBody[0] - desired_velocity_robot_frame[0])
            + (0.5 * state_estimator_result.position[2] / 9.81)
            * (state_estimator_result.vBody[1] * self.desired_yaw_rate)
        )

        foot_y_offset_relative = (
            state_estimator_result.vBody[1] * 0.5 * stance_time * self.mpc_dt
            + 0.03 * (state_estimator_result.vBody[1] - desired_velocity_robot_frame[1])
            + (0.5 * state_estimator_result.position[2] / 9.81)
            * (-state_estimator_result.vBody[0] * self.desired_yaw_rate)
        )

        # Clamp offsets to prevent extreme foot placement
        foot_x_offset_relative = min(
            max(foot_x_offset_relative, -max_relative_position), max_relative_position
        )
        foot_y_offset_relative = min(
            max(foot_y_offset_relative, -max_relative_position), max_relative_position
        )

        # Apply offsets and set final position
        foot_position_global[0] += foot_x_offset_relative
        foot_position_global[1] += foot_y_offset_relative
        foot_position_global[2] = -0.003

        self.foot_swing_trajectories[i].setFinalPosition(foot_position_global)

    def _get_gait(self, gait_number: int) -> GaitABC:
        # pick gait
        gait = self.trotting
        if gait_number == 1:
            gait = self.bounding
        elif gait_number == 2:
            gait = self.pronking
        elif gait_number == 3:
            gait = self.pacing
        elif gait_number == 5:
            gait = self.galloping
        elif gait_number == 6:
            gait = self.walking
        elif gait_number == 7:
            gait = self.trot_running
        return gait

    def run(self, data: ControlFSMData):
        # Command Setup
        self.setup_command(data)
        gait_number = Parameters.cmpc_gait.value
        state_estimator_result = data._stateEstimator.getResult()

        gait = self._get_gait(gait_number)

        self.current_gait = gait_number
        gait.setIterations(self.iterations_between_mpc, self.iteration_counter)

        self.recompute_timing(self.default_iterations_between_mpc)

        for i in range(4):
            self.foot_positions_body_frame[i] = (
                data._quadruped.getHipLocation(i) + data._legController.datas[i].p
            )
            self.foot_positions_global[i] = (
                self.foot_positions_body_frame[i] + state_estimator_result.position
            )
            # np.copyto(self.foot_positions_global[i], state_estimator_result.position + \
            # (data._quadruped.getHipLocation(i)+
            # data._legController.datas[i].p))
        # self.foot_positions_body_frame = np.array([self.foot_positions_global[i] - state_estimator_result.position for i in range(4)], dtype=DTYPE).reshape((4,3,1))

        # * first time initialization
        if self.is_first_run:
            self.is_first_run = False
            data._stateEstimator._init_contact_history(self.foot_positions_body_frame)
            for i in range(4):
                self.foot_swing_trajectories[i].setHeight(0.05)
                self.foot_swing_trajectories[i].setInitialPosition(
                    self.foot_positions_global[i]
                )
                self.foot_swing_trajectories[i].setFinalPosition(
                    self.foot_positions_global[i]
                )

        if Parameters.flat_ground:
            data._stateEstimator._update_com_position_ground_frame(
                self.foot_positions_body_frame
            )
        else:
            data._stateEstimator._compute_ground_normal_and_com_position(
                self.foot_positions_body_frame
            )

        # * foot placement
        for leg_idx in range(4):
            self.swing_times[leg_idx] = gait.getCurrentSwingTime(self.mpc_dt, leg_idx)

        desired_velocity_robot_frame = np.array(
            [self.desired_x_velocity, self.desired_y_velocity, 0], dtype=DTYPE
        ).reshape((3, 1))

        for i in range(4):
            self._update_footstep_placement(
                i=i,
                gait=gait,
                data=data,
                state_estimator_result=state_estimator_result,
                desired_velocity_robot_frame=desired_velocity_robot_frame,
            )

        # calc gait
        self.iteration_counter += 1

        # gait
        contact_states = gait.getContactPhase()
        swing_states = gait.getSwingPhase()
        mpc_table = gait.getMpcTable()

        # * update MPC
        self.update_mpc_if_needed(mpc_table, data)

        state_estimator_contact_state = np.array([0, 0, 0, 0], dtype=DTYPE).reshape(
            (4, 1)
        )

        for foot in range(4):
            contact_state = contact_states[foot]
            swing_state = swing_states[foot]
            if swing_state > 0:  # * foot is in swing
                if self.first_swing_flags[foot]:
                    self.first_swing_flags[foot] = False
                    self.foot_swing_trajectories[foot].setInitialPosition(
                        self.foot_positions_global[foot]
                    )

                self.foot_swing_trajectories[foot].computeSwingTrajectoryBezier(
                    swing_state, self.swing_times[foot].item()
                )
                desired_foot_position_global = self.foot_swing_trajectories[
                    foot
                ].getPosition()
                desired_foot_velocity_global = self.foot_swing_trajectories[
                    foot
                ].getVelocity()

                desired_leg_position = (
                    desired_foot_position_global - state_estimator_result.position
                ) - data._quadruped.getHipLocation(foot)
                desired_leg_velocity = (
                    desired_foot_velocity_global - state_estimator_result.vBody
                )

                # data._legController.commands[foot].pDes = desired_leg_position
                # data._legController.commands[foot].vDes = desired_leg_velocity
                # data._legController.commands[foot].kpCartesian = self.position_gains_swing
                # data._legController.commands[foot].kdCartesian = self.damping_gains_swing

                np.copyto(
                    data._legController.commands[foot].pDes,
                    desired_leg_position,
                    casting=CASTING,
                )
                np.copyto(
                    data._legController.commands[foot].vDes,
                    desired_leg_velocity,
                    casting=CASTING,
                )
                np.copyto(
                    data._legController.commands[foot].kpCartesian,
                    self.kp_swing,
                    casting=CASTING,
                )
                np.copyto(
                    data._legController.commands[foot].kdCartesian,
                    self.kd_swing,
                    casting=CASTING,
                )
                np.copyto(
                    data._legController.commands[foot].forceFeedForward,
                    self.feedforward_forces[foot] * 0.0,
                    casting=CASTING,
                )
                np.copyto(
                    data._legController.commands[foot].kdJoint,
                    np.identity(3, dtype=DTYPE) * 0.0,
                    casting=CASTING,
                )

            else:  # * foot is in stance
                self.first_swing_flags[foot] = True
                desired_foot_position_global = self.foot_swing_trajectories[
                    foot
                ].getPosition()
                desired_foot_velocity_global = self.foot_swing_trajectories[
                    foot
                ].getVelocity()

                desired_leg_position = (
                    desired_foot_position_global - state_estimator_result.position
                ) - data._quadruped.getHipLocation(foot)
                desired_leg_velocity = (
                    desired_foot_velocity_global - state_estimator_result.vBody
                )

                np.copyto(
                    data._legController.commands[foot].pDes,
                    desired_leg_position,
                    casting=CASTING,
                )
                np.copyto(
                    data._legController.commands[foot].vDes,
                    desired_leg_velocity,
                    casting=CASTING,
                )
                np.copyto(
                    data._legController.commands[foot].kpCartesian,
                    self.kp_stance,
                    casting=CASTING,
                )
                np.copyto(
                    data._legController.commands[foot].kdCartesian,
                    self.kd_stance,
                    casting=CASTING,
                )
                np.copyto(
                    data._legController.commands[foot].forceFeedForward,
                    self.feedforward_forces[foot],
                    casting=CASTING,
                )
                np.copyto(
                    data._legController.commands[foot].kdJoint,
                    np.identity(3, dtype=DTYPE) * 0.6,
                    casting=CASTING,
                )

                state_estimator_contact_state[foot] = contact_state

        data._stateEstimator.setContactPhase(state_estimator_contact_state)
