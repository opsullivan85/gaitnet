import sys
from pathlib import Path

module_path = Path(__file__).parent.parent
sys.path.append(str(module_path))

import numpy as np

from gaitnet.simulation.util import (
    interface_to_isaac_torques,
    isaac_body_to_interface,
    isaac_joints_to_interface,
)

isaac_joints = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_knee_joint",
    "FR_knee_joint",
    "RL_knee_joint",
    "RR_knee_joint",
    "FL_ankle_joint",
    "FR_ankle_joint",
    "RL_ankle_joint",
    "RR_ankle_joint",
]

isaac_body_pos = [
    "pos_x",
    "pos_y",
    "pos_z",
]

isaac_body_quat = [
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
]

isaac_body_vel = [
    "vel_x",
    "vel_y",
    "vel_z",
]

isaac_body_omega = [
    "omega_x",
    "omega_y",
    "omega_z",
]


def test_interface_to_isaac_torques():
    num_robots = 2
    joints = [np.char.add(f"{i}_", isaac_joints) for i in range(num_robots)]
    joints = np.asarray(joints)
    joints = joints.reshape(-1, 4, 3, order="F")

    assert np.all(
        joints[0, 0]
        == np.asarray(["0_FL_hip_joint", "0_FL_knee_joint", "0_FL_ankle_joint"])
    ), "test data incorrect"

    interface_joints = interface_to_isaac_torques(joints)

    expected_joints = [np.char.add(f"{i}_", isaac_joints) for i in range(num_robots)]
    expected_joints = np.asarray(expected_joints)

    assert np.all(interface_joints == expected_joints), "torques don't match"


def test_isaac_body_to_interface():
    num_robots = 2
    isaac_body_state = (
        isaac_body_pos + isaac_body_quat + isaac_body_vel + isaac_body_omega
    )
    states = [np.char.add(f"{i}_", isaac_body_state) for i in range(num_robots)]
    states = np.asarray(states)

    assert np.all(
        states[0]
        == np.asarray(
            [
                "0_pos_x",
                "0_pos_y",
                "0_pos_z",
                "0_quat_w",
                "0_quat_x",
                "0_quat_y",
                "0_quat_z",
                "0_vel_x",
                "0_vel_y",
                "0_vel_z",
                "0_omega_x",
                "0_omega_y",
                "0_omega_z",
            ]
        )
    ), "test data incorrect"

    interface_states = isaac_body_to_interface(states)

    expected_states = np.asarray(
        [
            [
                "0_pos_x",
                "0_pos_y",
                "0_pos_z",
                "0_quat_x",
                "0_quat_y",
                "0_quat_z",
                "0_quat_w",
                "0_vel_x",
                "0_vel_y",
                "0_vel_z",
                "0_omega_x",
                "0_omega_y",
                "0_omega_z",
            ],
            [
                "1_pos_x",
                "1_pos_y",
                "1_pos_z",
                "1_quat_x",
                "1_quat_y",
                "1_quat_z",
                "1_quat_w",
                "1_vel_x",
                "1_vel_y",
                "1_vel_z",
                "1_omega_x",
                "1_omega_y",
                "1_omega_z",
            ],
        ]
    )

    assert np.all(interface_states == expected_states), "body states don't match"


def test_isaac_joint_to_interface():
    num_robots = 2
    joints = [np.char.add(f"{i}_", isaac_joints) for i in range(num_robots)]
    joints = np.asarray(joints)
    joint_pos_isaac = np.char.add(joints, "_pos")
    joint_vel_isaac = np.char.add(joints, "_vel")

    joint_states_interface = isaac_joints_to_interface(joint_pos_isaac, joint_vel_isaac)

    expected_states = np.asarray(
        [
            [
                [
                    ["0_FL_hip_joint_pos", "0_FL_hip_joint_vel"],
                    ["0_FL_knee_joint_pos", "0_FL_knee_joint_vel"],
                    ["0_FL_ankle_joint_pos", "0_FL_ankle_joint_vel"],
                ],
                [
                    ["0_FR_hip_joint_pos", "0_FR_hip_joint_vel"],
                    ["0_FR_knee_joint_pos", "0_FR_knee_joint_vel"],
                    ["0_FR_ankle_joint_pos", "0_FR_ankle_joint_vel"],
                ],
                [
                    ["0_RL_hip_joint_pos", "0_RL_hip_joint_vel"],
                    ["0_RL_knee_joint_pos", "0_RL_knee_joint_vel"],
                    ["0_RL_ankle_joint_pos", "0_RL_ankle_joint_vel"],
                ],
                [
                    ["0_RR_hip_joint_pos", "0_RR_hip_joint_vel"],
                    ["0_RR_knee_joint_pos", "0_RR_knee_joint_vel"],
                    ["0_RR_ankle_joint_pos", "0_RR_ankle_joint_vel"],
                ],
            ],
            [
                [
                    ["1_FL_hip_joint_pos", "1_FL_hip_joint_vel"],
                    ["1_FL_knee_joint_pos", "1_FL_knee_joint_vel"],
                    ["1_FL_ankle_joint_pos", "1_FL_ankle_joint_vel"],
                ],
                [
                    ["1_FR_hip_joint_pos", "1_FR_hip_joint_vel"],
                    ["1_FR_knee_joint_pos", "1_FR_knee_joint_vel"],
                    ["1_FR_ankle_joint_pos", "1_FR_ankle_joint_vel"],
                ],
                [
                    ["1_RL_hip_joint_pos", "1_RL_hip_joint_vel"],
                    ["1_RL_knee_joint_pos", "1_RL_knee_joint_vel"],
                    ["1_RL_ankle_joint_pos", "1_RL_ankle_joint_vel"],
                ],
                [
                    ["1_RR_hip_joint_pos", "1_RR_hip_joint_vel"],
                    ["1_RR_knee_joint_pos", "1_RR_knee_joint_vel"],
                    ["1_RR_ankle_joint_pos", "1_RR_ankle_joint_vel"],
                ],
            ],
        ]
    )

    assert np.all(joint_states_interface == expected_states), "joint states don't match"
