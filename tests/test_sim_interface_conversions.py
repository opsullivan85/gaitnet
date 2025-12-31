import sys
from pathlib import Path

module_path = Path(__file__).parent.parent
sys.path.append(str(module_path))

import numpy as np

from gaitnet.sim2real import SimInterface

_control_joint_states = [
    ["FL_hip_pos", "FL_hip_vel"],
    ["FL_knee_pos", "FL_knee_vel"],
    ["FL_ankle_pos", "FL_ankle_vel"],
    ["FR_hip_pos", "FR_hip_vel"],
    ["FR_knee_pos", "FR_knee_vel"],
    ["FR_ankle_pos", "FR_ankle_vel"],
    ["RL_hip_pos", "RL_hip_vel"],
    ["RL_knee_pos", "RL_knee_vel"],
    ["RL_ankle_pos", "RL_ankle_vel"],
    ["RR_hip_pos", "RR_hip_vel"],
    ["RR_knee_pos", "RR_knee_vel"],
    ["RR_ankle_pos", "RR_ankle_vel"],
]
control_joint_states = np.asarray(_control_joint_states)

_control_joint_torques = [
    "FL_hip_torque",
    "FL_knee_torque",
    "FL_ankle_torque",
    "FR_hip_torque",
    "FR_knee_torque",
    "FR_ankle_torque",
    "RL_hip_torque",
    "RL_knee_torque",
    "RL_ankle_torque",
    "RR_hip_torque",
    "RR_knee_torque",
    "RR_ankle_torque",
]
control_joint_torques = np.asarray(_control_joint_torques)


_interface_joint_states = [
    [
        ["FL_hip_pos", "FL_hip_vel"],
        ["FL_knee_pos", "FL_knee_vel"],
        ["FL_ankle_pos", "FL_ankle_vel"],
    ],
    [
        ["FR_hip_pos", "FR_hip_vel"],
        ["FR_knee_pos", "FR_knee_vel"],
        ["FR_ankle_pos", "FR_ankle_vel"],
    ],
    [
        ["RL_hip_pos", "RL_hip_vel"],
        ["RL_knee_pos", "RL_knee_vel"],
        ["RL_ankle_pos", "RL_ankle_vel"],
    ],
    [
        ["RR_hip_pos", "RR_hip_vel"],
        ["RR_knee_pos", "RR_knee_vel"],
        ["RR_ankle_pos", "RR_ankle_vel"],
    ],
]
interface_joint_states = np.asarray(_interface_joint_states)


_interface_joint_torques = [
    [
        "FL_hip_torque",
        "FL_knee_torque",
        "FL_ankle_torque",
    ],
    [
        "FR_hip_torque",
        "FR_knee_torque",
        "FR_ankle_torque",
    ],
    [
        "RL_hip_torque",
        "RL_knee_torque",
        "RL_ankle_torque",
    ],
    [
        "RR_hip_torque",
        "RR_knee_torque",
        "RR_ankle_torque",
    ],
]
interface_joint_torques = np.asarray(_interface_joint_torques)


def test_convert_joint_states():
    joint_states = SimInterface._convert_joint_states(interface_joint_states)
    assert np.all(joint_states == control_joint_states)


def test_convert_torques():
    joint_torques = SimInterface._convert_torques(torques_control=control_joint_torques)
    assert np.all(joint_torques == interface_joint_torques)
