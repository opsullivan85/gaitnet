import sys
from pathlib import Path

module_path = Path(__file__).parent.parent
sys.path.append(str(module_path))

from types import SimpleNamespace

import numpy as np
from nptyping import Float32, NDArray, Shape

import gaitnet.sim2real.siminterface as SimInterface
from gaitnet.util import VectorPool
from gaitnet.simulation.util import (
    interface_to_isaac_torques,
    isaac_body_to_interface,
    isaac_joints_to_interface,
)

isaac_joints = [
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_knee",
    "FR_knee",
    "RL_knee",
    "RR_knee",
    "FL_ankle",
    "FR_ankle",
    "RL_ankle",
    "RR_ankle",
]

isaac_body = [
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "omega_x",
    "omega_y",
    "omega_z",
]

control_torques = [
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

isaac_expected_torque = [
    "FL_hip_torque",
    "FR_hip_torque",
    "RL_hip_torque",
    "RR_hip_torque",
    "FL_knee_torque",
    "FR_knee_torque",
    "RL_knee_torque",
    "RR_knee_torque",
    "FL_ankle_torque",
    "FR_ankle_torque",
    "RL_ankle_torque",
    "RR_ankle_torque",
]

control_expected_dof_states = [
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

control_expected_body_states = [
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
    "vel_x",
    "vel_y",
    "vel_z",
    "omega_x",
    "omega_y",
    "omega_z",
]


def fake_robot_runner(
    dof_states: NDArray[Shape["12, 2"], Float32],
    body_states: NDArray[Shape["13"], Float32],
    commands: NDArray[Shape["3"], Float32],
) -> NDArray[Shape["12"], Float32]:
    assert np.all(dof_states == np.asarray(control_expected_dof_states))
    assert np.all(body_states == np.asarray(control_expected_body_states))
    return np.asarray(control_torques)


def test_mock_sim():
    args_cli = SimpleNamespace()
    args_cli.num_envs = 2

    sim = SimpleNamespace()
    sim.get_physics_dt = lambda: 0.05

    control_interface = VectorPool(
        dt=sim.get_physics_dt(),
        instances=args_cli.num_envs,
        cls=SimInterface.SimInterface,
        debug_logging=False,
    )
    for interface in control_interface.interfaces:
        interface.robot_runner.run = fake_robot_runner

    #### MOCK DATA
    # joint_pos = scene["robot"].data.joint_pos.cpu().numpy()
    joint_pos = [np.char.add(isaac_joints, "_pos") for i in range(args_cli.num_envs)]
    joint_pos = np.asarray(joint_pos)
    # joint_vel = scene["robot"].data.joint_vel.cpu().numpy()
    joint_vel = [np.char.add(isaac_joints, "_vel") for i in range(args_cli.num_envs)]
    joint_vel = np.asarray(joint_vel)
    # body_state = scene["robot"].data.root_state_w.cpu().numpy()
    body_state = [np.char.add("", isaac_body) for i in range(args_cli.num_envs)]
    body_state = np.asarray(body_state)

    joint_states = isaac_joints_to_interface(joint_pos, joint_vel)

    body_state = isaac_body_to_interface(body_state)

    command = np.zeros((args_cli.num_envs, 3), dtype=np.float32)

    torques_interface = control_interface.get_torques(
        joint_states=joint_states,
        body_states=body_state,
        commands=command,
    )
    torques_isaac = interface_to_isaac_torques(torques_interface)

    assert np.all(torques_isaac == isaac_expected_torque)
