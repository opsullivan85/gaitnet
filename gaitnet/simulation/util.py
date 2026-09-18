import numpy as np
import torch

from isaaclab.scene import InteractiveScene
from gaitnet.sim2real.siminterface import SimInterface
from gaitnet.util.vectorpool import VectorPool


def isaac_joints_to_interface(
    joint_pos_isaac: np.ndarray, joint_vel_isaac: np.ndarray
) -> np.ndarray:
    """
    Convert Isaac Gym joint positions and velocities to the interface format.

    Parameters:
    - joint_pos_isaac: np.ndarray of shape (num_envs, 12)
    - joint_vel_isaac: np.ndarray of shape (num_envs, 12)

    Returns:
    - joint_states: np.ndarray of shape (num_envs, 4, 3, 2)
    """
    # Stack positions and velocities along the last axis
    joint_pos_vel = np.stack(
        [joint_pos_isaac, joint_vel_isaac], axis=-1
    )  # shape (:, 12, 2)

    # Reshape to (num_envs, 4, 3, 2) with Fortran-like index order
    joint_states_interface = joint_pos_vel.reshape(-1, 4, 3, 2, order="F")

    return joint_states_interface


def isaac_body_to_interface(body_state_isaac: np.ndarray) -> np.ndarray:
    """
    Convert Isaac Gym body position, orientation (quaternion), and velocity to the interface format.

    Parameters:
    - body_state_isaac: np.ndarray of shape (num_envs, 13)
        [
            pos_x, pos_y, pos_z,
            quat_w, quat_x, quat_y, quat_z,
            vel_x, vel_y, vel_z,
            omega_x, omega_y, omega_z
        ]

    Returns:
    - body_states_interface: np.ndarray of shape (num_envs, 13)
        [
            pos_x, pos_y, pos_z,
            quat_x, quat_y, quat_z, quat_w
            vel_x, vel_y, vel_z,
            omega_x, omega_y, omega_z
        ]
    """
    # move quatw to end and shift xyz to left
    body_states_interface = np.concatenate(
        [
            body_state_isaac[:, :3],  # pos_x, pos_y, pos_z
            body_state_isaac[:, 4:7],  # quat_x, quat_y, quat_z
            body_state_isaac[:, 3:4],  # quat_w
            body_state_isaac[:, 7:],  # rest
        ],
        axis=1,
    )

    return body_states_interface


def interface_to_isaac_torques(torques_interface: np.ndarray) -> np.ndarray:
    """
    Convert torques from the interface format back to Isaac Gym format.

    Parameters:
    - torques_interface: np.ndarray of shape (num_envs, 4, 3)

    Returns:
    - torques_isaac: np.ndarray of shape (num_envs, 12)
    """
    # Reshape to (num_envs, 12) with Fortran-like index order
    torques_isaac = torques_interface.reshape(-1, 12, order="F")

    return torques_isaac


def controls_to_joint_efforts(
    controls: np.ndarray, controllers: VectorPool, scene: InteractiveScene, asset_name: str = "robot"
) -> torch.Tensor:
    asset_data = scene[asset_name].data
    n_joint_pos = asset_data.joint_pos.shape[-1]
    n_joint_vel = asset_data.joint_vel.shape[-1]

    # concatenate on-GPU and do a single transfer instead of three, since each
    # separate .cpu() call forces its own CUDA sync
    combined = torch.cat(
        [asset_data.joint_pos, asset_data.joint_vel, asset_data.root_state_w], dim=-1
    ).cpu().numpy()
    joint_pos = combined[:, :n_joint_pos]
    joint_vel = combined[:, n_joint_pos : n_joint_pos + n_joint_vel]
    body_state = combined[:, n_joint_pos + n_joint_vel :]

    joint_states = isaac_joints_to_interface(joint_pos, joint_vel)
    body_state = isaac_body_to_interface(body_state)

    torques_interface = controllers.call(
        function=SimInterface.get_torques,
        mask=None,
        joint_states=joint_states,
        body_state=body_state,
        command=controls,
    )
    torques_isaac_np = interface_to_isaac_torques(torques_interface)
    torques_isaac = torch.from_numpy(torques_isaac_np).to(scene.device)
    return torques_isaac
