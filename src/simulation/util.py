import numpy as np
import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp import joint_pos
from isaaclab.scene import InteractiveScene
from gaitnet.sim2real import SimInterface
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
    joint_pos = scene[asset_name].data.joint_pos.cpu().numpy()
    joint_vel = scene[asset_name].data.joint_vel.cpu().numpy()
    joint_states = isaac_joints_to_interface(joint_pos, joint_vel)

    body_state = scene[asset_name].data.root_state_w.cpu().numpy()
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


def _ensure_shape(arr: torch.Tensor, expected_shape: tuple):
    if arr.shape == expected_shape[1:]:
        return arr.expand(expected_shape[0], -1)
    elif arr.shape == expected_shape:
        return arr
    else:
        raise ValueError(
            f"Array must have shape {expected_shape[1:]} or {expected_shape}, got {arr.shape}"
        )


def reset_all_to(
    env: ManagerBasedEnv,
    joint_pos_isaac: torch.Tensor,
    joint_vel_isaac: torch.Tensor,
    body_state_isaac: torch.Tensor,
):
    """
    Reset all environments in the ManagerBasedEnv to the specified joint positions, velocities, and body states.

    WARNING:
        If you try to pass the tensors as a reference to the env state, it will not work correctly.
        You MUST pass in a copy of the data. If you pass a reference, the env will modify the data
        when it resets, and you will not get the desired behavior.

    Parameters:
    - env: ManagerBasedEnv instance
    - joint_pos_isaac: torch.Tensor of shape (num_envs, 12), or if shape (12,), will be broadcasted
    - joint_vel_isaac: torch.Tensor of shape (num_envs, 12), or if shape (12,), will be broadcasted
    - body_state_isaac: torch.Tensor of shape (num_envs, 13), or if shape (13,), will be broadcasted
    """
    joint_pos_isaac = _ensure_shape(joint_pos_isaac, (env.num_envs, 12))
    joint_vel_isaac = _ensure_shape(joint_vel_isaac, (env.num_envs, 12))
    body_state_isaac = _ensure_shape(body_state_isaac, (env.num_envs, 13)) 

    # Reset the environments
    env.reset()

    # make sure controller is there
    if not hasattr(env.cfg, "controllers"):
        raise AttributeError("The env cfg does not have a controllers field.")
    if env.cfg.controllers is None:  # type: ignore
        raise ValueError("The env cfg controllers field is None.")
    
    # TODO: I'm 50% sure that we can just reset the controllers without
    # manually initializing the state. There may be issues with the contact history in the state estimator.
    controllers: VectorPool[SimInterface] = env.cfg.controllers  # type: ignore
    controllers.call(
        SimInterface.reset,
        mask=None,
    )

    # update base state
    env.scene["robot"].write_root_pose_to_sim(body_state_isaac[:, :7])
    env.scene["robot"].write_root_velocity_to_sim(body_state_isaac[:, 7:])

    # update joint state
    env.scene["robot"].write_joint_state_to_sim(joint_pos_isaac, joint_vel_isaac)

    # reset forces and torques
    env.scene["robot"].set_external_force_and_torque(
        torch.zeros((env.num_envs, env.scene["robot"].num_bodies, 3), device=env.scene.device),
        torch.zeros((env.num_envs, env.scene["robot"].num_bodies, 3), device=env.scene.device),
    )

    # write the changes to the simulator
    env.scene["robot"].write_data_to_sim()
