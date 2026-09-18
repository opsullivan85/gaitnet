import numpy as np
import torch
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_shape,
)
import torch.nn.functional as F
import isaaclab.utils.math as math_utils
from gaitnet_mpc.pool import VectorPool
from gaitnet.sim2real.abstractinterface import Sim2RealInterface
from gaitnet import get_logger
import gaitnet.constants as const

logger = get_logger()


@generic_io_descriptor(
    units="m",
    axes=["X", "Y", "Z"],
    observation_type="RootState",
    on_inspect=[record_shape, record_dtype],
)
def foot_position_xy_b(
    env: ManagerBasedEnv,
    transform_name: SceneEntityCfg = SceneEntityCfg("foot_transforms"),
    flatten: bool = False,
) -> torch.Tensor:
    """Get foot xy positions in the base frame.
        Assumes transform_name sensor exists in the scene, ex:

    .. code-block:: python

        foot_transforms = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk",
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/FL_foot"),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/FR_foot"),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/RL_foot"),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/RR_foot"),
            ],
            debug_vis=False,
        )

    Args:
        env: The environment instance.
        transform_name: The name of the FrameTransformer sensor in the scene.
        flatten: Whether to flatten the output to (N, 8) instead of (N, 4, 2).
    """
    foot_positions_b = env.scene[transform_name.name].data.target_pos_source[:, :, :2]
    if flatten:
        foot_positions_b = foot_positions_b.reshape(foot_positions_b.shape[0], -1)
    return foot_positions_b


@generic_io_descriptor(
    units="m",
    observation_type="RootState",
    on_inspect=[record_shape, record_dtype],
)
def foot_position_z_b(
    env: ManagerBasedEnv,
    transform_name: SceneEntityCfg = SceneEntityCfg("foot_transforms"),
) -> torch.Tensor:
    """Get foot z positions in the base frame, (N, 4).

    Uses the same FrameTransformer sensor as `foot_position_xy_b`.
    """
    return env.scene[transform_name.name].data.target_pos_source[:, :, 2]


@generic_io_descriptor(
    units="m/s",
    observation_type="BodyState",
    on_inspect=[record_shape, record_dtype],
)
def foot_velocity_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, flatten: bool = False
) -> torch.Tensor:
    """Get foot velocities relative to the base, expressed in the base frame.

    This is the quantity leg kinematics measure on hardware (J(q) * qd), so it
    excludes the base's own motion: a planted foot reads -(v_base + w x r_foot).

    Args:
        env: The environment instance.
        asset_cfg: The robot, with body_names set to the feet in the desired order.
        flatten: Whether to flatten the output to (N, 3 * num_feet) instead of (N, num_feet, 3).
    """
    data = env.scene[asset_cfg.name].data
    foot_pos_w = data.body_link_pos_w[:, asset_cfg.body_ids]
    foot_vel_w = data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    base_ang_vel_w = data.root_link_ang_vel_w.unsqueeze(1)
    foot_offset_w = foot_pos_w - data.root_link_pos_w.unsqueeze(1)
    relative_vel_w = (
        foot_vel_w
        - data.root_link_lin_vel_w.unsqueeze(1)
        - torch.cross(base_ang_vel_w.expand_as(foot_offset_w), foot_offset_w, dim=-1)
    )
    base_quat_w = data.root_link_quat_w.unsqueeze(1).expand(-1, relative_vel_w.shape[1], -1)
    relative_vel_b = math_utils.quat_apply_inverse(base_quat_w, relative_vel_w)
    if flatten:
        relative_vel_b = relative_vel_b.reshape(relative_vel_b.shape[0], -1)
    return relative_vel_b


@generic_io_descriptor(
    observation_type="RootState",
    on_inspect=[record_shape, record_dtype],
)
def contact_state_sensors(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Get the measured contact state from a contact sensor, (N, num_bodies) float.

    Args:
        env: The environment instance.
        sensor_cfg: The ContactSensor, with body_names set to the feet in the desired order.
    """
    sensor = env.scene[sensor_cfg.name]
    contact_forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    contacts = contact_forces.norm(dim=-1) > sensor.cfg.force_threshold
    return contacts.float()


@generic_io_descriptor(
    observation_type="RootState",
    on_inspect=[record_shape, record_dtype],
)
def gait_timing_controller(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the scheduled gait timing from the controller, (N, 12).

    Feature grouped, legs in FL, FR, RL, RR order:
    [swing phase (4), remaining swing time (4), time since touchdown (4)].
    Time since touchdown is clipped to `const.gait_net.max_stance_time_obs`.
    """
    controllers: VectorPool[Sim2RealInterface] = env.cfg.robot_controllers  # type: ignore

    # see contact_state_controller
    if controllers is None:
        logger.warning(
            "Controllers are not initialized, returning fake data. Normal 1 time only."
        )
        return torch.zeros((env.num_envs, 3 * const.robot.num_legs), device=env.device)

    timing: np.ndarray = controllers.call(
        Sim2RealInterface.get_gait_timing, mask=None
    )  # (N, 4, 3), already in FL, FR, RL, RR order
    timing[:, :, 2] = np.minimum(timing[:, :, 2], const.gait_net.max_stance_time_obs)
    # (N, 4, 3) -> (N, 3, 4) so the output is grouped by feature
    timing = np.ascontiguousarray(timing.transpose(0, 2, 1)).reshape(timing.shape[0], -1)
    return torch.from_numpy(timing).to(env.device)


@generic_io_descriptor(
    observation_type="RootState",
    on_inspect=[record_shape, record_dtype],
)
def contact_state_controller(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the contact state from the controller."""
    controllers: VectorPool[Sim2RealInterface] = env.cfg.robot_controllers  # type: ignore

    # controllers won't be initilized when the on_inspect is called, in that case return fake data.
    # this should only happen once in the beginning of training when the env is created
    if controllers is None:
        logger.warning(
            "Controllers are not initialized, returning fake data. Normal 1 time only."
        )
        return torch.zeros((env.num_envs, const.robot.num_legs), device=env.device, dtype=torch.bool)

    contacts: np.ndarray = controllers.call(
        Sim2RealInterface.get_contact_state, mask=None
    )  # already in FL, FR, RL, RR order
    # logger.info(f"contact: {contacts[0]}")
    contacts_gpu = torch.from_numpy(contacts).to(env.device)
    return contacts_gpu


def cspace_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    Note that the input shape (const.footstep_scanner.sensor_grid_size) is larger than the output shape
    (const.footstep_scanner.grid_size) to account for c-space dialation.

    assumes all sensor_cfgs point to RayCaster sensors with the same grid size

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    height_scan = mdp.height_scan(env=env, sensor_cfg=sensor_cfg, offset=offset)
    # reshape to (N, H, W)
    height_scan = height_scan.reshape((-1, *const.footstep_scanner.sensor_grid_size))

    # apply cspace dialation
    kernel_size = const.gait_net.cspace_dialation * 2 + 1
    # TODO: maybe we consider expanding our sensor size by padding
    # so we aren't getting misleading values at the edges?
    padding = 0
    height_scan = F.max_pool2d(
        height_scan, kernel_size=kernel_size, stride=1, padding=padding
    )

    assert height_scan.shape[1:] == tuple(const.footstep_scanner.grid_size), (
        f"Expected height scan shape to be {const.footstep_scanner.grid_size}, "
        f"but got {height_scan.shape[1:]}"
    )

    # flatten to (N, H*W)
    height_scan = height_scan.reshape(height_scan.shape[0], -1)
    # replace any -inf or inf with 1.0 (this roughly corresponds to a void in the terrain)
    height_scan[height_scan == -float("inf")] = 1.0
    height_scan[height_scan == float("inf")] = 1.0
    return height_scan


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        foot_position_xy_b = ObsTerm(
            func=foot_position_xy_b,
            noise=Unoise(n_min=-0.01, n_max=0.01),  # roughly 1/20 of the typical range
            params={"flatten": True},
        )

        base_pos_z = ObsTerm(
            func=mdp.base_pos_z,
            noise=Unoise(n_min=-0.01, n_max=0.01),  # just made up this number
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(
                n_min=-0.01, n_max=0.01
            ),  # roughly 1/10 of the max control input
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(
                n_min=-0.02, n_max=0.02
            ),  # roughly 1/10 of the max control input
        )

        control = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # the term order here defines the layout documented in observations_utils

        # measured contact, to match deployment where contact is estimated rather than scheduled
        contact_state_sensor = ObsTerm(
            func=contact_state_sensors,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                    preserve_order=True,
                )
            },
        )
        # contact_state_controller = ObsTerm(
        #     func=contact_state_controller,
        #     params={},
        # )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )

        foot_position_z_b = ObsTerm(
            func=foot_position_z_b,
            noise=Unoise(n_min=-0.01, n_max=0.01),  # matches foot_position_xy_b
        )

        foot_velocity_b = ObsTerm(
            func=foot_velocity_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                    preserve_order=True,
                ),
                "flatten": True,
            },
        )

        gait_timing_controller = ObsTerm(
            func=gait_timing_controller,
        )

        # scanners are last, in FL, FR, RL, RR order so terrain channel i is footstep option leg i
        FL_foot_scanner = ObsTerm(
            func=cspace_height_scan,
            params={"sensor_cfg": SceneEntityCfg("FL_foot_scanner")},
        )

        FR_foot_scanner = ObsTerm(
            func=cspace_height_scan,
            params={"sensor_cfg": SceneEntityCfg("FR_foot_scanner")},
        )

        RL_foot_scanner = ObsTerm(
            func=cspace_height_scan,
            params={"sensor_cfg": SceneEntityCfg("RL_foot_scanner")},
        )

        RR_foot_scanner = ObsTerm(
            func=cspace_height_scan,
            params={"sensor_cfg": SceneEntityCfg("RR_foot_scanner")},
        )


        def __post_init__(self):
            self.enable_corruption = False
            # self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


