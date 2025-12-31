import math

# import isaaclab.envs.mdp as mdp  # type: ignore
from isaaclab.envs.mdp import bad_orientation
from isaaclab.envs import mdp  # type: ignore
import torch
from isaaclab.envs import ManagerBasedEnv  # type: ignore
from isaaclab.managers import EventTermCfg as EventTerm  # type: ignore
from isaaclab.managers import TerminationTermCfg as DoneTerm  # type: ignore
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # type: ignore
from isaaclab.managers import ObservationTermCfg as ObsTerm  # type: ignore
from isaaclab.managers import RewardTermCfg as RewTerm  # type: ignore
from isaaclab.managers import SceneEntityCfg  # type: ignore
from isaaclab.utils import configclass  # type: ignore
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # type: ignore
from gaitnet.simulation.cfg.events import reset_controller


def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=constant_commands)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        FR_foot_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("FR_foot_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        FL_foot_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("FL_foot_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        RL_foot_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("RL_foot_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        RR_foot_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("RR_foot_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        # time = ObsTerm(func=mdp.current_time_s)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (0.9, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
    #         "mass_distribution_params": (-1.0, 3.0),
    #         "operation": "add",
    #     },
    # )

    # on reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0.0, 0.0),
    #             "yaw": (0.0, 0.0),
    #         },
    #     },
    # )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 0.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_controllers = EventTerm(
        func=reset_controller,
        mode="reset",
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": math.radians(20),
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
