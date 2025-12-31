import torch
from isaaclab.assets import ArticulationData, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vmdp  # type: ignore
from isaaclab.utils import configclass
import math


def height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    data: ArticulationData = env.scene[asset_cfg.name].data
    # properly index the body positions for the specified body IDs
    body_pos = data.body_link_pos_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 3)
    heights = body_pos[:, :, 2]  # (num_envs, num_bodies)
    # if any body is below the minimum height, terminate
    dones = torch.any(heights < minimum_height, dim=1)
    return dones


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
    bad_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
        },
    )
    foot_below_ground = DoneTerm(
        func=height_below_minimum,
        params={
            "minimum_height": -0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    terrain_out_of_bounds = DoneTerm(
        func=vmdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 0.5},
        time_out=True,
    )
