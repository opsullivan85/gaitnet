from isaaclab.managers import EventTermCfg as EventTerm, termination_manager
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vmdp  # type: ignore
from isaaclab.envs import mdp

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, terrain_generator
from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    terrain: TerrainImporter = env.scene.terrain  # type: ignore
    termination_manager = env.termination_manager

    move_down_given_failure = torch.rand(len(env_ids), device=env.device) < 0.5  # P_d
    move_up_given_success = torch.rand(len(env_ids), device=env.device) < 0.1  # P_u
    # Given the two parameters above, we can expect the system to settle into
    # a steady state success rate while the terrain levels are changing.
    # The level will be $\frac{P_d}{P_u + P_d}$
    # The magnitude of these combined with the magnitude of noise will determine
    # how quickly the terrain levels change, and the variance around the mean level.
    
    # random chance to move up or down regardless of success/failure
    random_noise = torch.rand(len(env_ids), device=env.device) < 0.02

    # move up with some probability if you survived the whole time
    success = termination_manager.time_outs[env_ids]
    move_up = success & move_up_given_success
    # randomly move up with a small probability
    move_up = move_up | random_noise

    # move down with some probability if you terminated early
    terminated_early = termination_manager.terminated[env_ids]
    move_down = terminated_early & move_down_given_failure
    # randomly move down with a small probability
    move_down = move_down | random_noise

    # update terrain levels
    # note that move_up and move_down are not mutually exclusive because of the noise
    # in this case terrain.update_env_origins won't move
    terrain.update_env_origins(env_ids, move_up, move_down)

    # return the mean difficulty
    difficulty_range: tuple[float, float] = terrain.cfg.terrain_generator.difficulty_range  # type: ignore
    mean_terrain_level = torch.mean(terrain.terrain_levels.float())
    max_terrain_level = terrain.max_terrain_level
    # scale terrain level from [0, max_terrain_level] to [difficulty_range[0], difficulty_range[1]]
    mean_difficulty = difficulty_range[0] + (
        mean_terrain_level / max_terrain_level
    ) * (difficulty_range[1] - difficulty_range[0])
    return mean_difficulty


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # in the beginning we reward always moving a foot to teach
    # the model when to use/not the no-op action. Once it has
    # learned this, we remove the reward.
    # remove_foot_in_swing_reward = CurrTerm(
    #     func=mdp.modify_reward_weight,  # type: ignore
    #     params={
    #         "term_name": "a_foot_in_swing",
    #         "weight": 0,
    #         "num_steps": 10000,
    #     },
    # )

    # once the model has learned to move a foot, we add a small penalty on joint torques
    # to encourage energy efficiency
    # add_joint_torque_penalty = CurrTerm(
    #     func=mdp.modify_reward_weight,  # type: ignore
    #     params={
    #         "term_name": "joint_accelerations",
    #         "weight": -3e-8,
    #         "num_steps": 10000,
    #     },
    # )

    # # once the robot learns to walk, reward it for moving it's feet less
    # add_no_op_reward = CurrTerm(
    #     func=mdp.modify_reward_weight,  # type: ignore
    #     params={
    #         "term_name": "no_op",
    #         "weight": 0.1,
    #         "num_steps": 15000,
    #     },
    # )

    terrain_levels = CurrTerm(func=vmdp.terrain_levels_vel)
    # terrain_levels = CurrTerm(func=terrain_levels_vel)  # type: ignore
