from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Evaluate Contactnet"
)
parser.add_argument(
    "--difficulty", type=float, default=0.1, help="Terrain difficulty for the environment"
)
parser.add_argument(
    "--velocity", type=float, default=0.1, help="Base velocity for the environment"
)
parser.add_argument(
    "--trials", type=int, default=2, help="Number of evaluation trials"
)
parser.add_argument(
    "--num_envs", type=int, default=50, help="Number of parallel environments to run"
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, unused_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from gaitnet.eval.components.contactnet_env import ContactNetEnv
from gaitnet.eval.evaluator import Evaluator
from gaitnet.gaitnet.util import get_checkpoint_path
from isaaclab.terrains import TerrainGeneratorCfg
from gaitnet.gaitnet.components.gaitnet_env import GaitNetEnv
from gaitnet.gaitnet.env_cfg.gaitnet_env_cfg import get_env, get_env_cfg, update_controllers
from gaitnet.util import log_exceptions
from gaitnet.gaitnet import actions, gaitnet
import re
from pathlib import Path
import gaitnet.constants as const
from gaitnet.eval.components.fixed_velocity_command import FixedVelocityCommand, FixedVelocityCommandCfg
from gaitnet.gaitnet.env_cfg.observations import contact_state_indices
from gaitnet import GIT_COMMIT, get_logger

logger = get_logger()


def get_actions(obs: torch.Tensor) -> torch.Tensor:
    # pre-populate actions with all no-ops
    no_op_index = const.robot.num_legs
    no_op = torch.tensor((no_op_index, 0), device=obs.device, dtype=torch.float)  # (leg_index, step_duration)
    actions = torch.stack([no_op] * obs.shape[0], dim=0)  # (num_envs, 2)

    # if all legs in contact, pick the non-no-op action with the lowest cost value
    contact_states = obs[:, contact_state_indices].bool()
    num_legs_in_contact = contact_states.sum(dim=1)
    full_contact_mask = num_legs_in_contact == const.robot.num_legs
    if not torch.any(full_contact_mask):
        return actions

    # get the unique states from the observation
    remaining_obs_size = obs.shape[1] - const.gait_net.robot_state_dim
    unique_states_dim = remaining_obs_size / const.gait_net.footstep_option_dim
    assert (
        unique_states_dim.is_integer()
    ), f"Expected unique_state_size ({const.gait_net.footstep_option_dim}) to evenly divide the remaining observation size ({remaining_obs_size}), got {unique_states_dim}"
    unique_states_dim = int(unique_states_dim)
    unique_states = obs[:, const.gait_net.robot_state_dim :].view(
        obs.shape[0], unique_states_dim, const.gait_net.footstep_option_dim
    )
    non_no_op_actions = unique_states[:, :-1, :]  # exclude no-op actions

    # get the index of the best actions
    action_costs = non_no_op_actions[:, :, -1]  # (num_envs, num_footstep_options-1)
    best_action_indices = torch.argmin(action_costs, dim=1)  # (num_envs,)
    actions[full_contact_mask, 0] = best_action_indices[full_contact_mask].float()
    actions[full_contact_mask, 1] = 0.2 # step duration
    return actions
    

def main():
    # args_cli.device = "cpu"
    # args_cli.num_envs = 50
    device = torch.device(args_cli.device)

    """Get the environment configuration and the environment instance."""
    env_cfg = get_env_cfg(args_cli.num_envs, args_cli.device)
    env_cfg.events.reset_base.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (0, 0)}

    # change terrain to all be same level and very long
    # over-ride control to be straight forward
    terrain_generator: TerrainGeneratorCfg = env_cfg.scene.terrain.terrain_generator  # type: ignore
    terrain_generator.difficulty_range = (args_cli.difficulty, args_cli.difficulty)
    terrain_generator.curriculum = False
    terrain_generator.size = (40, 1)
    terrain_generator.num_cols = args_cli.num_envs
    terrain_generator.num_rows = 1

    env_cfg.terminations.terrain_out_of_bounds.params["distance_buffer"] = 0.0

    env_cfg.commands.base_velocity = FixedVelocityCommandCfg(  # type: ignore
        command=(args_cli.velocity, 0, 0)
    )

    env = ContactNetEnv(cfg=env_cfg)
    update_controllers(env_cfg, args_cli.num_envs)
    observations, info = env.reset()
    obs: torch.Tensor = observations["policy"]  # type: ignore

    # format difficulty and speed without decimal points
    log_name = f"contactnet_eval_d{args_cli.difficulty}_v{args_cli.velocity}_commit{GIT_COMMIT}.csv"
    evaluator = Evaluator(env, observations, trials=args_cli.trials, name=log_name)

    with torch.inference_mode():
        while not evaluator.done:
            actions = get_actions(obs)
            env_step_info = env.step(actions)
            observations, rew, terminated, truncated, info = env_step_info
            obs = observations["policy"]  # type: ignore
            evaluator.process(env_step_info)
    
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    with log_exceptions(logger):
        main()