from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate Gaitnet")
parser.add_argument(
    "--difficulty",
    type=float,
    default=0.1,
    help="Terrain difficulty for the environment",
)
parser.add_argument(
    "--velocity", type=float, default=0.1, help="Base velocity for the environment"
)
parser.add_argument("--trials", type=int, default=2, help="Number of evaluation trials")
parser.add_argument(
    "--num_envs", type=int, default=50, help="Number of parallel environments to run"
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, unused_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(launcher_args=args_cli)
simulation_app = app_launcher.app

import torch
from gaitnet.eval.evaluator import Evaluator
from gaitnet.gaitnet.util import get_checkpoint_path
from isaaclab.terrains import TerrainGeneratorCfg
from gaitnet.gaitnet.components.gaitnet_env import GaitNetEnv, GaitNetObservationManager
from gaitnet.gaitnet.env_cfg.gaitnet_env_cfg import get_env, get_env_cfg, update_controllers
from gaitnet.util import log_exceptions
from gaitnet.gaitnet import gaitnet
import re
from pathlib import Path
import gaitnet.constants as const
from gaitnet.eval.components.fixed_velocity_command import (
    FixedVelocityCommand,
    FixedVelocityCommandCfg,
)
from gaitnet import GIT_COMMIT, get_logger
from gaitnet.util.pga import generate_footstep_action
from gaitnet.gaitnet.env_cfg.observations import get_terrain_mask
from gaitnet.simulation.cfg.footstep_scanner_constants import xy_to_idx

logger = get_logger()


def load_model(checkpoint_path: Path, device: torch.device) -> gaitnet.GaitnetActor:
    model = gaitnet.GaitnetActor(
        shared_state_dim=const.gait_net.robot_state_dim,
        shared_layer_sizes=[128, 128, 128],
        unique_state_dim=const.gait_net.footstep_option_dim,
        unique_layer_sizes=[64, 64],
        trunk_layer_sizes=[128, 128, 128],
    )
    agent = model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {
        re.sub(r"^actor\.", "", k): v
        for k, v in state_dict.items()
        if k.startswith("actor.")
    }
    agent.load_state_dict(state_dict)
    agent.to(device)
    return agent


def main():
    args_cli.device = "cpu"
    args_cli.num_envs = 7
    device = torch.device(args_cli.device)
    model = load_model(get_checkpoint_path(), device)
    model.eval()

    """Get the environment configuration and the environment instance."""
    env_cfg = get_env_cfg(args_cli.num_envs, args_cli.device)
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "yaw": (0, 0),
    }

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

    env = GaitNetEnv(cfg=env_cfg)
    update_controllers(env_cfg, args_cli.num_envs)
    observations, info = env.reset()
    obs: torch.Tensor = observations["policy"]  # type: ignore

    # format difficulty and speed without decimal points
    log_name = f"gaitnet_eval_d{args_cli.difficulty}_v{args_cli.velocity}_commit{GIT_COMMIT}.csv"
    evaluator = Evaluator(env, observations, trials=args_cli.trials, name=log_name)

    # with torch.inference_mode():
    while not evaluator.done:
        # Due to a long series of unfortunate design choices, we have to completley hack together the
        # custom footstep action generation here. Normally, a set of candidates is generated in the 
        # observation manager, but we need to generate them here to use PGA, so we inject them back
        # into the observation manager after generating them. This is why the env.step() call below
        # looks so stupid
        best_actions = torch.full(
            (args_cli.num_envs, 4), float("nan"), device=device
        )
        best_logits = torch.full(
            (args_cli.num_envs, ), float("-inf"), device=device
        )
        footstep_option_manager: "GaitNetObservationManager" = env.observation_manager
        base_obs = obs[:, :const.gait_net.robot_state_dim]
        # another unfortunate design choice, the observation manager deletes the terrain data, so we need to add
        # it back in here. In my defense, these decisions were made before these libaries supported dictionary observations,
        # so I just had one big observation tensor to work with.
        terrain_obs = torch.cat([base_obs, footstep_option_manager.most_recent_terrain_obs], dim=1)
        for leg in range(const.robot.num_legs):
            action_, logit_ = generate_footstep_action(
                state=base_obs,
                terrain_mask=get_terrain_mask(const.gait_net.valid_height_range, terrain_obs),
                leg=leg,
                gaitnet=model,
            pos_to_idx=xy_to_idx,
            )
            # update best actions and logits where applicable
            better_mask = logit_.squeeze(-1) > best_logits
            best_logits = torch.where(better_mask, logit_.squeeze(-1), best_logits)
            best_actions = torch.where(
                better_mask.unsqueeze(-1),
                action_,
                best_actions,
            )
        # inject the best actions into the observation manager
        footstep_option_manager.footstep_options = best_actions.unsqueeze(1)  # (num_envs, 1, 4)

        # this looks dumb because we manaully inject the actual actions into the observation manager
        # then just pass their index (all zeros) and duration here
        action_indices = torch.zeros((args_cli.num_envs,), device=device)
        action_durations = best_actions[:, 3]
        env_step_info = env.step(torch.stack((action_indices, action_durations), dim=-1))

        observations, rew, terminated, truncated, info = env_step_info
        obs = observations["policy"]  # type: ignore
        evaluator.process(env_step_info)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
