from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Evaluate Gaitnet"
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
app_launcher = AppLauncher(launcher_args=args_cli)
simulation_app = app_launcher.app

import torch
from src.eval.evaluator import Evaluator
from src.gaitnet.util import get_checkpoint_path
from isaaclab.terrains import TerrainGeneratorCfg
from src.gaitnet.components.gaitnet_env import GaitNetEnv
from src.gaitnet.env_cfg.gaitnet_env_cfg import get_env, get_env_cfg, update_controllers
from src.util import log_exceptions
from src.gaitnet import gaitnet
import re
from pathlib import Path
import src.constants as const
from src.eval.components.fixed_velocity_command import FixedVelocityCommand, FixedVelocityCommandCfg
from src import GIT_COMMIT, get_logger

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
    state_dict = {re.sub(r"^actor\.", "", k): v for k, v in state_dict.items() if k.startswith("actor.")}
    agent.load_state_dict(state_dict)
    agent.to(device)
    return agent


def main():
    # args_cli.device = "cpu"
    # args_cli.num_envs = 50
    device = torch.device(args_cli.device)
    model = load_model(get_checkpoint_path(), device)
    model.eval()

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

    env = GaitNetEnv(cfg=env_cfg)
    update_controllers(env_cfg, args_cli.num_envs)
    observations, info = env.reset()
    obs: torch.Tensor = observations["policy"]  # type: ignore

    # format difficulty and speed without decimal points
    log_name = f"gaitnet_eval_d{args_cli.difficulty}_v{args_cli.velocity}_commit{GIT_COMMIT}.csv"
    evaluator = Evaluator(env, observations, trials=args_cli.trials, name=log_name)

    with torch.inference_mode():
        while not evaluator.done:
            actions = gaitnet.GaitnetActor.act_inference(model, obs)
            env_step_info = env.step(actions)
            observations, rew, terminated, truncated, info = env_step_info
            obs = observations["policy"]  # type: ignore
            evaluator.process(env_step_info)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    with log_exceptions(logger):
        main()