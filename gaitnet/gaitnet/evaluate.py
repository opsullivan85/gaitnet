from gaitnet import setup_logging

setup_logging()

from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Train an RL agent with Stable-Baselines3."
)
parser.add_argument(
    "--num_envs", type=int, default=100, help="Number of environments to simulate."
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=10000, help="RL Policy training iterations."
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint to resume training from.",
)
from gaitnet.gaitnet.util import add_checkpoint_arg

add_checkpoint_arg(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, unused_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from gaitnet.gaitnet.util import get_checkpoint_path
from gaitnet.gaitnet.components.gaitnet_env import GaitNetEnv
from gaitnet.gaitnet.components.gaitnet_observation_manager import (
    GaitNetObservationManager,
)
from gaitnet.gaitnet.env_cfg.gaitnet_env_cfg import get_env
from gaitnet.util import log_exceptions
from gaitnet.gaitnet.dense_eval import dense_actions, dense_candidates, load_actor, policy_obs
from gaitnet.gaitnet import gaitnet
from pathlib import Path
import gaitnet.constants as const
from gaitnet import get_logger
from gaitnet import PROJECT_ROOT

data_path = PROJECT_ROOT / "data" / "contact_schedule" / "contact_schedule.csv"
data_path.parent.mkdir(parents=True, exist_ok=True)

logger = get_logger()


def load_model(checkpoint_path: Path, device: torch.device, deterministic: bool):
    actor = load_actor(checkpoint_path, device)
    if deterministic:
        return actor
    # keep the learned duration std for stochastic rollouts; the critic isn't needed
    agent = gaitnet.GaitnetActorCritic(0, 0, 0, actor, None)  # type: ignore
    state_dict = torch.load(checkpoint_path, map_location=device)["model_state_dict"]
    agent.duration_log_std.data.copy_(state_dict["duration_log_std"])
    return agent.to(device)

def log_action(actions: torch.Tensor, env: GaitNetEnv):
    fsc = env.action_manager.get_term("footstep_controller")  # type: ignore
    action_data = fsc.action_indices_to_actions(actions)[0].cpu().numpy()
    with open(data_path, "a") as f:
        f.write(f"{action_data[0]},{action_data[1]},{action_data[2]},{action_data[3]}\n")



def main():
    args_cli.device = "cpu"
    deterministic = False
    args_cli.num_envs = 1
    device = torch.device(args_cli.device)
    model = load_model(get_checkpoint_path(args_cli.checkpoint_name), device, deterministic=deterministic)
    model.eval()

    env = get_env(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        manager_class=GaitNetEnv,
    )
    obs, info = env.reset()

    with torch.inference_mode():
        while True:
            # score every cell of every leg rather than the sampler's random subset
            option_manager: GaitNetObservationManager = env.observation_manager  # type: ignore
            # the observation manager replaces the terrain scan with the footstep
            # options it sampled, so rebuild the raw observation dense sampling needs
            raw_obs = torch.cat(
                [
                    obs["policy"][:, : const.gait_net.robot_state_dim],
                    option_manager.most_recent_terrain_obs,
                ],
                dim=1,
            )
            candidates = dense_candidates(raw_obs)
            # the action term resolves the chosen index against the manager's candidates,
            # so it has to see the dense set, not the one the sampler generated
            option_manager.candidates = candidates

            if deterministic:
                _, actions = dense_actions(model, raw_obs)
            else:
                actions = model.act({**obs, "policy": policy_obs(raw_obs, candidates)})
            log_action(actions, env)
            obs, rew, terminated, truncated, info = env.step(actions)


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
