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
from gaitnet.gaitnet.env_cfg.gaitnet_env_cfg import get_env
from gaitnet.util import log_exceptions
from gaitnet.gaitnet import gaitnet
import re
from pathlib import Path
import gaitnet.constants as const
from gaitnet import get_logger
from gaitnet import PROJECT_ROOT

data_path = PROJECT_ROOT / "data" / "contact_schedule" / "contact_schedule.csv"
data_path.parent.mkdir(parents=True, exist_ok=True)

logger = get_logger()


def load_model(checkpoint_path: Path, device: torch.device, deterministic: bool) -> gaitnet.GaitnetActorCritic | gaitnet.GaitnetActor:
    model = gaitnet.GaitnetActor(
        shared_state_dim=const.gait_net.robot_state_dim,
        shared_layer_sizes=[128, 128, 128],
        unique_state_dim=const.gait_net.footstep_option_dim,
        unique_layer_sizes=[64, 64],
        trunk_layer_sizes=[128, 128, 128],
    )
    if deterministic:
        agent = model
    else:
        agent = gaitnet.GaitnetActorCritic(0, 0, 0, model, None)  # type: ignore
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if deterministic:
        state_dict = {re.sub(r"^actor\.", "", k): v for k, v in state_dict.items() if k.startswith("actor.")}
    else:  # remove all critic keys
        state_dict = {k: v for k, v in state_dict.items() if k.startswith("actor.")}
    agent.load_state_dict(state_dict)
    agent.to(device)
    return agent

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
    model = load_model(get_checkpoint_path(), device, deterministic=deterministic)
    model.eval()

    env = get_env(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        manager_class=GaitNetEnv,
    )
    obs, info = env.reset()

    with torch.inference_mode():
        while True:
            if deterministic:
                actions = gaitnet.GaitnetActor.act_inference(model, obs)
            if not deterministic:
                actions = model.act(obs)
            log_action(actions, env)
            obs, rew, terminated, truncated, info = env.step(actions)


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
