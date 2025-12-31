from isaaclab.app import AppLauncher
import argparse

from gaitnet.control.mpc.convex_MPC import Gait

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
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import multiprocessing
import signal
import datetime
import os
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore
from gaitnet.gaitnet.components.gaitnet_env import GaitNetEnv
import gaitnet.simulation.cfg.footstep_scanner_constants as fs
from rsl_rl.runners import on_policy_runner
import rsl_rl.modules
from gaitnet.gaitnet.env_cfg.gaitnet_env_cfg import get_env
from gaitnet.util import log_exceptions
from gaitnet.gaitnet import gaitnet
import gaitnet.constants as const
from gaitnet import get_logger

logger = get_logger()

# # Global flag for graceful shutdown
# shutdown_requested = False


# def signal_handler(sig, frame):
#     global shutdown_requested
#     if multiprocessing.current_process().name == "MainProcess":
#         signal_name = signal.Signals(sig).name
#         logger.info(f"signal {signal_name} received in main process, shutting down...")
#     shutdown_requested = True


# # Set up the signal handler for SIGINT (Ctrl+C)
# signal.signal(signal.SIGINT, signal_handler)

###############################################################################
###############################################################################


def main():
    env = get_env(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        manager_class=GaitNetEnv,
    )
    episode_info = {}
    env.episode_info = episode_info

    gaitnet_actor = gaitnet.GaitnetActor(
        shared_state_dim=const.gait_net.robot_state_dim,
        shared_layer_sizes=[128, 128, 128],
        unique_state_dim=const.gait_net.footstep_option_dim,
        unique_layer_sizes=[64, 64],
        trunk_layer_sizes=[128, 128, 128],
    ).to(args_cli.device)

    gaitnet_critic = gaitnet.GaitnetCritic(
        shared_state_dim=const.gait_net.robot_state_dim,
        shared_layer_sizes=[64, 64, 64],
        num_unique_states=const.gait_net.num_footstep_options*const.robot.num_legs+1,  # +1 for the "no step" option
        unique_state_dim=const.gait_net.footstep_option_dim,
        unique_layer_sizes=[32, 32],
        trunk_layer_sizes=[64, 64, 64],
        trunk_combiner_head_sizes=[32, 32],
    ).to(args_cli.device)

    actor_critic_class = gaitnet.GaitnetActorCritic
    on_policy_runner.__dict__[actor_critic_class.__name__] = actor_critic_class

    # wrap for RL training
    env = RslRlVecEnvWrapper(env)

    # Create unique experiment name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"gaitnet_{timestamp}"

    # Create unique log directory
    log_dir = f"./training/gaitnet/runs/{experiment_name}"
    save_dir = f"./training/gaitnet/checkpoints/{experiment_name}"

    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare config dict for OnPolicyRunner
    train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "schedule": "fixed",  # we don't support adaptive
            # specified the proximal part of PPO - larger = faster at cost of stability
            "clip_param": 0.3,
            # how many times to use each batch of data for gradient updates
            "num_learning_epochs": 8,
            # number of minibatches to split one batch of data into
            "num_mini_batches": 4,
            "value_loss_coef": 0.5,
            # controls entropy - higher = more exploration, slower convergence
            "entropy_coef": 0.02,
            # learning rate
            "learning_rate": 3e-4,
            # clips gradients to prevent unstable training
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            # decay rate of future rewards. Closer to 1 = long horizon, closer to 0 = short horizon
            # H=1/(1-gamma); 0.98 corresponds to 2s, 0.99 to 5s
            "gamma": 0.99,
            "lam": 0.95,
        },
        "policy": {
            "class_name": actor_critic_class.__name__,
            "actor": gaitnet_actor,
            "critic": gaitnet_critic,
            "episode_info": episode_info,
        },
        "obs_groups": {  # no private observations
            "policy": ["policy",],
            "critic": ["policy",],
        },
        "log_dir": log_dir,
        "num_steps_per_env": 500,  # ~1 episodes per batch (episode = 20s = 500 iterations)
        "save_interval": 5,
        "empirical_normalization": False,
        "logger": "tensorboard",  # Explicitly set TensorBoard as logger
    }

    runner = on_policy_runner.OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=train_cfg["log_dir"],
        device=args_cli.device,
    )

    # Resume from checkpoint if specified
    if args_cli.resume is not None:
        logger.info(f"Loading checkpoint from {args_cli.resume}")
        runner.load(args_cli.resume)

    logger.info("Starting training...")

    # Let the runner handle the entire training loop
    # This ensures proper TensorBoard logging continuity
    try:
        runner.learn(num_learning_iterations=args_cli.max_iterations)
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("Closing environments")
    env.close()


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
