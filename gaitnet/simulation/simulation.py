import argparse
import signal

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import multiprocessing

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv

from gaitnet.sim2real.siminterface import SimInterface
from gaitnet.util import VectorPool
from gaitnet.simulation.util import controls_to_joint_efforts, reset_all_to
from gaitnet.util.data_logging import data_logger
from gaitnet.simulation.cfg.quadrupedenv import QuadrupedEnvCfg, get_quadruped_env_cfg

from gaitnet import get_logger
logger = get_logger()


def walk_in_place(count: int, control_interface: VectorPool):
    def step1():
        control_interface.call(
            SimInterface.initiate_footstep,
            mask=None,
            leg=np.repeat(np.array([0]), args_cli.num_envs),
            location_hip=np.repeat(
                np.asarray([0.05, 0.1])[None, :], args_cli.num_envs, axis=0
            ),
            duration=np.repeat(np.array([0.2]), args_cli.num_envs),
        )
        control_interface.call(
            SimInterface.initiate_footstep,
            mask=None,
            leg=np.repeat(np.array([3]), args_cli.num_envs),
            location_hip=np.repeat(
                np.asarray([-0.05, -0.1])[None, :], args_cli.num_envs, axis=0
            ),
            duration=np.repeat(np.array([0.2]), args_cli.num_envs),
        )

    def step2():
        control_interface.call(
            SimInterface.initiate_footstep,
            mask=None,
            leg=np.repeat(np.array([1]), args_cli.num_envs),
            location_hip=np.repeat(
                np.asarray([0.05, -0.1])[None, :], args_cli.num_envs, axis=0
            ),
            duration=np.repeat(np.array([0.2]), args_cli.num_envs),
        )
        control_interface.call(
            SimInterface.initiate_footstep,
            mask=None,
            leg=np.repeat(np.array([2]), args_cli.num_envs),
            location_hip=np.repeat(
                np.asarray([-0.05, 0.1])[None, :], args_cli.num_envs, axis=0
            ),
            duration=np.repeat(np.array([0.2]), args_cli.num_envs),
        )

    cycle_length_s = 0.4
    cycle_length_steps = int(100 * cycle_length_s)
    # on the first count of the cycle do step 1, on the second half do step 2
    if count % cycle_length_steps == 0:
        step1()
    elif count % cycle_length_steps == cycle_length_steps // 2:
        step2()


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    if multiprocessing.current_process().name == "MainProcess":
        signal_name = signal.Signals(sig).name
        logger.info(f"signal {signal_name} received in main process, shutting down...")
    shutdown_requested = True


# Set up the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


def main():
    """Main function."""
    # create environment configuration
    env_cfg: QuadrupedEnvCfg = get_quadruped_env_cfg(args_cli.num_envs, args_cli.device)
    # setup RL environment
    env = ManagerBasedEnv(cfg=env_cfg)
    iterations_between_mpc = 5  # 50 Hz MPC
    controllers = VectorPool(
        instances=args_cli.num_envs,
        cls=SimInterface,
        dt=env_cfg.sim.dt * env_cfg.decimation,  # 500 Hz leg PD control
        iterations_between_mpc=iterations_between_mpc,
        debug_logging=False,
    )
    data_logger.set_dt(
        sim_dt=env_cfg.sim.dt,
        control_dt=env_cfg.sim.dt * env_cfg.decimation,
        mpc_dt=env_cfg.sim.dt * env_cfg.decimation * iterations_between_mpc,
    )

    # simulate physics
    count = 0
    with controllers, torch.inference_mode():
        env_cfg.controllers = controllers
        while simulation_app.is_running() and not shutdown_requested:  # Add flag check
            # walk_in_place(count, controllers)
            command = np.zeros((args_cli.num_envs, 3), dtype=np.float32)
            command[:, 0] = 0.0
            command[:, 2] = 0.0
            joint_efforts: torch.Tensor = controls_to_joint_efforts(
                command, controllers, env.scene
            )

            # if count == 0:
            #     # step the enviornment a few times to make sure the feet are in contact
            #     for _ in range(20):
            #         zero_efforts = torch.zeros_like(joint_efforts)
            #         _ = env.step(zero_efforts)

            # step the environment
            # obs, rew, terminated, truncated, info = env.step(joint_efforts)  # type: ignore
            obs, _ = env.step(joint_efforts)  # type: ignore
            obs: dict[str, dict[str, torch.Tensor]] = obs
            # print(f"{obs['policy'].keys() = }, {rew = }, {terminated = }, {truncated = }, {info = }")
            data_logger.log(obs["policy"])

            if count % 500 == 0:
                # set state of all to robot 1
                
                # get state from env
                joint_pos_isaac = env.scene["robot"].data.joint_pos[0]
                joint_vel_isaac = env.scene["robot"].data.joint_vel[0]
                body_state_isaac = env.scene["robot"].data.root_state_w[0]

                height_scanner = env.scene["FR_foot_scanner"]

                reset_all_to(env, joint_pos_isaac, joint_vel_isaac, body_state_isaac)
                logger.info("reset all to robot 0 state.")

                # step the enviornment a few times to make sure the feet are in contact
                # for _ in range(50):
                #     zero_efforts = torch.zeros_like(joint_efforts)
                #     _ = env.step(zero_efforts)

            # update counter
            count += 1
    env_cfg.controllers = None
    del controllers

    # close the environment
    env.close()
