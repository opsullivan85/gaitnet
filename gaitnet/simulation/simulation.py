from gaitnet import setup_logging

setup_logging()

import argparse
import signal

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Smoke test for the MPC controller in simulation.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--num_steps", type=int, default=1000, help="Number of simulation steps to run."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
from isaaclab.terrains import TerrainGeneratorCfg

import gaitnet.constants as const
from gaitnet.gaitnet.components.gaitnet_env import GaitNetEnv
from gaitnet.gaitnet.env_cfg.gaitnet_env_cfg import (
    GaitNetEnvCfg,
    get_env_cfg,
    update_controllers,
)
from gaitnet.sim2real.siminterface import SimInterface
from gaitnet_mpc.pool import VectorPool

from gaitnet import get_logger
logger = get_logger()


def walk_in_place(
    count: int,
    control_interface: VectorPool,
    num_envs: int,
    cycle_length_steps: int,
    velocity_lead: np.ndarray,
):
    """Drive a hardcoded diagonal trot directly through the low-level controller,
    bypassing the (disabled) learned footstep policy entirely.

    Args:
        velocity_lead: (num_envs, 2) hip-frame (x, y) offset added to every
            footstep, proportional to the commanded body velocity so the feet
            lead in the direction of travel instead of stepping purely in place.
    """

    def step1():
        control_interface.call(
            SimInterface.initiate_footstep,
            mask=None,
            leg=np.repeat(np.array([0]), num_envs),
            location_hip=np.repeat(np.asarray([0.05, 0.1])[None, :], num_envs, axis=0) + velocity_lead,  # type: ignore
            duration=np.repeat(np.array([0.2]), num_envs),
        )
        control_interface.call(
            SimInterface.initiate_footstep,
            mask=None,
            leg=np.repeat(np.array([3]), num_envs),
            location_hip=np.repeat(np.asarray([-0.05, -0.1])[None, :], num_envs, axis=0) + velocity_lead,  # type: ignore
            duration=np.repeat(np.array([0.2]), num_envs),
        )

    def step2():
        control_interface.call(
            SimInterface.initiate_footstep,
            mask=None,
            leg=np.repeat(np.array([1]), num_envs),
            location_hip=np.repeat(np.asarray([0.05, -0.1])[None, :], num_envs, axis=0) + velocity_lead,  # type: ignore
            duration=np.repeat(np.array([0.2]), num_envs),
        )
        control_interface.call(
            SimInterface.initiate_footstep,
            mask=None,
            leg=np.repeat(np.array([2]), num_envs),
            location_hip=np.repeat(np.asarray([-0.05, 0.1])[None, :], num_envs, axis=0) + velocity_lead,  # type: ignore
            duration=np.repeat(np.array([0.2]), num_envs),
        )

    # on the first half of the cycle do step 1, on the second half do step 2
    if count % cycle_length_steps == 0:
        step1()
    elif count % cycle_length_steps == cycle_length_steps // 2:
        step2()


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    signal_name = signal.Signals(sig).name
    logger.info(f"signal {signal_name} received, shutting down...")
    shutdown_requested = True


# Set up the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


def main():
    """Smoke-test the underlying MPC controller inside the real training environment.

    This reuses the actual training env config (terrain, spawn/reset events,
    disturbances, random velocity commands, curriculum, ...) so that anything
    configured for training -- e.g. new disturbance events -- shows up here too.
    The learned footstep policy is disabled (a constant no-op action), so the
    robot is driven purely by the MPC's own gait scheduler tracking the
    randomly-sampled base_velocity command.
    """
    env_cfg: GaitNetEnvCfg = get_env_cfg(args_cli.num_envs, args_cli.device)

    # keep the same terrain generator (and friction) as training, just force it flat
    # so a broken/held terrain generator doesn't spawn the robot's feet in the ground
    terrain_generator: TerrainGeneratorCfg = env_cfg.scene.terrain.terrain_generator  # type: ignore
    terrain_generator.difficulty_range = (0.0, 0.0)
    terrain_generator.curriculum = False

    env = GaitNetEnv(cfg=env_cfg)
    update_controllers(env_cfg, args_cli.num_envs)

    # no-op footstep action: the no-op candidate index, duration=0. This lets the MPC's
    # own gait scheduler drive the robot from the base_velocity command without any
    # learned footstep placement.
    no_op_index = const.robot.num_legs * const.gait_net.num_footstep_options
    no_op_action = torch.tensor([no_op_index, 0.0], device=env.device).expand(
        args_cli.num_envs, -1
    )

    cycle_length_s = 0.4
    cycle_length_steps = max(1, round(cycle_length_s / env.step_dt))
    # each leg is only re-planted once per full cycle, so lead the footstep by the
    # distance the body actually travels over one cycle at the commanded velocity
    footstep_lead_time = cycle_length_steps * env.step_dt

    observations, _ = env.reset()

    count = 0
    success = True
    with torch.inference_mode():
        while (
            simulation_app.is_running()
            and not shutdown_requested
            and count < args_cli.num_steps
        ):
            command = env.command_manager.get_command("base_velocity")  # (num_envs, 3): vx, vy, wz
            velocity_lead = (command[:, :2] * footstep_lead_time).cpu().numpy()
            walk_in_place(
                count, env_cfg.robot_controllers, args_cli.num_envs, cycle_length_steps, velocity_lead
            )

            observations, _, _, _, _ = env.step(no_op_action)

            obs: torch.Tensor = observations["policy"]  # type: ignore
            if not torch.isfinite(obs).all():
                logger.error(f"non-finite observation at step {count}")
                success = False
                break

            if count % 100 == 0:
                logger.info(f"step {count}/{args_cli.num_steps}")

            count += 1

    env.close()

    if success and count >= args_cli.num_steps:
        logger.info(f"smoke test passed: ran {count} steps without error.")
    elif not shutdown_requested:
        logger.error(f"smoke test failed after {count} steps.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()
