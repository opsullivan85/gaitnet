import argparse
from ast import mod
import signal

from isaaclab.app import AppLauncher
from gaitnet.contactnet import tree
from gaitnet.contactnet.datagen import check_dones, args
from gaitnet.sim2real.siminterface import SimInterface
from gaitnet.simulation.util import controls_to_joint_efforts, reset_all_to

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Manual robot control.")

# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args, unused_args = parser.parse_known_args()

# # launch omniverse app
# app_launcher = AppLauncher(args)
# simulation_app = app_launcher.app

import argparse
import time
import multiprocessing

import numpy as np
import pygame
from pygame import joystick
import torch
import torch.nn as nn

from isaaclab.envs import ManagerBasedEnv
from gaitnet.contactnet.debug import view_footstep_cost_map
from gaitnet.contactnet.contactnet import FootstepDataset, ContactNet
from gaitnet.contactnet.util import get_checkpoint_path, get_dataset_paths
from gaitnet import get_logger
import pickle

from gaitnet.simulation.cfg.quadrupedenv import QuadrupedEnvCfg, get_quadruped_env_cfg
import gaitnet.simulation.cfg.footstep_scanner_constants as fs
from gaitnet.util import VectorPool

logger = get_logger()

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    if multiprocessing.current_process().name == "MainProcess":
        signal_name = signal.Signals(sig).name
        logger.info(f"signal {signal_name} received in main process, shutting down...")
    shutdown_requested = True


class Controller:
    def __init__(self) -> None:
        pygame.init()
        pygame.joystick.init()

        try:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        except pygame.error as e:
            raise RuntimeError("Could not find valid controller") from e

        data_set_paths = get_dataset_paths()
        with open(data_set_paths[0], "rb") as f:
            data = pickle.load(f)
        metadata = data['metadata']
        max_yaw = metadata["max_yaw"]
        max_control_input = metadata["max_control_input"]
        self.max_control = np.array([max_control_input, max_control_input, max_yaw])

    def get_control(self):
        pygame.event.pump()  # Process events

        y = -self.joystick.get_axis(0)
        x = -self.joystick.get_axis(1)
        omega = -self.joystick.get_axis(2)

        control = np.array([x, y, omega]) * self.max_control
        return control



def main():
    controller = Controller()

    num_envs = 1
    env_cfg: QuadrupedEnvCfg = get_quadruped_env_cfg(num_envs, args.device)
    # setup RL environment
    env = ManagerBasedEnv(cfg=env_cfg)
    iterations_between_mpc = 5  # 50 Hz MPC
    controllers = VectorPool(
        instances=num_envs,
        cls=SimInterface,
        dt=env_cfg.sim.dt * env_cfg.decimation,  # 500 Hz leg PD control
        iterations_between_mpc=iterations_between_mpc,
        debug_logging=False,
    )

    default_state = tree.IsaacStateTorch(
        joint_pos=env.scene["robot"].data.default_joint_pos[0],
        joint_vel=env.scene["robot"].data.default_joint_vel[0],
        body_state=env.scene["robot"].data.default_root_state[0],
        # this is a bad observation, but we ignore the root when saving the data so it should be fine
        obs=tree.Observation.from_idxs(env, np.asarray([0]))[0],
    )
    default_state.body_state[
        2
    ] += -0.075  # slightly above ground (default state is in the air)


    # Load model
    dataset = FootstepDataset(get_dataset_paths())
    model = ContactNet(
        input_dim=dataset.input_dim, output_dim_per_foot=dataset.output_dim
    )
    del dataset
    checkpoint = torch.load(get_checkpoint_path(), map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    with controllers, torch.inference_mode():
        env_cfg.controllers = controllers

        reset_all_to(
            env,
            default_state.joint_pos,
            default_state.joint_vel,
            default_state.body_state,
        )

        while True:
            if shutdown_requested:
                break
            
            # get joint efforts from controller
            control = controller.get_control()
            controls = np.expand_dims(control, axis=0)
            logger.info(f"control input: {control}")
            joint_efforts = controls_to_joint_efforts(controls, controllers, env.scene)

            env.step(joint_efforts)

            dones, done_states = check_dones(env, controls)
            done_state: tree.IsaacStateCPU = done_states[0]
            if not np.all(dones):
                continue

            # take footstep
            x = FootstepDataset.flatten_state(done_state)
            x = x.astype(np.float32)
            x = torch.from_numpy(x).unsqueeze(0).to(args.device)
            costmap = model(x).cpu().numpy().reshape(4, 5, 5)
            # get min arg of costmap
            best_idx = np.unravel_index(np.argmin(costmap), costmap.shape)
            if args.debug:
                view_footstep_cost_map(
                    costmap,
                    best_idx,
                    title="Footstep Cost Map",
                    save_figure=not args.interactive_plots,
                )
            # convert best_idx to foot and dx, dy
            foot, dx_idx, dy_idx = best_idx
            foot = np.asarray([foot])
            # convert dx_idx, dy_idx to dx, dy based off of fs.grid_resolution and fs.grid_size
            dx = (dx_idx - fs._depricated_grid_size[0] // 2) * fs._depricated_grid_resolution
            dy = (dy_idx - fs._depricated_grid_size[1] // 2) * fs._depricated_grid_resolution
            hip_offset = np.asarray([[dx, dy]])
            duration = np.asarray([0.2])  # seconds
            controllers.call(
                function=SimInterface.initiate_footstep,
                mask=None,
                leg=foot,
                location_hip=hip_offset,
                duration=duration,
            )



    env_cfg.controllers = None
    del controllers

    # close the environment
    env.close()





if __name__ == "__main__":
    from gaitnet.util import log_exceptions
    with log_exceptions(logger):
        main()