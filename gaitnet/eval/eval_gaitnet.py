"""Evaluate GaitNet across a grid of terrain difficulties and commanded velocities.

The whole sweep runs in one process against one scene. The difficulty axis is laid
out along the rows of the terrain (see `gaitnet.eval.components.eval_terrain`), so
every difficulty is simulated simultaneously and none of them needs its own scene.
The velocity axis needs nothing but a new value in the command term. That leaves one
Omniverse boot, one terrain cook and one controller pool for the entire sweep.
"""

from gaitnet import setup_logging

setup_logging()

from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate Gaitnet")
parser.add_argument(
    "--difficulties",
    type=float,
    nargs="+",
    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    help="Terrain difficulties to evaluate. Each one gets its own row of the terrain.",
)
parser.add_argument(
    "--velocities",
    type=float,
    nargs="+",
    default=[0.05, 0.1, 0.15, 0.2],
    help="Commanded forward velocities to evaluate, swept one after another.",
)
parser.add_argument(
    "--envs-per-difficulty",
    type=int,
    default=50,
    help="Environments per difficulty, i.e. samples behind each output file.",
)
parser.add_argument("--trials", type=int, default=1, help="Number of evaluation trials")
parser.add_argument(
    "--terrain-length",
    type=float,
    default=4.0,
    help="Length of a sub-terrain in metres. Defaults to the furthest a robot could"
    " walk in an episode, with margin. Robots spawn at the centre, so only half of"
    " this is forward runway.",
)
from gaitnet.gaitnet.util import add_checkpoint_arg

add_checkpoint_arg(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, unused_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(launcher_args=args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import SensorBaseCfg
from gaitnet.eval.evaluator import EvalGroup, Evaluator
from gaitnet.gaitnet.util import get_checkpoint_path
from gaitnet.gaitnet.components.gaitnet_env import GaitNetEnv, GaitNetObservationManager
from gaitnet.gaitnet.env_cfg.gaitnet_env_cfg import (
    GaitNetEnvCfg,
    get_env_cfg,
    update_controllers,
)
from gaitnet.util import log_exceptions
from gaitnet_core.networks import CandidateScorer
import gaitnet.constants as const
from gaitnet.eval.components.eval_terrain import envs_for_difficulty, make_eval_terrain
from gaitnet.eval.components.fixed_velocity_command import (
    FixedVelocityCommand,
    FixedVelocityCommandCfg,
)
from gaitnet.eval.components.terminations import out_of_sub_terrain
from gaitnet import GIT_COMMIT, get_logger
from gaitnet.gaitnet.dense_eval import dense_actions, load_actor

logger = get_logger()

# how much further than the furthest reachable point a sub-terrain extends
_terrain_length_margin = 1.25
_minimum_terrain_length = 4.0
# the collision mesh the old one-difficulty-per-process sweep cooked, and the largest
# that is known to work here
_triangle_budget = 6.4e6


def sub_terrain_length(velocities: list[float], episode_length_s: float) -> float:
    """How long a sub-terrain has to be to contain a whole episode.

    The robot spawns at the centre of its sub-terrain and walks in +x, so it can only
    use half the length. Terrain is the dominant cost of building the scene -- every
    sub-terrain is cooked into the collision mesh and the ray caster's BVH -- so this
    is sized to the episode rather than left at a generous constant.
    """
    reachable = max(velocities) * episode_length_s
    return max(_minimum_terrain_length, 2.0 * reachable * _terrain_length_margin)


def build_env_cfg(
    difficulties: list[float],
    envs_per_difficulty: int,
    terrain_length: float | None,
    device: str,
) -> GaitNetEnvCfg:
    """Build the config for a scene that holds every difficulty at once."""
    num_envs = len(difficulties) * envs_per_difficulty
    env_cfg = get_env_cfg(num_envs, device)

    if terrain_length is None:
        terrain_length = sub_terrain_length(args_cli.velocities, env_cfg.episode_length_s)
    # the whole grid is cooked into one collision mesh, and it is the thing that grows
    # when the sweep is packed into a single scene -- a lost-contact failure (robots
    # sinking through the ground) shows up here first
    generator = env_cfg.scene.terrain.terrain_generator
    scale = generator.horizontal_scale  # type: ignore
    triangles = num_envs * round((terrain_length / scale - 1) * (1.0 / scale - 1) * 2)
    logger.info(
        f"terrain: {len(difficulties)} difficulties x {envs_per_difficulty} envs,"
        f" {terrain_length:.1f}m sub-terrains, {num_envs} environments,"
        f" ~{triangles / 1e6:.1f}M collision triangles"
    )
    if triangles > _triangle_budget:
        logger.warning(
            f"~{triangles / 1e6:.1f}M collision triangles exceeds the {_triangle_budget / 1e6:.1f}M"
            " that the per-difficulty sweep used to cook. If robots terminate immediately on"
            " foot_below_ground, they are falling through the terrain: lower"
            " --terrain-length or --envs-per-difficulty."
        )
    make_eval_terrain(
        env_cfg.scene.terrain,
        difficulties=tuple(difficulties),
        envs_per_difficulty=envs_per_difficulty,
        sub_terrain_size=(terrain_length, 1.0),
    )

    # the row an environment sits on is the difficulty being measured, so it has to
    # stay put -- the curriculum would otherwise shuffle environments between rows
    env_cfg.curriculum.terrain_levels = None  # type: ignore

    # the stock bound is the extent of the whole grid, which now spans every difficulty
    env_cfg.terminations.terrain_out_of_bounds = DoneTerm(  # type: ignore
        func=out_of_sub_terrain,
        params={"distance_buffer": 0.0},
        time_out=True,
    )

    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "yaw": (0, 0),
    }

    # velocity is swept in-place, so the value here is only the starting one
    env_cfg.commands.base_velocity = FixedVelocityCommandCfg(  # type: ignore
        command=(args_cli.velocities[0], 0, 0)
    )

    # marker prims and a per-frame visualisation callback, for nothing: the sweep is
    # headless and this scales with the environment count
    for attr_name in env_cfg.scene.__dir__():
        attr = getattr(env_cfg.scene, attr_name)
        if isinstance(attr, SensorBaseCfg):
            attr.debug_vis = False

    return env_cfg


def eval_groups(difficulties: list[float], velocity: float, envs_per_difficulty: int) -> list[EvalGroup]:
    """One output file per difficulty, over the environments on that difficulty's row."""
    return [
        EvalGroup(
            file_name=f"gaitnet_eval_d{difficulty}_v{velocity}_commit{GIT_COMMIT}.csv",
            envs=envs_for_difficulty(index, envs_per_difficulty),
        )
        for index, difficulty in enumerate(difficulties)
    ]


def run_velocity(
    env: GaitNetEnv,
    model: CandidateScorer,
    velocity: float,
    difficulties: list[float],
    envs_per_difficulty: int,
    trials: int,
) -> None:
    """Evaluate every difficulty at one commanded velocity."""
    command_term: FixedVelocityCommand = env.command_manager.get_term("base_velocity")  # type: ignore
    command_term.set_command((velocity, 0.0, 0.0))

    observations, info = env.reset()
    obs: torch.Tensor = observations["policy"]  # type: ignore
    evaluator = Evaluator(
        env, trials=trials, groups=eval_groups(difficulties, velocity, envs_per_difficulty)
    )

    while not evaluator.done:
        footstep_option_manager: "GaitNetObservationManager" = env.observation_manager
        # the observation manager replaces the terrain scan with the footstep
        # options it sampled, so rebuild the raw observation dense sampling needs
        base_obs = obs[:, : const.gait_net.robot_state_dim]
        terrain_obs = torch.cat(
            [base_obs, footstep_option_manager.most_recent_terrain_obs], dim=1
        )

        # score every cell of every leg rather than the sampler's random subset
        candidates, actions = dense_actions(model, terrain_obs)
        # the action term resolves the chosen index against the manager's candidates,
        # so it has to see the dense set, not the one the sampler generated
        footstep_option_manager.candidates = candidates

        env_step_info = env.step(actions)
        observations, rew, terminated, truncated, info = env_step_info

        # a finished trial resets the environment, and the observations from that
        # reset supersede the ones the step returned
        reset_observations = evaluator.process(env_step_info)
        if reset_observations is not None:
            observations = reset_observations
        obs = observations["policy"]  # type: ignore


def main():
    device = torch.device(args_cli.device)
    model = load_actor(get_checkpoint_path(args_cli.checkpoint_name), device)
    model.eval()

    difficulties: list[float] = args_cli.difficulties
    velocities: list[float] = args_cli.velocities
    envs_per_difficulty: int = args_cli.envs_per_difficulty
    num_envs = len(difficulties) * envs_per_difficulty

    env_cfg = build_env_cfg(
        difficulties, envs_per_difficulty, args_cli.terrain_length, args_cli.device
    )
    env = GaitNetEnv(cfg=env_cfg)
    # after the environment, which is when the observation terms first tolerate a
    # missing pool; one pool serves the whole sweep since the count never changes
    with update_controllers(env_cfg, num_envs):
        try:
            with torch.inference_mode():
                for index, velocity in enumerate(velocities):
                    logger.info(f"[velocity {index + 1}/{len(velocities)}] evaluating v={velocity}")
                    print(f"[velocity {index + 1}/{len(velocities)}] evaluating v={velocity}")
                    run_velocity(
                        env, model, velocity, difficulties, envs_per_difficulty, args_cli.trials
                    )
        finally:
            # before the pool, whose workers the action terms still hold a handle to
            env.close()

    logger.info("Evaluation complete.")
    print("Evaluation complete.")


if __name__ == "__main__":
    try:
        with log_exceptions(logger):
            main()
    finally:
        # log_exceptions re-raises, and the pool's workers are not daemons, so without
        # this a crash leaves the process hanging onto the GPU instead of exiting
        simulation_app.close()
