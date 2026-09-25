"""The training env cfg, turned into an evaluation scene: every terrain difficulty on its
own row of one terrain, robots spawned at the -x end of their sub-terrain, a fixed forward
velocity command, and the policy bundle's contract (foothold grid and rules) in place of the
env's defaults."""

from __future__ import annotations

import logging

from isaaclab.managers import TerminationTermCfg as DoneTerm

from gaitnet_core.bundle import BundleError, PolicyBundle
from gaitnet_sim.env.commands import FixedVelocityCommandCfg
from gaitnet_sim.env.env_cfg import GaitNetEnvCfg
from gaitnet_sim.env.scene import SCANNER_NAMES, foothold_scanner_cfg
from gaitnet_sim.env.terminations import out_of_sub_terrain
from gaitnet_sim.robot import HIP_NAMES
from gaitnet_sim.terrains import make_eval_terrain

logger = logging.getLogger(__name__)


def runway(env_cfg: GaitNetEnvCfg) -> float:
    """How far a robot walks in a scene from `make_eval_env_cfg`, from the centre of the spawn
    platform to the far end of its sub-terrain (m), where its episode ends. The sweep sets
    each velocity's episode length to walk it."""
    generator = env_cfg.scene.terrain.terrain_generator
    platform = next(iter(generator.sub_terrains.values())).platform_size
    return generator.size[0] - 0.5 * platform


def apply_bundle_contract(env_cfg: GaitNetEnvCfg, bundle: PolicyBundle) -> None:
    """Scan terrain on the bundle's foothold grid and use its foothold rules."""
    if bundle.robot.name != env_cfg.gaitnet.robot:
        raise BundleError(f"the policy is for {bundle.robot.name}, the scene has {env_cfg.gaitnet.robot}")
    grid, rules = bundle.grid, bundle.rules
    env_cfg.gaitnet.grid_resolution = grid.resolution
    env_cfg.gaitnet.grid_size = tuple(grid.size)
    env_cfg.gaitnet.grid_border = grid.border
    env_cfg.gaitnet.grid_center = tuple(grid.center)
    env_cfg.gaitnet.step_threshold = rules.step_threshold
    env_cfg.gaitnet.edge_margin = rules.edge_margin
    env_cfg.gaitnet.min_stance_after_step = rules.min_stance_after_step
    update_period = env_cfg.decimation * env_cfg.sim.dt
    for leg, (name, hip) in enumerate(zip(SCANNER_NAMES, HIP_NAMES)):
        scanner = foothold_scanner_cfg(hip, grid, leg)
        scanner.update_period = update_period
        setattr(env_cfg.scene, name, scanner)


def make_eval_env_cfg(
    env_cfg: GaitNetEnvCfg,
    bundle: PolicyBundle,
    difficulties: list[float],
    velocities: list[float],
    envs_per_difficulty: int,
    terrain_length: float,
    randomize: bool = False,
) -> GaitNetEnvCfg:
    """Rewrite `env_cfg` (in place, and returned) for a sweep over `difficulties` x `velocities`.
    The episode length is left to the sweep, see `runway`.

    Args:
        terrain_length: sub-terrain length (m)
        randomize: keep training's randomization and observation noise; by default the
            sweep runs with nominal dynamics and exact observations (`play_mode`)
    """
    apply_bundle_contract(env_cfg, bundle)
    if not randomize:
        env_cfg.play_mode()
    env_cfg.scene.num_envs = len(difficulties) * envs_per_difficulty
    make_eval_terrain(
        env_cfg.scene.terrain,
        difficulties=tuple(difficulties),
        envs_per_difficulty=envs_per_difficulty,
        sub_terrain_size=(terrain_length, 1.0),
    )
    logger.info(
        f"terrain: {len(difficulties)} difficulties x {envs_per_difficulty} envs, {terrain_length:.1f} m"
        f" sub-terrains, {runway(env_cfg):.2f} m walked from the spawn platform"
    )
    # each robot stays on its difficulty's row
    env_cfg.curriculum.terrain_levels = None
    # the stock bound is the whole grid, which spans every difficulty
    env_cfg.terminations.terrain_out_of_bounds = DoneTerm(
        func=out_of_sub_terrain, params={"distance_buffer": 0.0}, time_out=True
    )
    # on the platform at the -x end, a runway short of the far end; relative to the centre
    spawn_x = 0.5 * terrain_length - runway(env_cfg)
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (spawn_x - 0.1, spawn_x + 0.1),
        "y": (-0.1, 0.1),
        "yaw": (0.0, 0.0),
    }
    # swept in place (FixedVelocityCommand.set_command); this is only the first value
    env_cfg.commands.base_velocity = FixedVelocityCommandCfg(command=(velocities[0], 0.0, 0.0))
    # the planner samples its own candidates from the terrain scan
    env_cfg.observations.candidates = None
    env_cfg.observations.terrain = None
    return env_cfg
