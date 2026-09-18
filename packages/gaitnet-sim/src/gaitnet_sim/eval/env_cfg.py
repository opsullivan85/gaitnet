"""The training env cfg, turned into an evaluation scene: every terrain difficulty on its
own row of one terrain, a fixed forward velocity command, and the policy bundle's contract
(foothold grid and rules) in place of the env's defaults."""

from __future__ import annotations

import logging

from isaaclab.managers import TerminationTermCfg as DoneTerm

from gaitnet_core.bundle import BundleError, PolicyBundle
from gaitnet_sim.env.commands import FixedVelocityCommandCfg
from gaitnet_sim.env.env_cfg import GaitNetHolesEnvCfg
from gaitnet_sim.env.scene import SCANNER_NAMES, foothold_scanner_cfg
from gaitnet_sim.env.terminations import out_of_sub_terrain
from gaitnet_sim.robot import HIP_NAMES
from gaitnet_sim.terrains import make_eval_terrain

logger = logging.getLogger(__name__)

# how much further than the furthest reachable point a sub-terrain extends
_TERRAIN_LENGTH_MARGIN = 1.25
_MIN_TERRAIN_LENGTH = 4.0
# the largest collision mesh known to work; beyond it robots have fallen through the ground
_TRIANGLE_BUDGET = 6.4e6


def sub_terrain_length(velocities: list[float], episode_length_s: float) -> float:
    """Long enough to hold a whole episode. Robots spawn at the centre and walk +x, so only
    half is runway. Terrain dominates scene build time, so it's sized to the episode."""
    reachable = max(velocities) * episode_length_s
    return max(_MIN_TERRAIN_LENGTH, 2.0 * reachable * _TERRAIN_LENGTH_MARGIN)


def apply_bundle_contract(env_cfg: GaitNetHolesEnvCfg, bundle: PolicyBundle) -> None:
    """Scan terrain on the bundle's foothold grid and use its foothold rules."""
    if bundle.robot.name != env_cfg.gaitnet.robot:
        raise BundleError(f"the policy is for {bundle.robot.name}, the scene has {env_cfg.gaitnet.robot}")
    grid, rules = bundle.grid, bundle.rules
    env_cfg.gaitnet.grid_resolution = grid.resolution
    env_cfg.gaitnet.grid_size = tuple(grid.size)
    env_cfg.gaitnet.grid_border = grid.border
    env_cfg.gaitnet.step_threshold = rules.step_threshold
    env_cfg.gaitnet.edge_margin = rules.edge_margin
    env_cfg.gaitnet.min_stance_after_step = rules.min_stance_after_step
    update_period = env_cfg.decimation * env_cfg.sim.dt
    for name, hip in zip(SCANNER_NAMES, HIP_NAMES):
        scanner = foothold_scanner_cfg(hip, grid)
        scanner.update_period = update_period
        setattr(env_cfg.scene, name, scanner)


def make_eval_env_cfg(
    env_cfg: GaitNetHolesEnvCfg,
    bundle: PolicyBundle,
    difficulties: list[float],
    velocities: list[float],
    envs_per_difficulty: int,
    terrain_length: float | None = None,
) -> GaitNetHolesEnvCfg:
    """Rewrite `env_cfg` (in place, and returned) for a sweep over `difficulties` x `velocities`."""
    apply_bundle_contract(env_cfg, bundle)
    num_envs = len(difficulties) * envs_per_difficulty
    env_cfg.scene.num_envs = num_envs
    if terrain_length is None:
        terrain_length = sub_terrain_length(velocities, env_cfg.episode_length_s)

    generator = env_cfg.scene.terrain.terrain_generator
    scale = generator.horizontal_scale
    triangles = num_envs * round((terrain_length / scale - 1) * (1.0 / scale - 1) * 2)
    logger.info(
        f"terrain: {len(difficulties)} difficulties x {envs_per_difficulty} envs, {terrain_length:.1f} m"
        f" sub-terrains, ~{triangles / 1e6:.1f}M collision triangles"
    )
    if triangles > _TRIANGLE_BUDGET:
        logger.warning(
            f"~{triangles / 1e6:.1f}M collision triangles is over the {_TRIANGLE_BUDGET / 1e6:.1f}M known to work."
            " If robots end on foot_below_ground at once, they are falling through the terrain:"
            " lower --terrain_length or --envs_per_difficulty."
        )
    make_eval_terrain(
        env_cfg.scene.terrain,
        difficulties=tuple(difficulties),
        envs_per_difficulty=envs_per_difficulty,
        sub_terrain_size=(terrain_length, 1.0),
    )
    # each robot stays on its difficulty's row
    env_cfg.curriculum.terrain_levels = None
    # the stock bound is the whole grid, which spans every difficulty
    env_cfg.terminations.terrain_out_of_bounds = DoneTerm(
        func=out_of_sub_terrain, params={"distance_buffer": 0.0}, time_out=True
    )
    env_cfg.events.reset_base.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (0.0, 0.0)}
    # swept in place (FixedVelocityCommand.set_command); this is only the first value
    env_cfg.commands.base_velocity = FixedVelocityCommandCfg(command=(velocities[0], 0.0, 0.0))
    # the planner samples its own candidates from the terrain scan
    env_cfg.observations.candidates = None
    env_cfg.observations.terrain = None
    return env_cfg
