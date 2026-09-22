"""Observation terms. Each builds on `gaitnet_core`, so the simulator and the deployment
runtime compute the same quantities.

Groups (see `env_cfg.ObservationsCfg`): `state` (the robot state vector, features chosen by
name), `candidates` (the footholds the policy scores this tick), and, when a preset turns
them on, `terrain` (per-leg height patches, for networks that read them), `privileged`
(sim-only inputs for the critic), `base_command` (the command before any nudge, for
feedback observers) and `footholds` (each leg's valid terrain fraction, for observers). Candidates are an observation so the RL library stores them with the
step and can recompute log-probabilities on exactly the set the action was drawn from.

The policy's groups read the action term's `planner_observation()`, which carries the
observation noise; privileged terms read the truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.managers import SceneEntityCfg

from gaitnet_core import planner
from gaitnet_core.features import state_vector
from gaitnet_core.samplers import make_sampler
from gaitnet_core.terrain import fill_unknown, inner_heights, valid_footholds

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster

    from gaitnet_sim.env.actions import FootstepControlAction


def footstep_action(env: "ManagerBasedRLEnv", name: str = "footstep") -> "FootstepControlAction":
    """The footstep action term, which owns the controller and reads the robot's state."""
    return env.action_manager.get_term(name)  # type: ignore[return-value]


def robot_state(env: "ManagerBasedRLEnv", features: list[str], action_name: str = "footstep") -> torch.Tensor:
    """(N, D) the named `gaitnet_core.features`, concatenated in order."""
    return state_vector(footstep_action(env, action_name).planner_observation().state, features)


def terrain_heights(env: "ManagerBasedRLEnv", action_name: str = "footstep") -> torch.Tensor:
    """(N, L, *patch_size) terrain heights relative to each hip (m).

    Unknown cells (no ray returned) read `gaitnet_core.terrain.UNKNOWN_HEIGHT` rather than
    -inf, as the networks that read terrain see them at deployment too.
    """
    return fill_unknown(footstep_action(env, action_name).planner_observation().terrain.heights)


def footstep_candidates(
    env: "ManagerBasedRLEnv",
    sampler: str = "uniform_jitter",
    sampler_kwargs: dict | None = None,
    action_name: str = "footstep",
) -> torch.Tensor:
    """(N, L, K, 5) packed `gaitnet_core.candidates.Candidates` from the named sampler.

    Only valid footholds (terrain rules and leg eligibility, `env.cfg.gaitnet`) are
    sampled, so the policy never scores a foothold it isn't allowed to take.
    """
    term = footstep_action(env, action_name)
    observation = term.planner_observation()
    valid = env.cfg.gaitnet.foothold_rules().valid(observation, term.spec)
    heights = inner_heights(observation.terrain.heights, term.grid)
    candidates = make_sampler(sampler, **(sampler_kwargs or {})).sample(valid, term.grid, heights=heights)
    return candidates.pack()


def base_command(env: "ManagerBasedRLEnv", action_name: str = "footstep") -> torch.Tensor:
    """(N, 3) the velocity command before any nudge, which feedback observers scale."""
    return footstep_action(env, action_name).base_command()


def foothold_fraction(env: "ManagerBasedRLEnv", action_name: str = "footstep") -> torch.Tensor:
    """(N, L) fraction of each leg's foothold grid that is valid terrain (terrain rules only, not
    leg eligibility), as the planner sees it. `PlanResult.foothold_fraction` for observers."""
    term = footstep_action(env, action_name)
    cells = env.cfg.gaitnet.foothold_rules().valid_cells(term.planner_observation(), term.spec)
    return planner.foothold_fraction(cells)


##
# Privileged: what the critic may know and the policy can't
##


def terrain_height_summary(env: "ManagerBasedRLEnv", cells: int = 5, action_name: str = "footstep") -> torch.Tensor:
    """(N, L * cells^2) each leg's candidate grid of terrain heights, averaged down to
    cells x cells. Unknown cells count as `UNKNOWN_HEIGHT`."""
    term = footstep_action(env, action_name)
    heights = fill_unknown(inner_heights(term.terrain().heights, term.grid))
    n, l, size_x, size_y = heights.shape
    pooled = F.adaptive_avg_pool2d(heights.reshape(n * l, 1, size_x, size_y), cells)
    return pooled.reshape(n, -1)


def foothold_validity_summary(env: "ManagerBasedRLEnv", cells: int = 5, action_name: str = "footstep") -> torch.Tensor:
    """(N, L * cells^2) fraction of valid footholds (terrain rules only, not leg eligibility)
    in each of cells x cells blocks of each leg's grid."""
    term = footstep_action(env, action_name)
    rules = env.cfg.gaitnet.foothold_rules()
    valid = valid_footholds(term.terrain().heights, term.spec, term.grid, rules.step_threshold, rules.edge_margin)
    n, l, size_x, size_y = valid.shape
    pooled = F.adaptive_avg_pool2d(valid.float().reshape(n * l, 1, size_x, size_y), cells)
    return pooled.reshape(n, -1)


def base_terrain_clearance(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_scanner"),
    max_clearance: float = 1.0,
) -> torch.Tensor:
    """(N, 1) the base's height above the highest surface under the trunk (m), at most
    `max_clearance` (also where no ray hit anything).

    `sensor_cfg` is a ray caster on the base, yaw aligned, whose pattern covers the trunk's
    footprint. On flat ground this is the base's height above the ground.
    """
    scanner: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits = scanner.data.ray_hits_w.torch[..., 2]
    hits = torch.where(torch.isfinite(hits), hits, torch.full_like(hits, float("-inf")))
    clearance = scanner.data.pos_w.torch[:, 2] - hits.amax(dim=1)
    return clearance.clamp(max=max_clearance).unsqueeze(-1)


def foot_contact_forces(env: "ManagerBasedRLEnv", scale: float = 0.01, action_name: str = "footstep") -> torch.Tensor:
    """(N, L) each foot's contact force magnitude, times `scale` (1/N)."""
    io = footstep_action(env, action_name).io
    return io.contact_sensor.data.net_normal_forces_w.torch[:, io.contact_ids].norm(dim=-1) * scale
