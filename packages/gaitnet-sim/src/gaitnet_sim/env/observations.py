"""Observation terms. Each builds on `gaitnet_core`, so the simulator and the deployment
runtime compute the same quantities.

Groups (see `env_cfg.ObservationsCfg`): `state` (the robot state vector, features chosen by
name), `terrain` (per-leg height patches) and `candidates` (the footholds the policy scores
this tick). Candidates are an observation so the RL library stores them with the step and
can recompute log-probabilities on exactly the set the action was drawn from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from gaitnet_core.features import state_vector
from gaitnet_core.samplers import make_sampler
from gaitnet_core.terrain import inner_heights

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from gaitnet_sim.env.actions import FootstepControlAction


def footstep_action(env: "ManagerBasedRLEnv", name: str = "footstep") -> "FootstepControlAction":
    """The footstep action term, which owns the controller and reads the robot's state."""
    return env.action_manager.get_term(name)  # type: ignore[return-value]


def robot_state(env: "ManagerBasedRLEnv", features: list[str], action_name: str = "footstep") -> torch.Tensor:
    """(N, D) the named `gaitnet_core.features`, concatenated in order."""
    return state_vector(footstep_action(env, action_name).robot_state(), features)


def terrain_heights(
    env: "ManagerBasedRLEnv", unknown_height: float = -1.0, action_name: str = "footstep"
) -> torch.Tensor:
    """(N, L, *patch_size) terrain heights relative to each hip (m).

    Cells no ray returned from read `unknown_height` rather than -inf, so networks can
    consume them; it should sit below the reach band so it reads as a drop.
    """
    heights = footstep_action(env, action_name).terrain().heights
    return torch.where(torch.isfinite(heights), heights, torch.full_like(heights, unknown_height))


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
    observation = term.observation()
    valid = env.cfg.gaitnet.foothold_rules().valid(observation, term.spec)
    heights = inner_heights(observation.terrain.heights, term.grid)
    candidates = make_sampler(sampler, **(sampler_kwargs or {})).sample(valid, term.grid, heights=heights)
    return candidates.pack()
