"""Exhaustive (dense) candidate scoring for evaluation on the current sim layer.

Training samples a random subset of each leg's footholds. Evaluation scores every cell and
picks deterministically with `gaitnet_core.selection.select_deterministic`, which gates on
each leg's average score rather than on a single candidate, so it isn't biased toward the
no-op by the larger candidate count.
"""

from __future__ import annotations

from pathlib import Path
import re

import torch

import gaitnet.constants as const
from gaitnet.gaitnet.env_cfg.observations_utils import legacy_valid_footholds
from gaitnet_core.candidates import Candidates
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.networks import CandidateScorer
from gaitnet_core.samplers import Dense
from gaitnet_core.selection import Scores, select_deterministic

GRID = FootholdGrid(
    resolution=const.footstep_scanner.grid_resolution,
    size=tuple(int(n) for n in const.footstep_scanner.grid_size),
    border=0,
)

# Cap on (envs * candidates) scored in one forward pass, bounding peak activation memory.
_max_rows_per_forward = 65536


def load_actor(checkpoint_path: Path, device: torch.device) -> CandidateScorer:
    """The actor from an rsl_rl checkpoint of `gaitnet.gaitnet.train`, set up for evaluation."""
    actor = CandidateScorer(
        state_dim=const.gait_net.robot_state_dim,
        num_legs=const.robot.num_legs,
        candidate_features="xy",
        shared_sizes=[128, 128, 128],
        candidate_sizes=[64, 64],
        trunk_sizes=[128, 128, 128],
        # dense scoring compares thousands of nearby candidates, where bf16's ~3
        # significant digits would perturb the choice; checkpointing only helps backward
        checkpoint_chunk_size=None,
        use_bf16=False,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)["model_state_dict"]
    actor.load_state_dict({re.sub(r"^actor\.", "", k): v for k, v in state_dict.items() if k.startswith("actor.")})
    return actor.to(device).eval()


def dense_candidates(raw_obs: torch.Tensor) -> Candidates:
    """Every valid cell of every leg.

    Args:
        raw_obs: (N, robot_state_dim + footstep scanner features) policy observation with
            the scanner values still attached.
    """
    return Dense().sample(legacy_valid_footholds(raw_obs), GRID)


def score(actor: CandidateScorer, state: torch.Tensor, candidates: Candidates) -> Scores:
    per_env = candidates.num_legs * candidates.per_leg + 1
    chunk = max(1, _max_rows_per_forward // per_env)
    parts = [actor(state[i : i + chunk], candidates[i : i + chunk]) for i in range(0, state.shape[0], chunk)]
    return Scores(
        step_logits=torch.cat([p.step_logits for p in parts]),
        noop_logit=torch.cat([p.noop_logit for p in parts]),
        duration=torch.cat([p.duration for p in parts]),
    )


@torch.no_grad()
def dense_actions(actor: CandidateScorer, raw_obs: torch.Tensor) -> tuple[Candidates, torch.Tensor]:
    """Score every cell and pick deterministically.

    Returns:
        candidates: assign to the observation manager's `candidates` so the action term
            resolves the chosen index against the dense set
        actions: (N, 2) of (candidate index, duration), ready for `env.step`
    """
    candidates = dense_candidates(raw_obs)
    state = raw_obs[:, : const.gait_net.robot_state_dim]
    selection = select_deterministic(score(actor, state, candidates), candidates)
    return candidates, torch.stack([selection.index.float(), selection.duration], dim=-1)


def policy_obs(raw_obs: torch.Tensor, candidates: Candidates) -> torch.Tensor:
    """The policy observation the actor-critic consumes: state then packed candidates."""
    state = raw_obs[:, : const.gait_net.robot_state_dim]
    return torch.cat([state, candidates.pack().flatten(start_dim=1)], dim=1)
