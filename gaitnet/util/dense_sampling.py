"""Exhaustive footstep sampling for evaluation.

During training `FootstepCandidateSampler` draws `num_footstep_options` candidates
per leg from that leg's footstep scanner grid, so the chosen action is the best of
a random subset. At evaluation time there is no reason to subsample: the grid is
small enough to score every cell, which removes the sampling variance and gives
the policy's true argmax over the discretized foothold surface.

The option set built here has the same contract as
`FootstepCandidateSampler.get_footstep_options`: (leg, x, y) tuples, leg-major,
with a trailing no-op, and with filtered-out cells carrying the no-op encoding. That
means the scoring can go straight through `GaitnetActor.act_inference`, so the
per-leg ``log(N_valid)`` normalization in `masked_option_logits` is applied exactly
as it is in training.

This replaces `gaitnet.util.pga`, which optimized a handful of continuous samples
per leg with projected gradient ascent. PGA is left in the tree but unused.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import gaitnet.constants as const
from gaitnet.constants import NO_STEP
from gaitnet.gaitnet.components.footstep_candidate_sampler import FootstepCandidateSampler
from gaitnet.gaitnet.gaitnet import GaitnetActor
from gaitnet.simulation.cfg.footstep_scanner_constants import idx_to_xy

# Cap on (envs * options) scored in one forward pass. The trunk runs one row per
# option per env, so without this the peak activation grows with num_envs.
_max_rows_per_forward = 65536


def dense_footstep_options(obs: torch.Tensor) -> torch.Tensor:
    """Build the exhaustive footstep option set for each environment.

    Args:
        obs: (num_envs, robot_state_dim + total_robot_features) policy observation
            with the raw footstep scanner values still attached, i.e. the layout
            `FootstepCandidateSampler.get_footstep_options` expects.

    Returns:
        (num_envs, num_legs * H * W + 1, 3) options as (leg, x, y), leg-major
        with the no-op last. Cells rejected by the terrain / swing / minimum-contact
        filters carry the no-op encoding so `masked_option_logits` masks them out.
    """
    num_envs = obs.shape[0]
    num_legs = const.robot.num_legs
    height, width = (int(n) for n in const.footstep_scanner.grid_size)
    device = obs.device

    valid = FootstepCandidateSampler.valid_footholds(obs)  # (N, 4, H, W)
    invalid = ~valid.reshape(num_envs, num_legs * height * width)

    # cell centers in the hip frame, identical for every env and leg
    grid_idx = torch.stack(
        torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        ),
        dim=-1,
    )  # (H, W, 2)
    grid_xy = idx_to_xy(grid_idx)  # (H, W, 2)

    legs = torch.arange(num_legs, device=device, dtype=grid_xy.dtype)
    options = torch.empty((num_legs, height, width, 3), device=device, dtype=grid_xy.dtype)
    options[..., 0] = legs.view(num_legs, 1, 1)
    options[..., 1:3] = grid_xy
    options = options.reshape(1, num_legs * height * width, 3).repeat(num_envs, 1, 1)

    options[:, :, 0] = torch.where(invalid, float(NO_STEP), options[:, :, 0])
    options[:, :, 1:3] = torch.where(invalid.unsqueeze(-1), 0.0, options[:, :, 1:3])

    no_op = torch.zeros((num_envs, 1, 3), device=device, dtype=options.dtype)
    no_op[:, 0, 0] = NO_STEP
    return torch.cat([options, no_op], dim=1)


def options_to_policy_obs(
    robot_state: torch.Tensor, options: torch.Tensor
) -> torch.Tensor:
    """Assemble the observation the actor consumes from a robot state and option set.

    Mirrors `GaitNetObservationManager.footstep_options_to_one_hot` plus the
    flattening `_modify_obs` does, without depending on the isaaclab managers.

    Args:
        robot_state: (num_envs, robot_state_dim)
        options: (num_envs, num_options, 3) as (leg, x, y)

    Returns:
        (num_envs, robot_state_dim + num_options * footstep_option_dim)
    """
    # +1 remaps the leg index from [NO_STEP, 0..3] to [0..4]
    one_hot = F.one_hot(
        options[:, :, 0].long() + 1, num_classes=const.robot.num_legs + 1
    ).to(options.dtype)
    encoded = torch.cat([one_hot, options[:, :, 1:]], dim=-1)
    return torch.cat([robot_state, encoded.flatten(start_dim=1)], dim=1)


def dense_policy_obs(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap the footstep scanner values for the exhaustive option set.

    This is what `GaitNetObservationManager._modify_obs` does, but with every grid
    cell as a candidate instead of the sampler's random subset.

    Args:
        obs: (num_envs, robot_state_dim + total_robot_features) policy observation
            with the raw footstep scanner values still attached.

    Returns:
        options: (num_envs, num_options, 3) as (leg, x, y). Assign this to the
            observation manager's `footstep_options` so the action term can resolve
            the index the policy picks.
        policy_obs: (num_envs, robot_state_dim + num_options * footstep_option_dim)
            observation for the actor.
    """
    options = dense_footstep_options(obs)
    robot_state = obs[:, : const.gait_net.robot_state_dim]
    return options, options_to_policy_obs(robot_state, options)


@torch.no_grad()
def dense_footstep_actions(
    actor: GaitnetActor,
    obs: torch.Tensor,
    max_rows_per_forward: int = _max_rows_per_forward,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score every cell of every leg and pick the policy's best option.

    Args:
        actor: The GaitNet actor.
        obs: (num_envs, robot_state_dim + total_robot_features) policy observation
            with the raw footstep scanner values still attached.
        max_rows_per_forward: Envs are scored in chunks small enough to keep
            (envs * options) under this, bounding peak activation memory.

    Returns:
        options: (num_envs, num_options, 3) to assign to the observation manager's
            `footstep_options`, so the action term can resolve the chosen index.
        actions: (num_envs, 2) of (option index, duration), ready for `env.step`.
    """
    options, policy_obs = dense_policy_obs(obs)

    num_envs, num_options, _ = options.shape
    chunk = max(1, max_rows_per_forward // num_options)
    actions = torch.cat(
        [
            GaitnetActor.act_inference(actor, policy_obs[start : start + chunk])
            for start in range(0, num_envs, chunk)
        ]
    )
    return options, actions
