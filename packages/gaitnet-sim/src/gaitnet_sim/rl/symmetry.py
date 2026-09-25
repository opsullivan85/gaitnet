"""Left/right symmetry in training: RSL-RL's symmetry extension, fitted to GaitNet.

RSL-RL (`rsl_rl.extensions.symmetry`, after Mittal et al., "Symmetry Considerations for
Learning Task Symmetric Robot Policies", ICRA 2024) takes a function that mirrors a batch of
observations and actions. With data augmentation it appends every PPO minibatch's mirror
image, so the actor and critic learn from both; the mirrored half reuses the original's
advantages, returns and old log-probabilities. `augment` is that function for the GaitNet
env: every observation group is mirrored term by term (`TERM_MIRRORS`, built on
`gaitnet_core.symmetry`) and the action's choice moves to the mirrored candidate.

RSL-RL's mirror loss, and the `symmetry` metric it logs, compare the actor's outputs by MSE.
`GaitNetActor`'s outputs are encoded choices, not a distribution, so `MirrorSymmetry`
replaces it with the KL between the mirrored policy and the policy at the mirrored
observation, and `SymmetricPPO` is PPO with that in place. The `symmetry` preset
(`gaitnet_sim.rl.agent_cfg`) selects both.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.extensions import Symmetry
from tensordict import TensorDict

from gaitnet_core.action_layout import EnvAction
from gaitnet_core.candidates import Candidates
from gaitnet_core.symmetry import (
    LEG_MIRROR,
    action_index_mirror,
    mirror_candidates,
    mirror_command,
    mirror_env_action,
    mirror_leg_maps,
    mirror_legs,
    mirror_state_vector,
)
from gaitnet_sim.env import observations


def _leg_summary(x: torch.Tensor) -> torch.Tensor:
    """(B, L * c * c) per-leg c x c summaries of the foothold grid."""
    n, num_legs = x.shape[0], len(LEG_MIRROR)
    cells = round((x.shape[-1] / num_legs) ** 0.5)
    return mirror_leg_maps(x.reshape(n, num_legs, cells, cells)).reshape(n, -1)


TERM_MIRRORS: dict[Callable, Callable[[torch.Tensor, dict], torch.Tensor]] = {
    observations.robot_state: lambda x, params: mirror_state_vector(x, params["features"]),
    observations.footstep_candidates: lambda x, params: mirror_candidates(Candidates.unpack(x)).pack(),
    observations.terrain_heights: lambda x, params: mirror_leg_maps(x),
    observations.base_command: lambda x, params: mirror_command(x),
    observations.foothold_fraction: lambda x, params: mirror_legs(x),
    observations.terrain_height_summary: lambda x, params: _leg_summary(x),
    observations.foothold_validity_summary: lambda x, params: _leg_summary(x),
    # the highest surface under the trunk's (symmetric) footprint
    observations.base_terrain_clearance: lambda x, params: x,
    observations.foot_contact_forces: lambda x, params: mirror_legs(x),
}
"""How each observation term mirrors, `(term output (B, ...), term params) -> mirrored`.
Every term of every group the env has must be here; `mirror_observations` refuses others."""


def _group_terms(env, group: str) -> list[tuple[str, Callable, dict, tuple[int, ...]]]:
    """(name, func, params, shape without the batch) of each of a group's terms, in order."""
    manager = env.unwrapped.observation_manager
    group_cfg = getattr(env.unwrapped.cfg.observations, group)
    names = manager.active_terms[group]
    shapes = manager.group_obs_term_dim[group]
    terms = []
    for name, shape in zip(names, shapes):
        term = getattr(group_cfg, name)
        if term.func not in TERM_MIRRORS:
            raise KeyError(
                f"observation term '{group}.{name}' ({getattr(term.func, '__name__', term.func)}) has no mirror;"
                " add one to gaitnet_sim.rl.symmetry.TERM_MIRRORS"
            )
        terms.append((name, term.func, term.params, tuple(shape)))
    return terms


def mirror_group(env, group: str, x: torch.Tensor) -> torch.Tensor:
    """(B, ...) one observation group's batch, mirrored."""
    terms = _group_terms(env, group)
    if len(terms) == 1:
        _, func, params, _ = terms[0]
        return TERM_MIRRORS[func](x, params)
    if any(len(shape) != 1 for *_, shape in terms):
        raise ValueError(f"group '{group}' concatenates terms that aren't flat, which can't be split to mirror")
    parts = x.split([shape[0] for *_, shape in terms], dim=-1)
    return torch.cat([TERM_MIRRORS[func](part, params) for part, (_, func, params, _) in zip(parts, terms)], dim=-1)


def mirror_observations(env, obs: TensorDict) -> TensorDict:
    """Every group of a batch of observations, mirrored."""
    mirrored = {group: mirror_group(env, group, obs[group]) for group in obs.keys()}
    return TensorDict(mirrored, batch_size=obs.batch_size, device=obs.device)


def candidate_layout(env) -> tuple[int, int]:
    """(legs, candidates per leg) of the env's candidate set, which lays out the action index."""
    manager = env.unwrapped.observation_manager
    for group in manager.active_terms:
        for _, func, _, shape in _group_terms(env, group):
            if func is observations.footstep_candidates:
                return shape[0], shape[1]
    raise ValueError("the env has no footstep_candidates observation term")


def mirror_actions(env, actions: torch.Tensor) -> torch.Tensor:
    """(B, action_layout.DIM) actions, each the mirror image of the original's choice."""
    num_legs, per_leg = candidate_layout(env)
    return mirror_env_action(EnvAction.decode(actions), num_legs, per_leg).encode()


@torch.no_grad()
def augment(
    env, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """RSL-RL's `data_augmentation_func`: the batch followed by its mirror image, (2B, ...)."""
    obs_aug = None if obs is None else torch.cat([obs, mirror_observations(env, obs)], dim=0)
    actions_aug = None if actions is None else torch.cat([actions, mirror_actions(env, actions)], dim=0)
    return obs_aug, actions_aug


METRIC_CHUNK_SIZE = 4096
"""Robots per actor pass when the mirrored batch is evaluated only for the logged metric."""


class MirrorSymmetry(Symmetry):
    """RSL-RL's symmetry extension with a mirror loss for GaitNet's categorical policy.

    The loss is KL(mirror of pi(. | s) || pi(. | mirror of s)), the full policy's KL as
    `GaitNetActor.get_kl_divergence` computes it, with the first (target) side detached as in
    RSL-RL's own loss. The duration std is left out of it: it is state-independent, so
    symmetric already, and the KL would otherwise push it up to shrink the duration term.

    With data augmentation, PPO has already evaluated the actor on both halves of the batch,
    so the loss costs nothing extra. Without, it evaluates the actor on the mirrored batch,
    under no_grad and in chunks unless the mirror loss is on. It is logged as `symmetry`
    either way.
    """

    def compute_loss(self, actor, batch, original_batch_size: int) -> torch.Tensor:
        if self.use_data_augmentation:
            params = actor.output_distribution_params
            original = tuple(p[:original_batch_size] for p in params)
            mirrored = tuple(p[original_batch_size:] for p in params)
        else:
            original = actor.output_distribution_params
            mirrored_obs = mirror_observations(self.env, batch.observations)
            if self.use_mirror_loss and torch.is_grad_enabled():
                mirrored = actor.distribution_params(actor.distribution_for(mirrored_obs))
            else:
                # the networks only checkpoint in chunks with gradients on; a whole minibatch
                # in one no_grad pass holds several GB of activations at once
                with torch.no_grad():
                    parts = [
                        actor.distribution_params(actor.distribution_for(chunk))
                        for chunk in mirrored_obs.split(METRIC_CHUNK_SIZE)
                    ]
                mirrored = tuple(torch.cat(p) for p in zip(*parts))
        num_legs, per_leg = candidate_layout(self.env)
        perm = action_index_mirror(num_legs, per_leg, device=original[0].device)
        log_probs, duration_mean, duration_std = original
        target = (log_probs[:, perm].detach(), duration_mean[:, perm].detach(), duration_std.detach())
        prediction = (mirrored[0], mirrored[1], mirrored[2].detach())
        loss = actor.get_kl_divergence(target, prediction).mean()
        return loss if self.use_mirror_loss else loss.detach()


class SymmetricPPO(PPO):
    """RSL-RL's PPO with `MirrorSymmetry` as its symmetry extension; PPO without a symmetry_cfg."""

    def __init__(self, *args, symmetry_cfg: dict | None = None, **kwargs):
        super().__init__(*args, symmetry_cfg=None, **kwargs)
        if symmetry_cfg is not None and (self.actor.is_recurrent or self.critic.is_recurrent):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = MirrorSymmetry(**symmetry_cfg) if symmetry_cfg else None
