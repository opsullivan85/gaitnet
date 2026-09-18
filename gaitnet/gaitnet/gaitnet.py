"""rsl_rl adapter over the gaitnet_core policy, for the current (pre-P3) sim layer.

The policy observation is the robot state followed by the packed candidate set, see
`GaitNetObservationManager`. Actions are (flat candidate index, swing duration).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic

import gaitnet.constants as const
from gaitnet import get_logger
from gaitnet_core.candidates import Candidates
from gaitnet_core.networks import CandidateScorer, Critic
from gaitnet_core.selection import FootstepDistribution, Selection, select_deterministic

logger = get_logger()


def split_policy_obs(observations: torch.Tensor) -> tuple[torch.Tensor, Candidates]:
    """(N, robot_state_dim + L * K * 5) policy observation -> state vector and candidates."""
    num_envs = observations.shape[0]
    state = observations[:, : const.gait_net.robot_state_dim]
    packed = observations[:, const.gait_net.robot_state_dim :].reshape(
        num_envs, const.robot.num_legs, -1, Candidates.PACKED_DIM
    )
    return state, Candidates.unpack(packed)


def act_inference(actor: CandidateScorer, observations: torch.Tensor) -> torch.Tensor:
    """Deterministic (candidate index, duration) actions, (N, 2)."""
    state, candidates = split_policy_obs(observations)
    selection = select_deterministic(actor(state, candidates), candidates)
    return torch.stack([selection.index.float(), selection.duration], dim=-1)


class GaitnetActorCritic(ActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor: CandidateScorer,
        critic: Critic,
        episode_info: dict[str, Any] | None = None,
        duration_std: float = 0.05,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        **kwargs,
    ):
        """
        Args:
            episode_info: Shared dictionary to dump per-step metrics into
            duration_std: Initial duration std, then learned
        """
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.episode_info = episode_info
        nn.Module.__init__(self)
        if kwargs:
            logger.warning(f"GaitnetActorCritic received unused kwargs: {kwargs}")

        self.actor = actor
        self.critic = critic
        # log-parameterized so it can't go negative and collapse the distribution
        self.duration_log_std = nn.Parameter(torch.log(torch.tensor(duration_std, dtype=torch.float32)))
        self.distribution: FootstepDistribution | None = None

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def duration_std(self) -> torch.Tensor:
        return self.duration_log_std.exp()

    def _require_distribution(self) -> FootstepDistribution:
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution first.")
        return self.distribution

    @property
    def action_mean(self):
        """(N, 2) deterministic action, for logging."""
        selection = self._require_distribution().deterministic()
        return torch.stack([selection.index.float(), selection.duration], dim=-1)

    @property
    def action_std(self):
        """(N, 2), 1 for the discrete choice and the learned duration std."""
        dist = self._require_distribution()
        n = dist.candidates.num_robots
        return torch.stack(
            [torch.ones(n, device=self.duration_std.device), self.duration_std.detach().expand(n)], dim=-1
        )

    @property
    def entropy(self):
        return self._require_distribution().entropy()

    def update_distribution(self, observations):
        state, candidates = split_policy_obs(observations["policy"])
        scores = self.actor(state, candidates)
        self.distribution = FootstepDistribution(scores, candidates, self.duration_std)

        if self.episode_info is not None:
            valid = candidates.valid
            self.episode_info["step_prob"] = self.distribution.step_probability.mean().item()
            self.episode_info["valid_options"] = valid.sum(dim=(1, 2)).float().mean().item()
            # spread of the scores within each leg's valid candidates
            logits = scores.step_logits.masked_fill(~valid, float("nan"))
            spread = torch.nanmean((logits - torch.nanmean(logits, dim=-1, keepdim=True)) ** 2, dim=-1).sqrt()
            spread = spread[valid.any(dim=-1)]
            self.episode_info["leg_option_std"] = spread.mean().item() if spread.numel() else 0.0
            self.episode_info["duration_std_param"] = self.duration_std.item()

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        selection = self.distribution.sample()  # type: ignore

        if self.episode_info is not None:
            is_step = selection.index != self.distribution.candidates.noop_index  # type: ignore
            steps = selection.duration[is_step]
            self.episode_info["ops_per_step"] = is_step.float().mean().item()
            self.episode_info["duration_mean"] = steps.mean().item() if steps.numel() else 0.0
            self.episode_info["duration_std"] = steps.std(unbiased=False).item() if steps.numel() else 0.0

        return torch.stack([selection.index.float(), selection.duration], dim=-1)

    def get_actions_log_prob(self, actions):
        selection = Selection(index=actions[:, 0].round().long(), duration=actions[:, 1])
        return self._require_distribution().log_prob(selection)

    def act_inference(self, observations):
        return act_inference(self.actor, observations["policy"])

    def evaluate(self, critic_observations, **kwargs):
        state = critic_observations["policy"][:, : const.gait_net.robot_state_dim]
        return self.critic(state)
