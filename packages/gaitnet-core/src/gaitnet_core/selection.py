"""Turning network scores into an action.

The policy is defined over a continuous space: the no-op, or leg l at foothold x with
a swing duration. The network scores points, f(s, l, x), plus a no-op score f_0(s). The
step probability of a leg is the *average* of exp(f) over its valid footholds, so:

    P(no-op)   ∝ exp(f_0)
    P(leg l)   ∝ mean over valid x of exp(f(s, l, x))
    P(x | l)   ∝ exp(f(s, l, x))

A candidate set is a Monte-Carlo sample of each leg's footholds. With proposal density
q (relative to uniform over the valid set) and N_l valid samples, the estimator of the
mean is (1 / N_l) Σ exp(f_i) / q_i, so each candidate's logit is f_i - log q_i - log N_l.
That makes the stochastic policy the same, in expectation, for any sampler.

The deterministic policy picks the most likely leg first, then the best foothold within
it. Taking an argmax over individual candidate logits instead compares one foothold,
penalized by log N_l, against the no-op, which biases toward the no-op more and more as
the number of candidates grows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical, Normal

from gaitnet_core.candidates import Candidates


@dataclass
class Scores:
    """Network outputs for one candidate set."""

    step_logits: torch.Tensor
    """(N, L, K) f(s, l, x) for each candidate."""
    noop_logit: torch.Tensor
    """(N,) f_0(s)."""
    duration: torch.Tensor
    """(N, L, K) mean swing duration for each candidate (s)."""


def corrected_step_logits(scores: Scores, candidates: Candidates) -> torch.Tensor:
    """(N, L, K) f - log q - log N_valid, -inf for invalid slots."""
    num_valid = candidates.num_valid().clamp(min=1).to(scores.step_logits.dtype)
    logits = scores.step_logits - candidates.log_q - torch.log(num_valid).unsqueeze(-1)
    return logits.masked_fill(~candidates.valid, float("-inf"))


def action_logits(scores: Scores, candidates: Candidates) -> torch.Tensor:
    """(N, L * K + 1) categorical logits over the flat action index, no-op last."""
    step = corrected_step_logits(scores, candidates)
    return torch.cat([step.flatten(1), scores.noop_logit.unsqueeze(-1)], dim=-1)


def leg_marginals(scores: Scores, candidates: Candidates) -> torch.Tensor:
    """(N, L) log of each leg's unnormalized step probability, -inf if it has no valid
    candidate. Comparable to `scores.noop_logit`."""
    return torch.logsumexp(corrected_step_logits(scores, candidates), dim=-1)


@dataclass
class Selection:
    index: torch.Tensor
    """(N,) flat action index, `candidates.noop_index` for the no-op."""
    duration: torch.Tensor
    """(N,) swing duration (s), 0 for the no-op."""


def select_deterministic(scores: Scores, candidates: Candidates) -> Selection:
    """Most likely leg (or the no-op), then that leg's highest scoring foothold."""
    marginals = leg_marginals(scores, candidates)  # (N, L)
    gate = torch.argmax(torch.cat([marginals, scores.noop_logit.unsqueeze(-1)], dim=-1), dim=-1)
    is_step = gate < candidates.num_legs
    leg = torch.where(is_step, gate, torch.zeros_like(gate))

    rows = torch.arange(gate.shape[0], device=gate.device)
    leg_logits = scores.step_logits[rows, leg].masked_fill(~candidates.valid[rows, leg], float("-inf"))
    slot = torch.argmax(leg_logits, dim=-1)

    index = torch.where(is_step, leg * candidates.per_leg + slot, torch.full_like(gate, candidates.noop_index))
    duration = torch.where(is_step, scores.duration[rows, leg, slot], torch.zeros_like(scores.noop_logit))
    return Selection(index=index, duration=duration)


class FootstepDistribution:
    """The stochastic policy: categorical over (candidates + no-op), and a Gaussian swing
    duration for the chosen candidate."""

    def __init__(self, scores: Scores, candidates: Candidates, duration_std: torch.Tensor | None):
        """`duration_std` None means the duration is fixed by the network: it is the mean,
        never sampled, and not part of the log-probability."""
        self.scores = scores
        self.candidates = candidates
        self.duration_std = duration_std
        self.categorical = Categorical(logits=action_logits(scores, candidates))
        # (N, L * K + 1), the no-op's entry is never used
        self._duration_mean = torch.cat(
            [scores.duration.flatten(1), torch.zeros_like(scores.noop_logit).unsqueeze(-1)], dim=-1
        )

    def _is_step(self, index: torch.Tensor) -> torch.Tensor:
        return index < self.candidates.noop_index

    def _duration(self, index: torch.Tensor) -> Normal:
        mean = self._duration_mean.gather(-1, index.long().unsqueeze(-1)).squeeze(-1)
        return Normal(mean, self.duration_std.expand_as(mean))

    def sample(self) -> Selection:
        index = self.categorical.sample()
        if self.duration_std is None:
            duration = self._duration_mean.gather(-1, index.long().unsqueeze(-1)).squeeze(-1)
        else:
            duration = self._duration(index).sample()
        return Selection(index=index, duration=torch.where(self._is_step(index), duration, 0.0))

    def log_prob(self, selection: Selection) -> torch.Tensor:
        """(N,) joint log-probability. A no-op's duration is never used, so it isn't part
        of the action."""
        index = selection.index.long()
        if self.duration_std is None:
            return self.categorical.log_prob(index)
        duration_log_prob = self._duration(index).log_prob(selection.duration)
        duration_log_prob = torch.where(self._is_step(index), duration_log_prob, 0.0)
        return self.categorical.log_prob(index) + duration_log_prob

    def entropy(self) -> torch.Tensor:
        """(N,) entropy of the discrete choice only. The duration std is learned, and a
        Normal's entropy grows with log(std) regardless of return, so including it would
        push the duration noise up."""
        return self.categorical.entropy()

    def deterministic(self) -> Selection:
        return select_deterministic(self.scores, self.candidates)

    @property
    def step_probability(self) -> torch.Tensor:
        """(N,) probability of stepping rather than holding."""
        return 1.0 - self.categorical.probs[:, -1]
