"""Value function over the robot state (and, for a privileged critic, extra sim-only inputs)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from gaitnet_core.networks.candidate_scorer import make_mlp


class Critic(nn.Module):
    """MLP value function.

    It takes only the state, not the candidates: those are i.i.d. draws given the state
    (their order comes from the sampler), so they carry ~no information about V(s).
    """

    def __init__(self, state_dim: int, hidden_sizes: Sequence[int] = (64, 64, 64, 64, 64, 64)):
        super().__init__()
        self.config = dict(state_dim=state_dim, hidden_sizes=list(hidden_sizes))
        self.net = make_mlp(state_dim, hidden_sizes, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(N, state_dim) -> (N, 1)"""
        return self.net(state)
