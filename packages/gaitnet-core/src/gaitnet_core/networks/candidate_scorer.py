"""GaitNet's candidate-scoring actor: one shared state embedding, one embedding per
candidate, and a trunk that scores each (state, candidate) pair."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from gaitnet_core.candidates import Candidates
from gaitnet_core.selection import Scores


def make_mlp(input_size: int, hidden_sizes: Sequence[int], output_size: int, activation=nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_size = input_size
    for h in hidden_sizes:
        layers += [nn.Linear(in_size, h), activation()]
        in_size = h
    layers.append(nn.Linear(in_size, output_size))
    return nn.Sequential(*layers)


def _leg_one_hot(candidates: Candidates) -> torch.Tensor:
    n, l, k, _ = candidates.xyz.shape
    eye = torch.eye(l, device=candidates.xyz.device, dtype=candidates.xyz.dtype)
    return eye.view(1, l, 1, l).expand(n, l, k, l)


CANDIDATE_FEATURES: dict[str, tuple[Callable[[Candidates], torch.Tensor], Callable[[int], int]]] = {
    # name: (candidates -> (N, L, K, F), num_legs -> F)
    "xy": (lambda c: torch.cat([_leg_one_hot(c), c.xyz[..., :2]], dim=-1), lambda l: l + 2),
    "xyz": (lambda c: torch.cat([_leg_one_hot(c), c.xyz], dim=-1), lambda l: l + 3),
}
"""Per-candidate input features. Add an entry to try a new encoding."""


class CandidateScorer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_legs: int = 4,
        candidate_features: str = "xyz",
        shared_sizes: Sequence[int] = (128, 128, 128),
        candidate_sizes: Sequence[int] = (64, 64),
        trunk_sizes: Sequence[int] = (128, 128, 128),
        duration_range: tuple[float, float] = (0.1, 0.3),
        checkpoint_chunk_size: int | None = 1024,
        use_bf16: bool = True,
    ):
        """
        Args:
            state_dim: Length of the robot state vector (see `gaitnet_core.features`).
            candidate_features: Key of `CANDIDATE_FEATURES`.
            duration_range: Swing durations are squashed into this range (s).
            checkpoint_chunk_size: When gradients are enabled, split the batch into chunks
                of this size and checkpoint each, so backward holds one chunk's activations
                at a time. Memory otherwise grows with batch_size * num_candidates. None
                disables it.
            use_bf16: Run under bf16 autocast on CUDA.
        """
        super().__init__()
        self.config = dict(
            state_dim=state_dim,
            num_legs=num_legs,
            candidate_features=candidate_features,
            shared_sizes=list(shared_sizes),
            candidate_sizes=list(candidate_sizes),
            trunk_sizes=list(trunk_sizes),
            duration_range=list(duration_range),
            checkpoint_chunk_size=checkpoint_chunk_size,
            use_bf16=use_bf16,
        )
        self.encode_candidates, feature_dim = CANDIDATE_FEATURES[candidate_features]
        self.duration_range = tuple(duration_range)
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self.use_bf16 = use_bf16

        self.shared_encoder = make_mlp(state_dim, shared_sizes[:-1], shared_sizes[-1])
        self.candidate_encoder = make_mlp(feature_dim(num_legs), candidate_sizes[:-1], candidate_sizes[-1])
        self.noop_embedding = nn.Parameter(torch.randn(candidate_sizes[-1]))
        self.trunk = make_mlp(shared_sizes[-1] + candidate_sizes[-1], trunk_sizes[:-1], trunk_sizes[-1])
        self.logit_head = nn.Linear(trunk_sizes[-1], 1)
        self.duration_head = nn.Linear(trunk_sizes[-1], 1)

    def forward(self, state: torch.Tensor, candidates: Candidates, terrain: torch.Tensor | None = None) -> Scores:
        """Score every candidate and the no-op.

        Args:
            state: (N, state_dim) robot state vector
            candidates: N robots' candidates
            terrain: (N, L, *patch_size) terrain heights, unused by this network
        """
        n, l, k, _ = candidates.xyz.shape
        features = self.encode_candidates(candidates).reshape(n, l * k, -1).to(state.dtype)

        with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, enabled=self.use_bf16 and state.is_cuda):
            chunk = self.checkpoint_chunk_size
            if chunk and torch.is_grad_enabled() and n > chunk:
                outputs = [
                    checkpoint(self._forward, s, f, use_reentrant=False)
                    for s, f in zip(state.split(chunk), features.split(chunk))
                ]
                logits = torch.cat([o[0] for o in outputs])
                duration = torch.cat([o[1] for o in outputs])
            else:
                logits, duration = self._forward(state, features)

        logits, duration = logits.float(), duration.float()
        return Scores(
            step_logits=logits[:, :-1].reshape(n, l, k),
            noop_logit=logits[:, -1],
            duration=duration[:, :-1].reshape(n, l, k),
        )

    def _forward(self, state: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, S) state and (N, M, F) candidate features -> (N, M + 1) logits and durations,
        the no-op last."""
        shared = self.shared_encoder(state)  # (N, E)
        per_candidate = self.candidate_encoder(features)  # (N, M, U)
        noop = self.noop_embedding.to(per_candidate.dtype).expand(per_candidate.shape[0], 1, -1)
        per_candidate = torch.cat([per_candidate, noop], dim=1)  # (N, M + 1, U)

        trunk_in = torch.cat([shared.unsqueeze(1).expand(-1, per_candidate.shape[1], -1), per_candidate], dim=-1)
        trunk_out = self.trunk(trunk_in)
        logits = self.logit_head(trunk_out).squeeze(-1)

        low, high = self.duration_range
        duration = torch.sigmoid(self.duration_head(trunk_out).squeeze(-1).float()) * (high - low) + low
        return logits, duration
