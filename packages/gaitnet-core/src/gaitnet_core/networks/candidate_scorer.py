"""GaitNet's candidate-scoring actor: one shared state embedding, one embedding per
candidate, and a trunk that scores each (state, candidate) pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from gaitnet_core.candidates import Candidates
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.selection import Scores
from gaitnet_core.terrain import fill_unknown, sample_patch


def make_mlp(input_size: int, hidden_sizes: Sequence[int], output_size: int, activation=nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_size = input_size
    for h in hidden_sizes:
        layers += [nn.Linear(in_size, h), activation()]
        in_size = h
    layers.append(nn.Linear(in_size, output_size))
    return nn.Sequential(*layers)


@dataclass
class EncoderInputs:
    xyz: torch.Tensor
    """(N, L, K, 3) candidate positions in each leg's hip yaw frame (m)."""
    terrain: torch.Tensor | None
    """(N, L, *grid.patch_size) terrain heights relative to each hip, unknown cells filled;
    None for networks whose encoding doesn't read terrain."""
    grid: FootholdGrid | None
    crop_radius: int

    def leg_one_hot(self) -> torch.Tensor:
        n, l, k, _ = self.xyz.shape
        eye = torch.eye(l, device=self.xyz.device, dtype=self.xyz.dtype)
        return eye.view(1, l, 1, l).expand(n, l, k, l)


def _terrain_crop(inputs: EncoderInputs) -> torch.Tensor:
    """(N, L, K, (2r + 1)^2) terrain heights on a square of cells around each candidate,
    relative to the candidate's own height."""
    xyz, grid, r = inputs.xyz, inputs.grid, inputs.crop_radius
    steps = torch.arange(-r, r + 1, device=xyz.device, dtype=xyz.dtype) * grid.resolution
    offsets = torch.stack(torch.meshgrid(steps, steps, indexing="ij"), dim=-1).reshape(-1, 2)  # (P, 2)
    n, l, k, _ = xyz.shape
    points = (xyz[..., None, :2] + offsets).reshape(n, l, k * offsets.shape[0], 2)
    heights = sample_patch(inputs.terrain.unsqueeze(2), points, grid).reshape(n, l, k, -1)
    return heights - xyz[..., 2:3]


@dataclass(frozen=True)
class CandidateEncoding:
    encode: Callable[[EncoderInputs], torch.Tensor]
    """-> (N, L, K, dim)"""
    dim: Callable[[int, int], int]
    """(num_legs, crop_radius) -> dim"""
    uses_terrain: bool = False


CANDIDATE_FEATURES: dict[str, CandidateEncoding] = {
    "xy": CandidateEncoding(lambda c: torch.cat([c.leg_one_hot(), c.xyz[..., :2]], dim=-1), lambda l, r: l + 2),
    "xyz": CandidateEncoding(lambda c: torch.cat([c.leg_one_hot(), c.xyz], dim=-1), lambda l, r: l + 3),
    # the local-crop encoder: what the terrain looks like around the foothold
    "xyz_crop": CandidateEncoding(
        lambda c: torch.cat([c.leg_one_hot(), c.xyz, _terrain_crop(c)], dim=-1),
        lambda l, r: l + 3 + (2 * r + 1) ** 2,
        uses_terrain=True,
    ),
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
        grid: dict | None = None,
        crop_radius: int = 2,
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
            grid: The foothold grid (`FootholdGrid.to_dict()`) the terrain patches are on,
                for encodings that read terrain.
            crop_radius: The "xyz_crop" encoding reads (2 r + 1)^2 cells around a candidate.
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
            grid=grid,
            crop_radius=crop_radius,
        )
        self.encoding = CANDIDATE_FEATURES[candidate_features]
        if self.encoding.uses_terrain and grid is None:
            raise ValueError(f"candidate features '{candidate_features}' read terrain, which needs the foothold grid")
        self.grid = FootholdGrid.from_dict(grid) if grid is not None else None
        self.crop_radius = crop_radius
        self.duration_range = tuple(duration_range)
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self.use_bf16 = use_bf16

        self.shared_encoder = make_mlp(state_dim, shared_sizes[:-1], shared_sizes[-1])
        feature_dim = self.encoding.dim(num_legs, crop_radius)
        self.candidate_encoder = make_mlp(feature_dim, candidate_sizes[:-1], candidate_sizes[-1])
        self.noop_embedding = nn.Parameter(torch.randn(candidate_sizes[-1]))
        self.trunk = make_mlp(shared_sizes[-1] + candidate_sizes[-1], trunk_sizes[:-1], trunk_sizes[-1])
        self.logit_head = nn.Linear(trunk_sizes[-1], 1)
        self.duration_head = nn.Linear(trunk_sizes[-1], 1)

    @property
    def uses_terrain(self) -> bool:
        return self.encoding.uses_terrain

    def forward(self, state: torch.Tensor, candidates: Candidates, terrain: torch.Tensor | None = None) -> Scores:
        """Score every candidate and the no-op.

        Args:
            state: (N, state_dim) robot state vector
            candidates: N robots' candidates
            terrain: (N, L, *patch_size) terrain heights, -inf where unknown; read only by
                encodings that use terrain
        """
        n, l, k, _ = candidates.xyz.shape
        xyz = candidates.xyz.to(state.dtype)
        if self.uses_terrain:
            if terrain is None:
                raise ValueError("this network's candidate encoding reads terrain; pass the terrain patches")
            terrain = fill_unknown(terrain).to(state.dtype)
        else:
            terrain = None

        with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, enabled=self.use_bf16 and state.is_cuda):
            chunk = self.checkpoint_chunk_size
            if chunk and torch.is_grad_enabled() and n > chunk:
                terrain_chunks = terrain.split(chunk) if terrain is not None else [None] * len(state.split(chunk))
                outputs = [
                    checkpoint(self._forward, s, x, t, use_reentrant=False)
                    for s, x, t in zip(state.split(chunk), xyz.split(chunk), terrain_chunks)
                ]
                logits = torch.cat([o[0] for o in outputs])
                duration = torch.cat([o[1] for o in outputs])
            else:
                logits, duration = self._forward(state, xyz, terrain)

        logits, duration = logits.float(), duration.float()
        return Scores(
            step_logits=logits[:, :-1].reshape(n, l, k),
            noop_logit=logits[:, -1],
            duration=duration[:, :-1].reshape(n, l, k),
        )

    def _forward(
        self, state: torch.Tensor, xyz: torch.Tensor, terrain: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, S) state, (N, L, K, 3) candidates and terrain -> (N, L K + 1) logits and
        durations, the no-op last."""
        n, l, k, _ = xyz.shape
        features = self.encoding.encode(EncoderInputs(xyz, terrain, self.grid, self.crop_radius)).reshape(n, l * k, -1)
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
