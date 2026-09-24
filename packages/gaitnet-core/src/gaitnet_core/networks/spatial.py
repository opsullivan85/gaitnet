"""D1, the dense spatial policy: a small CNN reads each leg's terrain patch, conditioned on the
robot state and the leg, and outputs a score map and a swing duration map over the candidate
grid (the patch's border is the convolutions' context).

A candidate's score is the map bilinearly interpolated at its (x, y). With dense sampling
that is scoring every cell; with sampled or refined candidates it is the same function at
other points, so the network stays sampler-agnostic and differentiable in the foothold. The
no-op is scored from the state and the pooled terrain features.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from gaitnet_core.candidates import Candidates
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.networks.candidate_scorer import make_mlp
from gaitnet_core.selection import Scores
from gaitnet_core.terrain import fill_unknown, sample_patch


class DenseSpatialCNN(nn.Module):
    uses_terrain = True

    def __init__(
        self,
        state_dim: int,
        grid: dict,
        num_legs: int = 4,
        channels: Sequence[int] = (16, 16, 16),
        state_sizes: Sequence[int] = (128, 64),
        noop_sizes: Sequence[int] = (64,),
        height_scale: float = 0.1,
        duration_range: tuple[float, float] = (0.1, 0.3),
        checkpoint_chunk_size: int | None = 1024,
        use_bf16: bool = True,
        fixed_duration: float | None = None,
    ):
        """
        Args:
            state_dim: Length of the robot state vector (see `gaitnet_core.features`).
            grid: The foothold grid (`FootholdGrid.to_dict()`) the terrain patches are on.
            channels: Output channels of each 3x3 convolution. The terrain features they
                compute don't depend on the state; the last layer's are modulated by the
                state and leg (FiLM) before the score and duration heads.
            state_sizes: The state encoder MLP; its last size is the embedding.
            noop_sizes: Hidden sizes of the no-op head.
            height_scale: Heights are divided by this before the first convolution (m).
            duration_range: Swing durations are squashed into this range (s).
            checkpoint_chunk_size: As for `CandidateScorer`: with gradients enabled, run and
                checkpoint the batch in chunks of this many robots. None disables it.
            use_bf16: Run under bf16 autocast on CUDA.
            fixed_duration: If set, every step gets this swing duration (s) and the map head
                outputs the score map only.
        """
        super().__init__()
        self.config = dict(
            state_dim=state_dim,
            grid=dict(grid),
            num_legs=num_legs,
            channels=list(channels),
            state_sizes=list(state_sizes),
            noop_sizes=list(noop_sizes),
            height_scale=height_scale,
            duration_range=list(duration_range),
            checkpoint_chunk_size=checkpoint_chunk_size,
            use_bf16=use_bf16,
            fixed_duration=fixed_duration,
        )
        self.fixed_duration = fixed_duration
        self.grid = FootholdGrid.from_dict(grid)
        self.height_scale = height_scale
        self.duration_range = tuple(duration_range)
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self.use_bf16 = use_bf16

        self.state_encoder = make_mlp(state_dim, state_sizes[:-1], state_sizes[-1])
        condition_dim = state_sizes[-1] + num_legs
        # Convolutions spend the patch's border cells as context rather than zero padding
        # their edges, down to the candidate grid; any further ones are padded.
        self.paddings = [0 if i < self.grid.border else 1 for i in range(len(channels))]
        # The first layer reads the height, plus planes of each cell's x and y in the grid
        # and the leg (which together fix its position relative to the hip). Those planes are the same for every
        # robot, so their share of the first convolution is one map per leg, computed once
        # per call (`position_conv`) rather than convolved for every robot.
        self.height_conv = nn.Conv2d(1, channels[0], kernel_size=3, padding=self.paddings[0])
        self.position_conv = nn.Conv2d(2 + num_legs, channels[0], kernel_size=3, padding=self.paddings[0], bias=False)
        self.convs = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            for in_channels, out_channels, padding in zip(channels[:-1], channels[1:], self.paddings[1:])
        )
        # only the last layer is conditioned: per-layer FiLM's elementwise work outweighed the
        # convolutions themselves
        self.film = nn.Linear(condition_dim, 2 * channels[-1])
        # score, and unless the duration is fixed, duration (unsquashed)
        self.map_head = nn.Conv2d(channels[-1], 1 if fixed_duration is not None else 2, kernel_size=1)
        self.noop_head = make_mlp(state_sizes[-1] + channels[-1], noop_sizes, 1)

    def forward(self, state: torch.Tensor, candidates: Candidates, terrain: torch.Tensor | None = None) -> Scores:
        """Score every candidate and the no-op.

        Args:
            state: (N, state_dim) robot state vector
            candidates: N robots' candidates
            terrain: (N, L, *grid.patch_size) terrain heights relative to each hip, -inf
                where unknown
        """
        if terrain is None:
            raise ValueError("DenseSpatialCNN reads the terrain; pass the terrain patches")
        if tuple(terrain.shape[-2:]) != self.grid.patch_size:
            raise ValueError(f"expected terrain patches of {self.grid.patch_size}, got {tuple(terrain.shape[-2:])}")
        terrain = fill_unknown(terrain).to(state.dtype)
        xy = candidates.xyz[..., :2].to(state.dtype)

        with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, enabled=self.use_bf16 and state.is_cuda):
            chunk = self.checkpoint_chunk_size
            n = state.shape[0]
            if chunk and torch.is_grad_enabled() and n > chunk:
                outputs = [
                    checkpoint(self._forward, s, x, t, use_reentrant=False)
                    for s, x, t in zip(state.split(chunk), xy.split(chunk), terrain.split(chunk))
                ]
                logits, duration, noop = (torch.cat(parts) for parts in zip(*outputs))
            else:
                logits, duration, noop = self._forward(state, xy, terrain)

        return Scores(step_logits=logits.float(), noop_logit=noop.float(), duration=duration.float())

    def _forward(
        self, state: torch.Tensor, xy: torch.Tensor, terrain: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (N, L, K) logits, (N, L, K) durations, (N,) no-op logits"""
        n, l, size_x, size_y = terrain.shape
        embedding = self.state_encoder(state)  # (N, E)
        legs = torch.eye(l, device=state.device, dtype=terrain.dtype)
        condition = torch.cat(
            [embedding.unsqueeze(1).expand(n, l, -1), legs.to(embedding.dtype).expand(n, l, l)], dim=-1
        ).reshape(n * l, -1)

        coords_x = torch.linspace(-1.0, 1.0, size_x, device=state.device, dtype=terrain.dtype)
        coords_y = torch.linspace(-1.0, 1.0, size_y, device=state.device, dtype=terrain.dtype)
        coords = torch.stack(torch.meshgrid(coords_x, coords_y, indexing="ij"))  # (2, X, Y)
        planes = torch.cat([coords.expand(l, 2, size_x, size_y), legs.view(l, l, 1, 1).expand(l, l, size_x, size_y)], dim=1)
        position = self.position_conv(planes)  # (L, C, X', Y')

        heights = (terrain / self.height_scale).reshape(n * l, 1, size_x, size_y)
        x = self.height_conv(heights)
        x.view(n, l, *x.shape[1:]).add_(position)
        for conv in self.convs:
            x = conv(x.relu_())
        scale, shift = self.film(condition).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        x = x.mul(1 + scale).add_(shift).relu_()

        maps = self.map_head(x)
        maps = maps.reshape(n, l, -1, *maps.shape[-2:])
        sampled = sample_patch(maps.float(), xy.float(), self.grid)  # (N, L, K, 1 or 2)
        if self.fixed_duration is not None:
            duration = torch.full_like(sampled[..., 0], self.fixed_duration)
        else:
            low, high = self.duration_range
            duration = torch.sigmoid(sampled[..., 1]) * (high - low) + low

        pooled = x.mean(dim=(-2, -1)).reshape(n, l, -1).mean(dim=1)  # (N, C)
        noop = self.noop_head(torch.cat([embedding, pooled.to(embedding.dtype)], dim=-1)).squeeze(-1)
        return sampled[..., 0], duration, noop
