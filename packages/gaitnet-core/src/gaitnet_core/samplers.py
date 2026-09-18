"""Candidate samplers: turn a valid-foothold mask into a finite candidate set.

GaitNet scores a point, not a cell, so any sampler works with any trained network.
Selection (`gaitnet_core.selection`) corrects for how many candidates were drawn and, via
`log_q`, for non-uniform proposals, so the policy's step probability doesn't depend on
the sampler.
"""

from __future__ import annotations

from typing import Protocol

import torch

from gaitnet_core.candidates import Candidates
from gaitnet_core.grid import FootholdGrid


class CandidateSampler(Protocol):
    def sample(
        self,
        valid: torch.Tensor,
        grid: FootholdGrid,
        heights: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Candidates:
        """Draw candidates.

        Args:
            valid: (N, L, *grid.size) bool valid footholds
            heights: (N, L, *grid.size) cell heights relative to the hip, for candidate z.
                None gives z = 0.
            generator: random source, for reproducible sampling
        """
        ...


def _cells_to_candidates(
    cells: torch.Tensor,
    filled: torch.Tensor,
    grid: FootholdGrid,
    heights: torch.Tensor | None,
    offset: torch.Tensor | None = None,
) -> Candidates:
    """(N, L, K, 2) cell indices -> Candidates, z from the cell's height."""
    xy = grid.cell_to_xy(cells)
    if offset is not None:
        xy = xy + offset
    if heights is None:
        z = torch.zeros_like(xy[..., 0])
    else:
        n, l, k, _ = cells.shape
        flat = heights.reshape(n, l, -1)
        z = torch.gather(flat, 2, cells[..., 0] * grid.size[1] + cells[..., 1])
        z = torch.where(filled, z, torch.zeros_like(z))
    xyz = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
    xyz = torch.where(filled.unsqueeze(-1), xyz, torch.zeros_like(xyz))
    return Candidates(xyz=xyz, valid=filled, log_q=torch.zeros_like(z))


def _uniform_cells(
    valid: torch.Tensor, k: int, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Up to k distinct valid cells per leg, uniformly at random (without replacement).

    Returns:
        cells: (N, L, k, 2) long
        filled: (N, L, k) bool, False where the leg had fewer than k valid cells
    """
    n, l, h, w = valid.shape
    k = min(k, h * w)
    # the k smallest of iid uniform keys over the valid cells
    keys = torch.rand(valid.shape, device=valid.device, generator=generator)
    keys = keys.masked_fill(~valid, float("inf"))
    keys, flat = torch.topk(keys.view(n, l, h * w), k, largest=False)
    cells = torch.stack(torch.unravel_index(flat, (h, w)), dim=-1)
    return cells, ~torch.isinf(keys)


class UniformLattice:
    """k valid cell centres per leg, uniformly without replacement."""

    def __init__(self, per_leg: int = 64):
        self.per_leg = per_leg

    def sample(self, valid, grid, heights=None, generator=None) -> Candidates:
        cells, filled = _uniform_cells(valid, self.per_leg, generator)
        return _cells_to_candidates(cells, filled, grid, heights)


class UniformJitter:
    """k valid cells per leg uniformly without replacement, then a uniform point within
    each cell. Covers the continuous foothold surface, not just cell centres."""

    def __init__(self, per_leg: int = 64):
        self.per_leg = per_leg

    def sample(self, valid, grid, heights=None, generator=None) -> Candidates:
        cells, filled = _uniform_cells(valid, self.per_leg, generator)
        offset = torch.rand(
            (*cells.shape[:-1], 2), device=valid.device, generator=generator
        )
        offset = (offset - 0.5) * grid.resolution
        return _cells_to_candidates(cells, filled, grid, heights, offset)


class Dense:
    """Every cell centre of every leg. Exhaustive, so there is no sampling variance."""

    def sample(self, valid, grid, heights=None, generator=None) -> Candidates:
        n, l, h, w = valid.shape
        cells = torch.stack(
            torch.meshgrid(
                torch.arange(h, device=valid.device),
                torch.arange(w, device=valid.device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(1, 1, h * w, 2).expand(n, l, -1, -1)
        return _cells_to_candidates(cells, valid.reshape(n, l, h * w), grid, heights)


SAMPLERS: dict[str, type] = {
    "uniform_lattice": UniformLattice,
    "uniform_jitter": UniformJitter,
    "dense": Dense,
}


def make_sampler(name: str, **kwargs) -> CandidateSampler:
    return SAMPLERS[name](**kwargs)
