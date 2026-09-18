"""Which cells of a terrain patch a foot may be placed on."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gaitnet_core.grid import FootholdGrid
from gaitnet_core.robot_spec import RobotSpec


def _window_max(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Max over a (2 radius + 1)^2 window, same size output (edges padded with -inf)."""
    if radius == 0:
        return x
    n, l, h, w = x.shape
    out = F.max_pool2d(
        x.reshape(n * l, 1, h, w), kernel_size=2 * radius + 1, stride=1, padding=radius
    )
    return out.reshape(n, l, h, w)


def valid_footholds(
    heights: torch.Tensor,
    spec: RobotSpec,
    grid: FootholdGrid,
    step_threshold: float = 0.02,
    edge_margin: int = 2,
) -> torch.Tensor:
    """Cells within the leg's reach and at least `edge_margin` cells from any height step.

    Args:
        heights: (N, L, *grid.patch_size) terrain heights relative to each hip (m), -inf
            where unknown.
        step_threshold: A height difference above this between neighbouring cells is an
            edge (m). Both cells of the step count as edge cells.
        edge_margin: Cells within this Chebyshev distance of an edge cell are invalid.
            Needs `grid.border >= edge_margin + 1` to see edges just outside the grid.

    Returns:
        (N, L, *grid.size) bool, True where a foot may be placed.
    """
    if tuple(heights.shape[-2:]) != grid.patch_size:
        raise ValueError(f"expected terrain patches of {grid.patch_size}, got {tuple(heights.shape[-2:])}")

    lowest, highest = spec.reach_band
    reachable = (heights >= lowest) & (heights <= highest)

    # height range over each cell's 3x3 neighbourhood; unknown (-inf) cells make it inf
    finite = torch.where(torch.isfinite(heights), heights, torch.full_like(heights, -1e6))
    local_max = _window_max(finite, 1)
    local_min = -_window_max(-finite, 1)
    edge = (local_max - local_min) > step_threshold
    near_edge = _window_max(edge.float(), edge_margin) > 0

    valid = reachable & ~near_edge
    b = grid.border
    return valid[..., b : b + grid.size[0], b : b + grid.size[1]]


def inner_heights(heights: torch.Tensor, grid: FootholdGrid) -> torch.Tensor:
    """Crop a (…, *patch_size) terrain patch to the (…, *size) candidate grid."""
    b = grid.border
    return heights[..., b : b + grid.size[0], b : b + grid.size[1]]
